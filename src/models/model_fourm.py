"""4M model adapters for real and mock activation extraction."""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


class FourMMockEncoder:
    """Deterministic mock encoder used for smoke tests."""

    def __init__(self, layers: list[str], embedding_dim: int) -> None:
        """Store layer names and embedding size."""
        self.layers = layers
        self.embedding_dim = embedding_dim

    def encode_path(self, file_path: Path, modality: str) -> dict[str, np.ndarray]:
        """Generate deterministic mock vectors for one file."""
        seed_material = f"{file_path.as_posix()}::{modality}"
        seed = int(hashlib.md5(seed_material.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)
        return {
            layer: rng.standard_normal(self.embedding_dim).astype(np.float32)
            for layer in self.layers
        }


class FourMEncoder:
    """4M-backed encoder wrapper for real feature extraction."""

    def __init__(
        self,
        model_dir: Path,
        hf_repo: str,
        layers: list[str] | str,
        runtime_cfg: dict[str, Any],
        model_cfg: dict[str, Any],
    ) -> None:
        """Initialize model, tokenizers, and runtime settings."""
        self.model_dir = model_dir
        self.hf_repo = hf_repo
        self.layers = layers
        self.runtime_cfg = runtime_cfg
        self.model_cfg = model_cfg
        self._torch = self._import_torch()
        self._model = self._load_fourm_model()
        self._modality_info = self._load_modality_info()
        self._init_full_input_modality = self._load_init_input_helper()
        self._device = self._select_device()
        self._rgb_input_mode = str(self.model_cfg.get("rgb_input_mode", "pixel")).lower()
        if self._rgb_input_mode not in {"pixel", "tokenized"}:
            raise ValueError("model.rgb_input_mode must be one of: pixel, tokenized")
        self._rgb_tokenizer = self._load_rgb_tokenizer()
        self._depth_tokenizer = self._load_depth_tokenizer()
        self._normal_tokenizer = self._load_normal_tokenizer()
        # self._semseg_tokenizer = self._load_semseg_tokenizer()  # semseg disabled
        self._model = self._move_and_wrap_model(self._model)
        self._model.eval()

    def _import_torch(self):
        """Import torch and raise a clear dependency error."""
        try:
            import torch
        except ImportError as exc:
            raise ImportError(
                "torch is required for model.backend=fourm. Install dependencies first."
            ) from exc
        return torch

    def _select_device(self):
        """Resolve the runtime device from config and availability."""
        device_cfg = str(self.runtime_cfg.get("device", "auto")).lower()
        if device_cfg == "auto":
            return self._torch.device("cuda" if self._torch.cuda.is_available() else "cpu")
        if device_cfg in {"cuda", "gpu"}:
            if not self._torch.cuda.is_available():
                raise RuntimeError("CUDA requested but no GPU is available.")
            return self._torch.device("cuda")
        return self._torch.device("cpu")

    def _load_fourm_model(self):
        """Load the 4M checkpoint from local path or Hugging Face."""
        try:
            from fourm.models.fm import FM
        except ImportError as exc:
            raise ImportError(
                "fourm package is missing. Install from apple/ml-4m before using backend=fourm."
            ) from exc

        local_hint = str(self.model_dir)
        if Path(local_hint).exists():
            try:
                return FM.from_pretrained(local_hint)
            except Exception:
                pass
        return FM.from_pretrained(self.hf_repo)

    def _load_modality_info(self):
        """Load 4M modality metadata used by input builders."""
        from fourm.data.modality_info import MODALITY_INFO

        return MODALITY_INFO

    def _load_init_input_helper(self):
        """Load helper that fills masks for modality inputs."""
        from fourm.models.generate import init_full_input_modality

        return init_full_input_modality

    def _load_depth_tokenizer(self):
        """Load the depth tokenizer when configured."""
        depth_repo = self.model_cfg.get("tokenizers", {}).get("depth_repo")
        if not depth_repo:
            return None
        from fourm.vq.vqvae import DiVAE

        tok = DiVAE.from_pretrained(depth_repo).to(self._device)
        tok.eval()
        return tok

    def _load_normal_tokenizer(self):
        """Load the surface-normal tokenizer when configured."""
        normal_repo = self.model_cfg.get("tokenizers", {}).get("normal_repo")
        if not normal_repo:
            return None
        from fourm.vq.vqvae import DiVAE

        tok = DiVAE.from_pretrained(normal_repo).to(self._device)
        tok.eval()
        return tok


    def _load_rgb_tokenizer(self):
        """Load the RGB tokenizer for tokenized RGB mode."""
        if self._rgb_input_mode != "tokenized":
            return None
        rgb_repo = self.model_cfg.get("tokenizers", {}).get("rgb_repo")
        if not rgb_repo:
            raise RuntimeError(
                "RGB tokenized mode requires model.tokenizers.rgb_repo in config."
            )
        from fourm.vq.vqvae import DiVAE

        tok = DiVAE.from_pretrained(rgb_repo).to(self._device)
        tok.eval()
        return tok

    def _move_and_wrap_model(self, model):
        """Move model to target device and apply multi-GPU wrapper."""
        model = model.to(self._device)
        strategy = str(self.runtime_cfg.get("multi_gpu_strategy", "none")).lower()
        if self._device.type == "cuda" and self._torch.cuda.device_count() > 1:
            if strategy == "data_parallel":
                model = self._torch.nn.DataParallel(model)
            elif strategy == "ddp":
                raise NotImplementedError(
                    "DDP strategy requires launcher/process-group setup and is not implemented yet."
                )
        return model

    def _read_hdf5_array(self, file_path: Path) -> np.ndarray:
        """Read the first dataset array from an HDF5 file."""
        try:
            import h5py
        except ImportError as exc:
            raise ImportError("h5py is required to read Hypersim .hdf5 files.") from exc

        with h5py.File(file_path, "r") as handle:
            keys = list(handle.keys())
            if not keys:
                raise ValueError(f"No datasets found in {file_path}")
            arr = handle[keys[0]][()]
        return np.asarray(arr, dtype=np.float32)

    def _to_rgb_tensor(self, arr: np.ndarray):
        """Convert an RGB array to normalized model-ready tensor."""
        tensor = self._torch.from_numpy(arr)
        if tensor.ndim != 3:
            raise ValueError(f"RGB array must be HxWxC, got {tuple(tensor.shape)}")
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()
        input_size = int(self.model_cfg.get("input_size", 224))
        tensor = self._torch.nn.functional.interpolate(
            tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
        tensor = self._torch.nan_to_num(tensor)
        scale = self._torch.quantile(tensor.flatten(), 0.99)
        tensor = (tensor / (scale + 1e-6)).clamp(-3, 3)
        return tensor.to(self._device)

    def _to_depth_tokens(self, arr: np.ndarray):
        """Convert depth array to flattened tokenizer ids."""
        if self._depth_tokenizer is None:
            raise RuntimeError("Depth tokenizer is missing. Set model.tokenizers.depth_repo in config.")
        tensor = self._torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()
        input_size = int(self.model_cfg.get("input_size", 224))
        tensor = self._torch.nn.functional.interpolate(
            tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
        tensor = self._torch.nan_to_num(tensor).to(self._device)
        with self._torch.no_grad():
            token_grid = self._depth_tokenizer.tokenize(tensor)  # [B,H',W']
        return token_grid.reshape(token_grid.shape[0], -1)

    def _to_rgb_tokens(self, arr: np.ndarray):
        """Convert RGB array to flattened tokenizer ids."""
        if self._rgb_tokenizer is None:
            raise RuntimeError("RGB tokenizer is missing. Set model.tokenizers.rgb_repo in config.")
        tensor = self._to_rgb_tensor(arr)
        with self._torch.no_grad():
            token_grid = self._rgb_tokenizer.tokenize(tensor)  # [B,H',W']
        return token_grid.reshape(token_grid.shape[0], -1)

    def _to_normal_tokens(self, arr: np.ndarray):
        if self._normal_tokenizer is None:
            raise RuntimeError("Normal tokenizer missing. Set model.tokenizers.normal_repo in config.")
        
        tensor = self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        input_size = int(self.model_cfg.get("input_size", 224))
        tensor = self._torch.nn.functional.interpolate(
            tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
        # Hypersim normals are already in [-1,1] — matches NormalTransform output exactly
        tensor = self._torch.nan_to_num(tensor).clamp(-1.0, 1.0)
        tensor = tensor.to(self._device)
        with self._torch.no_grad():
            token_grid = self._normal_tokenizer.tokenize(tensor)
        return token_grid.reshape(token_grid.shape[0], -1)


    def _build_mod_dict(self, modality: str, arr: np.ndarray) -> dict[str, dict[str, Any]]:
        """Build a single-modality input dictionary for 4M."""
        if modality == "rgb":
            if self._rgb_input_mode == "tokenized":
                domain = "tok_rgb@224"
                tensor = self._to_rgb_tokens(arr)
            else:
                domain = "rgb@224"
                tensor = self._to_rgb_tensor(arr)
        elif modality == "depth":
            domain = "tok_depth@224"
            tensor = self._to_depth_tokens(arr)
        elif modality == "normals":
            domain = "tok_normal@224"
            tensor = self._to_normal_tokens(arr)
        else:
            raise ValueError(f"Unsupported modality for FourMEncoder: {modality}")

        mod_dict = {domain: {"tensor": tensor}}
        mod_dict = self._init_full_input_modality(
            mod_dict,
            self._modality_info,
            domain,
            self._device,
        )
        return mod_dict

    def _extract_from_encoder(self, mod_dict) -> dict[str, np.ndarray]:
        """Run encoder and return pooled vectors per layer."""
        model = self._model.module if hasattr(self._model, "module") else self._model
        domain = next(iter(mod_dict.keys()))
        token_count = int(mod_dict[domain]["tensor"].shape[1]) if domain.startswith("tok_") else int(
            (int(self.model_cfg.get("input_size", 224)) // 16) ** 2
        )
        block_outputs: dict[int, Any] = {}
        hooks = []

        def _make_hook(idx: int):
            """Create a hook that stores block outputs by index."""

            def _hook(_module, _inputs, output):
                """Capture one encoder block output."""
                block_outputs[idx] = output[0] if isinstance(output, tuple) else output

            return _hook

        for idx, block in enumerate(model.encoder):
            hooks.append(block.register_forward_hook(_make_hook(idx)))

        with self._torch.no_grad():
            encoder_mod_dict = {
                mod: model.encoder_embeddings[mod](d)
                for mod, d in mod_dict.items()
                if mod in model.encoder_embeddings
            }
            encoder_tokens, encoder_emb, encoder_mask, _ = model.forward_mask_encoder(
                encoder_mod_dict,
                num_encoder_tokens=token_count,
            )
            x = encoder_tokens + encoder_emb
            model.forward_encoder(x, encoder_mask)

        for h in hooks:
            h.remove()

        vectors: dict[str, np.ndarray] = {}
        if not block_outputs:
            pooled = encoder_emb.mean(dim=1).squeeze(0).float().cpu().numpy().astype(np.float32)
            return {"layer_00": pooled}

        for idx in sorted(block_outputs.keys()):
            tensor = block_outputs[idx]
            pooled = tensor.mean(dim=1).squeeze(0).float().cpu().numpy().astype(np.float32)
            vectors[f"layer_{idx:02d}"] = pooled

        return vectors

    def encode_path(self, file_path: Path, modality: str) -> dict[str, np.ndarray]:
        """Encode one file and return selected layer vectors."""
        arr = self._read_hdf5_array(file_path)
        mod_dict = self._build_mod_dict(modality=modality, arr=arr)
        vectors = self._extract_from_encoder(mod_dict=mod_dict)
        if self.layers == "all":
            return vectors

        selected: dict[str, np.ndarray] = {}
        ordered_keys = sorted(vectors.keys())
        for wanted in self.layers:
            if wanted in vectors:
                selected[wanted] = vectors[wanted]
            else:
                try:
                    idx = int(str(wanted).split("_")[-1])
                    if 0 <= idx < len(ordered_keys):
                        selected[wanted] = vectors[ordered_keys[idx]]
                except Exception:
                    continue
        return selected if selected else vectors
