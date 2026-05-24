"""4M model adapter for real activation extraction."""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

log = logging.getLogger(__name__)


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
        dist_cfg = self.runtime_cfg.get("distributed", {}) or {}
        dist_enabled = bool(dist_cfg.get("enabled", False)) or int(os.environ.get("WORLD_SIZE", "1")) > 1
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        device_cfg = str(self.runtime_cfg.get("device", "auto")).lower()
        if dist_enabled:
            if not self._torch.cuda.is_available():
                raise RuntimeError("Distributed mode requires CUDA-enabled GPUs.")
            return self._torch.device(f"cuda:{local_rank}")
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
        dist_cfg = self.runtime_cfg.get("distributed", {}) or {}
        dist_enabled = bool(dist_cfg.get("enabled", False)) or int(os.environ.get("WORLD_SIZE", "1")) > 1
        if dist_enabled:
            return model
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

    def _read_png_array(self, file_path: Path) -> np.ndarray:
        """Read a PNG image as float32 HxWx3 array with values in [0, 255]."""
        try:
            from PIL import Image
        except ImportError as exc:
            raise ImportError("Pillow is required to read DIODE .png files.") from exc
        img = np.array(Image.open(file_path).convert("RGB")).astype(np.float32)
        return img

    def _read_npy_array(self, file_path: Path) -> np.ndarray:
        """Read a .npy file and squeeze trailing singleton dimensions."""
        arr = np.load(file_path).astype(np.float32)
        return arr.squeeze()

    def _read_diode_depth(self, file_path: Path) -> np.ndarray:
        """Read DIODE depth .npy and zero out invalid pixels using the companion mask.

        Returns raw depth values in meters with invalid pixels set to 0.
        Normalization (truncated z-score) is applied downstream in _to_depth_tokens,
        consistent with how Hypersim depth is handled.
        """
        depth = np.load(file_path).astype(np.float32).squeeze()   # (H, W)
        mask_path = file_path.parent / (file_path.stem + "_mask.npy")
        if mask_path.exists():
            mask = np.load(mask_path).astype(np.float32).squeeze()  # (H, W)
            depth = depth * mask   # zero out invalid pixels, keep raw meters
        else:
            log.warning(
                "DIODE depth mask not found at %s; using raw depth values.", mask_path
            )
        return depth.astype(np.float32)

    def _load_array(self, file_path: Path, modality: str) -> np.ndarray:
        """Dispatch array loading based on file extension and modality."""
        suffix = file_path.suffix.lower()
        if suffix == ".hdf5":
            return self._read_hdf5_array(file_path)
        if suffix == ".png":
            return self._read_png_array(file_path)
        if suffix == ".npy":
            if modality == "depth":
                return self._read_diode_depth(file_path)
            return self._read_npy_array(file_path)
        raise ValueError(
            f"Unsupported file format '{suffix}' for modality '{modality}'. "
            "Supported: .hdf5 (Hypersim), .png (DIODE RGB), .npy (DIODE depth/normals)."
        )

    def _to_rgb_tensor(self, arr: np.ndarray, file_format: str = "hdf5"):
        """Convert an RGB array to a normalized, model-ready tensor.

        Handles two input formats:
        - DIODE .png  → uint8 values in [0, 255]: divide by 255 to reach [0, 1].
        - Hypersim .hdf5 → linear HDR floats (can contain inf): percentile
          scaling (0.99 quantile) to reach [0, 1].

        Dispatch is based on ``file_format`` (the source file extension), NOT on
        pixel magnitude. The magnitude-based heuristic (``tensor.max() > 2.0``)
        was unreliable because ``nan_to_num`` converts HDR ``inf`` pixels to
        3.4e38 before the check, incorrectly triggering the DIODE branch and
        producing activations of magnitude ~10^34.

        Both paths then apply standard ImageNet mean/std normalization.
        """
        tensor = self._torch.from_numpy(arr)
        if tensor.ndim != 3:
            raise ValueError(f"RGB array must be HxWxC, got {tuple(tensor.shape)}")
        tensor = tensor.permute(2, 0, 1).unsqueeze(0).float()  # (1, 3, H, W)
        input_size = int(self.model_cfg.get("input_size", 224))
        tensor = self._torch.nn.functional.interpolate(
            tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
        tensor = self._torch.nan_to_num(tensor).to(self._device)

        # Bring to [0, 1] — dispatch on source file format
        if file_format == "png":
            # DIODE path: uint8 PNG, values in [0, 255]
            tensor = tensor / 255.0
        else:
            # Hypersim path: linear HDR floats — percentile scaling
            scale = self._torch.quantile(tensor.flatten(), 0.99)
            tensor = (tensor / (scale + 1e-6)).clamp(0.0, 1.0)

        # ImageNet normalization — required by the rgb@224 encoder embedding
        # (confirmed in the 4M README and official demo code)
        mean = self._torch.tensor(
            [0.485, 0.456, 0.406], device=self._device
        ).view(1, 3, 1, 1)
        std = self._torch.tensor(
            [0.229, 0.224, 0.225], device=self._device
        ).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std

        return tensor

    def _truncated_depth_standardization(self, depth, thresh: float = 0.1):
        """Replicates 4M's DepthTransform.truncated_depth_standardization exactly."""
        trunc = self._torch.sort(depth.reshape(-1))[0]
        trunc = trunc[int(thresh * trunc.shape[0]): int((1 - thresh) * trunc.shape[0])]
        return (depth - trunc.mean()) / self._torch.sqrt(trunc.var() + 1e-6)

    def _to_depth_tokens(self, arr: np.ndarray):
        """Convert depth array to flattened tokenizer ids."""
        if self._depth_tokenizer is None:
            raise RuntimeError("Depth tokenizer missing. Set model.tokenizers.depth_repo in config.")
        tensor = self._torch.from_numpy(arr).unsqueeze(0).unsqueeze(0).float()  # (1, 1, H, W)
        input_size = int(self.model_cfg.get("input_size", 224))
        tensor = self._torch.nn.functional.interpolate(
            tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
        tensor = self._torch.nan_to_num(tensor)
        # Replicate 4M's DepthTransform — truncated z-score, no /65535 (input is float32 meters)
        tensor = self._truncated_depth_standardization(tensor)
        tensor = tensor.to(self._device)
        with self._torch.no_grad():
            token_grid = self._depth_tokenizer.tokenize(tensor)
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
        """Convert surface normals array to flattened tokenizer ids.

        Normals from both Hypersim (.hdf5) and DIODE (.npy) are in [-1, 1].
        This matches the NormalTransform expected by the tokenizer directly —
        no remapping needed for either dataset.
        """
        if self._normal_tokenizer is None:
            raise RuntimeError("Normal tokenizer missing. Set model.tokenizers.normal_repo in config.")

        tensor = self._torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0).float()
        input_size = int(self.model_cfg.get("input_size", 224))
        tensor = self._torch.nn.functional.interpolate(
            tensor, size=(input_size, input_size), mode="bilinear", align_corners=False
        )
        # Both Hypersim and DIODE normals are in [-1, 1] — clamp defensively
        tensor = self._torch.nan_to_num(tensor).clamp(-1.0, 1.0)
        tensor = tensor.to(self._device)
        with self._torch.no_grad():
            token_grid = self._normal_tokenizer.tokenize(tensor)
        return token_grid.reshape(token_grid.shape[0], -1)

    def _build_mod_dict(self, modality: str, arr: np.ndarray, file_format: str = "hdf5") -> dict[str, dict[str, Any]]:
        """Build a single-modality input dictionary for 4M.

        Thin wrapper around :meth:`_build_mod_entry` kept for backward
        compatibility with :meth:`encode_path`. The joint code path uses
        :meth:`_build_mod_entry` directly and merges multiple entries.
        """
        entry, domain = self._build_mod_entry(modality=modality, arr=arr, file_format=file_format)
        return {domain: entry}

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
        """Encode one file and return selected layer vectors.

        Supports Hypersim (.hdf5) and DIODE (.png for RGB, .npy for depth /
        normals) file formats, dispatched automatically from the file extension.
        """
        arr = self._load_array(file_path, modality)
        file_format = file_path.suffix.lower().lstrip(".")
        mod_dict = self._build_mod_dict(modality=modality, arr=arr, file_format=file_format)
        vectors = self._extract_from_encoder(mod_dict=mod_dict)
        return self._select_requested_layers(vectors)

    def encode_pair_joint(
        self,
        file_paths: dict[str, Path],
        modalities: tuple[str, str],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Encode both modalities together in a single joint forward pass.

        Tokenizes each modality independently, then concatenates the input
        tokens **in alphabetical order of the modality name** so the slice
        boundaries inside the encoder output are deterministic for the lifetime
        of this repo. For ``modalities=("rgb", "depth")`` the joint sequence is
        always ``[depth_tokens | rgb_tokens]``.

        Hooks capture each transformer block's output of shape
        ``(1, n_left + n_right, D)``. The output is split at the recorded
        boundary and mean-pooled per slice, yielding one vector per modality
        per layer.

        Parameters
        ----------
        file_paths:
            Mapping ``{modality: path}`` containing both input files. Must
            contain entries for both members of ``modalities``.
        modalities:
            Two distinct modality names. Order is preserved for the return
            keys but the *encoder* always sees them in alphabetical order.

        Returns
        -------
        dict
            ``{layer_name: {modality_a: np.ndarray, modality_b: np.ndarray}}``.
            Vectors are pooled (mean over the modality's token slice) and have
            shape ``(D,)``.
        """
        if len(modalities) != 2 or modalities[0] == modalities[1]:
            raise ValueError(
                f"encode_pair_joint requires two distinct modalities, got {modalities!r}"
            )
        sorted_mods = tuple(sorted(modalities))

        slices: list[tuple[str, str, int]] = []   # (modality, domain, n_tokens)
        merged_mod_dict: dict[str, dict[str, Any]] = {}
        for mod in sorted_mods:
            path = file_paths[mod]
            arr = self._load_array(path, mod)
            file_format = path.suffix.lower().lstrip(".")
            entry, domain = self._build_mod_entry(modality=mod, arr=arr, file_format=file_format)
            merged_mod_dict[domain] = entry
            n_tokens = self._domain_token_count(domain=domain, entry=entry)
            slices.append((mod, domain, n_tokens))

        per_layer = self._extract_joint_from_encoder(merged_mod_dict, slices)
        return self._select_requested_layers_joint(per_layer)

    def _build_mod_entry(
        self,
        modality: str,
        arr: np.ndarray,
        file_format: str = "hdf5",
    ) -> tuple[dict[str, Any], str]:
        """Build one ``mod_dict`` entry for a single modality.

        Identical to the body of :meth:`_build_mod_dict` but returns just the
        single ``(entry, domain)`` pair so the joint path can merge multiple
        entries before calling ``init_full_input_modality`` on each.
        """
        if modality == "rgb":
            if self._rgb_input_mode == "tokenized":
                domain = "tok_rgb@224"
                tensor = self._to_rgb_tokens(arr)
            else:
                domain = "rgb@224"
                tensor = self._to_rgb_tensor(arr, file_format=file_format)
        elif modality == "depth":
            domain = "tok_depth@224"
            tensor = self._to_depth_tokens(arr)
        elif modality == "normals":
            domain = "tok_normal@224"
            tensor = self._to_normal_tokens(arr)
        else:
            raise ValueError(f"Unsupported modality for FourMEncoder: {modality}")

        single = {domain: {"tensor": tensor}}
        single = self._init_full_input_modality(
            single,
            self._modality_info,
            domain,
            self._device,
        )
        return single[domain], domain

    def _domain_token_count(self, domain: str, entry: dict[str, Any]) -> int:
        """Resolve the number of encoder tokens contributed by one modality."""
        if domain.startswith("tok_"):
            return int(entry["tensor"].shape[1])
        input_size = int(self.model_cfg.get("input_size", 224))
        return (input_size // 16) ** 2

    def _select_requested_layers(self, vectors: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Filter the layer dict down to the configured ``self.layers``."""
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

    def _select_requested_layers_joint(
        self,
        per_layer: dict[str, dict[str, np.ndarray]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Layer-filter for the joint dict (mirrors :meth:`_select_requested_layers`)."""
        if self.layers == "all":
            return per_layer
        selected: dict[str, dict[str, np.ndarray]] = {}
        ordered_keys = sorted(per_layer.keys())
        for wanted in self.layers:
            if wanted in per_layer:
                selected[wanted] = per_layer[wanted]
            else:
                try:
                    idx = int(str(wanted).split("_")[-1])
                    if 0 <= idx < len(ordered_keys):
                        selected[wanted] = per_layer[ordered_keys[idx]]
                except Exception:
                    continue
        return selected if selected else per_layer

    def _extract_joint_from_encoder(
        self,
        merged_mod_dict: dict[str, dict[str, Any]],
        slices: list[tuple[str, str, int]],
    ) -> dict[str, dict[str, np.ndarray]]:
        """Run one joint forward pass and pool per-modality token slices.

        ``slices`` is a list of ``(modality, domain, n_tokens)`` triples whose
        order **must match** the insertion order of ``merged_mod_dict``. The
        encoder concatenates tokens in dict insertion order, so the first
        ``slices[0][2]`` output tokens belong to ``slices[0][0]``, etc.
        """
        model = self._model.module if hasattr(self._model, "module") else self._model
        total_tokens = sum(n for _, _, n in slices)
        block_outputs: dict[int, Any] = {}
        hooks = []

        def _make_hook(idx: int):
            def _hook(_module, _inputs, output):
                block_outputs[idx] = output[0] if isinstance(output, tuple) else output
            return _hook

        for idx, block in enumerate(model.encoder):
            hooks.append(block.register_forward_hook(_make_hook(idx)))

        try:
            with self._torch.no_grad():
                encoder_mod_dict = {
                    mod: model.encoder_embeddings[mod](d)
                    for mod, d in merged_mod_dict.items()
                    if mod in model.encoder_embeddings
                }
                encoder_tokens, encoder_emb, encoder_mask, _ = model.forward_mask_encoder(
                    encoder_mod_dict,
                    num_encoder_tokens=total_tokens,
                )
                # First-call sanity check: token sequence must equal the sum of
                # per-modality counts AND nothing may be masked out (otherwise
                # the slice boundaries are no longer trustworthy).
                if int(encoder_tokens.shape[1]) != total_tokens:
                    raise RuntimeError(
                        f"Joint pass token count mismatch: forward_mask_encoder "
                        f"returned {int(encoder_tokens.shape[1])} tokens, expected {total_tokens}. "
                        f"Slice boundaries would be wrong; aborting."
                    )
                if encoder_mask is not None and bool(encoder_mask.any()):
                    raise RuntimeError(
                        "Joint pass encountered a non-empty encoder_mask. "
                        "init_full_input_modality is expected to leave all input "
                        "tokens unmasked; slice boundaries are unsafe otherwise."
                    )
                x = encoder_tokens + encoder_emb
                model.forward_encoder(x, encoder_mask)
        finally:
            for h in hooks:
                h.remove()

        if not block_outputs:
            # No transformer blocks captured → fall back to the embedding tensor
            # but still split it correctly.
            return {"layer_00": self._pool_token_slices(encoder_emb, slices)}

        out: dict[str, dict[str, np.ndarray]] = {}
        for idx in sorted(block_outputs.keys()):
            tensor = block_outputs[idx]
            out[f"layer_{idx:02d}"] = self._pool_token_slices(tensor, slices)
        return out

    def _pool_token_slices(
        self,
        tensor: Any,
        slices: list[tuple[str, str, int]],
    ) -> dict[str, np.ndarray]:
        """Mean-pool ``tensor`` of shape ``(1, T, D)`` per modality slice."""
        out: dict[str, np.ndarray] = {}
        start = 0
        for modality, _domain, n_tokens in slices:
            end = start + n_tokens
            pooled = (
                tensor[:, start:end, :]
                .mean(dim=1)
                .squeeze(0)
                .float()
                .cpu()
                .numpy()
                .astype(np.float32)
            )
            out[modality] = pooled
            start = end
        return out