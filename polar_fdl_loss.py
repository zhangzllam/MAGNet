import math
from pathlib import Path
import torch
from torch import nn
from torch.nn import functional as F


class PolarFDLLoss(nn.Module):
    def __init__(
        self,
        image_size,
        center_xy=None,
        n_angles=400,
        n_radii=None,
        radius_min=0.0,
        radius_max=None,
        vgg_weights_path=None,
        feature_levels=(0, 1, 2),
        num_proj=64,
        patch_size=3,
        projection_stride=1,
        projection_chunk_size=32,
        radial_bands=4,
        amplitude_weight=1.0,
        cross_phase_weight=0.01,
        radial_offsets=(1, 4, 16),
        phase_energy_floor=1e-05,
        phase_eps=1e-06,
        input_range="0_1",
        projection_seed=1234,
        circular_vgg=True,
        raw_amplitude_weight=1.0,
    ):
        super().__init__()
        if isinstance(image_size, int):
            image_size = (image_size, image_size)
        if len(image_size) != 2 or any((int(v) != v or v < 2 for v in image_size)):
            raise ValueError("image_size must be (height, width), each >= 2")
        self.image_size = tuple((int(v) for v in image_size))
        h, w = self.image_size
        self.center_xy = tuple(center_xy) if center_xy is not None else (w // 2, h // 2)
        if len(self.center_xy) != 2:
            raise ValueError("center_xy must be (x, y), in input-image pixels")
        cx, cy = self.center_xy
        if not all((math.isfinite(v) for v in (cx, cy))) or not (
            0 <= cx <= w - 1 and 0 <= cy <= h - 1
        ):
            raise ValueError("center_xy must lie inside the image")
        safe_radius = min(cx, w - 1 - cx, cy, h - 1 - cy)
        radius_max = safe_radius if radius_max is None else float(radius_max)
        if not 0 <= radius_min < radius_max <= safe_radius + 1e-06:
            raise ValueError(
                f"Require 0 <= radius_min < radius_max <= {safe_radius}; no padded rings"
            )
        n_radii = int(radius_max - radius_min) + 1 if n_radii is None else n_radii
        for name, value in (
            ("n_angles", n_angles),
            ("n_radii", n_radii),
            ("num_proj", num_proj),
            ("patch_size", patch_size),
            ("projection_stride", projection_stride),
            ("projection_chunk_size", projection_chunk_size),
            ("radial_bands", radial_bands),
        ):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if n_angles < 4 or n_radii < 2:
            raise ValueError("At least 4 angles and 2 radii are required")
        self.feature_levels = tuple(feature_levels)
        if (
            not self.feature_levels
            or tuple(sorted(set(self.feature_levels))) != self.feature_levels
        ):
            raise ValueError("feature_levels must be an increasing, nonempty sequence")
        if any(
            (
                isinstance(i, bool) or not isinstance(i, int) or (not 0 <= i < 5)
                for i in self.feature_levels
            )
        ):
            raise ValueError("feature_levels use zero-based VGG stage indices 0..4")
        factor = 2 ** max(self.feature_levels)
        if amplitude_weight > 0 and n_angles % factor:
            raise ValueError(
                "n_angles must be divisible by the deepest VGG downsampling factor"
            )
        if amplitude_weight > 0 and (
            n_angles // factor // 2 + 1 < patch_size
            or n_radii // factor // radial_bands < patch_size
        ):
            raise ValueError(
                "Polar grid too small for feature_levels / radial_bands / patch_size"
            )
        for name, weight in (
            ("amplitude_weight", amplitude_weight),
            ("cross_phase_weight", cross_phase_weight),
            ("raw_amplitude_weight", raw_amplitude_weight),
        ):
            if not math.isfinite(weight) or weight < 0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if input_range not in ("0_1", "minus1_1", "0_255"):
            raise ValueError("input_range must be 0_1, minus1_1 or 0_255")
        if (
            not math.isfinite(phase_eps)
            or not math.isfinite(phase_energy_floor)
            or min(phase_eps, phase_energy_floor) <= 0
        ):
            raise ValueError("Phase eps and energy floor must be positive and finite")
        radial_offsets = tuple(radial_offsets)
        if not radial_offsets or any(
            (
                isinstance(d, bool) or not isinstance(d, int) or (not 0 < d < n_radii)
                for d in radial_offsets
            )
        ):
            raise ValueError(
                "radial_offsets must be positive integers below n_radii (polar samples)"
            )
        if len(set(radial_offsets)) != len(radial_offsets):
            raise ValueError("radial_offsets must not repeat")
        self.n_angles, self.n_radii = (n_angles, n_radii)
        self.radius_min, self.radius_max = (float(radius_min), radius_max)
        self.num_proj, self.patch_size = (num_proj, patch_size)
        self.projection_stride, self.projection_chunk_size = (
            projection_stride,
            projection_chunk_size,
        )
        self.radial_bands = radial_bands
        self.amplitude_weight, self.cross_phase_weight = (
            amplitude_weight,
            cross_phase_weight,
        )
        self.raw_amplitude_weight = float(raw_amplitude_weight)
        self.radial_offsets = radial_offsets
        self.phase_energy_floor, self.phase_eps = (phase_energy_floor, phase_eps)
        self.input_range, self.circular_vgg = (input_range, circular_vgg)
        angles = torch.arange(n_angles, dtype=torch.float32) * (2 * math.pi / n_angles)
        radii = torch.linspace(radius_min, radius_max, n_radii)
        xx = cx + angles.cos()[:, None] * radii[None, :]
        yy = cy + angles.sin()[:, None] * radii[None, :]
        grid = torch.stack((2 * xx / (w - 1) - 1, 2 * yy / (h - 1) - 1), dim=-1)
        self.register_buffer("polar_grid", grid.unsqueeze(0))
        self.register_buffer(
            "mean", torch.tensor([0.485, 0.456, 0.406])[None, :, None, None]
        )
        self.register_buffer(
            "std", torch.tensor([0.229, 0.224, 0.225])[None, :, None, None]
        )
        self.stages = nn.ModuleList()
        if amplitude_weight > 0:
            from torchvision.models import VGG19_Weights
            from torchvision.models.vgg import cfgs, make_layers

            with torch.random.fork_rng(devices=[]):
                backbone = make_layers(cfgs["E"], batch_norm=False)
            if vgg_weights_path:
                state = torch.load(
                    Path(vgg_weights_path), map_location="cpu", weights_only=True
                )
            else:
                state = VGG19_Weights.IMAGENET1K_V1.get_state_dict(
                    progress=True, check_hash=True
                )
            feature_state = {
                k[len("features.") :]: v
                for k, v in state.items()
                if k.startswith("features.")
            }
            if not feature_state:
                raise ValueError("Expected the official torchvision VGG19 state_dict")
            backbone.load_state_dict(feature_state, strict=True)
            edges = (0, 4, 9, 18, 27, 36)
            self.stages = nn.ModuleList(
                [
                    backbone[edges[i] : edges[i + 1]]
                    for i in range(max(self.feature_levels) + 1)
                ]
            )
            self.stages.requires_grad_(False)
            self.stages.eval()
            generator = torch.Generator().manual_seed(projection_seed)
            for i in self.feature_levels:
                channels = (64, 128, 256, 512, 512)[i]
                weights = torch.randn(
                    num_proj, channels, patch_size, patch_size, generator=generator
                )
                weights /= weights.flatten(1).norm(dim=1)[:, None, None, None]
                self.register_buffer(f"projection_{i}", weights)

    def train(self, mode=True):
        super().train(mode)
        self.stages.eval()
        return self

    def _image_values(self, image):
        image = image.float()
        if self.input_range == "minus1_1":
            image = (image + 1) / 2
        elif self.input_range == "0_255":
            image = image / 255
        return image

    def _validate(self, prediction, target, spatial_shape):
        if prediction.ndim != 4 or prediction.shape != target.shape:
            raise ValueError("Prediction and GT must be equal-shaped [B,C,H,W] tensors")
        if prediction.shape[0] < 1 or prediction.shape[1] not in (1, 3):
            raise ValueError("A nonempty batch with 1 or 3 channels is required")
        if tuple(prediction.shape[-2:]) != tuple(spatial_shape):
            raise ValueError(
                f"Expected spatial shape {spatial_shape}, got {tuple(prediction.shape[-2:])}"
            )
        if (
            prediction.device != target.device
            or prediction.device != self.polar_grid.device
        ):
            raise ValueError("Move loss, prediction and GT to the same device")
        if self.polar_grid.dtype != torch.float32:
            raise ValueError(
                "Keep PolarFDLLoss in FP32, even when training the generator with AMP"
            )

    def to_polar(self, image):
        if image.ndim != 4 or tuple(image.shape[-2:]) != self.image_size:
            raise ValueError(
                f"Expected [B,C,{self.image_size[0]},{self.image_size[1]}]"
            )
        return F.grid_sample(
            image.float(),
            self.polar_grid.expand(image.shape[0], -1, -1, -1),
            mode="bilinear",
            padding_mode="zeros",
            align_corners=True,
        )

    def _stage(self, x, stage):
        for layer in stage:
            if self.circular_vgg and isinstance(layer, nn.Conv2d):
                ph, pw = layer.padding
                if pw:
                    x = F.pad(x, (pw, pw, 0, 0))
                if ph:
                    x = F.pad(x, (0, 0, ph, ph), mode="circular")
                x = F.conv2d(
                    x,
                    layer.weight,
                    layer.bias,
                    layer.stride,
                    0,
                    layer.dilation,
                    layer.groups,
                )
            else:
                x = layer(x)
        return x

    def _swd(self, x, y, weights):
        score = x.new_zeros(x.shape[0])
        for kernel in weights.split(self.projection_chunk_size):
            px = (
                F.conv2d(x, kernel, stride=self.projection_stride)
                .flatten(2)
                .sort(-1)
                .values
            )
            py = (
                F.conv2d(y, kernel, stride=self.projection_stride)
                .flatten(2)
                .sort(-1)
                .values
            )
            score = score + (px - py).abs().mean(-1).sum(-1)
        return score / weights.shape[0]

    def _amplitude_loss(self, prediction_polar, target_polar):
        x, y = (prediction_polar, target_polar)
        if x.shape[1] == 1:
            x, y = (x.repeat(1, 3, 1, 1), y.repeat(1, 3, 1, 1))
        x = (x - self.mean) / self.std
        with torch.no_grad():
            y = (y - self.mean) / self.std
        scores = []
        for i, stage in enumerate(self.stages):
            x = self._stage(x, stage)
            with torch.no_grad():
                y = self._stage(y, stage)
            if i not in self.feature_levels:
                continue
            ax = torch.log1p(torch.fft.rfft(x, dim=-2, norm="forward").abs())
            with torch.no_grad():
                ay = torch.log1p(torch.fft.rfft(y, dim=-2, norm="forward").abs())
            radius_count = ax.shape[-1]
            for band in range(self.radial_bands):
                left = band * radius_count // self.radial_bands
                right = (band + 1) * radius_count // self.radial_bands
                scores.append(
                    self._swd(
                        ax[..., left:right],
                        ay[..., left:right],
                        getattr(self, f"projection_{i}"),
                    )
                )
        return torch.stack(scores).mean()

    def _raw_amplitude_loss(self, prediction_polar, target_polar):
        ax = torch.fft.rfft(prediction_polar, dim=-2, norm="forward").abs()
        ax = torch.log1p(ax[:, :, 1:, :])
        with torch.no_grad():
            ay = torch.fft.rfft(target_polar, dim=-2, norm="forward").abs()
            ay = torch.log1p(ay[:, :, 1:, :])
        return F.l1_loss(ax, ay)

    def _cross_phase_loss(self, prediction_polar, target_polar):
        zx = torch.fft.rfft(prediction_polar, dim=-2, norm="forward")[:, :, 1:, :]
        with torch.no_grad():
            zy = torch.fft.rfft(target_polar, dim=-2, norm="forward")[:, :, 1:, :]
        scores = []
        for offset in self.radial_offsets:
            a, b = (zx[..., :-offset], zx[..., offset:])
            cx = a * b.conj() / (a.abs() * b.abs() + self.phase_eps)
            with torch.no_grad():
                ta, tb = (zy[..., :-offset], zy[..., offset:])
                energy = ta.abs() * tb.abs()
                cy = ta * tb.conj() / (energy + self.phase_eps)
                confidence = energy / (energy + self.phase_energy_floor)
            error = (cx - cy).abs()
            dims = (1, 2, 3)
            scores.append(
                (
                    (confidence * error).sum(dims) / confidence.sum(dims).clamp_min(1.0)
                ).mean()
            )
        return torch.stack(scores).mean()

    def _loss_from_values(self, x, y, return_components):
        amp = self._amplitude_loss(x, y) if self.amplitude_weight else x.sum() * 0
        phase = self._cross_phase_loss(x, y) if self.cross_phase_weight else x.sum() * 0
        raw_amp = (
            self._raw_amplitude_loss(x, y) if self.raw_amplitude_weight else x.sum() * 0
        )
        total = (
            self.amplitude_weight * amp
            + self.cross_phase_weight * phase
            + self.raw_amplitude_weight * raw_amp
        )
        if return_components:
            return (
                total,
                {
                    "amplitude": amp.detach(),
                    "cross_phase": phase.detach(),
                    "raw_amplitude": raw_amp.detach(),
                    "total": total.detach(),
                    "weighted_amplitude": (self.amplitude_weight * amp).detach(),
                    "weighted_cross_phase": (self.cross_phase_weight * phase).detach(),
                    "weighted_raw_amplitude": (
                        self.raw_amplitude_weight * raw_amp
                    ).detach(),
                },
            )
        return total

    def forward(self, prediction, target, return_components=False):
        self._validate(prediction, target, self.image_size)
        if not (
            self.amplitude_weight
            or self.raw_amplitude_weight
            or self.cross_phase_weight
        ):
            return self._loss_from_values(
                prediction.float(), target.detach().float(), return_components
            )
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            x = self.to_polar(self._image_values(prediction))
            with torch.no_grad():
                y = self.to_polar(self._image_values(target.detach()))
            return self._loss_from_values(x, y, return_components)

    def forward_polar(self, prediction, target, return_components=False):
        self._validate(prediction, target, (self.n_angles, self.n_radii))
        if not (
            self.amplitude_weight
            or self.raw_amplitude_weight
            or self.cross_phase_weight
        ):
            return self._loss_from_values(
                prediction.float(), target.detach().float(), return_components
            )
        with torch.autocast(device_type=prediction.device.type, enabled=False):
            x = self._image_values(prediction)
            with torch.no_grad():
                y = self._image_values(target.detach())
            return self._loss_from_values(x, y, return_components)
