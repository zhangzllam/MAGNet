"""MAGNet for kernel-conditioned rotational blur generation."""

import cv2
import kornia
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from pytorch_wavelets import DWTForward


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class FusionBlock(nn.Module):
    def __init__(self, channels, expansion=1, dilations=(1, 1)):
        super().__init__()
        self.dw_channel = expansion * channels
        self.dilation1 = dilations[0]
        self.branch1 = nn.Conv2d(
            in_channels=self.dw_channel,
            out_channels=self.dw_channel,
            kernel_size=3,
            padding=self.dilation1,
            stride=1,
            groups=self.dw_channel,
            bias=True,
            dilation=self.dilation1,
        )
        self.dilation2 = dilations[1]
        self.branch2 = nn.Conv2d(
            in_channels=self.dw_channel,
            out_channels=self.dw_channel,
            kernel_size=3,
            padding=self.dilation2,
            stride=1,
            groups=self.dw_channel,
            bias=True,
            dilation=self.dilation2,
        )
        self.gate = SimpleGate()
        self.channel_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(
                in_channels=self.dw_channel // 2,
                out_channels=self.dw_channel // 2,
                kernel_size=1,
                padding=0,
                stride=1,
                groups=1,
                bias=True,
                dilation=1,
            ),
        )
        self.projection = nn.Conv2d(
            in_channels=self.dw_channel // 2,
            out_channels=channels // 2,
            kernel_size=1,
            padding=0,
            stride=1,
            groups=1,
            bias=True,
            dilation=1,
        )

    def forward(self, x1, x2):
        x_all = torch.cat([x1, x2], dim=1)
        z1 = self.branch1(x_all)
        z2 = self.branch2(x_all)
        z = z1 + z2
        z = self.gate(z)
        x = self.channel_attention(z) * z
        x = self.projection(x)
        return x


class WBM(nn.Module):
    """Wavelet Blur Module."""

    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.dwt = DWTForward(J=1, mode="zero", wave="haar")
        self.subband_attention = nn.Conv2d(4, 4, 7, padding=3)
        self.projection = nn.Sequential(
            nn.Conv2d(in_ch * 4, out_ch, kernel_size=1, stride=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        yL, yH = self.dwt(x)
        y_HL = yH[0][:, :, 0, :]
        y_LH = yH[0][:, :, 1, :]
        y_HH = yH[0][:, :, 2, :]
        x = torch.cat([yL, y_HL, y_LH, y_HH], dim=1)
        avg_w = torch.mean(x, dim=1, keepdim=True)
        max_w, _ = torch.max(x, dim=1, keepdim=True)
        agg = torch.cat([avg_w, max_w, avg_w, max_w], dim=1)
        sig = self.subband_attention(agg).sigmoid()
        x = torch.cat(
            [
                yL * sig[:, 0, :, :].unsqueeze(1),
                y_HL * sig[:, 1, :, :].unsqueeze(1),
                y_LH * sig[:, 2, :, :].unsqueeze(1),
                y_HH * sig[:, 3, :, :].unsqueeze(1),
            ],
            dim=1,
        )
        x = self.projection(x)
        return x


class ConvBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        stride,
        bias=True,
        norm=False,
        activation=None,
        padding=None,
    ):
        super().__init__()
        if bias and norm:
            bias = False
        padding = kernel_size // 2 if padding is None else padding
        layers = []
        layers.append(
            nn.Conv2d(
                in_channel,
                out_channel,
                kernel_size,
                padding=padding,
                stride=stride,
                bias=bias,
            )
        )
        if norm:
            layers.append(nn.BatchNorm2d(out_channel))
        if activation is not None:
            layers.append(activation)
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class ResidualBlock(nn.Module):
    def __init__(
        self, in_channel, out_channel, activation=nn.ReLU(inplace=True), norm=True
    ):
        super().__init__()
        self.body = nn.Sequential(
            ConvBlock(
                in_channel,
                out_channel,
                kernel_size=3,
                stride=1,
                activation=activation,
                norm=norm,
            ),
            ConvBlock(out_channel, out_channel, kernel_size=3, stride=1),
        )

    def forward(self, x):
        return self.body(x) + x


class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel, kernel_size=3, ratio=2):
        super().__init__()
        modules = [
            ConvBlock(
                in_channel,
                out_channel * ratio * ratio,
                kernel_size=kernel_size,
                stride=1,
            ),
            nn.PixelShuffle(ratio),
        ]
        self.layers = nn.Sequential(*modules)

    def forward(self, x):
        return self.layers(x)


class KBU(nn.Module):
    """Kernel-guided Blur U-Net."""

    def __init__(
        self,
        in_channel,
        out_channel,
        n_feats,
        activation=nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.wbm1 = WBM(n_feats, n_feats)
        self.wbm2 = WBM(n_feats, n_feats)
        self.wbm3 = WBM(n_feats, n_feats)
        self.wbm4 = WBM(n_feats, n_feats)
        self.up1 = UpsampleBlock(n_feats, n_feats)
        self.up2 = UpsampleBlock(n_feats, n_feats)
        self.up3 = UpsampleBlock(n_feats, n_feats)
        self.up4 = UpsampleBlock(n_feats, n_feats)
        self.input_projection = ConvBlock(
            in_channel, n_feats, kernel_size=3, activation=activation, stride=1
        )
        self.output_projection = ConvBlock(
            n_feats, out_channel, kernel_size=3, activation=None, stride=1
        )
        self.kernel_encoder1 = KernelEncoder(n_feats)
        self.kernel_encoder2 = KernelEncoder(n_feats)
        self.kernel_encoder3 = KernelEncoder(n_feats)
        self.kernel_encoder4 = KernelEncoder(n_feats)
        self.cross_attn1 = CrossAttention(n_feats)
        self.cross_attn2 = CrossAttention(n_feats)
        self.cross_attn3 = CrossAttention(n_feats)
        self.cross_attn4 = CrossAttention(n_feats)

    def forward(self, x, kernel):
        x = self.input_projection(x)
        x_1 = self.wbm1(x)
        x_2 = self.wbm2(x_1)
        k_feat1 = self.kernel_encoder1(kernel)
        x_2 = self.cross_attn1(x_2, k_feat1)
        x_3 = self.wbm3(x_2)
        x_4 = self.wbm4(x_3)
        k_feat2 = self.kernel_encoder2(kernel)
        x_4 = self.cross_attn2(x_4, k_feat2)
        out = self.up1(x_4)
        out = self.up2(out + x_3)
        k_feat3 = self.kernel_encoder3(kernel)
        out = self.cross_attn3(out, k_feat3)
        out = self.up3(out + x_2)
        out = self.up4(out + x_1)
        k_feat4 = self.kernel_encoder4(kernel)
        out = self.cross_attn4(out, k_feat4)
        out = self.output_projection(out + x)
        return (out, [k_feat1, k_feat2, k_feat3, k_feat4])


class KernelEncoder(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(1, 32, 3, padding=1), nn.ReLU(), nn.Conv1d(32, dim, 3, padding=1)
        )

    def forward(self, kernel):
        kernel = kernel.mean(1)
        feat = self.net(kernel)
        feat = feat.permute(0, 2, 1)
        feat = feat / (feat.norm(dim=-1, keepdim=True) + 1e-06)
        return feat


class CrossAttention(nn.Module):
    def __init__(self, dim, num_heads=4):
        super().__init__()
        if num_heads <= 0 or dim % num_heads != 0:
            raise ValueError("dim must be divisible by a positive num_heads")
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, x, k_feat):
        B, C, H, W = x.shape
        L = k_feat.shape[1]
        x_flat = x.flatten(2).transpose(1, 2)
        q = self.q_proj(x_flat)
        k = self.k_proj(k_feat)
        v = self.v_proj(k_feat)
        q = q.reshape(B, H * W, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(B, L, self.num_heads, self.head_dim).transpose(1, 2)
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)
        out = out.transpose(1, 2).reshape(B, H * W, C)
        out = self.out_proj(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)
        return x + out


class OutputRefinement(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        n_feats,
        num_basic=3,
        activation=nn.ReLU(inplace=True),
    ):
        super().__init__()
        self.input_projection = ConvBlock(
            in_channel, n_feats, kernel_size=3, activation=activation, stride=1
        )
        self.output_projection = ConvBlock(
            n_feats, out_channel, kernel_size=3, activation=None, stride=1
        )
        self.body = nn.Sequential(
            *[
                ResidualBlock(n_feats, n_feats, activation=activation, norm=False)
                for _ in range(num_basic)
            ]
        )

    def forward(self, x):
        x0 = self.input_projection(x)
        x0 = self.body(x0)
        return self.output_projection(x0) + x


class MAGNet(nn.Module):
    """Return a blurred image and two lists of four kernel-token tensors."""

    def __init__(
        self,
        in_channel,
        out_channel,
        n_feats,
        device,
        img_length,
        activation=nn.ReLU(inplace=True),
        norm=True,
    ):
        super().__init__()
        self.img_length = img_length
        self.img_size = (self.img_length, self.img_length)
        self.center = (int(self.img_length / 2), int(self.img_length / 2))
        self.max_radius = self.img_length // 2
        self.transformed_width = 400
        self.data_type = torch.float32
        map_x, map_y = self.build_polar_map()
        inverse_x, inverse_y = self.build_cartesian_map()
        for name, values in (
            ("polar_map_x", map_x),
            ("polar_map_y", map_y),
            ("cartesian_map_x", inverse_x),
            ("cartesian_map_y", inverse_y),
        ):
            self.register_buffer(
                name,
                torch.tensor(values, device=device, dtype=self.data_type).unsqueeze(0),
                persistent=False,
            )
        self.sharp_stem0 = ConvBlock(
            in_channel,
            n_feats // 2,
            kernel_size=3,
            stride=1,
            activation=activation,
            norm=norm,
        )
        self.sharp_stem1 = ConvBlock(
            n_feats // 2,
            n_feats,
            kernel_size=3,
            stride=1,
            activation=activation,
            norm=norm,
        )
        self.kbu1 = KBU(n_feats, n_feats, 2 * n_feats)
        self.kbu2 = KBU(n_feats, n_feats, 2 * n_feats)
        self.blur_stem = ConvBlock(
            in_channel,
            n_feats,
            kernel_size=3,
            stride=1,
            activation=activation,
            norm=norm,
        )
        self.fusion = FusionBlock(2 * n_feats)
        self.polar_head = ConvBlock(
            n_feats, out_channel, kernel_size=3, stride=1, activation=None, norm=False
        )
        self.output_refinement = OutputRefinement(
            in_channel=out_channel,
            out_channel=out_channel,
            n_feats=n_feats,
            num_basic=5,
        )
        nn.init.zeros_(self.polar_head.layers[0].weight)
        nn.init.zeros_(self.polar_head.layers[0].bias)
        nn.init.zeros_(self.output_refinement.output_projection.layers[0].weight)
        nn.init.zeros_(self.output_refinement.output_projection.layers[0].bias)

    def angular_blur(self, x, kernel):
        B, C, A, R = x.shape
        if kernel.ndim != 4 or kernel.shape[:3] != (B, C, 1):
            raise ValueError(
                f"Expected kernel [{B}, {C}, 1, K], got {tuple(kernel.shape)}"
            )
        K = kernel.shape[-1]
        if K < 1 or K % 2 == 0:
            raise ValueError("Kernel length must be a positive odd integer")
        kernel = kernel.to(device=x.device, dtype=x.dtype)
        pad = K // 2
        x = F.pad(x, (0, 0, pad, pad), mode="circular")
        x = x.reshape(1, B * C, A + 2 * pad, R)
        weight = kernel.reshape(B * C, 1, K, 1)
        out = F.conv2d(x, weight, groups=B * C)
        return out.reshape(B, C, A, R)

    def build_polar_map(self, center=None, max_radius=None, height=None):
        center = self.center if center is None else center
        max_radius = self.max_radius if max_radius is None else max_radius
        h = self.transformed_width if height is None else height
        w = max_radius
        dsize = (h, w)
        mapx = np.zeros(dsize, dtype=np.float32)
        mapy = np.zeros(dsize, dtype=np.float32)
        Kangle = 2 * np.pi / h
        rhos = np.zeros((w,))
        Kmag = max_radius / w
        for rho in range(0, w):
            rhos[rho] = rho * Kmag
        for phi in range(0, h):
            KKy = Kangle * phi
            cp = np.cos(KKy)
            sp = np.sin(KKy)
            for rho in range(0, w):
                x = rhos[rho] * cp + center[1]
                y = rhos[rho] * sp + center[0]
                mapx[phi, rho] = x
                mapy[phi, rho] = y
        return (mapx, mapy)

    def build_cartesian_map(
        self, dsize=None, max_radius=None, center=None, src_size=None
    ):
        center = self.center if center is None else center
        max_radius = self.max_radius if max_radius is None else max_radius
        dsize = self.img_size if dsize is None else dsize
        src_size = (
            (self.transformed_width, self.max_radius) if src_size is None else src_size
        )
        w = dsize[1]
        h = dsize[0]
        angle_border = 1
        ssize_w = src_size[1]
        ssize_h = src_size[0]
        mapx = np.zeros(dsize, dtype=np.float32)
        mapy = np.zeros(dsize, dtype=np.float32)
        Kangle = 2 * np.pi / ssize_h
        Kmag = max_radius / ssize_w
        bufx = np.zeros(w, dtype=np.float32)
        bufy = np.zeros(w, dtype=np.float32)
        for x in range(0, w):
            bufx[x] = x - center[1]
        for y in range(0, h):
            for x in range(0, w):
                bufy[x] = y - center[0]
            bufp, bufa = cv2.cartToPolar(bufx, bufy, angleInDegrees=False)
            for x in range(0, w):
                rho = bufp[x] / Kmag
                phi = bufa[x] / Kangle
                mapx[y, x] = rho
                mapy[y, x] = phi + angle_border
        return (mapx, mapy)

    def cartesian_to_polar(self, img_tensor):
        img_tensor = img_tensor.to(device=self.polar_map_x.device, dtype=self.data_type)
        return kornia.geometry.transform.remap(
            img_tensor, self.polar_map_x, self.polar_map_y, align_corners=True
        )

    def polar_to_cartesian(self, img_tensor):
        img_tensor = img_tensor.to(
            device=self.cartesian_map_x.device, dtype=self.data_type
        )
        img_tensor = F.pad(img_tensor, pad=[0, 0, 1, 1], mode="circular")
        return kornia.geometry.transform.remap(
            img_tensor, self.cartesian_map_x, self.cartesian_map_y, align_corners=True
        )

    def forward(self, x, kernel):
        kernel = kernel.to(device=x.device, dtype=torch.float32)
        if torch.any(kernel < 0):
            raise ValueError("Blur kernel weights must be nonnegative")
        kernel_sum = kernel.sum(dim=-1, keepdim=True)
        if torch.any(kernel_sum <= 0):
            raise ValueError("Each blur kernel must have a positive weight sum")
        kernel = kernel / kernel_sum
        x_p = self.cartesian_to_polar(x)
        x_blur_p = self.angular_blur(x_p, kernel)
        baseline = x.float() + self.polar_to_cartesian(x_blur_p - x_p)
        blur_feat = self.blur_stem(x_blur_p)
        sharp_feat = self.sharp_stem0(x_p)
        sharp_feat = self.sharp_stem1(sharp_feat)
        x1, k_feat1 = self.kbu1(sharp_feat, kernel)
        x2, k_feat2 = self.kbu2(sharp_feat + blur_feat, kernel)
        fused = self.fusion(x1, x2)
        delta_p = self.polar_head(fused)
        delta = self.polar_to_cartesian(delta_p)
        delta = self.output_refinement(delta)
        prediction = baseline + delta
        return (prediction, k_feat1, k_feat2)

    Cart2PolarTensor = cartesian_to_polar
    Polar2CartTensor = polar_to_cartesian
