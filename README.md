# MAGNet

MAGNet generates rotational motion blur from sharp images and blur kernels. It combines a physical blur branch with a learnable correction branch built from Kernel-guided Blur U-Nets (KBU) and Wavelet Blur Modules (WBM).

- `magnet.py`: MAGNet architecture.
- `polar_fdl_loss.py`: Polar-FDL loss combining VGG-feature amplitude distributions, raw-image amplitude differences, and cross-radius relative phase, with default weights **1, 1, and 0.01**.

Dependencies: PyTorch, torchvision, NumPy, OpenCV, Kornia, and pytorch-wavelets.

```python
from polar_fdl_loss import PolarFDLLoss

criterion = PolarFDLLoss(image_size=512).to(device)
loss = criterion(prediction, target)
```

The loss expects Cartesian images in `[0, 1]`. VGG19 ImageNet weights are downloaded on first use unless a local `vgg_weights_path` is supplied. Keep the loss in FP32.
