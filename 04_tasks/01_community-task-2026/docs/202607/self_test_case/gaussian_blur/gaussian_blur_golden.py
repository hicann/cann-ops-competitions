import numpy as np
import cv2


def gaussian_blur_golden(self, ksize_x, ksize_y, sigmaX, sigmaY):
    """
    高斯模糊标杆函数 (对齐 cv2.GaussianBlur)
    Args:
        self:   输入图像, numpy.ndarray, float32, shape (H,W) 或 (H,W,C)
        ksize_x: 高斯核宽 (x方向), int, 正奇数
        ksize_y: 高斯核高 (y方向), int, 正奇数
        sigmaX:  X方向标准差, float, 0 表示由 ksize 自动计算
        sigmaY:  Y方向标准差, float, 0 表示与 sigmaX 相同
    Returns:
        [out]: 模糊后的图像, numpy.ndarray, float32, shape 与输入一致
    """
    img = self.astype(np.float32)
    kx = int(ksize_x)
    ky = int(ksize_y)
    sx = float(sigmaX)
    sy = float(sigmaY)
    out = cv2.GaussianBlur(img, (kx, ky), sx, sigmaY=sy)
    # cv2 会把 (H,W,1) 降维为 (H,W)，需还原原始 shape
    if out.shape != img.shape:
        out = out.reshape(img.shape)
    return [out.astype(np.float32)]
