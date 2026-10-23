import os

import cv2
import numpy as np
from skimage import data


def main():
    os.makedirs("data/real", exist_ok=True)

    # 1. Ảnh thiếu sáng (Underexposed) - Dùng ảnh quán cafe
    coffee = data.coffee()  # RGB
    coffee_bgr = cv2.cvtColor(coffee, cv2.COLOR_RGB2BGR)
    under = (coffee_bgr.astype(np.float32) * 0.35).clip(0, 255).astype(np.uint8)
    cv2.imwrite("data/real/coffee_underexposed.jpg", under)
    print("✅ Đã tạo: data/real/coffee_underexposed.jpg (Ảnh thật bị thiếu sáng)")

    # 2. Ảnh nhiễu hạt (High ISO) - Dùng ảnh phi hành gia
    astro = data.astronaut()
    astro_bgr = cv2.cvtColor(astro, cv2.COLOR_RGB2BGR)
    noise = np.random.normal(0, 30, astro_bgr.shape)
    noisy = (astro_bgr.astype(np.float32) + noise).clip(0, 255).astype(np.uint8)
    cv2.imwrite("data/real/astro_noisy.jpg", noisy)
    print("✅ Đã tạo: data/real/astro_noisy.jpg (Ảnh thật bị nhiễu hạt ISO cao)")

    # 3. Ảnh ám màu (Color cast) - Dùng ảnh con mèo Chelsea
    cat = data.chelsea()
    cat_bgr = cv2.cvtColor(cat, cv2.COLOR_RGB2BGR).astype(np.float32)
    # Ám vàng/đỏ (Warm)
    cat_bgr[:, :, 2] += 60  # Red
    cat_bgr[:, :, 1] += 30  # Green
    cat_bgr[:, :, 0] -= 20  # Blue
    cast = cat_bgr.clip(0, 255).astype(np.uint8)
    cv2.imwrite("data/real/cat_warm_cast.jpg", cast)
    print("✅ Đã tạo: data/real/cat_warm_cast.jpg (Ảnh thật bị ám màu vàng ấm)")


if __name__ == "__main__":
    main()
