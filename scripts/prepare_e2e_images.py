import os
import cv2
import numpy as np
from skimage import data

os.makedirs("data/real/e2e", exist_ok=True)

# Image 1: Chelsea
chelsea = data.chelsea()
chelsea_dark = (chelsea * 0.4).astype(np.uint8)
noise = np.random.normal(0, 25, chelsea_dark.shape)
chelsea_noisy = np.clip(chelsea_dark + noise, 0, 255).astype(np.uint8)
cv2.imwrite("data/real/e2e/test_chelsea_dark_noisy.jpg", cv2.cvtColor(chelsea_noisy, cv2.COLOR_RGB2BGR))

# Image 2: Astronaut
astro = data.astronaut()
astro_noisy = np.clip(astro + np.random.normal(0, 30, astro.shape), 0, 255).astype(np.uint8)
cv2.imwrite("data/real/e2e/test_astro_noisy.jpg", cv2.cvtColor(astro_noisy, cv2.COLOR_RGB2BGR))

# Image 3: Coffee cup
coffee = data.coffee()
coffee_blue = coffee.copy().astype(np.float32)
coffee_blue[:, :, 2] = np.clip(coffee_blue[:, :, 2] * 1.5, 0, 255)
cv2.imwrite("data/real/e2e/test_coffee_blue_cast.jpg", cv2.cvtColor(coffee_blue.astype(np.uint8), cv2.COLOR_RGB2BGR))
print("Done generating test images.")
