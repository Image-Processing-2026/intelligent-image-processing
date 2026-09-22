"""
Benchmark tham số cho Module 3 (denoise, gamma) bằng PSNR/SSIM
trên dữ liệu synthetic tự sinh (có ground-truth).
Dùng lại evaluate_reference() của Module 1 để nhất quán chỉ số.
"""
import csv
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np

from src.analyzer_evaluator.reference_eval import evaluate_reference
from src.processing_engine.denoise import apply_denoise
from src.processing_engine.exposure_contrast import apply_gamma
 
 
def make_clean_image(seed: int, size: int = 256) -> np.ndarray:
    """Ảnh sạch có cấu trúc rõ (để SSIM có ý nghĩa), không phải nhiễu thuần."""
    rng = np.random.default_rng(seed)
    x, y = np.meshgrid(np.linspace(0, 4 * np.pi, size), np.linspace(0, 4 * np.pi, size))
    pattern = (np.sin(x) * np.cos(y) + 1) / 2  # 0..1, có cấu trúc lượn sóng
    base = (pattern * 180 + 40).astype(np.uint8)  # dải sáng vừa phải, không cháy/tối
    img = np.stack([base] * 3, axis=-1)
    # thêm vài khối hình chữ nhật ngẫu nhiên mô phỏng "vật thể"
    for _ in range(5):
        x0, y0 = rng.integers(0, size - 40, 2)
        w, h = rng.integers(20, 40, 2)
        color = rng.integers(50, 220, 3)
        img[y0 : y0 + h, x0 : x0 + w] = color
    return img.astype(np.uint8)
 
 
def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.default_rng(0).normal(0, sigma, img.shape)
    return np.clip(img.astype(float) + noise, 0, 255).astype(np.uint8)
 
 
def darken(img: np.ndarray, factor: float) -> np.ndarray:
    return np.clip(img.astype(float) * factor, 0, 255).astype(np.uint8)
 
 
def main():
    rows = []
    clean_images = [make_clean_image(seed) for seed in range(3)]  # 3 ảnh synthetic khác nhau
 
    # ---- Benchmark DENOISE ----
    methods = ["gaussian", "median", "bilateral", "nlm"]
    strengths = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    for method in methods:
        for strength in strengths:
            psnrs, ssims = [], []
            for clean in clean_images:
                noisy = add_gaussian_noise(clean, sigma=20)
                result = apply_denoise(noisy, method=method, strength=strength)
                m = evaluate_reference(result, clean)
                psnrs.append(m["psnr"])
                ssims.append(m["ssim"])
            rows.append(
                {
                    "task": "denoise",
                    "param": f"{method}, strength={strength}",
                    "psnr": round(np.mean(psnrs), 2),
                    "ssim": round(np.mean(ssims), 4),
                }
            )
 
    # ---- Benchmark GAMMA ----
    for gamma in [1.0, 1.2, 1.5, 1.8, 2.0, 2.2, 2.5, 3.0]:
        psnrs, ssims = [], []
        for clean in clean_images:
            dark = darken(clean, factor=0.35)
            result = apply_gamma(dark, gamma=gamma)
            m = evaluate_reference(result, clean)
            psnrs.append(m["psnr"])
            ssims.append(m["ssim"])
        rows.append(
            {
                "task": "gamma",
                "param": f"gamma={gamma}",
                "psnr": round(np.mean(psnrs), 2),
                "ssim": round(np.mean(ssims), 4),
            }
        )
 
    # In kết quả + lưu CSV
    with open("benchmark_results.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["task", "param", "psnr", "ssim"])
        writer.writeheader()
        writer.writerows(rows)
 
    for task in ["denoise", "gamma"]:
        print(f"\n=== {task.upper()} — xếp theo SSIM giảm dần ===")
        task_rows = sorted([r for r in rows if r["task"] == task], key=lambda r: -r["ssim"])
        for r in task_rows[:5]:
            print(f"  {r['param']:35s} PSNR={r['psnr']:6.2f}  SSIM={r['ssim']:.4f}")
 
    print("\nĐã lưu chi tiết đầy đủ vào benchmark_results.csv")
 
 
if __name__ == "__main__":
    main()
 