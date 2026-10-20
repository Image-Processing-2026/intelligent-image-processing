import os
import urllib.request


def download_image(url, filename):
    req = urllib.request.Request(
        url,
        data=None,
        headers={
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
        },
    )
    try:
        with urllib.request.urlopen(req) as response, open(filename, "wb") as out_file:
            out_file.write(response.read())
        print(f"✅ Đã tải: {filename}")
    except Exception as e:
        print(f"❌ Lỗi tải {filename}: {e}")


def main():
    os.makedirs("data/real", exist_ok=True)

    images = {
        # Ảnh chụp thiếu sáng (Underexposed)
        "data/real/underexposed_street.jpg": "https://upload.wikimedia.org/wikipedia/commons/4/40/Underexposed_photo.JPG",
        # Ảnh bị ám màu vàng (Color cast - Indoor Tungsten lighting)
        "data/real/color_cast_indoor.jpg": "https://upload.wikimedia.org/wikipedia/commons/3/3b/Tungsten_lighting_color_cast.jpg",
        # Ảnh bị nhiễu hạt (High ISO / Noise)
        "data/real/noisy_night.jpg": "https://upload.wikimedia.org/wikipedia/commons/1/1a/High_ISO_noise_example.jpg",
    }

    print("Đang tải các ảnh thật từ Wikimedia Commons...")
    for filename, url in images.items():
        download_image(url, filename)


if __name__ == "__main__":
    main()
