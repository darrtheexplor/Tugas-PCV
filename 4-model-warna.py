import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# PENGATURAN

IMAGE_PATH = "pic.jpg"
OUTPUT_DIR = "hasil_pcv_spyder"
DEMO_IF_MISSING = True
SHOW_CHANNELS = True

# Kernel Gaussian 3x3 dari PDF 4
GAUSSIAN = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float64) / 16.0

EPS = 1e-12

# 1. KONVERSI RGB KE CMYK

def rgb_to_cmyk(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)

    k = 1.0 - np.max(rgb, axis=-1)
    pembagi = 1.0 - k

    cmy = np.divide(
        1.0 - rgb - k[..., None],
        pembagi[..., None],
        out=np.zeros_like(rgb),
        where=pembagi[..., None] > EPS
    )

    cmyk = np.concatenate(
        (cmy, k[..., None]), axis=-1
    )

    return np.clip(cmyk, 0, 1)


# 2. KONVERSI CMYK KE RGB

def cmyk_to_rgb(cmyk):
    cmyk = np.asarray(cmyk, dtype=np.float64)

    cmy = cmyk[..., :3]
    k = cmyk[..., 3:4]

    rgb = (1.0 - cmy) * (1.0 - k)

    return np.clip(rgb, 0, 1)


# 3. KONVERSI RGB KE HSI

def rgb_to_hsi(rgb):
    rgb = np.asarray(rgb, dtype=np.float64)

    r = rgb[..., 0]
    g = rgb[..., 1]
    b = rgb[..., 2]

    # Intensity
    intensity = (r + g + b) / 3.0

    # Saturation
    total = r + g + b
    min_rgb = np.min(rgb, axis=-1)

    saturation = np.divide(
        total - 3.0 * min_rgb,
        total,
        out=np.zeros_like(total),
        where=total > EPS
    )

    # Hue menggunakan atan2
    hue = np.degrees(
        np.arctan2(
            np.sqrt(3.0) * (g - b),
            2.0 * r - g - b
        )
    ) % 360.0

    # Hue warna abu-abu tidak terdefinisi
    hue = np.where(saturation > EPS, hue, 0.0)

    hsi = np.stack(
        (hue, saturation, intensity), axis=-1
    )

    return hsi


# 4. KONVERSI HSI KE RGB

def hsi_to_rgb(hsi):
    hsi = np.asarray(hsi, dtype=np.float64)

    h = np.deg2rad(hsi[..., 0] % 360.0)
    s = hsi[..., 1]
    i = hsi[..., 2]

    r = np.empty_like(i)
    g = np.empty_like(i)
    b = np.empty_like(i)

    # Sektor 0 sampai 120 derajat
    sektor1 = h < 2 * np.pi / 3

    hh = h[sektor1]

    b[sektor1] = i[sektor1] * (1 - s[sektor1])

    r[sektor1] = i[sektor1] * (
        1 + s[sektor1] * np.cos(hh) /
        np.cos(np.pi / 3 - hh)
    )

    g[sektor1] = (
        3 * i[sektor1] -
        r[sektor1] -
        b[sektor1]
    )

    # Sektor 120 sampai 240 derajat
    sektor2 = (
        (h >= 2 * np.pi / 3) &
        (h < 4 * np.pi / 3)
    )

    hh = h[sektor2] - 2 * np.pi / 3

    r[sektor2] = i[sektor2] * (1 - s[sektor2])

    g[sektor2] = i[sektor2] * (
        1 + s[sektor2] * np.cos(hh) /
        np.cos(np.pi / 3 - hh)
    )

    b[sektor2] = (
        3 * i[sektor2] -
        r[sektor2] -
        g[sektor2]
    )

    # Sektor 240 sampai 360 derajat
    sektor3 = ~(sektor1 | sektor2)

    hh = h[sektor3] - 4 * np.pi / 3

    g[sektor3] = i[sektor3] * (1 - s[sektor3])

    b[sektor3] = i[sektor3] * (
        1 + s[sektor3] * np.cos(hh) /
        np.cos(np.pi / 3 - hh)
    )

    r[sektor3] = (
        3 * i[sektor3] -
        g[sektor3] -
        b[sektor3]
    )

    rgb = np.stack((r, g, b), axis=-1)

    return np.clip(rgb, 0, 1)


# 5. KONVERSI RGB KE HSV

def rgb_to_hsv(rgb):
    rgb32 = np.asarray(rgb, dtype=np.float32)

    hsv = cv2.cvtColor(
        rgb32, cv2.COLOR_RGB2HSV
    )

    return hsv.astype(np.float64)


# 6. KONVERSI HSV KE RGB

def hsv_to_rgb(hsv):
    hsv32 = np.asarray(hsv, dtype=np.float32)

    rgb = cv2.cvtColor(
        hsv32, cv2.COLOR_HSV2RGB
    )

    return np.clip(rgb.astype(np.float64), 0, 1)


# 7. GAUSSIAN FILTER

def filter_kanal(kanal):
    hasil = cv2.filter2D(
        kanal.astype(np.float64),
        -1,
        GAUSSIAN,
        borderType=cv2.BORDER_REPLICATE
    )
    return hasil


def filter_hue(hue, saturation):
    # Hue bersifat melingkar, misalnya
    # 359 derajat berdekatan dengan 1 derajat.

    sudut = np.deg2rad(hue)

    x = filter_kanal(
        np.cos(sudut) * saturation
    )
    y = filter_kanal(
        np.sin(sudut) * saturation
    )

    hasil = np.degrees(
        np.arctan2(y, x)
    ) % 360.0

    return np.where(
        np.hypot(x, y) > EPS, hasil, 0.0
    )


# 8. FILTER DI SETIAP RUANG WARNA

def gaussian_di_ruang_warna(rgb, mode):

    if mode == "RGB":
        hasil = cv2.filter2D(
            rgb, -1, GAUSSIAN,
            borderType=cv2.BORDER_REPLICATE
        )
        return np.clip(hasil, 0, 1)

    if mode == "CMYK":
        data = rgb_to_cmyk(rgb)

        filtered = np.stack(
            [filter_kanal(data[..., c])
             for c in range(4)],
            axis=-1
        )

        return cmyk_to_rgb(np.clip(filtered, 0, 1))

    if mode == "HSI":
        data = rgb_to_hsi(rgb)
        filtered = np.empty_like(data)

        filtered[..., 0] = filter_hue(
            data[..., 0], data[..., 1]
        )
        filtered[..., 1] = np.clip(
            filter_kanal(data[..., 1]), 0, 1
        )
        filtered[..., 2] = np.clip(
            filter_kanal(data[..., 2]), 0, 1
        )

        return hsi_to_rgb(filtered)

    if mode == "HSV":
        data = rgb_to_hsv(rgb)
        filtered = np.empty_like(data)

        filtered[..., 0] = filter_hue(
            data[..., 0], data[..., 1]
        )
        filtered[..., 1] = np.clip(
            filter_kanal(data[..., 1]), 0, 1
        )
        filtered[..., 2] = np.clip(
            filter_kanal(data[..., 2]), 0, 1
        )

        return hsv_to_rgb(filtered)

    raise ValueError("Mode tidak dikenal: " + mode)


# 9. FUNGSI SIMPAN GAMBAR

def simpan_rgb(path, rgb):
    gambar8 = np.clip(
        np.rint(rgb * 255), 0, 255
    ).astype(np.uint8)

    gambar_bgr = cv2.cvtColor(
        gambar8, cv2.COLOR_RGB2BGR
    )

    cv2.imwrite(str(path), gambar_bgr)


# 10. MEMBUAT GAMBAR DEMO

def gambar_demo():
    yy, xx = np.mgrid[0:240, 0:360]

    rgb = np.stack((
        xx / 359,
        yy / 239,
        0.5 + 0.5 * np.sin(xx / 25)
    ), axis=-1)

    rng = np.random.default_rng(7)

    noise = rng.normal(0, 0.055, rgb.shape)

    return np.clip(rgb + noise, 0, 1)


# 11. PROGRAM UTAMA

def main():

    folder = Path(OUTPUT_DIR)
    folder.mkdir(parents=True, exist_ok=True)

    bgr = cv2.imread(IMAGE_PATH)

    if bgr is None:
        if DEMO_IF_MISSING:
            print("Gambar tidak ditemukan.")
            print("Menggunakan gambar demo.")
            rgb = gambar_demo()
        else:
            raise FileNotFoundError(IMAGE_PATH)
    else:
        rgb = cv2.cvtColor(
            bgr, cv2.COLOR_BGR2RGB
        ).astype(np.float64) / 255.0

    # Konversi ke setiap ruang warna
    cmyk = rgb_to_cmyk(rgb)
    hsi = rgb_to_hsi(rgb)
    hsv = rgb_to_hsv(rgb)

    print("Ukuran citra:", rgb.shape)

    print("Piksel pertama RGB:",
          np.round(rgb[0, 0], 4))
    print("Piksel pertama CMYK:",
          np.round(cmyk[0, 0], 4))
    print("Piksel pertama HSI:",
          np.round(hsi[0, 0], 4))
    print("Piksel pertama HSV:",
          np.round(hsv[0, 0], 4))

    # Uji konversi balik
    for nama, balik in [
        ("CMYK", cmyk_to_rgb(cmyk)),
        ("HSI", hsi_to_rgb(hsi)),
        ("HSV", hsv_to_rgb(hsv))
    ]:
        error = np.max(np.abs(rgb - balik))
        print("Error", nama, ":", error)

    # Penerapan Gaussian Filter
    hasil = {
        "Asli RGB": rgb
    }

    for mode in ["RGB", "CMYK", "HSI", "HSV"]:
        hasil["Gaussian " + mode] = (
            gaussian_di_ruang_warna(rgb, mode)
        )

    # Simpan hasil
    for nama, gambar in hasil.items():
        filename = nama.lower().replace(" ", "_")
        simpan_rgb(
            folder / (filename + ".png"), gambar
        )

    # Visualisasi perbandingan
    fig, axs = plt.subplots(
        1, 5, figsize=(17, 4)
    )

    for ax, (nama, gambar) in zip(
        axs, hasil.items()
    ):
        ax.imshow(gambar)
        ax.set_title(nama)
        ax.axis("off")

    fig.suptitle(
        "Perbandingan Gaussian Filter 3x3"
    )
    fig.tight_layout()

    fig.savefig(
        folder / "perbandingan.png",
        dpi=150
    )

    # Visualisasi setiap kanal warna
    if SHOW_CHANNELS:

        fig2, axs2 = plt.subplots(
            3, 4, figsize=(13, 9)
        )

        data = [
            ("CMYK", cmyk, ["C", "M", "Y", "K"]),
            ("HSI", hsi, ["H", "S", "I"]),
            ("HSV", hsv, ["H", "S", "V"])
        ]

        for row, (nama, arr, channels) in enumerate(data):

            for col, ch_name in enumerate(channels):

                ch = arr[..., col]

                if ch_name == "H":
                    ch = ch / 360.0

                axs2[row, col].imshow(
                    ch,
                    cmap="gray",
                    vmin=0,
                    vmax=1
                )
                axs2[row, col].set_title(
                    nama + " - " + ch_name
                )
                axs2[row, col].axis("off")

            for col in range(len(channels), 4):
                axs2[row, col].axis("off")

        fig2.suptitle(
            "Visualisasi Kanal Warna"
        )
        fig2.tight_layout()

        fig2.savefig(
            folder / "kanal_warna.png",
            dpi=150
        )

    print("Hasil tersimpan di:", folder.resolve())

    plt.show()


if __name__ == "__main__":
    main()
