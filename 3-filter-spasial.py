import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. MEMBACA CITRA

im = cv2.imread("pic.jpg", cv2.IMREAD_GRAYSCALE)

if im is None:
    raise FileNotFoundError(
        "Gambar tidak ditemukan! Periksa nama dan lokasi gambar."
    )

imf = im.astype(np.float64)

print("Gambar berhasil dibaca.")


# 2. FUNGSI KONVOLUSI MANUAL

def konvolusi(f, kernel, tepi="replicate"):
    m, n = kernel.shape
    a, b = m // 2, n // 2

    # Membalik kernel 180 derajat untuk konvolusi
    kernel_putar = kernel[::-1, ::-1]

    # Penanganan tepi
    if tepi == "zero":
        f_pad = np.pad(
            f, ((a, a), (b, b)),
            mode="constant", constant_values=0
        )
    elif tepi == "replicate":
        f_pad = np.pad(
            f, ((a, a), (b, b)), mode="edge"
        )
    elif tepi == "reflect":
        f_pad = np.pad(
            f, ((a, a), (b, b)), mode="reflect"
        )
    else:
        raise ValueError("Metode tepi tidak dikenal.")

    tinggi, lebar = f.shape
    hasil = np.zeros((tinggi, lebar), dtype=np.float64)

    # Menggeser kernel ke seluruh posisi citra
    for x in range(tinggi):
        for y in range(lebar):
            jendela = f_pad[x:x + m, y:y + n]

            total = 0
            for i in range(m):
                for j in range(n):
                    total += (
                        jendela[i, j] * kernel_putar[i, j]
                    )

            hasil[x, y] = total

    return hasil


# 3. FUNGSI KONVERSI KE UINT8

def ke_uint8(g):
    # Membulatkan dan membatasi nilai ke rentang 0-255
    g = np.floor(g + 0.5)
    g = np.clip(g, 0, 255)

    return g.astype(np.uint8)


# 4. FILTER LOWPASS: BOX FILTER

kernel_box = np.ones((3, 3), dtype=np.float64) / 9.0

hasil_box = konvolusi(imf, kernel_box)
hasil_box = ke_uint8(hasil_box)


# 5. FILTER LOWPASS: GAUSSIAN FILTER

kernel_gauss = np.array([
    [1, 2, 1],
    [2, 4, 2],
    [1, 2, 1]
], dtype=np.float64) / 16.0

hasil_gauss = konvolusi(imf, kernel_gauss)
hasil_gauss = ke_uint8(hasil_gauss)


# 6. FILTER MEDIAN

tinggi, lebar = im.shape
hasil_median = np.zeros((tinggi, lebar), dtype=np.uint8)

# Median merupakan filter nonlinear, bukan konvolusi
im_pad = np.pad(im, ((1, 1), (1, 1)), mode="edge")

for x in range(tinggi):
    for y in range(lebar):
        jendela = []

        for i in range(3):
            for j in range(3):
                jendela.append(int(im_pad[x + i, y + j]))

        jendela.sort()
        hasil_median[x, y] = jendela[4]


# 7. FILTER HIGHPASS: LAPLACIAN

kernel_lap = np.array([
    [0, 1, 0],
    [1, -4, 1],
    [0, 1, 0]
], dtype=np.float64)

lap = konvolusi(imf, kernel_lap)

# Respons Laplacian untuk visualisasi
hasil_lap = ke_uint8(np.abs(lap))


# 8. PENAJAMAN DENGAN LAPLACIAN

# Kernel Laplacian memiliki nilai pusat -4,
# sehingga citra tajam = citra asli - Laplacian
tajam = imf - lap

hasil_tajam = ke_uint8(tajam)


# 9. UNSHARP MASKING DAN HIGHBOOST

# Mask adalah selisih citra asli dan citra halus
mask = imf - hasil_box.astype(np.float64)

# Unsharp masking: k = 1
unsharp = imf + 1.0 * mask
hasil_unsharp = ke_uint8(unsharp)

# Highboost: k > 1
highboost = imf + 2.5 * mask
hasil_highboost = ke_uint8(highboost)


# 10. FILTER SOBEL

kernel_sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float64)

kernel_sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]
], dtype=np.float64)

gx = konvolusi(imf, kernel_sobel_x)
gy = konvolusi(imf, kernel_sobel_y)

# Hampiran besar gradien sesuai materi
besar = np.abs(gx) + np.abs(gy)
hasil_sobel = ke_uint8(besar)


# 11. MENAMPILKAN HASIL

judul = [
    "Citra Asli",
    "Box Filter",
    "Gaussian Filter",
    "Median Filter",
    "Laplacian",
    "Penajaman Laplacian",
    "Unsharp Masking",
    "Highboost",
    "Sobel"
]

daftar_hasil = [
    im,
    hasil_box,
    hasil_gauss,
    hasil_median,
    hasil_lap,
    hasil_tajam,
    hasil_unsharp,
    hasil_highboost,
    hasil_sobel
]

plt.figure(figsize=(14, 12))

for i in range(len(daftar_hasil)):
    plt.subplot(3, 3, i + 1)
    plt.imshow(daftar_hasil[i], cmap="gray", vmin=0, vmax=255)
    plt.title(judul[i])
    plt.axis("off")

plt.tight_layout()
plt.show()

# 12. MENYIMPAN HASIL

cv2.imwrite("hasil_box.jpg", hasil_box)
cv2.imwrite("hasil_gaussian.jpg", hasil_gauss)
cv2.imwrite("hasil_median.jpg", hasil_median)
cv2.imwrite("hasil_laplacian.jpg", hasil_lap)
cv2.imwrite("hasil_penajaman.jpg", hasil_tajam)
cv2.imwrite("hasil_unsharp.jpg", hasil_unsharp)
cv2.imwrite("hasil_highboost.jpg", hasil_highboost)
cv2.imwrite("hasil_sobel.jpg", hasil_sobel)

print("Semua proses filter spasial selesai.")

