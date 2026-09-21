#Nama: Dien Fadillah Prihardini
#NRP: 5024241051

import cv2
import numpy as np
import matplotlib.pyplot as plt


# 1. READ IMAGE

image = cv2.imread("pic.jpg", cv2.IMREAD_GRAYSCALE)

if image is None:
    print("Gambar tidak ditemukan!")
    exit()

print("Gambar berhasil dibaca.")


# 2. FUNGSI HISTOGRAM MANUAL

def histogram_manual(image):
    hist = [0] * 256

    tinggi = image.shape[0]
    lebar = image.shape[1]

    for i in range(tinggi):
        for j in range(lebar):
            nilai = int(image[i, j])
            hist[nilai] += 1

    return hist


# 3. TRANSFORMASI NEGATIF
#    s = 255 - r

def transformasi_negatif(image):
    hasil = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = int(image[i, j])
            hasil[i, j] = 255 - r

    return hasil


# 4. TRANSFORMASI LOGARITMIK
#    s = c * log(1 + r)
#    c = 255 / log(256)

def transformasi_log(image):
    hasil = np.zeros_like(image)

    c = 255.0 / np.log(256.0)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = int(image[i, j])

            s = c * np.log(1 + r)

            # pembulatan manual
            s = int(np.floor(s + 0.5))

            if s > 255:
                s = 255
            if s < 0:
                s = 0

            hasil[i, j] = s

    return hasil


# 5. TRANSFORMASI GAMMA
#    s = 255 * (r / 255)^gamma

def transformasi_gamma(image, gamma):
    hasil = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            r = int(image[i, j])

            s = 255.0 * ((r / 255.0) ** gamma)

            s = int(np.floor(s + 0.5))

            if s > 255:
                s = 255
            if s < 0:
                s = 0

            hasil[i, j] = s

    return hasil


# 6. CONTRAST STRETCHING
#
# r1 = 80
# s1 = 20
# r2 = 175
# s2 = 240

def contrast_stretching(image, r1=80, s1=20, r2=175, s2=240):
    hasil = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            r = int(image[i, j])

            if r < r1:
                s = (s1 / r1) * r

            elif r < r2:
                s = ((s2 - s1) / (r2 - r1)) * (r - r1) + s1

            else:
                s = ((255 - s2) / (255 - r2)) * (r - r2) + s2

            s = int(np.floor(s + 0.5))

            if s > 255:
                s = 255
            if s < 0:
                s = 0

            hasil[i, j] = s

    return hasil


# 7. THRESHOLDING
#    r < T  -> 0
#    r >= T -> 255

def thresholding(image, T=128):
    hasil = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            r = int(image[i, j])

            if r < T:
                hasil[i, j] = 0
            else:
                hasil[i, j] = 255

    return hasil


# 8. EKUALISASI HISTOGRAM MANUAL

def ekualisasi_histogram(image):

    tinggi = image.shape[0]
    lebar = image.shape[1]

    jumlah_piksel = tinggi * lebar

    # Hitung histogram secara manual

    hist = [0] * 256

    for i in range(tinggi):
        for j in range(lebar):

            nilai = int(image[i, j])
            hist[nilai] += 1

    # Hitung probabilitas setiap intensitas

    prob = [0.0] * 256

    for k in range(256):
        prob[k] = hist[k] / jumlah_piksel

    # Hitung CDF secara manual

    cdf = [0.0] * 256

    cdf[0] = prob[0]

    for k in range(1, 256):
        cdf[k] = cdf[k - 1] + prob[k]

    # Buat LUT ekualisasi
    #
    # s = (L - 1) * CDF

    lut = [0] * 256

    for k in range(256):

        s = 255 * cdf[k]

        # pembulatan setengah ke atas
        s = int(np.floor(s + 0.5))

        if s > 255:
            s = 255

        if s < 0:
            s = 0

        lut[k] = s

    # Terapkan LUT ke citra

    hasil = np.zeros_like(image)

    for i in range(tinggi):
        for j in range(lebar):

            nilai = int(image[i, j])
            hasil[i, j] = lut[nilai]

    return hasil


# 9. PROSES SEMUA TRANSFORMASI

hasil_negatif = transformasi_negatif(image)

hasil_log = transformasi_log(image)

hasil_gamma = transformasi_gamma(image, 0.4)

hasil_stretch = contrast_stretching(image)

hasil_threshold = thresholding(image, 128)

hasil_equalization = ekualisasi_histogram(image)


# 10. HISTOGRAM MANUAL

hist_original = histogram_manual(image)
hist_equalization = histogram_manual(hasil_equalization)


# 11. MENAMPILKAN HASIL

plt.figure(figsize=(12, 8))

plt.subplot(2, 4, 1)
plt.imshow(image, cmap="gray")
plt.title("Original")
plt.axis("off")

plt.subplot(2, 4, 2)
plt.imshow(hasil_negatif, cmap="gray")
plt.title("Negative")
plt.axis("off")

plt.subplot(2, 4, 3)
plt.imshow(hasil_log, cmap="gray")
plt.title("Log")
plt.axis("off")

plt.subplot(2, 4, 4)
plt.imshow(hasil_gamma, cmap="gray")
plt.title("Gamma = 0.4")
plt.axis("off")

plt.subplot(2, 4, 5)
plt.imshow(hasil_stretch, cmap="gray")
plt.title("Contrast Stretching")
plt.axis("off")

plt.subplot(2, 4, 6)
plt.imshow(hasil_threshold, cmap="gray")
plt.title("Thresholding")
plt.axis("off")

plt.subplot(2, 4, 7)
plt.imshow(hasil_equalization, cmap="gray")
plt.title("Histogram Equalization")
plt.axis("off")

plt.subplot(2, 4, 8)
plt.bar(range(256), hist_equalization)
plt.title("Equalized Histogram")

plt.tight_layout()
plt.show()


# 12. TAMPILKAN HISTOGRAM ORIGINAL

plt.figure(figsize=(10, 4))
plt.bar(range(256), hist_original)
plt.title("Histogram Original")
plt.xlabel("Intensitas")
plt.ylabel("Jumlah Piksel")
plt.show()


# 13. SIMPAN HASIL

cv2.imwrite("hasil_negatif.jpg", hasil_negatif)
cv2.imwrite("hasil_log.jpg", hasil_log)
cv2.imwrite("hasil_gamma.jpg", hasil_gamma)
cv2.imwrite("hasil_stretch.jpg", hasil_stretch)
cv2.imwrite("hasil_threshold.jpg", hasil_threshold)
cv2.imwrite("hasil_equalization.jpg", hasil_equalization)

print("Semua proses selesai.")