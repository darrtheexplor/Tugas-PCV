import cv2
import numpy as np
import matplotlib.pyplot as plt

# 1. MEMBACA GAMBAR

gambar = cv2.imread("pic.jpg", cv2.IMREAD_GRAYSCALE)

if gambar is None:
    raise FileNotFoundError("File pic.jpg tidak ditemukan!")

# 2. KONVERSI KE CITRA BINER

_, biner = cv2.threshold(
    gambar, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# 3. STRUCTURING ELEMENT 3x3

kernel = cv2.getStructuringElement(
    cv2.MORPH_RECT, (3, 3)
)

print("Structuring Element:")
print(kernel)

# Sesuai PDF: luar citra dianggap latar
tepi = dict(
    borderType=cv2.BORDER_CONSTANT,
    borderValue=0
)

# 4. EROSI

erosi = cv2.erode(
    biner, kernel, iterations=1, **tepi
)

# 5. DILASI

dilasi = cv2.dilate(
    biner, kernel, iterations=1, **tepi
)

# 6. OPENING = EROSI + DILASI

opening = cv2.morphologyEx(
    biner, cv2.MORPH_OPEN, kernel, **tepi
)

# 7. CLOSING = DILASI + EROSI

closing = cv2.morphologyEx(
    biner, cv2.MORPH_CLOSE, kernel, **tepi
)

# 8. MENAMPILKAN HASIL

judul = [
    "Citra Asli",
    "Citra Biner",
    "Erosi",
    "Dilasi",
    "Opening",
    "Closing"
]

hasil = [
    gambar, biner, erosi,
    dilasi, opening, closing
]

plt.figure(figsize=(12, 8))

for i in range(6):
    plt.subplot(2, 3, i + 1)
    plt.imshow(hasil[i], cmap="gray",
               vmin=0, vmax=255)
    plt.title(judul[i])
    plt.axis("off")

plt.suptitle("Operasi Morfologi Citra")
plt.tight_layout()
plt.show()

# 9. MENYIMPAN HASIL

cv2.imwrite("hasil_erosi.png", erosi)
cv2.imwrite("hasil_dilasi.png", dilasi)
cv2.imwrite("hasil_opening.png", opening)
cv2.imwrite("hasil_closing.png", closing)

print("Semua operasi selesai.")
