#Nama: Dien Fadillah Prihardini
#NRP: 5024241051

import cv2

# 1. READ IMAGE
image = cv2.imread("pic.jpg")

if image is None:
    print("Gambar tidak ditemukan!")
else:
    print("Gambar berhasil dibaca.")

# 2. SHOW IMAGE
cv2.imshow("Original Image", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 3. FILTER COLOR IMAGE
red_image = cv2.imread("pic.jpg")

height, width, channel = red_image.shape

for row in range(height):
    for col in range(width):
        red_image[row, col, 0] = 0
        red_image[row, col, 1] = 0

cv2.imshow("Red Filter", red_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 4. FILTER COLOR VIDEO
camera = cv2.VideoCapture(0)

while camera.isOpened():
    success, frame = camera.read()

    if not success:
        print("Webcam tidak dapat diakses.")
        break

    height, width, channel = frame.shape

    for row in range(height):
        for col in range(width):
            frame[row, col, 0] = 0
            frame[row, col, 1] = 0

    cv2.imshow("Red Video Filter", frame)

    if cv2.waitKey(1) & 0xFF == ord("d"):
        break

camera.release()
cv2.destroyAllWindows()