import cv2
import numpy as np

# загрузка изображения
img = cv2.imread('variant-1.jpg')

# перевести изображение в полутоновое
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# сохранить полутоновое изображение
cv2.imwrite('variant-1-grayscale.jpg', gray_img)

# загрузка изображения мухи
fly_img = cv2.imread('fly64.png')

# определение параметров маркера
dp = 1
minDist = 20
param1 = 50
param2 = 30
minRadius = 5
maxRadius = 50

# видеопоток
cap = cv2.VideoCapture(0)

while True:
    # кадр с камеры
    ret, frame = cap.read()

    # перевод кадра в полутон
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # блюр для уменьшения шума
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)

    # находим маркер
    circles = cv2.HoughCircles(blurred, cv2.HOUGH_GRADIENT, dp, minDist, param1=param1, param2=param2, minRadius=minRadius, maxRadius=maxRadius)

    # если маркер найден
    if circles is not None:
        # координаты в переменные
        circles = np.round(circles[0, :]).astype("int")

        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 2)

            # поместим муху в центр маркера
            marker_size = int(r * 2)
            fly_img_resized = cv2.resize(fly_img, (marker_size, marker_size))
            x_offset = x - int(marker_size / 2)
            y_offset = y - int(marker_size / 2)
            frame[y_offset:y_offset+marker_size, x_offset:x_offset+marker_size] = fly_img_resized

            # отобразим координаты маркера
            cv2.putText(frame, f"({x}, {y})", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # отобразим кадр
    cv2.imshow('frame', frame)

    # кнопка q для выхода
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# закрыть окна и отключить камеру
cap.release()
cv2.destroyAllWindows()
