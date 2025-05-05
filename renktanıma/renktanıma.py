import cv2
import numpy as np
from collections import deque

# deque tespit edilen nesnenin ortası için
buffer_size = 16
pts = deque(maxlen=buffer_size)  # nesnenin merkez pointleri

# Mavi renk aralığı
bluelower =(90,100,0)
blueupper=(160,255,255)

# Kamera bağlantısını başlat
cap = cv2.VideoCapture(0)
cap.set(3, 960)
cap.set(4, 480)

while True:
    success, imgOriginal = cap.read()

    if success:  # kamera çalışıyor mu bakmak için
        # Blur
        blurred = cv2.GaussianBlur(imgOriginal, (11, 11), 0)
        # HSV dönüşümü
        hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)
        # Mavi için maske oluştur
        mask = cv2.inRange(hsv, bluelower, blueupper)
        # Maske üzerinde morfolojik işlemler uygula
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)
        # Konturları bul
        contours, _ = cv2.findContours(mask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        center = None

        if len(contours) > 0:
            # En büyük contouru al
            c = max(contours, key=cv2.contourArea)
            # Dikdörtgen al
            rect = cv2.minAreaRect(c)
            ((x, y), (width, height), rotation) = rect
            s = "x:{}, y:{}, width:{}, height:{}, rotation:{}".format(np.round(x), np.round(y), np.round(width),
                                                                      np.round(height), np.round(rotation))
            print(s)
            # Kutucuk
            box = cv2.boxPoints(rect)
            box = np.int64(box)
            # Moment
            M = cv2.moments(c)
            center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
            # Kontur çizdir sarı
            cv2.drawContours(imgOriginal, [box], 0, (0, 255, 255), 2)
            cv2.circle(imgOriginal, center, 5, (255, 0, 255), -1)
            # Bilgileri yaz
            cv2.putText(imgOriginal, s, (50, 50), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 0), 2)

        cv2.imshow("tespitli", imgOriginal)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# Kamera bağlantısını serbest bırak
cap.release()
cv2.destroyAllWindows()
