# -*- coding: utf-8 -*-
import numpy as np
import cv2


# calculate mask with RGB color space (it's very hard)
def calcMaskRGB(img_f):
    # set mask range and calc mask
    lower_color = np.array([0, 127, 0], dtype=np.uint8)
    upper_color = np.array([127, 255, 127], dtype=np.uint8)
    mask = cv2.inRange(img_f, lower_color, upper_color)
    # expand single channel to 3 channels
    mask = np.tile(mask[:, :, None], [1, 1, 3])
    return mask


# calculate mask with HSV color space
def calcMaskHSV(img_f, hkey):
    img_f_hsv = cv2.cvtColor(img_f, cv2.COLOR_BGR2HSV)
    # set mask range and calc mask
    lower_color = np.array([(hkey-30)/2, 30, 30], dtype=np.uint8)
    upper_color = np.array([(hkey+30)/2, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img_f_hsv, lower_color, upper_color)
    # expand single channel to 3 channels
    mask = np.tile(mask[:, :, None], [1, 1, 3])
    return mask


def main():
    # read image
    img_b = cv2.imread("img/chromakey/background.jpg")
    img_b = cv2.resize(img_b, (640, 480))

    # video setting
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)  # width
    cap.set(4, 480)  # height

    # capture loop
    while cap.isOpened():
        ret, frame = cap.read()

        mask = calcMaskRGB(frame)
        # hkey = 120 # green
        # mask = calcMaskHSV(frame, hkey)

        frame[mask == 255] = img_b[mask == 255]  # True == 255
        cv2.imshow("frame", frame)

        # stopping condition
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # press "q" key ==> stop
            break

    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
