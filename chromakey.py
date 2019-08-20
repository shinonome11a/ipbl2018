# -*- coding: utf-8 -*-
import numpy as np
import cv2


# resize the long side to the specified size
def resizeImg(img, size):
    h, w, _ = img.shape
    if max(h, w) < size:
        return img
    if h < w:
        newSize = (size, int(h*size/w))
    else:
        newSize = (int(w*size/h), size)
    return cv2.resize(img, newSize)


# resize img_f and Match two image sizes by resizing
def resizeForChromaKey(img_b, img_f, size):
    img_f = resizeImg(img_f.copy(), size)
    h, w, c = img_f.shape
    img_b = cv2.resize(img_b, (w, h))
    return img_b, img_f


# calculate mask by using chromakey(hkey)
def calcMask(img_f, hkey):
    # calc HSV
    img_f_hsv = cv2.cvtColor(img_f, cv2.COLOR_BGR2HSV)
    # set mask range and calc mask
    lower_color = np.array([(hkey-30)/2, 30, 30], dtype=np.uint8)
    upper_color = np.array([(hkey+30)/2, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(img_f_hsv, lower_color, upper_color)
    # expand single channel to 3 channels
    mask = np.tile(mask[:, :, None], [1, 1, 3])
    # cv2.imshow("mask", mask)
    return mask


def maskDenoising(mask):
    # Let's try implementing this function
    return mask



def main():
    # hue: 000 -- 060 -- 120 -- 180 -- 240 -- 300 -- 360
    #      red  yellow  green   cyan   blue magenta  red
    # red has two regions (??)
    hkey = 120

    # read images
    img_b = cv2.imread("img/chromakey/background.jpg")  # background
    img_f = cv2.imread("img/chromakey/GreenBG3.JPG")  # foreground
    [img_b, img_f] = resizeForChromaKey(img_b, img_f, 512)

    # calc mask
    mask = calcMask(img_f, hkey)

    # mask denoising
    mask = maskDenoising(mask)

    # masking
    img = img_b
    img[mask == 0] = img_f[mask == 0]

    # show result
    cv2.imshow("res", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
