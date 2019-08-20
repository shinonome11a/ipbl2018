# -*- coding: utf-8 -*-
import numpy as np
import cv2
import testMouseEvent as me

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
    h,w,c = img_f.shape
    img_b = cv2.resize(img_b, (w,h))
    return img_b, img_f

# calculate mask by using chromakey(hkey) 
def calcMask(img_b, img_f, hkey):
    # calc HSV
    img_f_hsv = cv2.cvtColor(img_f, cv2.COLOR_BGR2HSV)
    # set mask range and calc mask
    lower_color = np.array([(hkey-20)/2,30,30], dtype=np.uint8)
    upper_color = np.array([(hkey+20)/2,255,255], dtype=np.uint8)
    mask = cv2.inRange(img_f_hsv, lower_color, upper_color)
    # denoising
    mask = denoiseForMask(mask)
    # expand single channel to 3 channels
    mask = np.tile(mask[:,:,None], [1,1,3])
    # show mask image
    cv2.imshow("mask", mask)
    cv2.waitKey(0)
    return mask

def denoiseForMask(mask):
    # opening
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)
    # delete small connected elements by using labeling
    mask = 255-mask # only white areas (255) are recognized as connected elements
    nLabels, labelimg, contours, CoGs = cv2.connectedComponentsWithStats(mask)
    for label in range(nLabels):
        _,_,_,_,size = contours[label]
        if size < 100:
            mask[labelimg==label] = 0
    return 255-mask

def main():
    # read images
    img_b = cv2.imread("img/chromakey/background.jpg")
    img_f = cv2.imread("img/chromakey/GreenBG3.JPG")
    # resize images
    [img_b, img_f] = resizeForChromaKey(img_b, img_f, 512)

    #hkey = 120;
    hkey = me.main2(img_f)
    print(hkey)

    # calculate mask
    mask = calcMask(img_b, img_f, hkey)
    # masking
    img = img_b
    img[mask==0] = img_f[mask==0]
    # show res image
    cv2.imshow("res", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()