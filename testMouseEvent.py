# -*- coding: utf-8 -*-
import cv2
import numpy as np

dragFlag = False
ix, iy = -1, -1

def mouse_event(event, x, y, flags, param):
    global img, img_orig, hue, dragFlag, ix, iy

    if event ==cv2.EVENT_LBUTTONDOWN:
        dragFlag = True
        ix, iy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if dragFlag == True:
            img = img_orig.copy()
            cv2.rectangle(img, (ix,iy), (x,y), (0,255,0), 1)

    elif event == cv2.EVENT_LBUTTONUP:
        dragFlag = False
        cv2.rectangle(img, (ix,iy), (x,y), (0,0,255), 1)
        imgROI = img[iy:y, ix:x, :]
        meanBGR = img.mean(axis=0).mean(axis=0)
        meanBGR = np.reshape(meanBGR,(1,1,3))
        hsv = cv2.cvtColor(meanBGR.astype(np.uint8), cv2.COLOR_BGR2HSV)
        hue = hsv[0,0,0]

def main2(img_):
    global img, img_orig, hue

    img = img_
    img_orig = img.copy()

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("img", mouse_event)

    while (True):
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()
    return hue*2

def main():
    global img, img_orig

    img = cv2.imread("img.jpg", 1)
    img_orig = img.copy()

    cv2.namedWindow("img", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("img", mouse_event)

    while (True):
        cv2.imshow("img", img)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cv2.destroyAllWindows()

if __name__=='__main__':
    main()
