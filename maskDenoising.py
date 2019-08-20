import numpy as np
import cv2

def maskDenoising(mask):

    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    # res1 = cv2.dilate(mask, kernel, iterations=1)
    # res2 = cv2.erode(mask, kernel, iterations=1)
    # res = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # res = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    nLabels, labelimg, contours, CoGs = cv2.connectedComponentsWithStats(mask)

    res5 = mask
    for i in range(nLabels):
        if contours[i, 4] < 300:
            res5[labelimg == i] = 0

    # cv2.imwrite("./res/dilate.png", res1*255)
    # cv2.imwrite("./res/erode.png", res2*255)
    # cv2.imwrite("./res/opening.png", res3*255)
    # cv2.imwrite("./res/closing.png", res4*255)
    # cv2.imwrite("./res/removing.png", res5*255)

    return res5


def main():
    img = cv2.imread("./img/noisyBinaryImg.png", 0)
    mask = (img > 127).astype(np.uint8)

    # cv2.imshow("mask", mask*255)
    # cv2.waitKey(0)

    # mask_res = maskDenoising(mask)
    maskDenoising(mask)

    # cv2.imshow("res", mask_res*255)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
