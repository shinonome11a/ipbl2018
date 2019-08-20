# -*- coding: utf-8 -*-

import cv2
import numpy as np


def main():
    # input image files
    A = cv2.imread('./img/house/C.jpg')
    B = cv2.imread('./img/house/B.jpg')

    # image with direct connecting each half
    rows, cols, dpt = A.shape

    # short version (without mask)
#    rImg = np.hstack((A[:,:int(cols/2),:],B[:,int(cols/2):,:]))

    # version (with each masks)
    leftBlock = (rows, int(np.floor(cols/2)), dpt)
    rightBlock = (rows, int(np.ceil(cols/2)), dpt)
    maskA = np.hstack((np.ones(leftBlock), np.zeros(rightBlock)))
    maskB = np.ones((rows, cols, dpt)) - maskA
    rImg = maskA*A + maskB*B

    # show result image
    cv2.imshow('masking_half&half', rImg.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows

    cv2.imwrite('res/maskingCB.png', rImg)
    cv2.imwrite('res/maskA.png', maskA*255)
    cv2.imwrite('res/maskB.png', maskB*255)


if __name__ == '__main__':
    main()
