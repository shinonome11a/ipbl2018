# -*- coding: utf-8 -*-

import cv2
import numpy as np
import pyrFunc as pf


def main():
    # input image files
    A = cv2.imread('./img/house/C.jpg')
    B = cv2.imread('./img/house/B.jpg')

    # calculate pyramid depth (level)
    nlv = int(np.log(np.min(A.shape[:2]))/np.log(2))
    print('nlv:', nlv)

    # generate Gaussian pyramid for A
    gpA = pf.generateGussianPyr(A.copy(), nlv)
    pf.showPyr(gpA.copy())

    # generate Gaussian pyramid for B
    gpB = pf.generateGussianPyr(B.copy(), nlv)
    pf.showPyr(gpB.copy())

    # generate Laplacian Pyramid for A
    lpA = pf.generateLaplacianPyr(A.copy(), nlv)
    pf.showPyr(lpA.copy())

    # generate Laplacian Pyramid for B
    lpB = pf.generateLaplacianPyr(B.copy(), nlv)
    pf.showPyr(lpB.copy())

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls_ = np.hstack((la[:, :int(cols/2)], lb[:, int(cols/2):]))
        LS.append(ls_)
    pf.showPyr(LS.copy())

    # now reconstruct
    rImg = pf.reconstructLaplacianPyr(LS, nlv)
    rImg[rImg > 255] = 255

    # image with direct connecting each half
    rows, cols, dpt = A.shape
    rImg_direct = np.hstack((A[:, :int(cols/2)], B[:, int(cols/2):]))

    cv2.imshow('Pyramid_blending2', rImg)
    cv2.imshow('Direct_blending', rImg_direct)
    cv2.waitKey(0)
    cv2.destroyAllWindows

    # output image files
    cv2.imwrite('res/Pyramid_blendingCB.jpg', rImg)
#    cv2.imwrite('res/Direct_blending.jpg',rImg_direct)


if __name__ == '__main__':
    main()
