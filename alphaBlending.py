# -*- coding: utf-8 -*-

import numpy as np
import cv2


def main():
    # input image files
    folderNM = './img/house/'
    fNM = ['A.jpg', 'B.jpg', 'C.jpg', 'D.jpg']
    imgs = [cv2.imread(folderNM+fn) for fn in fNM]
    '''imgs[0] <- first image, imgs[1] <- second image, ...'''

    # decide alpha values (weight for each image)
    alphas = [0.2, 0.4, 0.3, 0.5]
    '''alphas[0] <- for first image, alphas[1] <- for second image, ...'''

    # initialize result image matrix
    rImg = np.zeros(imgs[0].shape)
    # simple alpha blending
    for img, alpha in zip(imgs, alphas):
        rImg += alpha*img
    '''i.e.) res = alphas[0]*imgs[0] + alphas[1]*imgs[1] + ...'''
    # normalization
    rImg /= np.sum(alphas)

    # show output image
    cv2.imshow('result', rImg.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite('res/ABCD.png', rImg)


if __name__ == '__main__':
    main()
