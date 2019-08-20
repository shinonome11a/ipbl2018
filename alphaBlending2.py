# -*- coding: utf-8 -*-

import numpy as np
import cv2


def main():
    # input image files
    folderNM = './img/house/'
    fNM = ['B.jpg', 'C.jpg']
    imgs = [cv2.imread(folderNM+fn) for fn in fNM]

    # decide alpha values (weight for each image)
    h, w = imgs[0].shape[:2]
    tmp = np.arange(w) / w
    tmp = tmp[:, None].T
    alphas = []
    alphas.append(np.tile(tmp[:, :, None], [h, 1, 3]))
    alphas.append(1-alphas[0])

    # initialize result image matrix
    rImg = np.zeros(imgs[0].shape)
    sumAlpha = np.zeros(imgs[0].shape)
    # simple alpha blending
    for img, alpha in zip(imgs, alphas):
        rImg += alpha*img
        sumAlpha += alpha
    # normalization
    rImg /= sumAlpha

    # show output image
    cv2.imshow('result', rImg.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
