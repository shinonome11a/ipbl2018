import numpy as np
import cv2
import matplotlib.pyplot as plt


# absolute Laplacian value
def contrast(imgs):
    n = len(imgs)
    h, w = imgs[0].shape[:2]
    W = np.zeros((h, w, n))

    # need to update W

    return W


# standard deviation of color (R,G,B)
def saturation(imgs):
    n = len(imgs)
    h, w = imgs[0].shape[:2]
    W = np.zeros((h, w, n))

    # need to update W

    return W


# middle value is the best of pixel values (gaussian weighting)
def exposure(imgs):
    n = len(imgs)
    h, w = imgs[0].shape[:2]
    W = np.zeros((h, w, n))

    # need to update W

    return W


def demoFeatures():
    img = cv2.imread('img/flash/ambient.jpg')
    f = contrast([img])

    cv2.imshow('contrast', f)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    demoFeatures()
