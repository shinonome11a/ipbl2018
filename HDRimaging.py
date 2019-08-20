# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import tmo_lib as tmo


# minimal value (for avoiding zero divide, and so on)
eps = 1.0e-16


def makeHDRI(imgs, imgExp):
    n = len(imgs)  # n is number of input images
    Wsum = np.zeros(imgs[0].shape, dtype=np.float32)
    hdr = np.zeros(imgs[0].shape, dtype=np.float32)
    for i in range(n):
        tmp = weightFunc(imgs[i].copy()/255)  # calc 2D weight map
        W = np.tile(tmp[:, :, None], [1, 1, 3])  # 1ch => 3ch (2D => 3D)

        ######################################
        # "you should change to correct code"
        # calc denominator
        # calc numerator
        ######################################

    hdr /= Wsum
    return hdr


def weightFunc(img):
    gray = rgb2gray(img)
    ######################################
    # "you should change to correct code!"
    ######################################


def inverseCRF(img):
    return np.power(img, 2.4)


# clipping with percentile (strong normalization)
def percentileCut(img, lprc, hprc):
    lth = np.nanpercentile(img, lprc)
    hth = np.nanpercentile(img, hprc)
    img = (img - lth)/(hth - lth)
    img[img < 0] = 0.0
    img[img > 1.0] = 1.0
    return img


def rgb2gray(img):
    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]


def getExifInfo(fNM):
    print(fNM)
    im = Image.open(fNM)
    exif = im._getexif()
    if exif is None:
        return None
    exifInfo = {}
    for tag_id, value in exif.items():
        tag = TAGS.get(tag_id, tag_id)
        exifInfo[tag] = value
    et = exifInfo.get("ExposureTime")
    return et[0]/et[1]


def main():
    folderNM = './img/europe/'
    fileNM = ["low.jpg", "middle.jpg", "high.jpg"]
    imgs = [cv2.imread(folderNM+fn) for fn in fileNM]

    # auto discover exposure time from Exif information of input image file
    exposureTimes = np.array([getExifInfo(folderNM+fn) for fn in fileNM], dtype=np.float32)
    # if Exif information is None, you should set exposure time manually
    # exposureTimes = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

    # make HDR image
    hdr = makeHDRI(imgs.copy(), exposureTimes.copy())

    # linear
    res1 = 255*hdr/hdr.max()
    # gamma
    res2 = 255*(hdr/hdr.max())**0.3
    # reinhardTM_global
    res3 = tmo.ReinhardGlobalTMO(hdr.copy(), [])
    # retinexTM_gaussian
    res4 = tmo.RetinexGaussianTMO(hdr.copy(), [1, 3], [])

    cv2.imshow('linear', res1.astype(np.uint8))
    cv2.imshow('gamma', res2.astype(np.uint8))
    cv2.imshow('reinhardGlobal', res3.astype(np.uint8))
    cv2.imshow('retinexGaussian', res4.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
