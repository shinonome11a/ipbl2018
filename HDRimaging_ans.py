# -*- coding: utf-8 -*-
import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS
import tmo_lib_ans as tmo


eps = 1.0e-16


def makeHDRI(imgs, imgExp):
    n = len(imgs)
    Wsum = np.zeros(imgs[0].shape, dtype=np.float32)
    hdr = np.zeros(imgs[0].shape, dtype=np.float32)
    for i in range(n):
        tmp = weightFunc(imgs[i].copy())
        W = np.tile(tmp[:, :, None], [1, 1, 3])
        Wsum += W
        hdr += W * inverseCRF(imgs[i].copy()) / imgExp[i]
    hdr /= Wsum
    hdr[hdr < 0] = 0
    return hdr


def weightFunc(img):
    if img.max() > 1:
        img = img/255
    return 1.0 - 0.99 * abs(2 * rgb2gray(img) - 1.0)


def inverseCRF(img):
    return np.power(img, 2.4)


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

    exposureTimes = np.array([getExifInfo(folderNM+fn) for fn in fileNM], dtype=np.float32)

    hdr = makeHDRI(imgs.copy(), exposureTimes.copy())

    # linear
    res1 = 255*hdr/hdr.max()
    # gamma
    res2 = 255*(hdr/hdr.max())**0.5
    # reinhardTM_global
    res3 = tmo.ReinhardGlobalTMO(hdr.copy(), [])
    # reinhardTM_local_L0Smoothing
    res4 = tmo.ReinhardLocalTMO_L0Smoothing(hdr.copy(), [])
    # retinexTM_gaussianFilter
    res5 = tmo.RetinexGaussianTMO(hdr.copy(), [1, 3], [])
    # retinexTM_bilateralFilter
    res6 = tmo.RetinexBilateralTMO(hdr.copy(), [1, 3], [])
    # retinexTM_L0Smoothing
    res7 = tmo.RetinexL0SmoothingTMO(hdr.copy(), [1, 3], [])

    cv2.imshow('linear', res1.astype(np.uint8))
    cv2.imshow('gamma', res2.astype(np.uint8))
    cv2.imshow('reinhardTM_global', res3.astype(np.uint8))
    cv2.imshow('reinhardTM_local', res4.astype(np.uint8))
    cv2.imshow('retinexTM_GF', res5.astype(np.uint8))
    cv2.imshow('retinexTM_BF', res6.astype(np.uint8))
    cv2.imshow('retinexTM_L0', res7.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
