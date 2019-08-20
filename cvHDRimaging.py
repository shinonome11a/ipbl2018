import cv2
import numpy as np
from PIL import Image
from PIL.ExifTags import TAGS


def main():
    # input multiple exposure images
    folderNM = './img/europe/'
    fileNM = ["low.jpg", "middle.jpg", "high.jpg"]
    imgs = [cv2.imread(folderNM+fn) for fn in fileNM]

    # auto discover exposure time from Exif information of input image file
    exposureTimes = np.array([getExifInfo(folderNM+fn) for fn in fileNM], dtype=np.float32)
    # if Exif information is None, you should set exposure time manually
    # exposureTimes = np.array([15.0, 2.5, 0.25, 0.0333], dtype=np.float32)

    # Estimate camera response function (CRF)
    mergeDebevec = cv2.createMergeDebevec()
    hdr = mergeDebevec.process(imgs, times=exposureTimes.copy())
    hdr[hdr < 0] = 0

    cv2.imwrite("res/hdrDebevec.hdr", hdr)

    # linear
    res1 = 255*hdr/hdr.max()
    # gamma
    res2 = 255*(hdr/hdr.max())**0.2

    # reinhardTM_global
    tonemap = cv2.createTonemapReinhard(gamma=1.5)
    res3 = tonemap.process(hdr.copy())
    res3 = percentileCut(res3, 0.5, 99.9)*255

    cv2.imshow('result_linear', res1.astype(np.uint8))
    cv2.imshow('result_gamma', res2.astype(np.uint8))
    cv2.imshow('result_reinhard', res3.astype(np.uint8))
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("res/tonemappedImg.png", res3)


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


def rgb2gray(img):
    return 0.2989 * img[:, :, 0] + 0.5870 * img[:, :, 1] + 0.1140 * img[:, :, 2]


def percentileCut(img, lprc, hprc):
    lth = np.nanpercentile(img, lprc)
    hth = np.nanpercentile(img, hprc)
    img = (img - lth)/(hth - lth)
    img[img < 0] = 0.0
    img[img > 1.0] = 1.0
    return img


if __name__ == '__main__':
    main()
