import numpy as np
import cv2
import matplotlib.pyplot as plt
import pyrFunc as pf
import imageFeatures as imf


def exposureFusion(imgs, param):
    print(param)
    conP = param[0]  # contrast parameter
    satP = param[1]  # saturation parameter
    expP = param[2]  # exposure parameter

    # resize the long side to the specified length (640)
    imgs = pf.resizeImgs(imgs.copy(), 640)

    h, w, c = imgs[0].shape[:3]
    n = len(imgs)

    W = np.ones((h, w, n))
    if conP > 0:
        print(' - contrast')
        W *= np.power(imf.contrast(imgs.copy()), conP)
    if satP > 0:
        print(' - saturation')
        W *= np.power(imf.saturation(imgs.copy()), satP)
    if expP > 0:
        print(' - exposure')
        W *= np.power(imf.exposure(imgs.copy()), expP)

    W[W == 0] += 1e-12  # avoids division by zero
    Wsum = np.zeros((h, w))
    for i in range(n):
        Wsum += W[:, :, i]
    # Normalization (dividing total weight -> total weight becomes 1.0)
    W /= np.tile(Wsum[:, :, None], [1, 1, n])

    # pf.showImage('W',W,0)
    for i in range(n):
        pf.showImage('W'+str(i), W[:, :, i], 0)

    # calc pyramid level
    m = int(np.log2(np.min((h, w)))) + 1

    # initialize pyr
    pyr = pf.generateGussianPyr(np.zeros((h, w, c)), m)

    # multi-scale processing
    for i in range(n):
        # generate Gaussian pyramid for Weight
        Gw = W[:, :, i].copy()
        gpW = pf.generateGussianPyr(Gw, m)

        # generate Laplacian pyramid for Image
        Gi = imgs[i].copy()
        lpI = pf.generateLaplacianPyr(Gi, m)

        # weighted average for pyramid
        for j in range(m):
            gpwtmp = gpW[j]
            w = np.tile(gpwtmp[:, :, None], [1, 1, c])
            pyr[j] += w*lpI[j]

    # reconstruct
    rImg = pf.reconstructLaplacianPyr(pyr.copy(), m)

    pf.showImage('results', rImg, 1)
    pf.showMultiImage('inputs', imgs)
    cv2.destroyAllWindows()

    pf.writeImage('res/test.png', rImg, 1)


if __name__ == '__main__':

    #######################
    # user settings
    #######################
    folderNM = './img/house/'
    fNM = ['A.jpg', 'B.jpg', 'C.jpg']
    con = 1.0  # contrast parameter
    sat = 1.0  # saturation parameter
    exp = 1.0  # exposure parameter

    imgs = [cv2.imread("{}{}".format(folderNM, fn))/255 for fn in fNM]

    exposureFusion(imgs, [con, sat, exp])
