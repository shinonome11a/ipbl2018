import cv2
import numpy as np
import L0_serial_func as L0


def RetinexGaussianTMO(hdr, TMOparam, param):
    eps = 10e-16
    a = TMOparam[0]
    b = TMOparam[1]

    if param == []:
        ksize = 51
        sigmaX = 0.5
    else:
        ksize = param[0]
        sigmaX = param[1]

    lnHDR = np.log(hdr+eps)
    lnL = cv2.GaussianBlur(lnHDR, (ksize, ksize), sigmaX)
    lnR = lnHDR - lnL
    lnLDR = a*lnL + b*lnR

    ldr = clipping(lnLDR, 1, 99)
    return (ldr*255).astype(np.uint8)


# def RetinexBilateralTMO(hdr, TMOparam, param):
#     eps = 10e-16
#     a = TMOparam[0]
#     b = TMOparam[1]

#     if param == []:
#         d = 51
#         sigmaColor = 1.0
#         sigmaSpace = 1.5
#     else:
#         d = param[0]
#         sigmaColor = param[1]
#         sigmaSpace = param[2]

#     # Let's try implementing this function

#     ldr = clipping(lnLDR, 1, 99)
#     return (ldr*255).astype(np.uint8)


# def RetinexL0SmoothingTMO(hdr, TMOparam, param):
#     eps = 10e-16
#     a = TMOparam[0]
#     b = TMOparam[1]

#     if param == []:
#         kappa = 2.0
#         _lambda = 2e-5
#     else:
#         kappa = param[0]
#         _lambda = param[1]

#     # Let's try implementing this function

#     ldr = clipping(hdr, 1, 99.9)
#     return (ldr*255).astype(np.uint8)


def ReinhardGlobalTMO(hdr, param):
    eps = 10e-16
    if param == []:
        Lmax = np.nanpercentile(hdr, 99)
        Lmin = np.nanpercentile(hdr, 1)
        log2max = np.log2(Lmax+1e-9)
        log2min = np.log2(Lmin+1e-9)
        m = np.exp(np.mean(np.log(hdr+eps)))-eps
        log2Average = np.log2(m+1e-9)
        alpha = 0.18*4**((2.0*log2Average-log2min-log2max)/(log2max-log2min))
    else:
        alpha = param[0]

    m = np.exp(np.mean(np.log(hdr+eps)))-eps
    hdr2 = alpha/m*hdr
    ldr = hdr2/(hdr2+1)

    ldr = clipping(ldr, 1, 99)**0.5
    return (ldr*255).astype(np.uint8)


# def ReinhardLocalTMO_L0Smoothing(hdr, param):
#     eps = 10e-16
#     if param == []:
#         kappa = 2.0
#         _lambda = 2e-5
#     else:
#         kappa = param[0]
#         _lambda = param[1]

#     # Let's try implementing this function

#     ldr = clipping(ldr, 1, 99)
#     return (ldr*255).astype(np.uint8)


def rgb2gray(img):
    return 0.299*img[:, :, 0] + 0.587*img[:, :, 1] + 0.114*img[:, :, 2]


def clipping(img, Lper, Hper):
    lth = np.nanpercentile(img, Lper)
    hth = np.nanpercentile(img, Hper)
    img = (img - lth)/(hth - lth)
    img[img < 0] = 0
    img[img > 1] = 1
    return img


if __name__ == '__main__':
    # Image File Path
    fileNM = "./res/hdrDebevec.hdr"

    # L0 minimization parameters
    kappa = 2.0
    _lambda = 2e-5  # 2e-2: <-strong-- smoothing effect --weak-> 2e-7

    # Read HDR image
    hdr = cv2.imread(fileNM, -1)

    res1 = RetinexGaussianTMO(hdr, [1, 3], [51, 1.0])
    # res2 = RetinexBilateralTMO(hdr, [1, 3], [51, 1.0, 1.5])
    # res3 = RetinexL0smoothingTMO(hdr, [1, 3], [kappa, _lambda])
    res4 = ReinhardGlobalTMO(hdr, [])
    # res5 = ReinhardGlobalTMO(hdr, [0.14])
    # res6 = ReinhardLocalTMO_L0Smooting(hdr, [kappa, _lambda])

    cv2.imshow("res_Retinex_Gaussian", res1)
    # cv2.imshow("res_Retinex_Bilateral", res2)
    # cv2.imshow("res_Retinex_L0smoothing", res3)
    cv2.imshow("res_Reinhard_Global_auto", res4)
    # cv2.imshow("res_Reinhard_Global_0.14", res5)
    # cv2.imshow("res_Reinhard_Local_L0Smoothing", res6)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
