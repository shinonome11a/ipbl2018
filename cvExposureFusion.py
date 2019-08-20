import cv2
import sys


def main(argv):
    # input image files
    folderNM = './img/house/'
    fileNM = ["A.jpg", "B.jpg", 'C.jpg', 'D.jpg']
    imgs = [cv2.imread(folderNM+fn) for fn in fileNM]

    # set parameters
    cp = 1.0  # contrast parameter
    sp = 1.0  # saturation parameter
    ep = 1.0  # exposure parameter

    # Exposure Fusion using OpenCV
    merge_mertens = cv2.createMergeMertens(cp, sp, ep)
    rImg = merge_mertens.process(imgs)

    # show output image
    cv2.imshow('results', rImg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main(sys.argv[1:])
