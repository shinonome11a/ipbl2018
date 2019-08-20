import cv2
import matplotlib.pyplot as plt
import numpy as np


def imageIOdemo():
    # read image files
    folderNM = './img/house/'
    fNM = ['A.jpg', 'B.jpg', 'C.jpg', 'D.jpg']
    imgs = [cv2.imread("{}{}".format(folderNM, fn)) for fn in fNM]

    # show multi-images (plt)
    showMultiFigures(imgs, [])
    showMultiFigures(imgs, fNM)


def showMultiFigures(imgs, wnames):
    m = len(imgs)
    h = np.round(np.sqrt(m))
    w = np.ceil(np.sqrt(m))
    for i in range(m):
        # write image files
        cv2.imwrite('res/mres'+str(i)+'.png', imgs[i])

        # show images
        plt.subplot(h, w, i+1)
        if len(imgs[i].shape) == 3:  # color
            b, g, r = cv2.split(imgs[i])
            plt.imshow(cv2.merge((r, g, b)), vmin=0, vmax=255)  # cv2:BGR <--> plt:RGB
        else:  # gray
            plt.imshow(imgs[i], 'gray', vmin=0, vmax=255)
        if wnames != []:
            plt.title(wnames[i])
        plt.axis('off')
    plt.show()


if __name__ == '__main__':
    imageIOdemo()
