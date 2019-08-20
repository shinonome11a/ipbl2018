import cv2
import matplotlib.pyplot as plt


def imageIOdemo():
    # read image file
    img_BGR = cv2.imread('img/flash/ambient.jpg')

    # some image processing ==============
    img_RGB = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    #   _____            _____
    #  /   B/__         /   R/__
    # /____/ G/__  ==> /____/ G/__
    #   /____/ R/        /____/ B/
    #     /____/           /____/
    # ====================================

    # write image file
    cv2.imwrite('res/res1.png', img_BGR)
    cv2.imwrite('res/res2.png', img_RGB)

    # show images (cv2)
    cv2.imshow('img_BGR', img_BGR)
    cv2.imshow('img_RGB', img_RGB)
#    cv2.waitKey(0) # pause until any key pressed

    # show multi-images (plt)
    plt.subplot(1, 2, 1), plt.imshow(img_BGR), plt.title('img_BGR'), plt.axis('off')
    plt.subplot(1, 2, 2), plt.imshow(img_RGB), plt.title('img_RGB'), plt.axis('off')
    plt.show()
    # plt's figures must be closed manually.

    # close all cv2 windows (cv2)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    imageIOdemo()
