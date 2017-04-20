#import cv2
#import numpy as np
#
#img = cv2.imread('OpenCV_Logo.png',0)
#img = cv2.medianBlur(img,5)
#cimg = cv2.cvtColor(img,cv2.COLOR_GRAY2BGR)
#
#circles = cv2.HoughCircles(img,cv2.HOUGH_GRADIENT,1,20,param1=50,param2=30,minRadius=0,maxRadius=0)
#
#circles = np.uint16(np.around(circles))
#for i in circles[0,:]:
#    # draw the outer circle
#    cv2.circle(cimg,(i[0],i[1]),i[2],(0,255,0),2)
#    # draw the center of the circle
#    cv2.circle(cimg,(i[0],i[1]),2,(0,0,255),3)
#
#cv2.imshow('detected circles',cimg)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# import the necessary packages
from skimage.exposure import rescale_intensity
import numpy as np
import cv2
from matplotlib import pyplot as plt

def convolve(imageL,imageR):
    # grab the spatial dimensions of the image, along with
    # the spatial dimensions of the kernel
    (iH, iW) = imageL.shape[:2]
    
    # allocate memory for the output image, taking care to
    # "pad" the borders of the input image so the spatial
    # size (i.e., width and height) are not reduced
    pad = iW / 2
    image_padded = cv2.copyMakeBorder(imageL, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    (iHp, iWp) = image_padded.shape[:2]
    plt.imshow(image_padded, cmap = 'gray', interpolation = 'bicubic')
    plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
    plt.show()
    output = np.zeros((iH, iW), dtype="float32")

    # loop over the input image, "sliding" the kernel across
    # each (x, y)-coordinate from left-to-right and top to
    # bottom
    for y in np.arange(pad, iH + pad):
        for x in np.arange(pad, iW + pad):
            # extract the ROI of the image by extracting the
            # *center* region of the current (x, y)-coordinates
            # dimensions
            roi = image_padded[y - pad:y + pad, x - pad:x + pad]
            
            # perform the actual convolution by taking the
            # element-wise multiplicate between the ROI and
            # the kernel, then summing the matrix
            k = (roi * imageR).sum()
            print k
            # store the convolved value in the output (x,y)-
            # coordinate of the output image
            output[y - pad, x - pad] = k


    # rescale the output image to be in the range [0, 255]
    output = rescale_intensity(output, in_range='image', out_range='uint8')

    # return the output image
    return output

## construct average blurring kernels used to smooth an image
#smallBlur = np.ones((7, 7), dtype="float") * (1.0 / (7 * 7))
#largeBlur = np.ones((21, 21), dtype="float") * (1.0 / (21 * 21))
#
## construct a sharpening filter
#sharpen = np.array((
#                    [0, -1, 0],
#                    [-1, 5, -1],
#                    [0, -1, 0]), dtype="int")
#
## construct the Laplacian kernel used to detect edge-like
## regions of an image
#laplacian = np.array((
#                      [0, 1, 0],
#                      [1, -4, 1],
#                      [0, 1, 0]), dtype="int")
#
## construct the Sobel x-axis kernel
#sobelX = np.array((
#                   [-1, 0, 1],
#                   [-2, 0, 2],
#                   [-1, 0, 1]), dtype="int")
#
## construct the Sobel y-axis kernel
#sobelY = np.array((
#                   [-1, -2, -1],
#                   [0, 0, 0],
#                   [1, 2, 1]), dtype="int")

# load the input image and convert it to grayscale
imageL = cv2.imread("L.png")
grayL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)

imageR = cv2.imread("R.png")
grayR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)

convoleOutput = convolve(grayL, grayR)

plt.imshow(convoleOutput, cmap = 'gray', interpolation = 'bicubic')
plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
plt.show()

# show the output images
cv2.imshow("Left", grayL)
cv2.imshow("Right", grayR)
cv2.imshow("convole", convoleOutput)
cv2.waitKey(0)
cv2.destroyAllWindows()
