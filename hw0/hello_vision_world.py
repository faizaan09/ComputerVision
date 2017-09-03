# Write an OpenCV program to do the following things:
#
# Read an image from a file and display it to the screen
# Add to, subtract from, multiply or divide each pixel with a scalar, display the result.
# Resize the image uniformly by 0.5

import cv2
import numpy as np


#given an image object and the window title, display the image in a window with that title
def displayImage(img, win_name):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


#given an image object and a scalar variable, multiply every pixel of the image with that variable
def performScalarOperation(img, op='+', scalar=1):
    temp = np.array(img)
    if op == '+':
        temp += scalar
        win_name = "added " + str(scalar)
    elif op == '-':
        temp -= scalar
        win_name = "subtracted " + str(scalar)
    elif op == '*':
        temp *= scalar
        win_name = "multiplied " + str(scalar)
    else:
        temp /= scalar
        win_name = "divided " + str(scalar)
    displayImage(temp, win_name)
    return temp


#given an image object and a scale, resize and display the resized image
def resizeImage(img, scale):
    win_name = "resized image to " + str(scale) + "x"
    # according to openCV docs, specific interpolations work better for shrinking and enlarging
    if scale >= 1:
        inter = cv2.INTER_CUBIC
    else:
        inter = cv2.INTER_AREA

    temp = cv2.resize(
        np.array(img), dsize=(0, 0), fx=scale, fy=scale, interpolation=inter)
    displayImage(temp, win_name)
    return temp


def main():
    img = cv2.imread('nyc_day_1.jpg')
    displayImage(img, "#shotOnOnePlus")  #original image

    #scalar operations on image
    added = performScalarOperation(img, '+', 50)
    subtracted = performScalarOperation(img[:], '-', 125)
    multiplied = performScalarOperation(img[:], '*', 5)
    divided = performScalarOperation(img,'/',2)
    #resize the image
    resizeImage(img, 0.5)


if __name__ == '__main__':
    main()
