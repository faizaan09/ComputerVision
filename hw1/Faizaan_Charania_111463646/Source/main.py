# Instructions:
# For question 1, only modify function: histogram_equalization
# For question 2, only modify functions: low_pass_filter, high_pass_filter, deconvolution
# For question 3, only modify function: laplacian_pyramid_blending

import os
import sys
import cv2
import numpy as np


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Histogram equalization")
    print("2 Frequency domain filtering")
    print("3 Laplacian pyramid blending")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image] " +
          "[output directory]")  # Single input, single output
    print(sys.argv[0] + " 2 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, three outputs
    print(sys.argv[0] + " 3 " + "[path to input image1] " +
          "[path to input image2] " +
          "[output directory]")  # Two inputs, single output


# ===================================================
# ======== Question 1: Histogram equalization =======
# ===================================================


def histogram_equalization(img_in):

    # Write histogram equalization here
    b, g, r = cv2.split(img_in)
    channels = [b, g, r]
    color = ('b', 'g', 'r')
    hist = {}
    cdf = {}
    for i, col in enumerate(color):
        hist[col] = cv2.calcHist([channels[i]], [0], None, [256], [0, 256])
        cdf[i] = np.cumsum(hist[col])
        cdf[i] = cdf[i] / (img_in.shape[0] * img_in.shape[1]) * 255
        channels[i] = cdf[i][channels[i]].astype('uint8')

    img_out = cv2.merge(channels)  # Histogram equalization result
    return True, img_out


def Question1():

    # Read in input images
    input_image = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)

    # Histogram equalization
    succeed, output_image = histogram_equalization(input_image)

    # Write out the result
    output_name = sys.argv[3] + "1.png"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# ===== Question 2: Frequency domain filtering ======
# ===================================================


def low_pass_filter(img_in):

    # Write low pass filter here
    new_channels = []
    for channel in cv2.split(img_in):
        rows, cols = channel.shape
        crow, ccol = rows / 2, cols / 2

        # create a mask first, center square is 1, remaining all zeros
        mask = np.zeros((rows, cols), np.uint8)
        mask[crow - 30:crow + 30, ccol - 30:ccol + 30] = 1

        f = np.fft.fft2(channel)
        # apply mask and inverse DFT
        fshift = np.fft.fftshift(f)
        fshift = fshift * mask
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        new_channels.append(img_back)

    img_out = cv2.merge(new_channels)  # Low pass filter result

    return True, img_out


def high_pass_filter(img_in):

    # Write high pass filter here
    new_channels = []
    for channel in cv2.split(img_in):
        f = np.fft.fft2(channel)
        fshift = np.fft.fftshift(f)

        rows, cols = channel.shape
        crow, ccol = rows / 2, cols / 2
        fshift[crow - 30:crow + 30, ccol - 30:ccol + 30] = 0
        f_ishift = np.fft.ifftshift(fshift)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)
        new_channels.append(img_back)

    img_out = cv2.merge(new_channels)  # High pass filter result

    return True, img_out


def deconvolution(img_in):

    # Write deconvolution codes here
    def ft(im, newsize=None):
        dft = np.fft.fft2(np.float32(im), newsize)
        return np.fft.fftshift(dft)

    def ift(shift):
        f_ishift = np.fft.ifftshift(shift)
        img_back = np.fft.ifft2(f_ishift)
        return np.abs(img_back)

    gk = cv2.getGaussianKernel(21, 5)
    gk = gk * gk.T

    imf = ft(img_in, (img_in.shape[0],
                      img_in.shape[1]))  # make sure sizes match
    gkf = ft(gk, (img_in.shape[0],
                  img_in.shape[1]))  # so we can multiple easily
    imconvf = ift(imf / gkf)

    # now for example we can reconstruct the blurred image from its FT

    img_out = (imconvf * 255).astype('uint8')
    return True, img_out


def Question2():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3],
                              cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)

    # Low and high pass filter
    succeed1, output_image1 = low_pass_filter(input_image1)
    succeed2, output_image2 = high_pass_filter(input_image1)

    # Deconvolution
    succeed3, output_image3 = deconvolution(input_image2)

    # Write out the result
    output_name1 = sys.argv[4] + "2LPF.png"
    output_name2 = sys.argv[4] + "2HPF.png"
    output_name3 = sys.argv[4] + "2deconv.png"
    cv2.imwrite(output_name1, output_image1)
    cv2.imwrite(output_name2, output_image2)
    cv2.imwrite(output_name3, output_image3)

    return True


# ===================================================
# ===== Question 3: Laplacian pyramid blending ======
# ===================================================


def laplacian_pyramid_blending(img_in1, img_in2):

    # Write laplacian pyramid blending codes here

    img_in1 = img_in1[:, :img_in1.shape[0]]
    img_in2 = img_in2[:img_in1.shape[0], :img_in1.shape[0]]

    # generate Gaussian pyramid for A
    G = img_in1.copy()
    gpA = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpA.append(G)

    # generate Gaussian pyramid for B
    G = img_in2.copy()
    gpB = [G]
    for i in xrange(6):
        G = cv2.pyrDown(G)
        gpB.append(G)

    # generate Laplacian Pyramid for A
    lpA = [gpA[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpA[i])
        L = cv2.subtract(gpA[i - 1], GE)
        lpA.append(L)

    # generate Laplacian Pyramid for B
    lpB = [gpB[5]]
    for i in xrange(5, 0, -1):
        GE = cv2.pyrUp(gpB[i])
        L = cv2.subtract(gpB[i - 1], GE)
        lpB.append(L)

    # Now add left and right halves of images in each level
    LS = []
    for la, lb in zip(lpA, lpB):
        rows, cols, dpt = la.shape
        ls = np.hstack((la[:, 0:cols / 2], lb[:, cols / 2:]))
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1, 6):
        ls_ = cv2.pyrUp(ls_)
        ls_ = cv2.add(ls_, LS[i])

    img_out = ls_  # Blending result
    return True, img_out


def Question3():

    # Read in input images
    input_image1 = cv2.imread(sys.argv[2], cv2.IMREAD_COLOR)
    input_image2 = cv2.imread(sys.argv[3], cv2.IMREAD_COLOR)

    # Laplacian pyramid blending
    succeed, output_image = laplacian_pyramid_blending(input_image1,
                                                       input_image2)

    # Write out the result
    output_name = sys.argv[4] + "3.png"
    cv2.imwrite(output_name, output_image)

    return True


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) < 4):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])

        if (question_number == 1 and not (len(sys.argv) == 4)):
            help_message()
            sys.exit()
        if (question_number == 2 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number == 3 and not (len(sys.argv) == 5)):
            help_message()
            sys.exit()
        if (question_number > 3 or question_number < 1 or len(sys.argv) > 5):
            print("Input parameters out of bound ...")
            sys.exit()

    function_launch = {
        1: Question1,
        2: Question2,
        3: Question3,
    }

    # Call the function
    function_launch[question_number]()
