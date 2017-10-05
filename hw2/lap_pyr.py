# adapted from http://docs.opencv.org/3.0-beta/doc/py_tutorials/py_imgproc/py_pyramids/py_pyramids.html

import cv2
import numpy as np


def Laplacian_Pyramid_Blending_with_mask(A, B, m, num_levels=6):
    # assume mask is float32 [0,1]

    # generate Gaussian pyramid for A,B and mask
    GA = A.copy()
    GB = B.copy()
    GM = m.copy()
    gpA = [GA]
    gpB = [GB]
    gpM = [GM]
    for i in xrange(num_levels):
        GA = cv2.pyrDown(GA)
        GB = cv2.pyrDown(GB)
        GM = cv2.pyrDown(GM)
        gpA.append(np.float32(GA))
        gpB.append(np.float32(GB))
        gpM.append(np.float32(GM))

    # generate Laplacian Pyramids for A,B and masks
    lpA = [gpA[num_levels - 1]
          ]  # the bottom of the Lap-pyr holds the last (smallest) Gauss level
    lpB = [gpB[num_levels - 1]]
    gpMr = [gpM[num_levels - 1]]
    for i in xrange(num_levels - 1, 0, -1):
        # Laplacian: subtarct upscaled version of lower level from current level
        # to get the high frequencies
        LA = np.subtract(gpA[i - 1],
                         cv2.pyrUp(
                             gpA[i],
                             dstsize=(gpA[i - 1].shape[1],
                                      gpA[i - 1].shape[0])))

        LB = np.subtract(gpB[i - 1],
                         cv2.pyrUp(
                             gpB[i],
                             dstsize=(gpB[i - 1].shape[1],
                                      gpB[i - 1].shape[0])))
        lpA.append(LA)
        lpB.append(LB)
        gpMr.append(gpM[i - 1])  # also reverse the masks

    # Now blend images according to mask in each level
    LS = []
    for la, lb, gm in zip(lpA, lpB, gpMr):
        ls = la * gm + lb * (1.0 - gm)
        LS.append(ls)

    # now reconstruct
    ls_ = LS[0]
    for i in xrange(1, num_levels):
        ls_ = cv2.resize(cv2.pyrUp(ls_), dsize=(LS[i].shape[1], LS[i].shape[0]))
        ls_ = cv2.add(ls_, LS[i].astype('float32'))

    return ls_


if __name__ == '__main__':
    A = cv2.imread("input1.png", 0)
    B = cv2.imread("input2.png", 0)
    m = np.zeros_like(A, dtype='float32')
    m[:, A.shape[1] / 2:] = 1  # make the mask half-and-half
    lpb = Laplacian_Pyramid_Blending_with_mask(A, B, m, 5)
    cv2.imwrite("lpb.png", lpb)
