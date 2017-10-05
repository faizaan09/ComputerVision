# Instructions:
# Do not change the output file names, use the helper functions as you see fit

import pdb
pdb.set_trace()

import os
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import numpy.ma as ma
from lap_pyr import Laplacian_Pyramid_Blending_with_mask


def help_message():
    print("Usage: [Question_Number] [Input_Options] [Output_Options]")
    print("[Question Number]")
    print("1 Perspective warping")
    print("2 Cylindrical warping")
    print("3 Bonus perspective warping")
    print("4 Bonus cylindrical warping")
    print("[Input_Options]")
    print("Path to the input images")
    print("[Output_Options]")
    print("Output directory")
    print("Example usages:")
    print(sys.argv[0] + " 1 " + "[path to input image1] " +
          "[path to input image2] " + "[path to input image3] " +
          "[output directory]")


'''
Detect, extract and match features between img1 and img2.
Using SIFT as the detector/extractor, but this is inconsequential to the user.

Returns: (pts1, pts2), where ptsN are points on image N.
    The lists are "aligned", i.e. point i in pts1 matches with point i in pts2.

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (pts1, pts2) = feature_matching(im1, im2)

    plt.subplot(121)
    plt.imshow(im1)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
    plt.subplot(122)
    plt.imshow(im2)
    plt.scatter(pts1[:,:,0],pts1[:,:,1], 0.5, c='r', marker='x')
'''


def feature_matching(img1, img2, savefig=False):
    # Initiate SIFT detector
    sift = cv2.xfeatures2d.SIFT_create()
    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # FLANN parameters
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)  # or pass empty dictionary
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches2to1 = flann.knnMatch(des2, des1, k=2)

    matchesMask_ratio = [[0, 0] for i in xrange(len(matches2to1))]
    match_dict = {}
    for i, (m, n) in enumerate(matches2to1):
        if m.distance < 0.7 * n.distance:
            matchesMask_ratio[i] = [1, 0]
            match_dict[m.trainIdx] = m.queryIdx

    good = []
    recip_matches = flann.knnMatch(des1, des2, k=2)
    matchesMask_ratio_recip = [[0, 0] for i in xrange(len(recip_matches))]

    for i, (m, n) in enumerate(recip_matches):
        if m.distance < 0.7 * n.distance:  # ratio
            if m.queryIdx in match_dict and match_dict[m.
                                                       queryIdx] == m.trainIdx:  #reciprocal
                good.append(m)
                matchesMask_ratio_recip[i] = [1, 0]

    if savefig:
        draw_params = dict(
            matchColor=(0, 255, 0),
            singlePointColor=(255, 0, 0),
            matchesMask=matchesMask_ratio_recip,
            flags=0)
        img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, recip_matches, None,
                                  **draw_params)

        plt.figure(), plt.xticks([]), plt.yticks([])
        plt.imshow(img3,)
        plt.savefig("feature_matching.png", bbox_inches='tight')

    return ([kp1[m.queryIdx].pt
             for m in good], [kp2[m.trainIdx].pt for m in good])


'''
Warp an image from cartesian coordinates (x, y) into cylindrical coordinates (theta, h)
Returns: (image, mask)
Mask is [0,255], and has 255s wherever the cylindrical images has a valid value.
Masks are useful for stitching

Usage example:

    im = cv2.imread("myimage.jpg",0) #grayscale
    h,w = im.shape
    f = 700
    K = np.array([[f, 0, w/2], [0, f, h/2], [0, 0, 1]]) # mock calibration matrix
    imcyl = cylindricalWarpImage(im, K)
'''


def cylindricalWarpImage(img1, K, savefig=False):
    f = K[0, 0]

    im_h, im_w = img1.shape

    # go inverse from cylindrical coord to the image
    # (this way there are no gaps)
    cyl = np.zeros_like(img1)
    cyl_mask = np.zeros_like(img1)
    cyl_h, cyl_w = cyl.shape
    x_c = float(cyl_w) / 2.0
    y_c = float(cyl_h) / 2.0
    for x_cyl in np.arange(0, cyl_w):
        for y_cyl in np.arange(0, cyl_h):
            theta = (x_cyl - x_c) / f
            h = (y_cyl - y_c) / f

            X = np.array([math.sin(theta), h, math.cos(theta)])
            X = np.dot(K, X)
            x_im = X[0] / X[2]
            if x_im < 0 or x_im >= im_w:
                continue

            y_im = X[1] / X[2]
            if y_im < 0 or y_im >= im_h:
                continue

            cyl[int(y_cyl), int(x_cyl)] = img1[int(y_im), int(x_im)]
            cyl_mask[int(y_cyl), int(x_cyl)] = 255

    if savefig:
        plt.imshow(cyl, cmap='gray')
        plt.savefig("cyl.png", bbox_inches='tight')

    return (cyl, cyl_mask)


'''
Calculate the geometric transform (only affine or homography) between two images,
based on feature matching and alignment with a robust estimator (RANSAC).

Returns: (M, pts1, pts2, mask)
Where: M    is the 3x3 transform matrix
       pts1 are the matched feature points in image 1
       pts2 are the matched feature points in image 2
       mask is a binary mask over the lists of points that selects the transformation inliers

Usage example:
    im1 = cv2.imread("image1.jpg", 0)
    im2 = cv2.imread("image2.jpg", 0)
    (M, pts1, pts2, mask) = getTransform(im1, im2)

    # for example: transform im1 to im2's plane
    # first, make some room around im2
    im2 = cv2.copyMakeBorder(im2,200,200,500,500, cv2.BORDER_CONSTANT)
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(im1, M, (im1.shape[1],im2.shape[0]), dst=im2.copy(), borderMode=cv2.BORDER_TRANSPARENT)

    plt.imshow(out, cmap='gray')
    plt.show()
'''


def getTransform(src, dst, method='affine'):
    pts1, pts2 = feature_matching(src, dst, savefig=True)

    src_pts = np.float32(pts1).reshape(-1, 1, 2)
    dst_pts = np.float32(pts2).reshape(-1, 1, 2)

    if method == 'affine':
        M, mask = cv2.estimateAffine2D(
            src_pts, dst_pts, cv2.RANSAC, ransacReprojThreshold=5.0)
        # M = np.append(M, [[0, 0, 1]], axis=0)

    if method == 'homography':
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    matchesMask = mask.ravel().tolist()

    return (M, pts1, pts2, mask)


# ===================================================
# ================ Perspective Warping ==============
# ===================================================
def Perspective_warping(img1, img2, img3):

    img1 = cv2.copyMakeBorder(img1, 200, 200, 500, 500, cv2.BORDER_CONSTANT)
    (M, pts1, pts2, mask) = getTransform(img2, img1, method='homography')
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(
        img2,
        M, (img1.shape[1], img1.shape[0]),
        dst=img1.copy(),
        borderMode=cv2.BORDER_TRANSPARENT)
    img1 = out
    (M, pts1, pts2, mask) = getTransform(img3, img1, method='homography')
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(
        img3,
        M, (img1.shape[1], img1.shape[0]),
        dst=img1.copy(),
        borderMode=cv2.BORDER_TRANSPARENT)

    plt.imshow(out, cmap='gray')
    plt.show()
    output_image = out  # This is dummy output, change it to your output

    print RMSD(1, out, cv2.imread("./example_output1.png", 0))

    # Write out the result
    output_name = sys.argv[5] + "output_homography.png"
    cv2.imwrite(output_name, output_image)

    return True


def Bonus_perspective_warping(img1, img2, img3):

    # Write your codes here
    img1 = cv2.copyMakeBorder(img1, 200, 200, 500, 500, cv2.BORDER_CONSTANT)
    (M, pts1, pts2, mask) = getTransform(img2, img1, method='homography')
    # then transform im1 with the 3x3 transformation matrix
    out = cv2.warpPerspective(img2, M, (img1.shape[1], img1.shape[0]))
    # dst=img1.copy(),
    # borderMode=cv2.BORDER_TRANSPARENT)

    out = Laplacian_Pyramid_Blending_with_mask(img1, img2, mask)
    img1 = out
    (M, pts1, pts2, mask) = getTransform(img3, img1, method='homography')
    # then transform im1 with the 3x3 transformation matrix
    # out = Laplacian_Pyramid_Blending_with_mask(img1, img3, mask)
    out = cv2.warpPerspective(img3, M, (img1.shape[1], img1.shape[0]))
    # dst=img1.copy(),
    # borderMode=cv2.BORDER_TRANSPARENT)
    mask = np.zeros_like(img1, dtype='float32')
    out = Laplacian_Pyramid_Blending_with_mask(img1, img3, mask)
    plt.imshow(out, cmap='gray')
    plt.show()
    output_image = out  # This is dummy output, change it to your output

    print RMSD(1, out, cv2.imread("./example_output1.png", 0))

    output_image = img1  # This is dummy output, change it to your output

    # Write out the result
    output_name = sys.argv[5] + "output_homography_lpb.png"
    cv2.imwrite(output_name, output_image)

    return True


# ===================================================
# =============== Cynlindrical Warping ==============
# ===================================================
def Cylindrical_warping(img1, img2, img3):

    # Write your codes here
    # img1 = cv2.copyMakeBorder(img1, 50, 50, 300, 300, cv2.BORDER_CONSTANT)
    h, w = img1.shape
    f = 410
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0,
                                                 1]])  # mock calibration matrix
    img1, mask1 = cylindricalWarpImage(img1, K)
    img1 = cv2.copyMakeBorder(img1, 50, 50, 300, 300, cv2.BORDER_CONSTANT)
    h, w = img3.shape
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0,
                                                 1]])  # mock calibration matrix

    img3, mask3 = cylindricalWarpImage(img3, K)

    (M, pts1, pts2, mask) = getTransform(img3,
                                         img1)  #perform the transformation

    im3_warp = cv2.warpAffine(
        img3,
        M, (img1.shape[1], img1.shape[0]),
        dst=img1.copy(),
        borderMode=cv2.BORDER_TRANSPARENT)

    m3_warp = cv2.warpAffine(
        mask3,
        M, (img1.shape[1], img1.shape[0]),
        dst=img1.copy(),
        borderMode=cv2.BORDER_TRANSPARENT)

    interim = img1.copy()
    interim[m3_warp != 0] = im3_warp[m3_warp != 0]
    img1 = interim.copy()
    plt.imshow(interim, cmap='gray')
    plt.show()

    img2, mask2 = cylindricalWarpImage(img2, K)

    (M, pts1, pts2, mask) = getTransform(img2,
                                         interim)  #perform the transformation

    im2_warp = cv2.warpAffine(
        img2,
        M, (interim.shape[1], interim.shape[0]),
        dst=interim.copy(),
        borderMode=cv2.BORDER_TRANSPARENT)

    m2_warp = cv2.warpAffine(
        mask2,
        M, (interim.shape[1], interim.shape[0]),
        dst=interim.copy(),
        borderMode=cv2.BORDER_TRANSPARENT)

    interim[m2_warp != 0] = im2_warp[m2_warp != 0]
    out = interim.copy()
    plt.imshow(out, cmap='gray')
    plt.show()

    output_image = out  # This is dummy output, change it to your output
    print RMSD(2, out, cv2.imread('example_output2.png', 0))
    # Write out the result
    output_name = sys.argv[5] + "output_cylindrical.png"
    cv2.imwrite(output_name, output_image)

    return True


def Bonus_cylindrical_warping(img1, img2, img3):

    # Write your codes here
    h, w = img1.shape
    f = 410
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0,
                                                 1]])  # mock calibration matrix
    img1, mask1 = cylindricalWarpImage(img1, K)
    # img1 = cv2.copyMakeBorder(img1, 50, 50, 300, 300, cv2.BORDER_CONSTANT)
    h, w = img3.shape
    K = np.array([[f, 0, w / 2], [0, f, h / 2], [0, 0,
                                                 1]])  # mock calibration matrix

    img3, mask3 = cylindricalWarpImage(img3, K)

    (M, pts1, pts2, mask) = getTransform(img3,
                                         img1)  #perform the transformation
    out = Laplacian_Pyramid_Blending_with_mask(img1, img3, mask3)
    output_image = img1  # This is dummy output, change it to your output

    # Write out the result
    output_name = sys.argv[5] + "output_cylindrical_lpb.png"
    cv2.imwrite(output_name, output_image)

    return True


'''
This exact function will be used to evaluate your results for HW2
Compare your result with master image and get the difference, the grading
criteria is posted on Piazza
'''


def RMSD(questionID, target, master):
    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or
            master_channel != target_channel):
        return -1
    else:
        nonZero_target = cv2.countNonZero(target)
        nonZero_master = cv2.countNonZero(master)

        if (questionID == 1):
            if (nonZero_target < 1200000):
                return -1
        elif (questionID == 2):
            if (nonZero_target < 700000):
                return -1
        else:
            return -1

        total_diff = 0.0
        master_channels = cv2.split(master)
        target_channels = cv2.split(target)

        for i in range(0, len(master_channels), 1):
            dst = cv2.absdiff(master_channels[i], target_channels[i])
            dst = cv2.pow(dst, 2)
            mean = cv2.mean(dst)
            total_diff = total_diff + mean[0]**(1 / 2.0)

        return total_diff


if __name__ == '__main__':
    question_number = -1

    # Validate the input arguments
    if (len(sys.argv) != 6):
        help_message()
        sys.exit()
    else:
        question_number = int(sys.argv[1])
    if (question_number > 4 or question_number < 1):
        print("Input parameters out of bound ...")
        sys.exit()

    input_image1 = cv2.imread(sys.argv[2], 0)
    input_image2 = cv2.imread(sys.argv[3], 0)
    input_image3 = cv2.imread(sys.argv[4], 0)

    function_launch = {
        1: Perspective_warping,
        2: Cylindrical_warping,
        3: Bonus_perspective_warping,
        4: Bonus_cylindrical_warping
    }
    # Call the function
    function_launch[question_number](input_image1, input_image2, input_image3)
