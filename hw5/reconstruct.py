# ================================================
# Skeleton codes for HW5
# Read the skeleton codes carefully and put all your
# codes into function "reconstruct_from_binary_patterns"
# ================================================

import cv2
import numpy as np
from math import log, ceil, floor
import matplotlib.pyplot as plt
import pickle
import sys


def help_message():
    # Note: it is assumed that "binary_codes_ids_codebook.pckl", "stereo_calibration.pckl",
    # and images folder are in the same root folder as your "generate_data.py" source file.
    # Same folder structure will be used when we test your program

    print("Usage: [Output_Directory]")
    print("[Output_Directory]")
    print("Where to put your output.xyz")
    print("Example usages:")
    print(sys.argv[0] + " ./")


def reconstruct_from_binary_patterns():
    scale_factor = 1.0
    ref_white = cv2.resize(
        cv2.imread("images/pattern000.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,
                                                                            0),
        fx=scale_factor,
        fy=scale_factor)
    ref_black = cv2.resize(
        cv2.imread("images/pattern001.jpg", cv2.IMREAD_GRAYSCALE) / 255.0, (0,
                                                                            0),
        fx=scale_factor,
        fy=scale_factor)
    ref_avg = (ref_white + ref_black) / 2.0
    ref_on = ref_avg + 0.05  # a threshold for ON pixels
    ref_off = ref_avg - 0.05  # add a small buffer region

    h, w = ref_white.shape

    # mask of pixels where there is projection
    proj_mask = (ref_white > (ref_black + 0.05))

    scan_bits = np.zeros((h, w), dtype=np.uint16)

    # analyze the binary patterns from the camera
    for i in range(0, 15):
        # read the file
        patt_gray = cv2.resize(
            cv2.imread("images/pattern%03d.jpg" %
                       (i + 2), cv2.IMREAD_GRAYSCALE) / 255.0, (0, 0),
            fx=scale_factor,
            fy=scale_factor)
        # mask where the pixels are ON

        on_mask = (patt_gray > ref_on) & proj_mask
        on_mask = (on_mask << i).astype(np.uint16)
        # this code corresponds with the binary pattern code
        bit_code = np.uint16(1 << i)
        scan_bits |= (bit_code & on_mask)
        # TODO: populate scan_bits by putting the bit_code according to on_mask

    # the codebook translates from <binary code> to (x,y) in projector screen space
    with open("binary_codes_ids_codebook.pckl", "r") as f:
        binary_codes_ids_codebook = pickle.load(f)

    camera_points = []
    projector_points = []
    correspondence = np.zeros((h, w, 3))
    a0 = cv2.imread('./images/pattern001.jpg', cv2.IMREAD_COLOR)
    color_points = []

    for x in range(w):
        for y in range(h):
            if not proj_mask[y, x]:
                continue  # no projection here
            if scan_bits[y, x] not in binary_codes_ids_codebook:
                continue  # bad binary code
            val = binary_codes_ids_codebook[scan_bits[y, x]]
            if val[0] >= 1279 or val[1] >= 799:  # filter
                continue
            # TODO: use binary_codes_ids_codebook[...] and scan_bits[y,x] to
            # TODO: find for the camera (x,y) the projector (p_x, p_y).
            # TODO: store your points in camera_points and projector_points
            # IMPORTANT!!! : due to differences in calibration and acquisition - divide the camera points by 2

            camera_points.append([x / 2.0, y / 2.0])
            projector_points.append([val[0] * 1.0, val[1] * 1.0])

            correspondence[y, x, 1] = val[1] / 960.0
            correspondence[y, x, 2] = val[0] / 1280.0

            color_points.append([a0[y, x, 2], a0[y, x, 1], a0[y, x, 0]])

    a, b = np.unique(correspondence, return_counts=True)
    correspondence = correspondence * 255
    cv2.imwrite('../Results/correspondence.jpg',
                correspondence.astype(np.uint8))

    # now that we have 2D-2D correspondances, we can triangulate 3D points!

    # load the prepared stereo calibration between projector and camera
    with open("camera_points.pkl", 'wb') as file:
        pickle.dump(camera_points, file)
    with open("projector_points.pkl", 'wb') as file:
        pickle.dump(projector_points, file)

    with open("stereo_calibration.pckl", "r") as f:
        d = pickle.load(f)
        camera_K = d['camera_K']
        camera_d = d['camera_d']
        projector_K = d['projector_K']
        projector_d = d['projector_d']
        projector_R = d['projector_R']
        projector_t = d['projector_t']

    # TODO: use cv2.undistortPoints to get normalized points for camera, use camera_K and camera_d

    camera_points = np.array([camera_points])

    # TODO: use cv2.undistortPoints to get normalized points for projector, use projector_K and projector_d

    projector_points = np.array([projector_points])
    norm_cam_points = cv2.undistortPoints(camera_points, camera_K, camera_d)
    norm_proj_points = cv2.undistortPoints(projector_points, projector_K,
                                           projector_d)

    # TODO: use cv2.triangulatePoints to triangulate the normalized points

    homo_points = cv2.triangulatePoints(
        np.column_stack((np.eye(3), [0, 0, 1])),
        np.column_stack((projector_R, projector_t)), norm_cam_points,
        norm_proj_points)

    # TODO: use cv2.convertPointsFromHomogeneous to get real 3D points
    # TODO: name the resulted 3D points as "points_3d"

    points_3d = cv2.convertPointsFromHomogeneous(homo_points.T)
    color_points = np.array(color_points).reshape(-1, 1, 3)
    points_3d = np.append(points_3d, color_points, 2)

    mask = (points_3d[:, :, 2] > 200) & (points_3d[:, :, 2] < 1400)

    points_3d = points_3d[mask].reshape(-1, 1, 6)

    return points_3d


def write_3d_points_in_color(points_3d):

    print("write colour output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output_color.xyz"
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d %d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2], p[0, 3],
                                             p[0, 4], p[0, 5]))


def write_3d_points(points_3d):

    # ===== DO NOT CHANGE THIS FUNCTION =====

    print("write output point cloud")
    print(points_3d.shape)
    output_name = sys.argv[1] + "output.xyz"
    with open(output_name, "w") as f:
        for p in points_3d:
            f.write("%d %d %d\n" % (p[0, 0], p[0, 1], p[0, 2]))


if __name__ == '__main__':

    # ===== DO NOT CHANGE THIS FUNCTION =====

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    points_3d = reconstruct_from_binary_patterns()
    write_3d_points(points_3d)
    write_3d_points_in_color(points_3d)
