import cv2
import numpy as np
from main import *

drawing = False
marked_bg_pixels = []
len_bg = 0

marked_fg_pixels = []
len_fg = 0
mode = None


def mark_seeds(event, x, y, flags, param):
    global h, w, c, img, img_marking, drawing, mode  #,marked_bg_pixels,marked_fg_pixels
    size = 4

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing == True:
            if mode == "ob":
                # if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
                # marked_fg_pixels.append((y,x))
                cv2.circle(img, (x, y), size, (255, 0, 0), -1)
                cv2.circle(img_marking, (x, y), size, (255, 0, 0), -1)
            else:
                # if(x>=0 and x<=w-1) and (y>0 and y<=h-1):
                # marked_bg_pixels.append((y,x))
                cv2.circle(img, (x, y), size, (0, 0, 255), -1)
                cv2.circle(img_marking, (x, y), size, (0, 0, 255), -1)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        if mode == "ob":
            cv2.circle(img, (x, y), size, (255, 0, 0), -1)
            cv2.circle(img_marking, (x, y), size, (255, 0, 0), -1)
        else:
            cv2.circle(img, (x, y), size, (0, 0, 255), -1)
            cv2.circle(img_marking, (x, y), size, (0, 0, 255), -1)
        # segment_image()

        # Create a black image, a window and bind the function to window


img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
h, w, c = img.shape
cv2.namedWindow('Mark the object and background')
cv2.setMouseCallback('Mark the object and background', mark_seeds)

centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(
    img)
img_marking = np.ones_like(img)
img_marking = np.uint8(img_marking * 255)
cv2.setMouseCallback('Mark the object and background', mark_seeds)

while (1):

    cv2.imshow('Mark the object and background', img_marking)
    cv2.imshow('Mark the object and background', img)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('o'):
        mode = "ob"
    elif k == ord('b'):
        mode = "bg"
    elif k == ord('g'):

        # centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors(
        # img)

        fg_segments, bg_segments = find_superpixels_under_marking(
            img_marking, superpixels)

        fg_cumulative_hist = cumulative_histogram_for_superpixels(
            fg_segments, color_hists)

        bg_cumulative_hist = cumulative_histogram_for_superpixels(
            bg_segments, color_hists)

        norm_hists = normalize_histograms(color_hists)

        graph_cut = do_graph_cut((fg_cumulative_hist, bg_cumulative_hist),
                                 (fg_segments,
                                  bg_segments), norm_hists, neighbors)

        segmask = pixels_for_segment_selection(superpixels,
                                               np.nonzero(graph_cut))
        segmask = np.uint8(segmask * 255)
        output_name = "./bonus_mask.png"
        cv2.imwrite(output_name, segmask)
        cv2.imshow("imgMask", img_marking)
        cv2.waitKey(1)
        cv2.imshow("segMask", segmask)
        cv2.waitKey(1)
        mask = segmask

    elif k == 27:
        img_marking = np.ones_like(img)
        img_marking = np.uint8(img_marking * 255)
        img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    elif k == ord('q'):
        break

cv2.destroyAllWindows()
