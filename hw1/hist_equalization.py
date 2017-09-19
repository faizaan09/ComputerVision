import cv2
import numpy as np
import matplotlib.pyplot as plt
# import pdb; pdb.set_trace()

def displayImage(img, win_name):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()



def main():
	img_path = './HW1-Filters/input1.jpg'
	img = cv2.imread(img_path)
	b,g,r = cv2.split(img)
	chnls = [b,g,r]
	color = ('b','g','r')
	hist = {}
	cdf = {}
	for i,col in enumerate(color):
		hist[col] = cv2.calcHist([chnls[i]],[0],None,[256],[0,256])
		cdf[i] = np.cumsum(hist[col])
		cdf[i] = cdf[i]/(img.shape[0]*img.shape[1])*255
		chnls[i] = cdf[i][chnls[i]].astype('uint8')

	img_out = cv2.merge(chnls)
	displayImage(img_out,'out')
	# n_b,a,e = cv2.split(img_out)
	# equ = cv2.equalizeHist(chnls[2])
	# res = np.hstack((e,equ))
	# displayImage(res,'merge')


if __name__ == '__main__':
	main()