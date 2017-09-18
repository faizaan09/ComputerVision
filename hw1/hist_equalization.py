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

		# plt.plot(hist[col],color = col)

	############ WE ARE NOT SUPPOSED TO USE FOR LOOPS, TRY ANOTHER WAY LATER ON ####################
	b,g,r = cv2.split(img)
	b = np.array([cdf[0][i] for i in b ])
	g = np.array([cdf[1][i] for i in g ])
	r = np.array([cdf[2][i] for i in r ])

	img_out = cv2.merge((b,g,r))
			
	n_b,a,e = cv2.split(img_out)
	equ = cv2.equalizeHist(chnls[0])
	res = np.hstack((n_b,equ))
	displayImage(res,'merge')


if __name__ == '__main__':
	main()