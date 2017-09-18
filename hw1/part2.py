import numpy as np
import cv2
from matplotlib import pyplot as plt
import pdb; pdb.set_trace()

def displayImage(img, win_name):
    cv2.namedWindow(win_name, cv2.WINDOW_NORMAL)
    cv2.imshow(win_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img = cv2.imread('./HW1-Filters/input2.png')
img = cv2.cvtColor(img,code = cv2.COLOR_BGR2GRAY)


def get_HPF(img):
	f = np.fft.fft2(img)
	fshift = np.fft.fftshift(f)
	magnitude_spectrum = 20*np.log(np.abs(fshift))
	# plt.subplot(121),plt.imshow(img, cmap = 'gray')
	# plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	# plt.subplot(122)
	plt.imshow(magnitude_spectrum, cmap = 'gray')
	plt.title('Magnitude Spectrum')
	# plt.show()


	rows, cols = img.shape
	crow,ccol = rows/2 , cols/2
	fshift[crow-30:crow+30, ccol-30:ccol+30] = 0
	f_ishift = np.fft.ifftshift(fshift)
	img_back = np.fft.ifft2(f_ishift)
	img_back = np.abs(img_back)
	# displayImage(img_back,"back")
	cv2.imwrite("img_back_HPF.png",img_back)

########### code for LPF starts here ###########


def get_LPF(img):
	rows, cols = img.shape
	crow,ccol = rows/2 , cols/2

	# create a mask first, center square is 1, remaining all zeros
	mask = np.zeros((rows,cols),np.uint8)
	mask[crow-30:crow+30, ccol-30:ccol+30] = 1

	f = np.fft.fft2(img)
	# apply mask and inverse DFT
	fshift = np.fft.fftshift(f)
	fshift = fshift*mask
	f_ishift = np.fft.ifftshift(fshift)
	img_back = np.fft.ifft2(f_ishift)
	img_back = np.abs(img_back)
	plt.subplot(121),plt.imshow(img, cmap = 'gray')
	plt.title('Input Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(img_back, cmap = 'gray')
	plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
	plt.show()

	cv2.imwrite("img_back_LPF.png",img_back)


# def get_deconvulted_image(img):
def ft(im, newsize=None):
	dft = np.fft.fft2(np.float32(im),newsize)
	return np.fft.fftshift(dft)

def ift(shift):
	f_ishift = np.fft.ifftshift(shift)
	img_back = np.fft.ifft2(f_ishift)#,cv2.DFT_SCALE)
	return np.abs(img_back)


def decon_img(img):
	im = cv2.imread('./HW1-Filters/blurred2.exr', cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
	# im = cv2.cvtColor(im,code = cv2.COLOR_BGR2GRAY)
	gk = cv2.getGaussianKernel(21,5)
	gk = gk * gk.T
	channels = cv2.split(im)
	imconvf = []
	for c in channels:
		imf = ft(c, (c.shape[0],c.shape[1])) # make sure sizes match
		gkf = ft(gk, (c.shape[0],c.shape[1])) # so we can multiple easily
		imconvf.append(imf / gkf)

	# now for example we can reconstruct the blurred image from its FT

	blurred = [ift(imc) for imc in imconvf]
	blurred = cv2.merge(tuple(blurred))
	cv2.imwrite('decon.exr',blurred)
	cv2.imshow('image',blurred)
	cv2.waitKey(0)
	cv2.destroyAllWindows()

get_HPF(img)