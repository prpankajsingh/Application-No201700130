import math
import numpy as np
from os import listdir
from PIL import Image as PImage
def vectorMag(S,S_target,n):
	sum=0,sqSum=0
	for i in range(n):
		sqSum = 0
		for j in range(36):
			sqSum+= (S[i][j]-S_target[j])^2
		sum+=sqSum
	return sum
 
def Calc_sigma(S,S_target):
	sigma = (vectorMag(S,S_target,n) - vectorMag(S,S_target,N))/(2*(-0.10536))
	return sigma
 
def Calc_weight(S, S_target):
	sqSum=0
	sigma = Calc_sigma(S,S_target);
	for i in range(N):
		sqSum = 0
		for j in range(36):
			sqSum+= (S[i][j]-S_target[j])^2
			p= sqSum/(2*sigma*sigma)
        w[i] = e^-p
 
alpha=0.9
w=[0]* N

def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        loadedImages.append(img)

    return loadedImages

def Calc_Median(I_k)
	path = "/path/to/images"     # the variable contains path to the image dataset

	
	I_i = loadImages(path)
	min_val=0;

	for i in range(N):
		pix1 = numpy.array(I_i[i]);
		pix2 = numpy.array(I_k);
		x= np.absolute(pix1-pix2)*w[i]
		if(i==1||min_val>x)
			min_val = x
			median_pix = pix1
			median_img = I_i[i]
			
			
		
