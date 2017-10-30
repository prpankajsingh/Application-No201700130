import math
import numpy as np
from os import listdir
from PIL import Image as PImage
from sklearn.decomposition import PCA
from imutils import face_utils
import argparse
import imutils
import dlib
import cv2
# TASK 2 STARTS
#function to calculate the square magnitude of each shape vector
def vectorMag(S,S_target,n):
	sum=0,sqSum=0
	for i in range(n):
		sqSum = 0
		for j in range(36):
			sqSum+= (S[i][j]-S_target[j])^2
		sum+=sqSum
	return sum
 #this function calculates the appropriate value of sigma as solved from equation 12
def Calc_sigma(S,S_target):
	sigma = (vectorMag(S,S_target,n) - vectorMag(S,S_target,N))/(2*(-0.10536))
	return sigma
#Calculation of weight based on equation 11
def Calc_weight(S, S_target):
	sqSum=0
	sigma = Calc_sigma(S,S_target);
	for i in range(N):
		sqSum = 0
		for j in range(36):
			sqSum+= (S[i][j]-S_target[j])^2
			p= sqSum/(2*sigma*sigma)
        w[i] = e^-p
 


#Calculating the weighted median as based on equation 9
def Calc_Median(I_k):
	
	min_val=0

	for i in range(N):
		pix1 = numpy.array(I_i[i])
		pix2 = numpy.array(I_k);
		x= np.absolute(pix1-pix2)*w[i]
		if(i==1 or min_val>x):
			min_val = x
			median_pix = pix1
			median_img = I_i[i]
			
#TASK 2 ENDS
			
#TASK 1 STARTS

def rect_to_bb(rect):
	# take a bounding prediction of face by dlib and convert it
	# to the format (x, y, w,h)
	x = rect.left()
	y = rect.top()
	w = rect.right() - x
	h = rect.bottom() - y
 
	# return a tuple of (x, y, w, h)
	return (x, y, w, h)

# the function returns the (x, y) coordinates of all the important landmarks of the face as a numpy array
def shape_to_np(shape, dtype="int"):
	# initialize the list of (x, y)-coordinates
	coords = np.zeros((68, 2), dtype=dtype)
 
	# loop over the 68 facial landmarks and convert them
	# to a 2-tuple of (x, y)-coordinates
	for i in range(0, 68):
		coords[i] = (shape.part(i).x, shape.part(i).y)
 
	# return the list of (x, y)-coordinates
	return coords

# the function computes and returns the landmark points of the mouth region
def landmark(image):
	ap = argparse.ArgumentParser()
	ap.add_argument("-p", "--shape-predictor", required=True,
		help="path to facial landmark predictor")
	ap.add_argument("-i", "--image", required=True,
		help="path to input image")
	args = vars(ap.parse_args())
	# initialize dlib's face detector (HOG-based) and then create
	# the facial landmark predictor
	detector = dlib.get_frontal_face_detector()
	predictor = dlib.shape_predictor(args["shape_predictor"])
	image = cv2.imread(args["image"])
	image = imutils.resize(image, width=500)
	gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
	# detect faces in the grayscale image
	rects = detector(gray, 1)
	for (i, rect) in enumerate(rects):
		# determine the facial landmarks for the face region, then
		# convert the facial landmark (x, y)-coordinates to a NumPy
		# array
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)
	k=0
	for i in range (49 , 68):
		if(i!=61 or i!=65):
			lip[k++] = shape[i]
	return lip

# this determines the 20 PCA components of mouth shape by dimensionality reduction
def PCA_Coeff():
	for i in range(N) :
		points = landmark(I_i[i]);
		S[i] = np.reshape(points, (np.product(points.shape),))
	pca = PCA(n_components=20)
	S_PCA = pca.fit_transform(S)
	
#function to load the images from the dataset	
def loadImages(path):
    # return array of images

    imagesList = listdir(path)
    loadedImages = []
    for image in imagesList:
        img = PImage.open(path + image)
        loadedImages.append(img)

    return loadedImages

#TASK 1 ENDS

alpha=0.9
w=[0]* N
path = "/path/to/images"     # the variable contains path to the image dataset
I_i = loadImages(path)
S = []
for i in range(N):
	S.append([])
	for j in range(36):
        	S[i].append(0)
		
