# USAGE
# python stitch.py --first images/bryce_left_01.png --second images/bryce_right_01.png

# import the necessary packages
from pyimagesearch.panorama import Stitcher
import argparse
import imutils
import cv2
import time
import glob
"""
# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-f", "--first", required=True,
	help="path to the first image")
ap.add_argument("-s", "--second", required=True,
	help="path to the second image")
args = vars(ap.parse_args())

# load the two images and resize them to have a width of 400 pixels
# (for faster processing)
imageA = cv2.imread(args["first"])
imageB = cv2.imread(args["second"])
imageA = imutils.resize(imageA, width=400)
imageB = imutils.resize(imageB, width=400)
print("imageA.shape: ", imageA.shape)
"""
left_image_paths = sorted(glob.glob("images/*.png"))
right_image_paths = []

while (len(left_image_paths) != 7):
	for i,path in enumerate(left_image_paths):
		if "right" in path:
			del left_image_paths[i]
			right_image_paths.append(path)
print(len(left_image_paths))
right_image_paths = sorted(right_image_paths)

stitcher = Stitcher()
while True:
	for i, (left_image_path, right_image_path) in enumerate(zip(left_image_paths, right_image_paths)):
		left_image = cv2.imread(left_image_path)
		right_image = cv2.imread(right_image_path)
		t1 = time.time()
		result = stitcher.stitch([left_image, right_image])
		t2 = time.time()
		print("delta time: ", t2-t1)
"""
# show the images
cv2.imshow("Image A", imageA)
cv2.imshow("Image B", imageB)
cv2.imshow("Keypoint Matches", vis)
cv2.imshow("Result", result)
cv2.waitKey(0)
"""
