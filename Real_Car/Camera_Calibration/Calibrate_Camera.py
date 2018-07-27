import matplotlib.pyplot as plt
import numpy as np
%matplotlib inline
import cv2 
import pickle
import glob

image_paths = glob.glob('Mobius_Dashcam_Camera_Cal/1920x1080/camera_cal/*.jpg')
#print('image paths: ',image_paths)
num_image_paths = len(image_paths)
print('number of image paths: ',num_image_paths)

nx = 7 # number of inside corners per row
ny = 7 # number of inside corners per col

"""
prepare object points:
[[0,0,0],
 [1,0,0],
 [2,0,0],
 ......,
 [7,7,0]
"""
objp = np.zeros((ny*nx,3),np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

# Lists to store object points and image points from all images
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image space

f, axs = plt.subplots(num_image_paths//4+1,4,figsize=(10,15))
axs = axs.ravel()

# Iterate through list of images and draw chessboard corners on image
for index,fileName  in enumerate(image_paths):
    print("index: ", index)
    image = cv2.imread(fileName)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx,ny), None)
    # If chessboard corners found
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw the corners
        cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
        axs[index].imshow(image)
print("done iterating")
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints,
                                                   gray.shape[::-1], # tuple (width,height)
                                                   None, None)
print('mtx: ',mtx)
print('dist: ', dist)
file_name = 'Mobius_Dashcam_Camera_Cal/1920x1080/mobius_dashcam_1920x1080_calibration_for_python2.p'

with open(file_name, 'wb') as f: 
    pickle.dump([mtx, dist], f)
    
# Try loading pickle file

file_name = 'Mobius_Dashcam_Camera_Cal/1920x1080/mobius_dashcam_1920x1080_calibration_for_python2.p'

with open(file_name, 'rb') as f:   
    mtx, dist = pickle.load(f)

print('mtx: ',mtx)
print('dist: ', dist)

# Display the images

f, axs = plt.subplots(num_image_paths,2,figsize=(15,70))
axs = axs.ravel()
plt.axis('off')
for i,image_path in enumerate(image_paths):
    image = cv2.imread(image_path)
    #print('image.shape: ',image.shape)
    image_undistorted = cv2.undistort(image, mtx, dist, None, mtx)
    axs[i*2].imshow(image)
    axs[i*2].axis('off')
    axs[i*2].set_title('original',fontsize=13)
    axs[i*2+1].imshow(image_undistorted)
    axs[i*2+1].axis('off')
    axs[i*2+1].set_title('undistorted',fontsize=13)

plt.show()