"""
This program extracts frames from videos
"""
import cv2

path = 'Real_Car_Images_And_Videos/Bay_Area_Videos/Highway_280_3-31-2018.mp4' # path to the input video you wish to extract frames frame
vidcap = cv2.VideoCapture(path)
success,image = vidcap.read()

count = 0
success = True

while (success and count <= 2179):
  success,image = vidcap.read()
  print('Read a new frame: ', count)
  if (count >= 1703):
      cv2.imwrite('Real_Car_Images_And_Videos/Bay_Area_Videos/subclip3_frames/%d.jpg' % (count-1703), image) # path you wish to save each of your frames to; save each frame as JPEG file
  count += 1
