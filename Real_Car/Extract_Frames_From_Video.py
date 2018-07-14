"""
This program extracts frames from videos
"""
import cv2

path = 'project_video_output_with_vehicle_detection.mp4' # path to the input video you wish to extract frames frame
vidcap = cv2.VideoCapture(path)
success,image = vidcap.read()

count = 0
success = True

while success:
  success,image = vidcap.read()
  print('Read a new frame: ', success)
  cv2.imwrite('frames/%d.jpg' % count, image) # path you wish to save each of your frames to; save each frame as JPEG file
  count += 1
