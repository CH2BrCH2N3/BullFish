import os
import cv2 as cv

with open('background_settings.txt', 'r') as f:
    fps = int(f.readline())

for file in os.listdir('.'):
     filename = os.fsdecode(file)
     split_tup = os.path.splitext(filename)
     if filename.endswith(".png"):
         image = cv.imread(filename)
         size = image.shape
         width = size[1]
         height = size[0]
         output = cv.VideoWriter(split_tup[0] + '.avi', cv.VideoWriter_fourcc('M','J','P','G'), fps, (width, height))
         for i in range(fps):
             output.write(image)
         output.release()

cv.destroyAllWindows()
