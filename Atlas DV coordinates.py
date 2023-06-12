import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
import csv
import time

stime = time.time()

seg = nib.load('C:/Users/User/OneDrive - The University of Hong Kong/Zebrafish drive/3D Zebrafish Brain Atlas/2021-08-22_AZBA_segmentation.nii.gz')
depth, length, width = seg.shape
seg_data = seg.get_fdata()

with open('azba_coordinates.csv', 'r') as csvfile:
    coord = [[int(number) for number in row] for row in csv.reader(csvfile)]

l = len(coord)
surface = []

for i in range(l):
    j = 0
    while j < depth - 1:
        if seg_data[j][coord[i][2]][coord[i][1]] > 0:
            surface.append(j)
            break
        j += 1
            
with open('azba_coordinates_for_inj.csv', 'w') as file:
    for i in range(l):
        file.write(str(coord[i][0]) + ', ' + str((coord[i][1] - 335) * 4) + ', ' 
                   + str((356 - coord[i][2]) * 4) + ', ' + str((coord[i][3] - surface[i]) * 4) + '\n')
    
etime = time.time()
print(etime - stime)