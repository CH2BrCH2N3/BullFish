import nibabel as nib
import numpy as np
import csv
import time

stime = time.time()

seg = nib.load('C:/Users/User/OneDrive - The University of Hong Kong/Zebrafish drive/3D Zebrafish Brain Atlas/2021-08-22_AZBA_segmentation.nii.gz')
depth, length, width = seg.shape
seg_data = seg.get_fdata()

surface = [[0 for j in range(width)] for i in range(length)]

for i in range(length):
    for j in range(width):
        k = 0
        while k < depth:
            if seg_data[k][i][j] > 0:
                surface[i][j] = k
                break
            k += 1

with open('surface_DV.csv', 'w') as file:
    for i in range(length):
        for j in range(width - 1):
            file.write(str(surface[i][j]) + ', ')
        file.write(str(surface[i][width - 1]) + '\n')

etime = time.time()
print(etime - stime)