import nibabel as nib
from decimal import Decimal
import time
stime = time.time()
seg = nib.load('2021-08-22_AZBA_segmentation.nii.gz')
depth, length, width = seg.shape
data = seg.get_fdata()
brain = [[[0 for k in range(590)] for i in range(900)] for j in range(450)]
for j in range(50, 450):
    for i in range(60, 900):
        for k in range(86, 590):
            brain[j][i][k] = int(data[j][i][k])
brain_regions = [[0 for j in range(9)] for i in range(203)]
for i in range(198):
    brain_regions[i][0] = i + 1
brain_regions[198][0] = 900
brain_regions[199][0] = 901
brain_regions[200][0] = 902
brain_regions[201][0] = 903
brain_regions[202][0] = 904
for i in range(60, 900):
    for j in range(50, 450):
        first = 0
        first_found = False
        last = 0
        for k in range(86, 590):
            if brain[j][i][k] != 0:
                if first_found == False:
                    first = k
                    first_found = True
                last = k
        if first > 0:
            midpt = (first + last) / 2
            for k in range(first, round(midpt + 0.5)):
                if brain[j][i][k] > 899:
                    brain_region = brain[j][i][k] - 702
                else:
                    brain_region = brain[j][i][k] - 1
                if k == round(midpt):
                    brain_regions[brain_region][1] += 0.5
                    brain_regions[brain_region][2] += j / 2
                    brain_regions[brain_region][3] += i / 2
                    brain_regions[brain_region][4] += k / 2
                else:
                    brain_regions[brain_region][1] += 1
                    brain_regions[brain_region][2] += j
                    brain_regions[brain_region][3] += i
                    brain_regions[brain_region][4] += k
            for k in range(round(midpt - 0.5), last + 1):
                if brain[j][i][k] > 899:
                    brain_region = brain[j][i][k] - 702
                else:
                    brain_region = brain[j][i][k] - 1
                if k == round(midpt):
                    brain_regions[brain_region][5] += 0.5
                    brain_regions[brain_region][6] += j / 2
                    brain_regions[brain_region][7] += i / 2
                    brain_regions[brain_region][8] += k / 2
                else:
                    brain_regions[brain_region][5] += 1
                    brain_regions[brain_region][6] += j
                    brain_regions[brain_region][7] += i
                    brain_regions[brain_region][8] += k
for i in range(203):
    brain_regions[i][2] = Decimal(brain_regions[i][2]) / Decimal(brain_regions[i][1])
    brain_regions[i][3] = Decimal(brain_regions[i][3]) / Decimal(brain_regions[i][1])
    brain_regions[i][4] = Decimal(brain_regions[i][4]) / Decimal(brain_regions[i][1])
    brain_regions[i][6] = Decimal(brain_regions[i][6]) / Decimal(brain_regions[i][5])
    brain_regions[i][7] = Decimal(brain_regions[i][7]) / Decimal(brain_regions[i][5])
    brain_regions[i][8] = Decimal(brain_regions[i][8]) / Decimal(brain_regions[i][5])
with open('Zebrafish_Brain_Atlas.csv', 'w') as f:
    header = ['Region', 'Left Volume', 'Left DV', 'Left AP', 'Left ML', 'Right Volume', 'Right DV', 'Right AP', 'Right ML']
    for word in header:
        f.write(word + ', ')
    f.write('\n')
    for i in range(203):
        for j in range(9):
            f.write(str(brain_regions[i][j]) + ', ')
        f.write('\n')
with open('Zebrafish_Brain_Stereotactic_Atlas.csv', 'w') as f:
    header = ['Region', 'Left Volume', 'Left DV', 'Left AP', 'Left ML', 'Right Volume', 'Right DV', 'Right AP', 'Right ML']
    for word in header:
        f.write(word + ', ')
    f.write('\n')
    for i in range(203):
        rounded_ap = round(brain_regions[i][3])
        rounded_ml = round(brain_regions[i][4])
        surface = 0
        for j in range(50, 450):
            if brain[j][rounded_ap][rounded_ml] != 0:
                surface = j
                break
        rounded_dv = round(brain_regions[i][2]) - surface
        row = [brain_regions[i][0], brain_regions[i][1] * 64, (rounded_ml - 335) * 4, (356 - rounded_ap) * 4, rounded_dv * 4]
        for item in row:
            f.write(str(item) + ', ')
        rounded_ap = round(brain_regions[i][7])
        rounded_ml = round(brain_regions[i][8])
        surface = 0
        for j in range(50, 450):
            if brain[j][rounded_ap][rounded_ml] != 0:
                surface = j
                break
        rounded_dv = round(brain_regions[i][6]) - surface
        row = [brain_regions[i][5] * 64, (rounded_ml - 335) * 4, (356 - rounded_ap) * 4, rounded_dv * 4]
        for item in row:
            f.write(str(item) + ', ')
        f.write('\n')
etime = time.time()
print(etime - stime)