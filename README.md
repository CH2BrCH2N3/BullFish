BullFish_tracker
v0.0
- New system of managing settings
- Record settings in metadata
- Added code for finding head position and tail beat amplitude
v1.0
- Two methods of thresholding for identifying the tip of the caudal fin
- Abandoned Decimal class to speed things up without losing precision
- Faster algorithm of drawing fish in thresholded frame
- Less clumsy code for video editing
v1.1
- Algorithm of finding head coordinates
- New algorithm of finding midline coordinates
- Faster algorithm of finding square of neighboring white pixels
- Calculation of tail bend amplitudes moved to analysis.
- Format of annotated video changed to 'MJPG'
- Binary video using Method 2 thresholding can now be saved.
BullFish_analysis
v0.0
- Output plot using matplotlib
v1.1
- Changed method of calculating tail bend angles
- Tail bend amplitudes are now calculated here.
