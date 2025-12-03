from pathlib import Path
from copy import deepcopy
from traceback import print_exc
import cv2 as cv
from BullFish_pkg.general import getfilepath, create_path, get_input
from BullFish_pkg.cv_editing import supported_formats, get_rm, frame_rotate
import pandas as pd

settings = {
    "naming": 1,
    "check": 1}

metadata_all = []
metadata_path = Path.cwd() / 'metadata.csv'
if metadata_path.exists():
    if input('metadata.csv already exists! Do you want to overwrite it? Press y to continue overwriting it, others to exit:') != 'y':
        from sys import exit
        exit()

while True:
    
    filepath = getfilepath()
    p = Path(filepath)
    if p.suffix not in supported_formats:
        print(f'{p.suffix} not supported')
        continue
    video = cv.VideoCapture(filepath)
    if not video.isOpened():
        print(f'{p.suffix} cannot be opened.')
        continue
    print(f'\nOpening {filepath}...')
    
    try:
        
        x_original = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        y_original = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        framenumber_original = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv.CAP_PROP_FPS)
        if settings['naming']:
            name = input('Enter the name for this video:')
        else:
            name = p.stem
        subpath = f'{p.parent}/{name}'
        create_path(subpath)
        group = input('Enter the group name for this video (e.g. Control, Treatment, 1, 2, etc.):')
        metadata = {
            'filepath': filepath,
            'name': name,
            'group': group,
            'fps': fps,
            'x_original': x_original,
            'y_original': y_original}
        
        print('Now, choose the video segment for tracking.')
        print('Be aware of the frame rate as you need to enter the frame numbers instead of the time.')
        video_start = get_input(int, 'What is the starting frame?')
        video_end = get_input(int, 'What is the ending frame?')
        while True:
            if video_end < framenumber_original and video_start < video_end:
                break
            else:
                print('Unacceptable range. Choose again.')
                video_start = get_input(int, 'What is the starting frame?')
                video_end = get_input(int, 'What is the ending frame?')
        
        video.set(cv.CAP_PROP_POS_FRAMES, video_start)
        ret, frame1 = video.read()
        cv.imwrite(f'{p.stem}_Frame1.png', frame1)
        video.release()
        print('The first frame of the selected video segment has been saved at the current directory.')
        print('The following part is to set the parameters for rotating and cropping the video to prepare for tracking.')
        
        rotate = get_input(float, 'Enter the video rotation angle (clockwise, enter within -179.9 to 180):')
        crop_tlx = get_input(int, 'Enter the x-coordinate of the top-left corner of the cropping area:')
        crop_tly = get_input(int, 'Enter the y-coordinate of the top-left corner of the cropping area:')
        crop_x = get_input(int, 'Enter the width (x) of the cropping area:')
        crop_y = get_input(int, 'Enter the height (y) of the cropping area:')
        
        while settings['check']:
            try:
                rm = get_rm(x_original, y_original, rotate)
                frame_edited = frame_rotate(frame1, x_original, y_original, rotate, rm)
                crop_brx = crop_tlx + crop_x
                crop_bry = crop_tly + crop_y
                if crop_x != 0 and crop_y != 0:
                    frame_edited = frame_edited[crop_tly:crop_bry, crop_tlx:crop_brx]
                cv.imwrite(f'{p.stem}_edited_frame.png', frame_edited)
                if input('The rotated and cropped frame is saved. Enter f to carry on, others to change:') == 'f':
                    break
            except Exception:
                print('An error occurred when editing the frame. Change parameters.')
                print_exc()
            rotate = get_input(float, 'Enter the video rotation angle (clockwise, enter within -179.9 to 180):')
            crop_tlx = get_input(int, 'Enter the x-coordinate of the top-left corner of the cropping area:')
            crop_tly = get_input(int, 'Enter the y-coordinate of the top-left corner of the cropping area:')
            crop_x = get_input(int, 'Enter the width (x) of the cropping area:')
            crop_y = get_input(int, 'Enter the height (y) of the cropping area:')
        
        size = frame_edited.shape
        x_current = size[1]
        y_current = size[0]
        
        swimarea_tlx = get_input(int, 'Enter the x-coordinate of the top-left corner of the swimming area:')
        swimarea_tly = get_input(int, 'Enter the y-coordinate of the top-left corner of the swimming area:')
        swimarea_x = get_input(int, 'Enter the width (x) of the swimming area:')
        swimarea_y = get_input(int, 'Enter the height (y) of the swimming area:')
        
        while settings['check']:
            frame_l = deepcopy(frame_edited)
            cv.rectangle(frame_l, (swimarea_tlx, swimarea_tly), (swimarea_tlx + swimarea_x, swimarea_tly + swimarea_y), (0, 0, 255), 3)
            cv.imwrite(f'{p.stem}_labeled_frame.png', frame_l)
            if input('The swimming area is labeled in a new figure. Enter f to carry on, others to change:') == 'f':
                break
            swimarea_tlx = get_input(int, 'Enter the x-coordinate of the top-left corner of the swimming area:')
            swimarea_tly = get_input(int, 'Enter the y-coordinate of the top-left corner of the swimming area:')
            swimarea_x = get_input(int, 'Enter the width (x) of the swimming area:')
            swimarea_y = get_input(int, 'Enter the height (y) of the swimming area:')
        
        metadata.update({
            'video_start': video_start,
            'video_end': video_end,
            'rotate': rotate,
            'crop_tlx': crop_tlx,
            'crop_tly': crop_tly,
            'crop_x': crop_x,
            'crop_y': crop_y,
            'swimarea_tlx': swimarea_tlx,
            'swimarea_tly': swimarea_tly,
            'swimarea_x': swimarea_x,
            'swimarea_y': swimarea_y,
            'x_current': x_current,
            'y_current': y_current})
        metadata_all.append(metadata)
        
    except Exception:
        
        print(f'An error occurred when setting up {p.name} for tracking:')
        print_exc()
    
    if input('Press f to finish videosetup, others to continue creating metadata:') == 'f':
        break

metadata_all = pd.DataFrame(metadata_all)
metadata_all.to_csv('metadata.csv', index=False)
