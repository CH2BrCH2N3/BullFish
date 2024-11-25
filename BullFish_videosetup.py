import os
import cv2 as cv
from BullFish_pkg.general import create_path, get_input, load_settings
from BullFish_pkg.cv_editing import get_rm, frame_rotate
from copy import deepcopy
from traceback import print_exc

default_settings = {
    "video_start": -1,
    "video_end": -1,
    "rotate": 181,
    "crop_tlx": -1,
    "crop_tly": -1,
    "crop_x": -1,
    "crop_y": -1,
    "swimarea_tlx": -1,
    "swimarea_tly": -1,
    "swimarea_x": -1,
    "swimarea_y": -1,
    "check": 1,
    'downsampling': 1
}

print('Welcome to BullFish. After recording your zebrafish videos, this program will guide you to set up the videos for tracking and analysis.')

settings = load_settings('videosetup', default_settings)

for file in os.listdir('.'):
     
    filename = os.fsdecode(file)
    filename_split = os.path.splitext(filename)
    supported_formats = {'.avi', '.mp4'}
    if filename_split[1] not in supported_formats:
        continue
    video = cv.VideoCapture(filename)
    if not video.isOpened():
        print(filename + ' cannot be opened.')
        continue
    print('\nOpening ' + filename + ' ...')
    
    try:
        
        videoname = filename_split[0]
        path = './' + videoname
        create_path(path)
        
        x_original = int(video.get(cv.CAP_PROP_FRAME_WIDTH))
        y_original = int(video.get(cv.CAP_PROP_FRAME_HEIGHT))
        framenumber_original = int(video.get(cv.CAP_PROP_FRAME_COUNT))
        fps = video.get(cv.CAP_PROP_FPS) / settings['downsampling']
        
        create_metadata = True
        if os.path.exists(path + '/' + videoname + '_metadata.csv'):
            while True:
                response = input(videoname + '_metadata.csv' + ' already exists. Enter y to use the existing metadata. Enter o to overwrite the metadata.')
                if response == 'y':        
                    create_metadata = False
                elif response != 'o':
                    print('Try again')
                    continue
                break
        
        if create_metadata:
            
            print('Creating metadata for ' + videoname + '...')
            
            if settings['video_start'] == -1:
                settings['video_start'] = get_input(int, 'What is the starting frame?')
            if settings['video_end'] == -1:
                settings['video_end'] = get_input(int, 'What is the ending frame?')
            while True:
                if settings['video_end'] < framenumber_original and settings['video_start'] < settings['video_end']:
                    break
                else:
                    print('Unacceptable range. Choose again.')
                    settings['video_start'] = get_input(int, 'What is the starting frame?')
                    settings['video_end'] = get_input(int, 'What is the ending frame?')
            
            video.set(cv.CAP_PROP_POS_FRAMES, settings['video_start'])
            ret, frame1 = video.read()
            cv.imwrite(videoname + '_Frame1.png', frame1)
            video.release()
            print('The first frame of the selected video segment has been saved at the current directory.')
            print('The following part is to set the parameters for rotating and cropping the video to prepare for tracking.')
            
            if settings['rotate'] == 181:
                settings['rotate'] = get_input(float, 'What is the video rotation angle?')
            if settings['crop_tlx'] == -1:
                settings['crop_tlx'] = get_input(int, 'Enter the x-coordinate of the top-left corner of the cropping area:')
            if settings['crop_tly'] == -1:
                settings['crop_tly'] = get_input(int, 'Enter the y-coordinate of the top-left corner of the cropping area:')
            if settings['crop_x'] == -1:
                settings['crop_x'] = get_input(int, 'Enter the width (x) of the cropping area:')
            if settings['crop_y'] == -1:
                settings['crop_y'] = get_input(int, 'Enter the height (y) of the cropping area:')
            
            while settings['check']:
                try:
                    rm = get_rm(x_original, y_original, settings['rotate'])
                    frame_t = frame_rotate(frame1, x_original, y_original, settings['rotate'], rm)
                    crop_brx = settings['crop_tlx'] + settings['crop_x']
                    crop_bry = settings['crop_tly'] + settings['crop_y']
                    if settings['crop_x'] != 0 and settings['crop_y'] != 0:
                        frame_t = frame_t[settings['crop_tly']:crop_bry, settings['crop_tlx']:crop_brx]
                    cv.imwrite(videoname + '_edited_frame.png', frame_t)
                    if input('The rotated and cropped frame is saved. Enter f to carry on, others to change:') == 'f':
                        break
                except Exception:
                    print('An error occurred when editing the frame. Change parameters.')
                    print_exc()
                settings['rotate'] = get_input(float, 'What is the video rotation angle?')
                settings['crop_tlx'] = get_input(int, 'Enter the x-coordinate of the top-left corner of the cropping area:')
                settings['crop_tly'] = get_input(int, 'Enter the y-coordinate of the top-left corner of the cropping area:')
                settings['crop_x'] = get_input(int, 'Enter the width (x) of the cropping area:')
                settings['crop_y'] = get_input(int, 'Enter the height (y) of the cropping area:')
            
            size = frame_t.shape
            x_current = size[1]
            y_current = size[0]
            
            if settings['swimarea_tlx'] == -1:
                settings['swimarea_tlx'] = get_input(int, 'Enter the x-coordinate of the top-left corner of the swimming area:')
            if settings['swimarea_tly'] == -1:
                settings['swimarea_tly'] = get_input(int, 'Enter the y-coordinate of the top-left corner of the swimming area:')
            if settings['swimarea_x'] == -1:
                settings['swimarea_x'] = get_input(int, 'Enter the width (x) of the swimming area:')
            if settings['swimarea_y'] == -1:
                settings['swimarea_y'] = get_input(int, 'Enter the height (y) of the swimming area:')
            
            while settings['check']:
                frame_l = deepcopy(frame_t)
                cv.rectangle(frame_l, (settings['swimarea_tlx'], settings['swimarea_tly']), (settings['swimarea_tlx'] + settings['swimarea_x'], settings['swimarea_tly'] + settings['swimarea_y']), (0, 0, 255), 3)
                cv.imwrite(videoname + '_labeled_frame.png', frame_l)
                if input('The swimming area is labeled in a new figure. Enter f to carry on, others to change:') == 'f':
                    break
                settings['swimarea_tlx'] = get_input(int, 'Enter the x-coordinate of the top-left corner of the swimming area:')
                settings['swimarea_tly'] = get_input(int, 'Enter the y-coordinate of the top-left corner of the swimming area:')
                settings['swimarea_x'] = get_input(int, 'Enter the width (x) of the swimming area:')
                settings['swimarea_y'] = get_input(int, 'Enter the height (y) of the swimming area:')
            
            metadata = {
                'filename': filename,
                'fps': fps,
                'x_original': x_original,
                'y_original': y_original,
                'x_current': x_current,
                'y_current': y_current}
            metadata.update(settings)
            with open(path + '/' + videoname + '_metadata.csv', 'w') as f:
                for key in metadata:
                    f.write(key + ',' + str(metadata[key]) + '\n')
            
    except Exception:
        
        print('An error occurred when setting up ' + filename + ' for tracking:')
        print_exc()
    