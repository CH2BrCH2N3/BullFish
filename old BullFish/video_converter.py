import os
import csv
import cv2 as cv

with open('settings_video_converter.csv', 'r') as f:
    settings = {row[0]: row[1] for row in csv.reader(f)}
    lr = bool(int(settings['Large rotation? (0/1)']))
    sr = bool(int(settings['Small rotation? (0/1)']))
    cr = bool(int(settings['Crop? (0/1)']))
    cr_fix = bool(int(settings['Crop with fixed width and height? (0/1)']))
    cr_fix_width = int(settings['Width of the cropped video'])
    cr_fix_height = int(settings['Height of the cropped video'])
    check = bool(int(settings['Check before production? (0/1)']))

class video_p:
    def __init__(self, filename):
        
        split_tup = os.path.splitext(filename)
        self.videoname = split_tup[0]
        self.filename = filename
        cap = cv.VideoCapture(self.videoname + '.mp4')
        print('Opening: ' + self.videoname + '.mp4')
        self.skip = False
        if input('Enter s to skip this video, any other things to continue:') == 's':
            self.skip = True
            return
        if not cap.isOpened():
            print('Unable to open ' + self.videoname + '.mp4')
            self.skip = True
            return
        self.width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))

        path = './' + self.videoname + '_p'
        if not os.path.exists(path):
            os.mkdir(path)
        
        while True:
            self.start = int(input('What is the starting frame?'))
            self.end = int(input('What is the ending frame?'))
            try:
                cap.set(cv.CAP_PROP_POS_FRAMES, self.start)
                ret, frame1 = cap.read()
                if ret:
                    cv.imwrite(self.videoname + '_Frame1.png', frame1)
                    print(self.videoname + '_Frame1.png saved.')
                else:
                    print('Unable to read ' + self.filename)
                    continue
                self.frame1 = frame1
                cap.release()
                break
            except:
                print('Try again')
        
        cr_tryagain = False
        while True:
            
            self.widthlr = self.width
            self.heightlr = self.height
            self.large_rotate = 0
            frame1lr = self.frame1
            if lr:
                self.large_rotate = int(input('Enter 90, 180, or 270 if the video needs to be rotated clockwise by 90, 180, or 270 deg, 0 if not needed:'))
                if self.large_rotate == 90:
                    frame1lr = cv.rotate(frame1, cv.ROTATE_90_CLOCKWISE)
                elif self.large_rotate == 180:
                    frame1lr = cv.rotate(frame1, cv.ROTATE_180)
                elif self.large_rotate == 270:
                    frame1lr = cv.rotate(frame1, cv.ROTATE_90_COUNTERCLOCKWISE)
            size = frame1lr.shape
            self.widthlr = size[1]
            self.heightlr = size[0]
            
            self.small_rotate = 0
            frame1sr = frame1lr
            if sr:
                self.small_rotate = -float(input('Enter the rotation angle if the video needs to be rotated clockwise by a small angle, 0 if this is not needed:'))
                if self.small_rotate != 0:
                    self.rm = cv.getRotationMatrix2D((self.widthlr, self.heightlr), self.small_rotate, 1.0)
                    frame1sr = cv.warpAffine(frame1lr, self.rm, (self.widthlr, self.heightlr))
            
            self.widthc = self.widthlr
            self.heightc = self.heightlr
            self.centerx = 0
            self.centery = 0
            frame1c = frame1sr
            if cr:
                self.centerx = int(input('Enter the x-coordinate of the center of the cropping area:'))
                self.centery = int(input('Enter the y-coordinate of the center of the cropping area:'))                
                if cr_fix and not cr_tryagain:
                    self.widthc = cr_fix_width
                    self.heightc = cr_fix_height
                    cr_tryagain = True
                else:
                    self.widthc = int(input('Enter the width of the cropped video:'))
                    self.heightc = int(input('Enter the height of the cropped video:'))
                self.tlx = self.centerx - self.widthc // 2
                self.tly = self.centery - self.heightc // 2
                self.brx = self.tlx + self.widthc
                self.bry = self.tly + self.heightc
                frame1c = frame1sr[self.tly:self.bry, self.tlx:self.brx]
                size = frame1c.shape
                self.widthc = size[1]
                self.heightc = size[0]
            
            self.bframe = cv.cvtColor(frame1c, cv.COLOR_BGR2GRAY)
            cv.imwrite(self.videoname + '_Frame1_processed.png', self.bframe)
            print(self.videoname + '_Frame1_processed.png saved.')
            
            if check == False:
                break
            else:
                finished = input('Enter f to carry on the video processing with the current parameters, others to edit the parameters:')
                if finished == 'f':
                    break
        
        with open(path + '/' + self.videoname + '_p.csv', 'w') as csvfile:
            header = ['Video name', 'Start frame', 'End frame', 'Large rotation (clockwise)', 'Small rotation (clockwise)',
                      'x-coordinate of center of cropping', 'y-coordinate of center of cropping', 'New width (pixels)', 'New height (pixels)']
            for cell in header:
                csvfile.write(cell + ', ')
            csvfile.write('\n')
            row = [self.filename, self.start, self.end, self.large_rotate, -self.small_rotate,
                   self.centerx, self.centery, self.widthc, self.heightc]
            for cell in row:
                csvfile.write(str(cell) + ', ')
            csvfile.write('\n')
            
        if not os.path.exists(self.videoname + '_p_info.csv'):
            with open(self.videoname + '_p_info.csv', 'w') as csvfile:
                csvfile.write('x pixel at the left border' + '\n' + 'x pixel at the right border' + '\n'
                              + 'y pixel at the top border' + '\n' + 'y pixel at the bottom border' + '\n')

video = []
for file in os.listdir('.'):
    filename = os.fsdecode(file)
    if filename.endswith('.mp4') or filename.endswith('.avi'):
        video.append(video_p(filename))
    
video_n = len(video)
i = 0
while i < video_n:
    
    if not video[i].skip:
        
        cap = cv.VideoCapture(video[i].filename)
        gray = cv.VideoWriter(video[i].videoname + '_p.avi', cv.VideoWriter_fourcc('F','F','V','1'), video[i].fps, (video[i].widthc, video[i].heightc), 0)
        print('Producing: ' + video[i].videoname + '_p.avi')
        
        j = video[i].start
        while j < video[i].end:
            cap.set(cv.CAP_PROP_POS_FRAMES, j)
            ret, frame = cap.read()
            if video[i].large_rotate == 90:
                frame = cv.rotate(frame, cv.ROTATE_90_CLOCKWISE)
            if video[i].large_rotate == 180:
                frame = cv.rotate(frame, cv.ROTATE_180)
            if video[i].large_rotate == 270:
                frame = cv.rotate(frame, cv.ROTATE_90_COUNTERCLOCKWISE)
            if video[i].small_rotate != 0:
                frame = cv.warpAffine(frame, video[i].rm, (video[i].widthlr, video[i].heightlr))
            if cr:
                frame = frame[video[i].tly:video[i].bry, video[i].tlx:video[i].brx]
            bframe = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
            gray.write(bframe)
            print('\rProgress:', j, end = '')
            j += 1
        
        print('\n' + video[i].videoname + '_p.avi produced.')
        gray.release() 
        cap.release()
        
    i += 1
    
cv.destroyAllWindows()
