import csv
import cv2 as cv

class video_p:
    def __init__(self, videoname, start, end):
        
        self.videoname = videoname
        cap = cv.VideoCapture(videoname + '.mp4')
        if cap.isOpened():
            print('Processing: ' + videoname + '.mp4')
        else:
            print('Unable to open ' + videoname + '.mp4')
            return
        
        self.width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
        self.height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
        self.fps = int(cap.get(cv.CAP_PROP_FPS))
        self.start = start
        self.end = end
        cap.set(cv.CAP_PROP_POS_FRAMES, start)
        ret, frame1 = cap.read()
        if ret:
            cv.imwrite(videoname + '_Frame1.png', frame1)
            print(videoname + '_Frame1.png saved.')
        else:
            print('Unable to read ' + videoname + '.mp4')
            return
        self.frame1 = frame1
        cap.release()
        
        self.widthlr = self.width
        self.heightlr = self.height     
        frame1lr_finished = 'n'
        while frame1lr_finished != 'f':
            self.large_rotate = int(input('Enter 90, 180, or 270 if the video needs to be rotated clockwise by 90, 180, or 270 deg, 0 if not needed:'))
            if self.large_rotate == 0:
                break
            elif self.large_rotate == 90:
                frame1lr = cv.rotate(frame1, cv.ROTATE_90_CLOCKWISE)
            elif self.large_rotate == 180:
                frame1lr = cv.rotate(frame1, cv.ROTATE_180)
            elif self.large_rotate == 270:
                frame1lr = cv.rotate(frame1, cv.ROTATE_90_COUNTERCLOCKWISE)
            else:
                print('Invalid input.')
                continue
            cv.imwrite(videoname + '_Frame1_processed.png', frame1lr)
            print(videoname + '_Frame1_processed.png saved.')
            frame1lr_finished = input('Enter f to carry on the video processing with the current settings, others to repeat the settings:')
        if frame1lr_finished == 'f':
            self.frame1 = frame1lr
            size = frame1lr.shape
            self.widthlr = size[1]
            self.heightlr = size[0]
        
        frame1sr_finished = 'n'
        while frame1sr_finished != 'f':
            self.small_rotate = -float(input('Enter the rotation angle if the video needs to be rotated clockwise by a small angle, 0 if this is not needed:'))
            if self.small_rotate != 0:
                self.rm = cv.getRotationMatrix2D((self.widthlr, self.heightlr), self.small_rotate, 1.0)
                frame1sr = cv.warpAffine(self.frame1, self.rm, (self.widthlr, self.heightlr))
                cv.imwrite(videoname + '_Frame1_processed.png', frame1sr)
                print(videoname + '_Frame1_processed.png saved.')
                frame1sr_finished = input('Enter f to carry on the video processing with the current settings, others to repeat the settings:')
            else:
                break
        if frame1sr_finished == 'f':
            self.frame1 = frame1sr
        
        self.crop = bool(int(input('Enter 0 if the video does not need to be cropped, others if yes:')))
        self.xl = 0
        self.xr = self.widthlr - 1
        self.yu = 0
        self.yl = self.heightlr - 1
        self.widthc = self.widthlr
        self.heightc = self.heightlr
        if self.crop:
            frame1c_finished = 'n'
            while frame1c_finished != 'f':
                self.xl = int(input('Enter the x-coordinate of the left boundary:'))
                self.xr = int(input('Enter the x-coordinate of the right boundary:'))
                self.yu = int(input('Enter the y-coordinate of the upper boundary:'))
                self.yl = int(input('Enter the y-coordinate of the lower boundary:'))
                frame1c = self.frame1[self.yu:self.yl, self.xl:self.xr]
                cv.imwrite(videoname + '_Frame1_processed.png', frame1c)
                print(videoname + '_Frame1_processed.png saved.')
                frame1c_finished = input('Enter f to carry on the video processing with the current settings, others to repeat the settings:')
            self.frame1 = frame1c
            size = frame1c.shape
            self.widthc = size[1]
            self.heightc = size[0] 

with open('videos.csv', 'r') as f:
    table = [[cell for cell in row] for row in csv.reader(f)]

l = len(table)
video = [0 for i in range(l)]
i = 1
while i < l:
    video[i] = video_p(table[i][0], int(table[i][1]), int(table[i][2]))
    i += 1

i = 1
while i < l:
    cap = cv.VideoCapture(video[i].videoname + '.mp4')
    output = cv.VideoWriter(video[i].videoname + '_p.avi', cv.VideoWriter_fourcc('M','J','P','G'), video[i].fps, (video[i].widthc, video[i].heightc), 0)
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
        if video[i].crop:
            frame = frame[video[i].yu:video[i].yl, video[i].xl:video[i].xr]
        output.write(cv.cvtColor(frame, cv.COLOR_BGR2GRAY))
        print('\rProgress:', j, end = '')
        j += 1
    output.release()
    cap.release()
    print()
    i += 1

with open('videos_p.csv', 'w') as f:
    header = ['Video name', 'Start frame', 'End frame', 'Large rotation (clockwise)', 'Small rotation (clockwise)',
              'Crop', 'Left boundary', 'Right boundary', 'Upper boundary', 'Lower boundary', 'New width (pixels)', 'New height (pixels)']
    for cell in header:
        f.write(cell + ', ')
    f.write('\n')
    for i in range(1, l):
        row = [video[i].videoname, video[i].start, video[i].end, video[i].large_rotate, -video[i].small_rotate,
               video[i].crop, video[i].xl, video[i].xr, video[i].yu, video[i].yl, video[i].widthc, video[i].heightc]
        for cell in row:
            f.write(str(cell) + ', ')
        f.write('\n')

cv.destroyAllWindows()