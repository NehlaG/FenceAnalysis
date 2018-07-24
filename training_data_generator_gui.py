import cv2
import numpy as np
import math
import time
import sys, os
from screeninfo import get_monitors
from scipy.stats import mode as sp_mode
from utilities import processArguments



params = {
    'root_dir': '/home/nehla/laser_code',
    'file_name': '/home/nehla/SharedFolder/fc2_save_2018-07-09-234632-0005.avi', # whole/incomplete barbed wire
    'show_img': 1,
    'resize_factor': 0.8,
    'codec': 'H264',
}
monitors = get_monitors()
curr_monitor = str(monitors[0])
resolution = curr_monitor.split('(')[1].split('+')[0].split('x')

res_width, res_height = [int(x) for x in resolution]

print('resolution: ', resolution)
for m in get_monitors():
    print(m)

processArguments(sys.argv[1:], params)

root_dir = params['root_dir']
file_name = params['file_name']
show_img = params['show_img']
resize_factor = params['resize_factor']
codec = params['codec']


file_path = os.path.join(root_dir, file_name)

if os.path.isdir(file_path):
    # Load sample video
    cap = cv2.VideoCapture(os.path.join(file_path, 'image%06d.jpg'))
else:
    if not os.path.exists(file_path):
        raise SystemError('Source video file: {} does not exist'.format(file_path))
    # Load sample video
    cap = cv2.VideoCapture(file_path)

if not cap:
    raise SystemError('Source video file: {} could not be opened'.format(file_path))

#try to read a frame
ret, curr_img = cap.read()

if not ret:
    print ("Fatal Error: Could not read/decode frames from specified file.")
    exit(-1)

width = cap.get(3)
height = cap.get(4)

original_width= int(width)
original_height=int (height)
if resize_factor != 1:
    width *= resize_factor
    height *= resize_factor

width = int(width)
height = int(height)

n_pts = height * width

print('width: ', width)
print('height: ', height)

wiudth_ratio = float(width) / float(res_width)
height_ratio = float(height) / float(res_height)

disp_resize_factor = 1
if wiudth_ratio > height_ratio:
    if wiudth_ratio > 1:
        disp_resize_factor = 1.0 / wiudth_ratio
else:
    if height_ratio > 1:
        disp_resize_factor = 1.0 / height_ratio


seq_name = os.path.basename(file_name)

#os.path.join(dir_name, base_filename + "." + filename_suffix)
frames_file_name=os.path.join(root_dir+'/FramesArrayFiles/',seq_name+ "." +'npy')
frames_array= []
### GUI and Callback
# enable the user to navigate through the video by using a trackbar and save a frame number by double-clicking on it. 
POS_TRACKBAR = "pos_trackbar"
CURRENT_FRAME_FLAG = cap.get(1)
TOTAL_FRAMES_FLAG = cap.get(7)

### STEP 1
# openCV function to create a named window
cv2.namedWindow(seq_name) 

### STEP 2
#create a trackbar callback function
def seek_callback(x):
    # we want to change the value of the frame variable in the global scope
    global curr_img,ret
    # by getting the position of the trackbar
    # POS_TRACKBAR is the name of the track bar which is going to be used to navigate through the video
    i = cv2.getTrackbarPos(POS_TRACKBAR, seq_name)
    # and skipping to the selected frame
    cap.set(int(CURRENT_FRAME_FLAG), i-1)
    ret, curr_img = cap.read()
    # we then update the window
    cv2.imshow(seq_name, curr_img)


# STEP 3
# we create the trackbar in the main window by using
# cv2.createTrackbar(trackbar_name: string, window_name: string,
#                    initial_value: int, max_value: int,
#                    callback_function: callable object)
cv2.createTrackbar(POS_TRACKBAR, seq_name, 0, int(TOTAL_FRAMES_FLAG), seek_callback)


# STEP 4
# we define a mouse callback function for mouse events. According to OpenCV docs
# it must take the following parameters:
# event  -> an integer flag describing the event which was triggered
# x, y   -> mouse x and y coordinates when the event was triggered
# flags  -> additional flags
# params -> optional parameters passed to the callback function

#We define a function which saves the current frame number to a file.
def save_image():
    #filename = "image_%0.5f.png" % t.time()
    #cv2.imwrite(filename, frame)
    frames_array.append(int(cap.get(1)))
    print ("saving into file ...")

def mouse_callback(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        save_image()

### STEP 5 add the mouse callback function
cv2.setMouseCallback(seq_name, mouse_callback)

while(True):
    #ret, curr_img = cap.read()
    #if not ret:
        #print ("Fatal Error: Could not read/decode frames from specified file.")
        #break
    # show images
    if show_img:
        #if resize_factor != 1:
            #curr_img = cv2.resize(curr_img, (0, 0), fx=resize_factor, fy=resize_factor)
        cv2.imshow(seq_name,curr_img)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# save frame numbers into file
print (frames_array)
np.save(frames_file_name, np.array(frames_array))

if show_img:
    cv2.destroyAllWindows()


