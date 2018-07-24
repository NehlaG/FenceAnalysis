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


fence_mask_params = list()
fence_mask_params.append(cv2.IMWRITE_PNG_COMPRESSION)
fence_mask_params.append(8)

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
pictures_file_name=os.path.join(root_dir+'/PicturesFiles/',seq_name)

frames_array= np.load(frames_file_name)
#print (frames_array)
#print('frames_array size :', frames_array.size)
minLineLength = 10
maxLineGap = height
curr_img_gray = None
buffer_binaries_size=0
img_binaries = np.zeros((height,width), np.uint8)
top_corner_x=0


if frames_array.size >=2 :
    #initialization
    total_frames_number=cap.get(7)
    first_frame_index=0
    second_frame_index=first_frame_index+1
    current_frame_id= frames_array.item(first_frame_index)
    window_width=frames_array.item(second_frame_index)-frames_array.item(first_frame_index)+1

    while (current_frame_id< total_frames_number and  window_width>0) :
        cap.set(1, current_frame_id)
        ret, curr_img = cap.read()
        #initialization
        x_list= []
        binary_line_img=np.zeros((height,10), np.uint8)
        # resizing
        curr_img = cv2.resize(curr_img, (0, 0), fx=resize_factor, fy=resize_factor)
        curr_img_gray = cv2.cvtColor(curr_img, cv2.COLOR_BGR2GRAY)
        curr_img_lines = np.copy(curr_img)
        cv2.bilateralFilter(curr_img_gray, 3, 10,10)
        curr_img_gray = cv2.GaussianBlur(curr_img_gray,(3,3),0)
        binImg = cv2.adaptiveThreshold(curr_img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, -7 )
        #image thresholding
        #cv2.adaptiveThreshold(src, maxvalue, adaptive method, threshold type, bock size, Constant subtracted from the mean or weighted mean )
        #binImg = cv2.adaptiveThreshold(curr_img_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, -3 )
        #ret,binImg = cv2.threshold(curr_img_gray,127,255,cv2.THRESH_BINARY)
    
        # line detection
        lines = cv2.HoughLinesP(image=binImg,rho=1,theta=np.pi/180, threshold=20,lines=np.array([]), minLineLength=minLineLength,maxLineGap=maxLineGap)
        lines_number=0
        if lines is not None:
            a,b,c = lines.shape
            for i in range(a):
                angle = math.atan2(lines[i][0][3] -  lines[i][0][1], lines[i][0][2] - lines[i][0][0]) 
                theta=(angle*180)/np.pi
            
                if math.fabs(theta)>85 and math.fabs(theta)<95:
                    #print('theta :', theta)
                    cv2.line(curr_img_lines, (lines[i][0][0], lines[i][0][1]), (lines[i][0][2], lines[i][0][3]), (0, 0, 255), 3, cv2.LINE_AA)
                    x_list.append((lines[i][0][0]+lines[i][0][2])/2)
                    lines_number +=1
        #print('number of vertical detected lines', lines_number)
        if lines_number>0 :
            # clustering
            x_list = np.array(x_list)
            x_list = x_list.reshape(lines_number,1)
            x_list = np.float32(x_list)
            # Define criteria = ( type, max_iter = 10 , epsilon = 1.0 )
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            # Set flags (Just to avoid line break in the code)
            flags = cv2.KMEANS_RANDOM_CENTERS
            # Apply KMeans
            compactness,labels,centers = cv2.kmeans(x_list,1,None,criteria,10,flags)
            #print('print the center point X', centers[0])
            # line binary image
            binary_line_img=  binImg[0:height, int(centers[0])-5:int(centers[0])+5]

        # create binary image from binary lines
        if buffer_binaries_size < int(window_width) :
            img_binaries[0:height,top_corner_x:top_corner_x+10] = binary_line_img
            top_corner_x+=10
            buffer_binaries_size+=1
            current_frame_id+=1
        if buffer_binaries_size == int(window_width) :
            # save image
            cv2.imwrite(pictures_file_name+"picture-index"+str(first_frame_index)+"." +'png',img_binaries,fence_mask_params)
            img_binaries = np.zeros((height,width), np.uint8)
            buffer_binaries_size=0
            top_corner_x=0
            first_frame_index=second_frame_index
            second_frame_index=first_frame_index+1
            current_frame_id= frames_array.item(first_frame_index)
            if (second_frame_index<frames_array.size):
                window_width=frames_array.item(second_frame_index)-frames_array.item(first_frame_index)+1
                #print("done")
            else :
                window_width=0
        # show images
        if show_img:
            #if resize_factor != 1:
                #curr_img = cv2.resize(curr_img, (0, 0), fx=resize_factor, fy=resize_factor)
                #curr_img_gray = cv2.resize(curr_img_gray, (0, 0), fx=resize_factor, fy=resize_factor)
                #binImg = cv2.resize(binImg, (0, 0), fx=resize_factor, fy=resize_factor)
                #curr_img_lines = cv2.resize(curr_img_lines, (0, 0), fx=resize_factor, fy=resize_factor)
                #binary_line_img = cv2.resize(binary_line_img, (0, 0), fx=resize_factor, fy=resize_factor)

            cv2.imshow(seq_name,curr_img)
            cv2.imshow('video converted to gray',curr_img_gray)
            cv2.imshow('video converted to binary',binImg)
            #cv2.imshow('detected lines',curr_img_lines)
            #cv2.imshow('detected binary line',binary_line_img)
            cv2.imshow('created binary image',img_binaries)
            curr_recorded_binaries = np.copy(img_binaries)
            curr_recorded_binaries=cv2.cvtColor(curr_recorded_binaries,cv2.COLOR_GRAY2RGB)
            #out_video.write(curr_recorded_binaries)
       
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break



if show_img:
    cv2.destroyAllWindows()
#out_video.release()


