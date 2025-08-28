# MIT License
# Copyright (c) 2019-2022 JetsonHacks

# A simple code snippet
# Using two  CSI cameras (such as the Raspberry Pi Version 2) connected to a
# NVIDIA Jetson Nano Developer Kit with two CSI ports (Jetson Nano, Jetson Xavier NX) via OpenCV
# Drivers for the camera and OpenCV are included in the base image in JetPack 4.3+

# This script will open a window and place the camera stream from each camera in a window
# arranged horizontally.
# The camera streams are each read in their own thread, as when done sequentially there
# is a noticeable lag

import cv2
import threading
import numpy as np


class CSI_Camera:

    def __init__(self):
        # Initialize instance variables
        # OpenCV video capture element
        self.video_capture = None
        # The last captured image from the camera
        self.frame = None
        self.grabbed = False
        # The thread where the video capture runs
        self.read_thread = None
        self.read_lock = threading.Lock()
        self.running = False

    def open(self, gstreamer_pipeline_string):
        try:
            self.video_capture = cv2.VideoCapture(
                gstreamer_pipeline_string, cv2.CAP_GSTREAMER
            )
            # Grab the first frame to start the video capturing
            self.grabbed, self.frame = self.video_capture.read()

        except RuntimeError:
            self.video_capture = None
            print("Unable to open camera")
            print("Pipeline: " + gstreamer_pipeline_string)


    def start(self):
        if self.running:
            print('Video capturing is already running')
            return None
        # create a thread to read the camera image
        if self.video_capture != None:
            self.running = True
            self.read_thread = threading.Thread(target=self.updateCamera)
            self.read_thread.start()
        return self

    def stop(self):
        self.running = False
        # Kill the thread
        self.read_thread.join()
        self.read_thread = None

    def updateCamera(self):
        # This is the thread to read images from the camera
        while self.running:
            try:
                grabbed, frame = self.video_capture.read()
                with self.read_lock:
                    self.grabbed = grabbed
                    self.frame = frame
            except RuntimeError:
                print("Could not read image from camera")
        # FIX ME - stop and cleanup thread
        # Something bad happened

    def read(self):
        with self.read_lock:
            frame = self.frame.copy()
            grabbed = self.grabbed
        return grabbed, frame

    def release(self):
        if self.video_capture != None:
            self.video_capture.release()
            self.video_capture = None
        # Now kill the thread
        if self.read_thread != None:
            self.read_thread.join()


""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080
"""

camera_width = 1920
camera_height = 1080

interocular_distance = 100 # (mm)
horizontal_fov = 160 # (degrees)
pixel_angular_width = np.deg2rad(horizontal_fov) / camera_width # (radians)

center_x = camera_width // 2
center_y = camera_height // 2

window_width = 80 # (pixels)
window_height = 80 # (pixels)

scope_width = 800
scope_height = 800

scope_xlim = (center_x - scope_width//2, center_x + scope_width//2)
scope_ylim = (center_y - scope_height//2, center_y + scope_height//2)

epipolar_lines = scope_height // window_height
columns = scope_width // window_width

# display stuff only
screen_width = 1920
screen_height = 1080
screen_ratio = (screen_width//camera_width, screen_height//camera_height)

def read_pixel(frame, x, y):   
    return frame[y, x]

def read_window(frame, x_left, y_top, window_dims=(window_width, window_height)):
    window = np.zeros(window_dims)
    for x in range(x_left, x_left + window_dims[0]):
        for y in range(y_top - window_dims[1], y_top):
            window[x][y] = read_pixel(frame, x, y)
    return window

def normalise(window):
    # how to normalise an individual window
    w, h = window.shape
    sumsquares = 0
    for wi in range(w):
        for hi in range(h):
            sumsquares += window[wi, hi]**2
    return window / sumsquares

def get_SSD(template_window, comparison_window):
    # where left_window and right_window are NxM arrays of pixels,
    # and N is window_width and M is window_height.

    # the actual cardinality of the window is not important. This function is symmetric.

    w, h = template_window.shape
    if (w, h) != comparison_window.shape:
        raise ValueError("Left and right windows must have the same shape")
    
    # First normalise (doing this double-counts a lot of pixels compared to normalising the whole image in one go, but it might make it easier to pick out patterns??):
    
    template = normalise(template_window)
    comparison = normalise(comparison_window)
    
    difference = template - comparison

    ssd = 0
    for wi in range(w):
        for hi in range(h):
            ssd += difference[wi, hi]**2

    return ssd

def scan_epipolar_line(left_image, right_image, template_top_left, window_dims=(window_width, window_height), scope_xlim=scope_xlim, scope_ylim=scope_ylim, yshift left_first=True):
    # window_top_left is [x, y] of the top left corner of the window providing the reference
    
    if left_first:
        scan_xlim = (scope_xlim[0], template_top_left[0])
        image_with_window = left_image
        image_to_scan = right_image
    else: # (i.e. right first)
        scan_xlim = (template_top_left[0], scope_xlim[1] - window_dims[0])
        image_with_window = right_image
        image_to_scan = left_image

    template = read_window(image_with_window, template_top_left[0], template_top_left[1], window_dims=window_dims)
    ssd_array = []
    x_range = range(scan_xlim[0], scan_xlim[1])
    #x_delta_array = x_range - scan_xlim[0]
    #x_delta_array = np.arange(len(x_range))
    for x in x_range:
        comparison = read_window(image_to_scan, x, template_top_left[1], window_dims=window_dims)
        ssd = get_SSD(template, comparison)
        ssd_array.append(ssd)
    return ssd_array, x_range

def get_distance(ssd_array, x_range, interocular_distance, pixel_angular_width):
    ssd_average = np.mean(ssd_array)
    ssd_stdev = np.std(ssd_array)
    ssd_threshold = ssd_average + 3*ssd_stdev # adding a "noise floor"
    ssd_min = min(ssd_array)
    if ssd_min < ssd_threshold:
        return 0, 0
    else:
        x_match = x_range[ssd_array.index(ssd_min)]
        x_delta_array = np.arange(len(x_range))
        x_delta = x_delta_array[x_range.index(x_match)] # this seems silly
        theta = x_delta * pixel_angular_width
        distance = interocular_distance / (2*np.tan(theta/2))
        return distance, x_match
    
def get_distance_matrix(left_image, right_image, interocular_distance, pixel_angular_width, window_dims=(window_width, window_height), scope_xlim=scope_xlim, scope_ylim=scope_ylim, yshift=0, left_first=True):
    distance_matrix = np.zeros((epipolar_lines, columns, 5))
    template_top_left = (scope_xlim[0], scope_ylim[0]+window_dims[1])
    for line in range(epipolar_lines):
        for column in range(columns):
            ssd_array, x_range = scan_epipolar_line(left_image, right_image, template_top_left, window_dims=window_dims, scope_xlim=scope_xlim, scope_ylim=scope_ylim, left_first=left_first)
            distance, x_match = get_distance(ssd_array, x_range, interocular_distance, pixel_angular_width)
            template_x = template_top_left[0] + window_dims[0] // 2
            template_y = template_top_left[1] - window_dims[1] // 2
            comparison_x = x_match + window_dims[0] // 2
            comparison_y = template_top_left[1] - window_dims[1] // 2 # it's the same, because of the epipolar lines constraint!
            distance_matrix[line][column] = [template_x, template_y, comparison_x, comparison_y, distance]

            template_top_left[0] += window_dims[0]
        template_top_left[1] += window_dims[1]
    
    return distance_matrix

# TODO: 
# - test that scope boxes are still plotting in the middle
# - work out how to disregard insignificant SSD minima (untested)
# - implement slight vertical shifts +/- epipolar lines to account for errors in the camera alignment
# - implement a way to check SSD in different colour channels independently, and then combine


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=camera_width,
    capture_height=camera_height,
    display_width=screen_width,
    display_height=screen_height,
    framerate=2, # frames per second
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )



def run_cameras():
    window_title = "Dual CSI Cameras"
    left_camera = CSI_Camera()
    left_camera.open(
        gstreamer_pipeline(
            sensor_id=0,
            capture_width=camera_width,
            capture_height=camera_height,
            flip_method=0,
            display_width=screen_width,
            display_height=screen_height,
        )
    )
    left_camera.start()

    right_camera = CSI_Camera()
    right_camera.open(
        gstreamer_pipeline(
            sensor_id=1,
            capture_width=camera_width,
            capture_height=camera_height,
            flip_method=0,
            display_width=screen_width,
            display_height=screen_height,
        )
    )
    right_camera.start()

    if left_camera.video_capture.isOpened() and right_camera.video_capture.isOpened():

        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        try:
            while True:
                _, left_image = left_camera.read()
                _, right_image = right_camera.read()

                left_image = cv2.cvtColor(left_image, cv2.COLOR_BGR2GRAY)
                right_image = cv2.cvtColor(right_image, cv2.COLOR_BGR2GRAY)

                # Use numpy to place images next to each other
                camera_images = np.hstack((left_image, right_image)) 
                
                # Draw scope boxes
                cv2.rectangle(left_image, (scope_xlim[0]*screen_ratio[0], scope_ylim[0]*screen_ratio[1]), (scope_xlim[1]*screen_ratio[0], scope_ylim[1]*screen_ratio[1]), (255, 0, 0), 8)
                cv2.rectangle(right_image, (scope_xlim[0]*screen_ratio[0], scope_ylim[0]*screen_ratio[1]), (scope_xlim[1]*screen_ratio[0], scope_ylim[1]*screen_ratio[1]), (255, 0, 0), 8)

                # Get distance matrix
                distance_matrix = get_distance_matrix(left_image, right_image, interocular_distance, pixel_angular_width, window_dims=(window_width, window_height), scope_xlim=scope_xlim, scope_ylim=scope_ylim, left_first=True)

                # Print distance matrix
                print('\n\n\n')
                print(distance_matrix)

                # Check to see if the user closed the window
                # Under GTK+ (Jetson Default), WND_PROP_VISIBLE does not work correctly. Under Qt it does
                # GTK - Substitute WND_PROP_AUTOSIZE to detect if window has been closed by user
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, camera_images)
                else:
                    break

                # This also acts as
                keyCode = cv2.waitKey(30) & 0xFF
                # Stop the program on the ESC key
                if keyCode == 27:
                    break
        finally:

            left_camera.stop()
            left_camera.release()
            right_camera.stop()
            right_camera.release()
        cv2.destroyAllWindows()
    else:
        print("Error: Unable to open both cameras")
        left_camera.stop()
        left_camera.release()
        right_camera.stop()
        right_camera.release()



if __name__ == "__main__":
    run_cameras()
