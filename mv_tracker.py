import argparse
import cv2
import imutils
import numpy as np
from rect_util import *

# Value used when thresholding the difference from the current frame to the
# base frame
diff_threshold = 20
# This value is used to resize the image to a default resolution
default_width = 480
# Diameter of the neighborhood of a pixel used on bilateral filter
neighborhood_size = 7
# Sigma color of bilateral filter ( A larger value of the parameter means
# that farther colors within the pixel neighborhood)
sigma_color = 50
# Sigma space of bilateral filter (A larger value of the parameter means
# that farther pixels will influence each other as long as their colors
# are close enough)
sigma_space = 50
# Minimum area of a contour that is highlighted
min_area = 1000


### KEYS ###
pause_key = 'p'
next_key = 'd'
previous_key = 'a'
quit_key = 'q'

# Background subtractor, used as an auxiliar tool to remove shadows
backSub = cv2.createBackgroundSubtractorKNN()
backSub.setDetectShadows(True)


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required = True, help='Video file to analize')
    args = parser.parse_args()
    video = cv2.VideoCapture(args.video)
    return video


def frame_normalization(frame):
    # Resizes all frames to a common width so that the detected areas can use
    # the same threshold
    frame = imutils.resize(frame, width=default_width)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return frame, gray_frame


# Receives the bounding boxes of movement areas and the foreground mask
# obtained with the background subtractor and filter the rectangles to
# not select those that represent shadows, once the background subtractor
# is far less sensitive to these regions

def non_shadow_rects(rectangles, fgMask):

    filtered = []

    for rectangle in rectangles:
        (x, y, w, h) = rectangle
        cropped = fgMask[y:y+h, x:x+w]
        white = cv2.countNonZero(cropped)
        # Only remain those rectangles with more than 10% of its area covered
        # by white pixels
        if white > 0.1 * w * h :
            filtered.append(rectangle)

    return filtered


def process_frame(frame, base_frame):
    frame, gray_frame = frame_normalization(frame)

    # Foreground mask calculation
    fgMask = backSub.apply(frame)
    ret, fgMask = cv2.threshold (fgMask, 200, 255, cv2.THRESH_BINARY)
    fgMask = cv2.erode(fgMask, None, iterations = 1)
    fgMask = cv2.dilate(fgMask, None, iterations = 1)

    # Movement detection
    frame_diff = cv2.absdiff(base_frame, gray_frame)
    ret, binary_frame = cv2.threshold (frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)
    binary_frame = cv2.erode(binary_frame, None, iterations = 1)
    binary_frame = cv2.dilate(binary_frame, None, iterations = 2)

    # Get contours
    contours = cv2.findContours(binary_frame.copy(), cv2.RETR_EXTERNAL,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Highlighted rectangles handling
    rectangles = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        rectangles.append((x, y, w, h))

    rectangles = non_shadow_rects(rectangles, fgMask)
    rectangles = adjust_rectangles(rectangles)

    for rectangle in rectangles:
        (x, y, w, h) = rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    # Images displayer
    cv2.imshow('Fgmask', fgMask)
    cv2.imshow('Camera', frame)
    cv2.imshow('Binary frame', binary_frame)
    cv2.imshow('Frame diff', frame_diff)
    cv2.imshow('Gray frame', gray_frame)


def process_pause(video, base_frame, current_frame_index):
    paused = True
    # It is required to decrement this value once the function
    # VideoCapture.set() considers the next frame
    frame_index = current_frame_index - 1
    while paused:
        key = cv2.waitKey(0)
        key = chr(key)
        if key == previous_key and frame_index > 0:
            video.set(1, frame_index - 1)
            ans,previous_frame = video.read()
            process_frame(previous_frame, base_frame)
            frame_index-=1
        elif key == next_key:
            video.set(1, frame_index + 1)
            valid_frame,next_frame = video.read()
            if valid_frame and next_frame is not None:
                process_frame(next_frame, base_frame)
                frame_index+=1
        elif key == pause_key:
            paused = False
        elif key == quit_key:
            exit()

    # Increment this value to compensate de decrement done before
    return frame_index + 1


def track_movements(video):
    base_frame = None
    current_frame_index = 0

    while True:
        valid_frame, frame = video.read()

        if not valid_frame or frame is None:
            # If the frame is not valid then the video ended
            break

        if base_frame is not None:
            process_frame(frame, base_frame)
        else:
            base_frame = frame_normalization(frame)[1]
            continue

        key = cv2.waitKey(1)
        if key != -1:
            key = chr(key).lower()
            if key == pause_key:
                frame_index = process_pause(video, base_frame, current_frame_index)
                if frame_index < current_frame_index:
                    while frame_index < current_frame_index:
                        frame_index+=1
                        ans, curr_frame = video.read()
                        process_frame(curr_frame, base_frame)
                        cv2.waitKey(1)
                else:
                    current_frame_index+=frame_index

        current_frame_index +=1



if __name__ == '__main__':
    video = parse_arguments()
    track_movements(video)
