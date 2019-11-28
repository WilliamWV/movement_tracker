import argparse
import cv2
import imutils
import numpy as np
import time
from rect_util import *

# Value used when thresholding the difference from the current frame to the
# base frame
diff_threshold = 30
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
# Limit to frame rate
fps_limit = 30

# Number of frames that a chunk may not contain movement to be considered
# part of the base frame
no_move_interval = 30
# Number of chunks in a line of the image
chunks_x = 16
# Number of chunks in a column of the image
chunks_y = 16

### KEYS ###
pause_key = 'p'
quit_key = 'q'

# Background subtractor, used as an auxiliar tool to remove shadows
backSub = cv2.createBackgroundSubtractorKNN()
backSub.setDetectShadows(True)

def frame_normalized_size(width, height):
    change_rate = default_width / width
    new_height = change_rate * height
    return default_width, int(new_height)

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required = True, help='Video file to analize')
    parser.add_argument('-s', '--save', required = False, type=bool, help='Boolean indicating if the result may be saved')
    args = parser.parse_args()
    video = cv2.VideoCapture(args.video)
    out = None

    if args.save is not None and args.save == True:
        base_name = args.video.replace('.mp4', '')
        out_file = base_name + '_' + time.asctime().replace(' ', '-').replace(':', '_') + '.avi'
        width, height = frame_normalized_size(int(video.get(3)), int(video.get(4)))
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(out_file, fourcc, 20.0, (width, height))

    return video, out


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
    return frame


def process_pause():
    paused = True
    while paused:
        key = cv2.waitKey(0)
        key = chr(key)
        if key == pause_key:
            paused = False
        elif key == quit_key:
            exit()


def update_base_frame(base_frame, old_frame, current_frame):
    h, w = base_frame.shape
    chunk_w = int(w / chunks_x)
    chunk_h = int(h / chunks_y)
    for i in range(chunks_y):
        for j in range(chunks_x):
            x_beg = j * chunk_w
            y_beg = i * chunk_h

            # Includes possible remainder pixels in the last chunk
            if i == chunks_y - 1:
                chunk_h = chunk_h + h % chunks_y
            if j == chunks_x - 1:
                chunk_w = chunk_w + w % chunks_x

            old_frame_chunk = old_frame[y_beg:y_beg+chunk_h, x_beg:x_beg+chunk_w]
            current_frame_chunk = current_frame[y_beg:y_beg+chunk_h, x_beg:x_beg+chunk_w]
            chunks_diff = cv2.absdiff(old_frame_chunk, current_frame_chunk)
            ret, binary_chunk = cv2.threshold (chunks_diff, diff_threshold, 255, cv2.THRESH_BINARY)
            white = cv2.countNonZero(binary_chunk)

            if white < 0.05 * chunk_w * chunk_h:
                # if less than 5% of the pixels changed update this chunk
                base_frame[y_beg:y_beg+chunk_h, x_beg:x_beg+chunk_w] = current_frame_chunk

    return base_frame


def track_movements(video, out):
    base_frame = None
    last_frames = []
    current_frame_index = 0

    while True:
        beginning = time.time()
        valid_frame, frame = video.read()

        if not valid_frame or frame is None:
            # If the frame is not valid then the video ended
            break

        normalized_frame = frame_normalization(frame)[1]

        if base_frame is not None:
            processed_frame = process_frame(frame, base_frame)
            if out is not None:
                out.write(processed_frame)
        else:
            base_frame = normalized_frame
            continue

        last_frames.append(normalized_frame)
        if current_frame_index >= no_move_interval:
            base_frame = update_base_frame(base_frame, last_frames[0], normalized_frame)
            last_frames.remove(last_frames[0])

        key = cv2.waitKey(1)
        if key != -1:
            key = chr(key).lower()
            if key == pause_key:
                frame_index = process_pause()


        current_frame_index +=1
        if time.time() - beginning < 1.0/fps_limit:
            time.sleep(1.0/fps_limit - (time.time() - beginning))


if __name__ == '__main__':
    video, out = parse_arguments()
    track_movements(video, out)
    video.release()
    if out is not None:
        out.release()
