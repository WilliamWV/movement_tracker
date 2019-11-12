import argparse
import cv2
import imutils
import numpy as np

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
# Luminance Value used when converting image to HSV and normalizing V
# to avoid shaddow detection
HSV_default_value = 150

### KEYS ###
pause_key = 'p'
next_key = 'd'
previous_key = 'a'

frames = []

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required = True, help='Video file to analize')
    args = parser.parse_args()
    video = cv2.VideoCapture(args.video)
    return video


def frame_normalization(frame):

    frame = imutils.resize(frame, width=default_width)

    prev_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    thresh_t = cv2.threshold(prev_gray, 20, 1, cv2.THRESH_BINARY)[1]

    frame[:,:,0] = np.multiply(frame[:,:,0], thresh_t)
    frame[:,:,1] = np.multiply(frame[:,:,1], thresh_t)
    frame[:,:,2] = np.multiply(frame[:,:,2], thresh_t)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    hsv[:,:,2] = np.multiply(HSV_default_value, thresh_t)
    hsv = cv2.GaussianBlur(hsv, (7, 7), 0)
    cv2.imshow("HSV", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

    gray_frame = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    gray_frame = cv2.bilateralFilter(gray_frame, neighborhood_size, \
        sigma_color, sigma_space)

    return frame, gray_frame


def process_frame(frame, base_frame):
    frame, gray_frame = frame_normalization(frame)

    frame_diff = cv2.absdiff(base_frame, gray_frame)
    ret, binary_frame = cv2.threshold (frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)

    binary_frame = cv2.dilate(binary_frame, None, iterations = 1)
    contours = cv2.findContours(binary_frame.copy(), cv2.RETR_EXTERNAL,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow('Camera', frame)
    cv2.imshow('Movements', binary_frame)
    cv2.imshow('Frame diff', frame_diff)
    cv2.imshow('Gray frame', gray_frame)


def process_pause(video, base_frame, current_frame_index):
    # Pause
    global frames
    paused = True
    frame_index = current_frame_index
    while paused:
        key = cv2.waitKey(0)
        key = chr(key)
        if key == previous_key and frame_index > 0:
            process_frame(frames[frame_index-1], base_frame)
            frame_index-=1
        elif key == next_key:
            if frame_index < current_frame_index:
                process_frame(frames[frame_index+1], base_frame)
                frame_index+=1
            else:
                valid_frame, frame = video.read()
                if valid_frame and frame is not None:
                    frames.append(frame.copy())
                    process_frame(frame, base_frame)
                    frame_index+=1
        elif key == pause_key:
            paused = False

    return frame_index


def track_movements(video):

    base_frame = None
    current_frame_index = 0

    while True:
        valid_frame, frame = video.read()
        if not valid_frame or frame is None:
            # If the frame is not valid then the video ended
            break

        frames.append(frame.copy())

        if base_frame is not None:
            process_frame(frame, base_frame)
        else:
            base_frame = frame_normalization(frame)[1]
            continue

        # Wait for 1 ms, this is to detect if the user type something
        key = cv2.waitKey(1)
        if key != -1:
            key = chr(key).lower()
            if key == pause_key:
                frame_index = process_pause(video, base_frame, current_frame_index)
                if frame_index < current_frame_index:
                    while frame_index < current_frame_index:
                        frame_index+=1
                        process_frame(frames[frame_index], base_frame)
                        cv2.waitKey(1)
                else:
                    current_frame_index+=frame_index

        current_frame_index +=1



if __name__ == '__main__':
    video = parse_arguments()
    track_movements(video)
