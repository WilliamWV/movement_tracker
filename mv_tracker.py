# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

import argparse
import cv2
import imutils

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


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required = True, help='Video file to analize')
    args = parser.parse_args()
    video = cv2.VideoCapture(args.video)
    return video


def track_movements(video):

    base_frame = None

    while True:
        valid_frame, frame = video.read()

        if not valid_frame or frame is None:
            # If the frame is not valid then the video ended
            break

        frame = imutils.resize(frame, width=default_width)
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_frame = cv2.bilateralFilter(gray_frame, neighborhood_size, \
            sigma_color, sigma_space)

        if base_frame is None:
            # If the baseFrame was not defined yet, define it.
            base_frame = gray_frame
            continue

        frame_diff = cv2.absdiff(base_frame, gray_frame)
        ret, binary_frame = cv2.threshold (frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)

        binary_frame = cv2.dilate(binary_frame, None, iterations = 1)

        cv2.imshow('Camera', frame)
        cv2.imshow('Movements', binary_frame)
        cv2.imshow('Frame diff', frame_diff)
        cv2.imshow('Gray frame', gray_frame)
        # Wait for 1 ms, this is to detect if the user type something
        key = cv2.waitKey(1)
        if key != -1:
            if chr(key) == 'p' or chr(key) == 'P':
                # Pause
                unpause_key = cv2.waitKey(0)
                while chr(unpause_key) != 'p' and chr(unpause_key) != 'P':
                    unpause_key = cv2.waitKey(0)



if __name__ == '__main__':
    video = parse_arguments()
    track_movements(video)
