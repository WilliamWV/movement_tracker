# USAGE
# python motion_detector.py
# python motion_detector.py --video videos/example_01.mp4

import argparse
import cv2

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required = True, help='Video file to analize')
    args = parser.parse_args()
    video = cv2.VideoCapture(args.video)
    return video


def track_movements(video):

    # Value used when thresholding the difference from the current frame to the
    # base frame
    diff_threshold = 50

    base_frame = None

    while True:
        valid_frame, frame = video.read()

        if not valid_frame or frame is None:
            # If the frame is not valid then the video ended
            break

        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

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
        # Wait for 1 ms, this is to detect if the user type something
        key = cv2.waitKey(1)
        if key != -1:
            print("Typed = " + str(key))


if __name__ == '__main__':
    video = parse_arguments()
    track_movements(video)
