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
HSV_default_value = 100
# Minimum distance to two detected bounding boxes to be considered differents
min_rec_dist = 10

### KEYS ###
pause_key = 'p'
next_key = 'd'
previous_key = 'a'
quit_key = 'q'


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-v', '--video', required = True, help='Video file to analize')
    args = parser.parse_args()
    video = cv2.VideoCapture(args.video)
    return video


def frame_normalization(frame):

    frame = imutils.resize(frame, width=default_width)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #thresh_t = cv2.threshold(prev_gray, 20, 1, cv2.THRESH_BINARY)[1]

    #frame[:,:,0] = np.multiply(frame[:,:,0], thresh_t)
    #frame[:,:,1] = np.multiply(frame[:,:,1], thresh_t)
    #frame[:,:,2] = np.multiply(frame[:,:,2], thresh_t)

    #hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)


    #hsv[:,:,2] = np.multiply(HSV_default_value, thresh_t)
    #hsv = cv2.GaussianBlur(hsv, (7, 7), 0)
    #cv2.imshow("HSV", cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR))

    #gray_frame = cv2.cvtColor(cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR), cv2.COLOR_BGR2GRAY)
    #gray_frame = cv2.bilateralFilter(gray_frame, neighborhood_size, \
    #    sigma_color, sigma_space)

    return frame, gray_frame


def superposition(rec1, rec2):
    (x1, y1, w1, h1) = rec1
    (x2, y2, w2, h2) = rec2

    return (x1 < x2 and x1 + w1 > x2 and y1 < y2 and y1 + h1 > y2) or \
        (x1 < x2 and x1 + w1 > x2 and y1 > y2 and y1 < y2 + h2) or \
        (x1 > x2 and x1 < x2 + w2 and y1 < y2 and y1 + h1 > y2) or \
        (x1 > x2 and x1 < x2 + w2 and y1 > y2 and y1 < y2 + h2)


def euclidian_distance(p1, p2):
    (x1, y1) = p1
    (x2, y2) = p2
    return sqrt((x1 - x2)**2 + (y1 - y2)**2)

def rec_distance(rec1, rec2):
    (x1, y1, w1, h1) = rec1
    (x2, y2, w2, h2) = rec2

    if superposition(rec1, rec2):
        return 0
    else:
        c1 = (x1 + w1/2, y1 + h1/2)
        c2 = (x2 + w2/2, y2 + h2/2)
        dx = abs(c1[0] - c2[0]) - w1/2 - w2/2
        dy = abs(c1[1] - c2[1]) - h1/2 - h2/2

        return max(dx, dy)

def merge_rects(rec1, rec2):
    (x1, y1, w1, h1) = rec1
    (x2, y2, w2, h2) = rec2
    x = min(x1, x2)
    y = min(y1, y2)
    w = max(x1 + w1, x2 + w2) - x
    h = max(y1 + h1, y2 + h2) - y
    return (x, y, w, h)


# Avoid superpositions and rectangles too close
# This code will check with all pairs of rectangles if they are too close
# When a pair that is too close is found the rectangles are merged and
# a recursive call is made with a reduced list of rectangles removing the previous
# pair and adding the merged result

def adjust_rectangles(rectangles):
    final_rects = []
    modifieds = []
    for i in range(len(rectangles)):
        for j in range(i+1, len(rectangles)):
            r0 = rectangles[i]
            r1 = rectangles[j]
            if rec_distance(r0, r1) < min_rec_dist:
                rec = merge_rects(r0, r1)
                rectangles.remove(r0)
                rectangles.remove(r1)
                rectangles.append(rec)
                return adjust_rectangles(rectangles)

    return rectangles


def process_frame(frame, base_frame):
    frame, gray_frame = frame_normalization(frame)

    frame_diff = cv2.absdiff(base_frame, gray_frame)
    ret, binary_frame = cv2.threshold (frame_diff, diff_threshold, 255, cv2.THRESH_BINARY)

    binary_frame = cv2.erode(binary_frame, None, iterations = 1)
    binary_frame = cv2.dilate(binary_frame, None, iterations = 1)
    contours = cv2.findContours(binary_frame.copy(), cv2.RETR_EXTERNAL,\
        cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    rectangles = []
    for c in contours:
        if cv2.contourArea(c) < min_area:
            continue

        (x, y, w, h) = cv2.boundingRect(c)
        rectangles.append((x, y, w, h))

    rectangles = adjust_rectangles(rectangles)

    for rectangle in rectangles:
        (x, y, w, h) = rectangle
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)

    cv2.imshow('Camera', frame)
    cv2.imshow('Binary frame', binary_frame)
    cv2.imshow('Frame diff', frame_diff)
    cv2.imshow('Gray frame', gray_frame)


def process_pause(video, base_frame, current_frame_index):
    # Pause
    paused = True
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

    return frame_index + 1


def track_movements(video):
    base_frame = None
    current_frame_index = 0

    while True:
        valid_frame, frame = video.read()
        cv2.waitKey(0)

        if not valid_frame or frame is None:
            # If the frame is not valid then the video ended
            break


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
                        ans, curr_frame = video.read()
                        process_frame(curr_frame, base_frame)
                        cv2.waitKey(1)
                else:
                    current_frame_index+=frame_index

        current_frame_index +=1



if __name__ == '__main__':
    video = parse_arguments()
    track_movements(video)
