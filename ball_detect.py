# Import libraries
import cv2
import os
import numpy as np
import imutils
from scipy.spatial import distance as dist


picked_ball_position = None


def distance(x, y): return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)


def is_ROI(center, pos, tolerance=100):
    """
    is tracked in roi?
    """
    x, y = pos
    if (center[0]-tolerance <= x) and (center[1]-tolerance <= y):
        if (center[0]+tolerance >= x) and (center[1]+tolerance >= y):
            return True
    return False


def set_ball_position(event, x, y, flags, param):
    global picked_ball_position
    if event == cv2.EVENT_LBUTTONDBLCLK:
        picked_ball_position = (x, y)
        print(picked_ball_position)
        cv2.circle(image, picked_ball_position, 2, (255, 255, 0), 2)


def find_min_distance_to_last_ball_position(candidate_points, last_ball_position):
    min_distance = 9999
    min_distance_index = 0
    for i, candidate_pos in enumerate(candidate_points):
        dist = np.sqrt((candidate_pos[0] - last_ball_position[0])**2 +
                       (candidate_pos[1] - last_ball_position[1]))
        if dist < min_distance:
            min_distance = dist
            min_distance_index = i
    return candidate_points[min_distance_index]


def find_min_distance_index(persistence_list, roi):
    """
    find index of min euclidian distance of roi objects
    """
    min_dist_index = 0
    min_dist = 999999
    x, y, _, _ = roi
    for i, obj in enumerate(persistence_list):
        if obj.distance(*(x, y)) <= min_dist:
            min_dist = obj.distance(*(x, y))
            min_dist_index = i
    return min_dist_index, min_dist


def find_max_persistence(persistence_list):
    """
    locate max persistence object
    """
    max_per = 0
    for obj in persistence_list:
        if obj.persistence_count > max_per:
            max_per = obj.persistence_count
    return max_per


def update_persistence(persistence_list, min_dist_index, roi):
    """
    update roi parameters of persistence
    """
    x, y, w, h = roi
    persistence_list[min_dist_index].persistence_count += 1
    persistence_list[min_dist_index].update_speed(*(x, y))
    persistence_list[min_dist_index].x = x
    persistence_list[min_dist_index].y = y
    persistence_list[min_dist_index].w = w
    persistence_list[min_dist_index].h = h


class Tracked:
    def __init__(self, x, y, w, h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h
        self.vel_x = 0
        self.vel_y = 0
        self.persistence_count = 0
        self.frames = 1

    def update_speed(self, x, y):
        self.vel_x = x-self.x
        self.vel_y = y-self.y

    def distance(self, x1, y1):
        return np.sqrt((self.x-x1)**2 + (self.y-y1)**2)

    def roi(self):
        return (self.x, self.y, self.w, self.h)


test_vide_file = 'FIFA 19  Chelsea vs Tottenham Hotspur  Premier League Gameplay.mp4'
# Reading the video
cap = cv2.VideoCapture(test_vide_file)
count = 0

idx = 0
tracker = cv2.TrackerCSRT_create()
tracker_bb = False
tracker_roi = None
tolerance = 120
kernel = np.ones((13, 13), np.uint8)
persistent = []
ball_positions, ball_speeds = [], [0]
acc_pos = (0, 0)

while True:
    idx += 1
    rate, image = cap.read()
    if idx < 120:
        continue
    else:
        break

image = imutils.resize(image, width=700)
while True:
    cv2.imshow('Match Detection', image)
    cv2.setMouseCallback('Match Detection', set_ball_position)
    if picked_ball_position is None:
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            break
        elif k == ord('a'):
            print(picked_ball_position)
            cv2.circle(image, picked_ball_position, 2, (255, 255, 0), 2)
    else:
        break

while True:
    idx += 1
    rate, image = cap.read()
    image = imutils.resize(image, width=700)
    (ih, iw) = image.shape[:2]
    cv2.imshow('Match Detection', image)
    # cv2.setMouseCallback('Match Detection', set_ball_position)
    new_ball_pos = picked_ball_position
    center = (int(iw/2), int(ih/2))

    # if idx <= 100:
    #     continue
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    sensitivity = 85
    lower_white = np.array([0, 0, 255-sensitivity])
    upper_white = np.array([255, sensitivity, 255])
    mask = cv2.inRange(hsv, lower_white, upper_white)
    # Do masking
    res = cv2.bitwise_and(image, image, mask=mask)
    res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
    # res = cv2.Canny(res, 150, 200)
    res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

    contours, hierarchy = cv2.findContours(
        res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = (x, y, w, h)
        if is_ROI(center, (x, y), tolerance=tolerance):
            if (h <= 12) and (w <= 12):
                bb = image[x:x+w, y:y+h, :]
                mean_color = np.mean(np.mean(bb, axis=0), axis=0)
                (x, y), radius = cv2.minEnclosingCircle(contour)
                if radius > 2:
                    continue
                area = np.pi*(radius**2)
                guessed_area = cv2.contourArea(contour)
                if guessed_area <= 7 and guessed_area >= 3:
                    center_x = (int(x), int(y))
                    candidates.append(center_x)
                    radius = int(radius)
                    min_distance_index, min_distance = find_min_distance_index(
                        persistent, roi)
                    if persistent == [] or min_distance > 4:
                        persistent.append(Tracked(*roi))
                    else:
                        update_persistence(persistent, min_distance_index, roi)
                    # we have caught the ball candidate, we track it now
                    cv2.circle(image, center_x, radius, (0, 255, 0), 2)

    print(candidates)
    if candidates != []:
        new_ball_pos = find_min_distance_to_last_ball_position(
            candidates, new_ball_pos)
        if ball_positions:
            ball_speeds.append(distance(new_ball_pos, ball_positions[-1]))
        ball_positions.append(new_ball_pos)
        cv2.circle(image, new_ball_pos, 2, (100, 0, 120), 2)
    # to_delete = []
    # ball = False
    # max_per = find_max_persistence(persistent)
    # for i in range(len(persistent)):
    #     persistent[i].frames += 1
    #     if persistent[i].persistence_count > 1:
    #         if (persistent[i].persistence_count == max_per) and not ball:
    #             x, y, w, h = persistent[i].roi()
    #             cv2.rectangle(image, (x, y), (x+w, y+h),
    #                           (0, 0, 255), 3)
    #             ball = True
    #             ball_positions.append([x, y])
    #             if len(ball_positions) != 1:
    #                 p_ball = ball_positions[-2]
    #                 ball_speeds.append((p_ball[0]-x, p_ball[1]-y))
    #                 acc_pos = (acc_pos[0]+p_ball[0]-x, acc_pos[1]+p_ball[1]-y)
    #     if ((persistent[i].persistence_count/persistent[i].frames) < 0.4):
    #         if persistent[i].frames > 1:
    #             to_delete.append(i)

    # for index in sorted(to_delete, reverse=True):
    #     del persistent[index]
    if ball_positions != [] and ball_speeds != []:
        field_half = 'left' if acc_pos[0] < 0 else 'right'
        info = {
            "ball pos": ball_positions[-1],
            "ball speed": np.around(ball_speeds[-1], decimals=2),
            "half": field_half
        }
        # loop over the info tuples and draw them on our frame
        for i, key in enumerate(info):
            text = f"{key}: {info[key]}"
            cv2.putText(image, text, (10, ih - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
    cv2.rectangle(
        image, (center[0]-tolerance, center[1]-tolerance),
        (center[0]+tolerance, center[1]+tolerance),
        (255, 0, 255), 3)
    cv2.imshow('Match Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
