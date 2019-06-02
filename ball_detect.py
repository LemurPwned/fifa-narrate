# Import libraries
import cv2
import os
import numpy as np
import imutils
from scipy.spatial import distance as dist


class BallDetector:
    """
    Object responsible for football tracking
    Registers the past and present ball positions
    Determines the params for tracking
    """

    def __init__(self):
        self.picked_ball_position = None
        self.count = 0

        self.idx = 0
        self.tolerance = 120
        self.kernel = np.ones((13, 13), np.uint8)
        self.persistent = []
        self.ball_positions, self.ball_speeds = [], [0]
        self.acc_pos = (0, 0)

    @staticmethod
    def distance(x, y): return np.sqrt((x[0]-y[0])**2 + (x[1]-y[1])**2)

    def is_ROI(self, center, pos, tolerance=100):
        """
        Verifies if a given point is in ROI 
        ROI is given by ROI center and tolerance respective to that center
        Parameters
        ----------
        center: tuple(int, int)
            center of ROI
        pos: tuple(int, int)
            position which is verified to be in ROI
        tolerance: int
            tolerance of ROI window (respective to center)

        Returns
        ----------
        bool
            true if pos is in ROI
        """
        x, y = pos
        if (center[0]-tolerance <= x) and (center[1]-tolerance <= y):
            if (center[0]+tolerance >= x) and (center[1]+tolerance >= y):
                return True
        return False

    def set_ball_position(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDBLCLK:
            self.picked_ball_position = (x, y)
            print(self.picked_ball_position)
            cv2.circle(self.image_draw, self.picked_ball_position,
                       2, (255, 255, 0), 2)

    def find_min_distance_to_last_ball_position(self, candidate_points, last_ball_position):
        """
        Calculates the distance relative to the last registered ball position
        The distance is calculated for each of the candidates in the set
        Parameters
        -------
        candidate_points: list(tuple(int, int))
            candidate ball positions
        last_ball_position: tuple(int, int)
            last registered ball position
        Returns
        -------
        tuple(int, int) 
            calculated ball position
        """
        min_distance = 9999
        min_distance_index = 0
        for i, candidate_pos in enumerate(candidate_points):
            dist = np.sqrt((candidate_pos[0] - last_ball_position[0])**2 +
                           (candidate_pos[1] - last_ball_position[1]))
            if dist < min_distance:
                min_distance = dist
                min_distance_index = i
        return candidate_points[min_distance_index]

    def find_min_distance_index(self, persistence_list, roi):
        """
        find index of min euclidian distance of roi objects
        Parameters 
        ------
        peristence list: list 
            list of candidates to find a minimal distance
        roi: tuple(int, int, int, int)
            region of interest

        Returns 
        ------
        min_dist_index: int
            index of persitence list with minimal distance
        min_dist int
            minimal distance from roi
        """
        min_dist_index = 0
        min_dist = 999999
        x, y, _, _ = roi
        for i, obj in enumerate(persistence_list):
            if obj.distance(*(x, y)) <= min_dist:
                min_dist = obj.distance(*(x, y))
                min_dist_index = i
        return min_dist_index, min_dist

    def find_max_persistence(self, persistence_list):
        """
        locate max persistence object
        Parameters 
        ------
        peristence list: list 
            list of candidates to find a minimal distance
        Returns
        ------
        max_per: int
            value of maximum persistence
        """
        max_per = 0
        for obj in persistence_list:
            if obj.persistence_count > max_per:
                max_per = obj.persistence_count
        return max_per

    def update_persistence(self, persistence_list, min_dist_index, roi):
        """
        update roi parameters of persistence using output from 
        find_max_persistence function
        Parameters 
        ------
        peristence list: list 
            list of candidates to find a minimal distance
        min_dist_index: int
            index of min distance from persistence list
        roi: tuple(x, y, w, h)
            region of ball interest (ball ROI)
        """
        x, y, w, h = roi
        persistence_list[min_dist_index].persistence_count += 1
        persistence_list[min_dist_index].update_speed(*(x, y))
        persistence_list[min_dist_index].x = x
        persistence_list[min_dist_index].y = y
        persistence_list[min_dist_index].w = w
        persistence_list[min_dist_index].h = h

    class Tracked:
        """
        Object that registers tracked objects
        Calculates frame persistence, velocity and position
        """

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

    def process_ball(self, rate, image_draw, image):
        """
        Main function to detect and update our prediction of ball location
        Parameters 
        ------
        rate: int
            rate that indicates processsing intensity
        image_draw: np.array
            image on which data will be drawn
        image: np.array
            image on which we will be calculating
        """
        self.idx += 1
        self.image_draw = image_draw
        self.image = image
        (ih, iw) = self.image.shape[:2]
        new_ball_pos = self.picked_ball_position
        center = (int(iw/2), int(ih/2))

        hsv = cv2.cvtColor(self.image, cv2.COLOR_BGR2HSV)
        sensitivity = 85
        lower_white = np.array([0, 0, 255-sensitivity])
        upper_white = np.array([255, sensitivity, 255])
        mask = cv2.inRange(hsv, lower_white, upper_white)
        # Do masking
        res = cv2.bitwise_and(self.image, self.image, mask=mask)
        res = cv2.cvtColor(res, cv2.COLOR_HSV2BGR)
        res = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)

        contours, _ = cv2.findContours(
            res, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
        candidates = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            roi = (x, y, w, h)
            if self.is_ROI(center, (x, y), tolerance=self.tolerance):
                if (h <= 12) and (w <= 12):
                    (x, y), radius = cv2.minEnclosingCircle(contour)
                    if radius > 2:
                        continue
                    guessed_area = cv2.contourArea(contour)
                    if guessed_area <= 7 and guessed_area >= 3:
                        center_x = (int(x), int(y))
                        candidates.append(center_x)
                        radius = int(radius)
                        min_distance_index, min_distance = self.find_min_distance_index(
                            self.persistent, roi)
                        if self.persistent == [] or min_distance > 4:
                            self.persistent.append(self.Tracked(*roi))
                        else:
                            self.update_persistence(
                                self.persistent, min_distance_index, roi)
                        # we have caught the ball candidate, we track it now
                        cv2.circle(self.image_draw, center_x,
                                   radius, (0, 255, 0), 2)

        if candidates != []:
            new_ball_pos = self.find_min_distance_to_last_ball_position(
                candidates, new_ball_pos)
            if self.ball_positions:
                self.ball_speeds.append(self.distance(
                    new_ball_pos, self.ball_positions[-1]))
            self.ball_positions.append(new_ball_pos)
            cv2.circle(self.image_draw, new_ball_pos, 2, (100, 0, 120), 2)

        if self.ball_positions != [] and self.ball_speeds != []:
            field_half = 'left' if self.acc_pos[0] < 0 else 'right'
            info = {
                "ball pos": self.ball_positions[-1],
                "ball speed": np.around(self.ball_speeds[-1], decimals=2),
                "half": field_half
            }
            # loop over the info tuples and draw them on our frame
            for i, key in enumerate(info):
                text = f"{key}: {info[key]}"
                cv2.putText(self.image_draw, text, (10, ih - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.rectangle(
            self.image, (center[0]-self.tolerance, center[1]-self.tolerance),
            (center[0]+self.tolerance, center[1]+self.tolerance),
            (255, 0, 255), 3)
