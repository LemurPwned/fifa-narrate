# This code is written at BigVision LLC. It is based on the OpenCV project. It is subject to the license terms in the LICENSE file found in this distribution and at http://opencv.org/license.html

# Usage example:  python3 object_detection_yolo.py --video=run.mp4
#                 python3 object_detection_yolo.py --image=bird.jpg

import cv2 as cv
#import argparse
import sys
import numpy as np
import os.path
import imutils
from scipy.spatial import distance as dist
from ball_detect import BallDetector
import boundary_detection

bd = BallDetector()
winName = "Football analysis"

# Initialize the parameters
confThreshold = 0.5  # Confidence threshold
nmsThreshold = 0.4  # Non-maximum suppression threshold
inpWidth = 416  # Width of network's input image
inpHeight = 416  # Height of network's input image

# Load names of classes
classesFile = "coco.names"
classes = None
with open(classesFile, 'rt') as f:
    classes = f.read().rstrip('\n').split('\n')

# Give the configuration and weight files for the model and load the network using them.
modelConfiguration = "yolov3.cfg"
modelWeights = "yolov3.weights"

net = cv.dnn.readNetFromDarknet(modelConfiguration, modelWeights)
net.setPreferableBackend(cv.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv.dnn.DNN_TARGET_CPU)


def getOutputsNames(net):
    """
    gets output names for labes from network    
    net: dnn network object
        network used to make predictions
    Returns
    ----
    list:
        all names generated by network
    """
    # Get the names of all the layers in the network
    layersNames = net.getLayerNames()
    # Get the names of the output layers, i.e. the layers with unconnected outputs
    return [layersNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]

# Draw the predicted bounding box


def drawPred(classId, conf, left, top, right, bottom):
    """
    draws predicted player box on image
    Parameters
    ------
    classId: int 
        id of class which was detected
    conf: int
        network output confidence level for this particular prediction
    left: int
        left x coordinate of player box
    top: int
        top y coordinate of player box
    right: int
        right x coordinate of player box
    bottom: int 
        bottom y coordinate of player box
    """
    # Draw a bounding box.
    if classes[classId] != "person":
        return
    cv.rectangle(frame, (left, top), (right, bottom), (255, 178, 50), 3)


def find_ball_player(players, ball):
    """
    finds player coordinates if ball is in player box
    players: list
        list of all players (bounding boxes) that are detected by 
        YOLO network and processed by postprocess      
        ball - coordinates x, y of ball
    Returns
    ----
    list:
        returns coordinates of player that is near ball, or None if 
        no such player has been found.

    """
    stickedPlayer = None
    for player in players:
        if ball[0] < player[2] and ball[0] > player[0] and ball[1] < player[3] and ball[1] > player[1]:
            stickedPlayer = player

    return stickedPlayer


def find_main_color_of_player(player, image):
    """
    finds main player color, it is work in progress
     player - 4 coordinates of playerBox
     image - clean image to analyse data
    """
    playerBox = image[player[0]:player[2], player[1]:player[3]]
    hist = playerBox[:, :, 0][playerBox[:, :, 0] > 230]

    # print(len(hist))


def postprocess(frame, outs):
    """
    processes output from neural network, drops objects with low confidence and add labels to image
    frame: np.array 
        frame of image that will be preprocessed
    outs: list 
        list of bounding boxes output from network
    Returns
    ----
    list:
         players coordinates that were found
    """
    frameHeight = frame.shape[0]
    frameWidth = frame.shape[1]

    # Scan through all the bounding boxes output from the network and keep only the
    # ones with high confidence scores. Assign the box's class label as the class with the highest score.
    classIds = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            classId = np.argmax(scores)
            confidence = scores[classId]
            if confidence > confThreshold:
                center_x = int(detection[0] * frameWidth)
                center_y = int(detection[1] * frameHeight)
                width = int(detection[2] * frameWidth)
                height = int(detection[3] * frameHeight)
                left = int(center_x - width / 2)
                top = int(center_y - height / 2)
                classIds.append(classId)
                confidences.append(float(confidence))
                boxes.append([left, top, width, height])

    # Perform non maximum suppression to eliminate redundant overlapping boxes with
    # lower confidences.
    indices = cv.dnn.NMSBoxes(boxes, confidences, confThreshold, nmsThreshold)
    players = []
    for i in indices:
        i = i[0]
        box = boxes[i]
        left = box[0]
        top = box[1]
        width = box[2]
        height = box[3]
        drawPred(classIds[i], confidences[i], left,
                 top, left + width, top + height)
        players.append([left, top, left+width, top + height])
    return players


cv.namedWindow(winName, cv.WINDOW_NORMAL)

cap = cv.VideoCapture("FIFA.mp4")

counter = 0
while True:
    counter += 1
    hasFrame, frame = cap.read()
    if counter < 120:
        continue
    else:
        break

scores_xpos = None
scores_ypos = None
surnames_xposL = None
surnames_yposL = None
surnames_xposR = None
surnames_yposR = None


while cv.waitKey(1) < 0:

    # get frame from the video
    hasFrame, frame = cap.read()
    counter += 1

    frame = imutils.resize(frame, width=700)
    frameCopy = frame.copy()
    if bd.picked_ball_position is None:
        while True:
            cv.imshow(winName, frame)
            cv.setMouseCallback(winName, bd.set_ball_position)
            if bd.picked_ball_position is None:
                k = cv.waitKey(20) & 0xFF
                if k == ord('q'):
                    break
                elif k == ord('a'):
                    # print(bd.picked_ball_position)
                    cv.circle(frame, bd.picked_ball_position,
                              2, (255, 255, 0), 2)
            else:
                break

    if not hasFrame:
        print("Done processing !!!")
        print("Output file is stored as ", outputFile)
        cv.waitKey(3000)
        # Release device
        cap.release()
        break

    if counter % 150 == 0:
        print(counter)
        scores_xpos, scores_ypos = boundary_detection.findScores(frameCopy)
        surnames_xposL, surnames_yposL, surnames_xposR, surnames_yposR = boundary_detection.findSurnames(
            frameCopy)

    if scores_xpos != None:
        cv.rectangle(frame, (scores_xpos[0], scores_ypos[0]),
                     (scores_xpos[1], scores_ypos[1]), (255, 0, 0), 3)
    if surnames_xposL != None:
        cv.rectangle(frame, (surnames_xposL[0], surnames_yposL[0]), (
            surnames_xposL[1], surnames_yposL[1]), (255, 0, 0), 3)
    if surnames_xposR != None:
        cv.rectangle(frame, (surnames_xposR[0], surnames_yposR[0]), (
            surnames_xposR[1], surnames_yposR[1]), (255, 0, 0), 3)

    # Create a 4D blob from a frame.
    blob = cv.dnn.blobFromImage(
        frame, 1/255, (inpWidth, inpHeight), [0, 0, 0], 1, crop=False)

    # Sets the input to the network
    net.setInput(blob)

    # Runs the forward pass to get output of the output layers
    outs = net.forward(getOutputsNames(net))

    players = postprocess(frame, outs)
    bd.process_ball(hasFrame, frame, frameCopy)
    # if len(bd.ball_positions) > 0:
    #    sticked_player = find_ball_player(players, bd.ball_positions[-1])
    #    find_main_color_of_player(sticked_player, frameCopy)
    cv.imshow(winName, frame)
