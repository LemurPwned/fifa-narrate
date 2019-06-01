import pytesseract
import cv2
import imutils
import re
import asyncio
from copy import deepcopy
import threading
import json
from queue import Queue

rois = {  # for 1000 width
    'time': (108, 69, 50, 23),
    'team1': (53, 52, 53, 21),
    'team2': (161, 52, 52, 16),
    'score': (110, 53, 48, 20),
    'player1': (80, 502, 129, 33),
    'player2': (788, 506, 133, 28)
}
# config may be shadowed in training
config = ('-l eng --oem 2 --psm 7 --tessdata-dir ./tessdata ')

win_name = 'Match Detection'

test_vide_file = 'FIFA 19  Chelsea vs Tottenham Hotspur  Premier League Gameplay.mp4'


def display_info_dict(image, custom_dict):
    if custom_dict:
        for i, key in enumerate(custom_dict):
            text = f"{key}: {custom_dict[key]}"
            cv2.putText(image, text, (10, ih - ((i * 20) + 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)


def waitForRoi():
    if cv2.waitKey(1) & 0xFF == ord('s'):
        r = cv2.selectROI(image)
        cv2.destroyAllWindows()
        textImage = image[int(r[1]):int(r[1]+r[3]), int(r[0]):int(r[0]+r[2])]
        text = pytesseract.image_to_string(textImage, config=config)
        cv2.imshow(win_name, textImage)
        if cv2.waitKey(5000):
            cv2.destroyAllWindows()


def validate_dict(persistence_dict, detection_dict):
    regex = {
        'time': re.compile('[0-9]{1,2}:[0-9]{2}'),
        'score': re.compile('[0-9]-[0-9]')
    }
    for key in detection_dict:
        if key in regex:
            if regex[key].match(detection_dict[key]) is None:
                print(f"unmatched {key} with {detection_dict[key]}")
            else:
                persistence_dict[key] = detection_dict[key]
        else:
            persistence_dict[key] = detection_dict[key]


def roi_text_recognition(queue, img_copy, cpy_persistence, rois):
    detected = {}
    config = ('-l eng --oem 2 --psm 7 --tessdata-dir ./tessdata ')
    for textObj in rois.keys():
        r = rois[textObj]
        textImage = img_copy[int(r[1]):int(r[1]+r[3]),
                             int(r[0]):int(r[0]+r[2])]
        text = pytesseract.image_to_string(textImage, config=config)
        detected[textObj] = text
    validate_dict(cpy_persistence, detected)
    queue.put(cpy_persistence)


cap = cv2.VideoCapture(test_vide_file)
idx = 0
detection_thread = None

queue = Queue()
persistence_dict = {}
while True:
    idx += 1
    rate, image = cap.read()
    if idx < 500:
        continue
    image = imutils.resize(image, width=1000)
    (ih, iw) = image.shape[:2]
    center = (int(iw/2), int(ih/2))

    if idx % 50 == 0:
        if not queue.empty():
            rcv_dict = queue.get()
            if rcv_dict is not None:
                persistence_dict = rcv_dict
        if detection_thread is None or not detection_thread.isAlive():
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            gray = cv2.bitwise_not(gray)
            detection_thread = threading.Thread(
                target=roi_text_recognition, args=(queue, gray, persistence_dict, rois))
            detection_thread.start()
    if persistence_dict:
        display_info_dict(image, persistence_dict)

    cv2.imshow(win_name, image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
