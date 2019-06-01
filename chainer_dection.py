from chainercv.datasets import voc_bbox_label_names
from chainercv.links import SSD300
from chainercv.utils import read_image
from chainercv.visualizations import vis_bbox
import cv2
import imutils
# Read an RGB image and return it in CHW format.
model = SSD300(pretrained_model='voc0712')

test_vide_file = 'FIFA 19  Chelsea vs Tottenham Hotspur  Premier League Gameplay.mp4'
cap = cv2.VideoCapture(test_vide_file)
idx = 0
while True:
    idx += 1
    rate, image = cap.read()
    if idx < 0:
        continue
    image = imutils.resize(image, width=700)
    (ih, iw) = image.shape[:2]
    print(image.shape)
    center = (int(iw/2), int(ih/2))
    bboxes, labels, scores = model.predict([image.T])
    print(bboxes)
    for box_id in range(len(bboxes)):
        (y_min, x_min, y_max, x_max) = bboxes[box_id][0]
        cv2.rectangle(image, (x_max, y_max), (x_min, y_min),
                      (0, 0, 255), 3)

    cv2.imshow('Match Detection', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
