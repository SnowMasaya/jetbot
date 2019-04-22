from jetbot import ObjectDetector
from jetbot import Camera
import torch
import torchvision
import torch.nn.functional as F
import cv2
import numpy as np
from jetbot import bgr8_to_jpeg
from enum import Enum


class ModelSettingValue(Enum):
    width = 300
    height = 300
    detect_label = 1


def model_define():
    """
    2 model define
    - object detection
    - collision avoid
    :return: object detect model, collision avoid model
    """
    model = ObjectDetector('ssd_mobilenet_v2_coco.engine')

    collision_model = torchvision.models.alexnet(pretrained=False)
    collision_model.classifier[6] = torch.nn.Linear(collision_model.classifier[6].in_features, 2)
    collision_model.load_state_dict(torch.load('../collision_avoidance/best_model.pth'))
    device = torch.device('cuda')
    collision_model = collision_model.to(device)

    return model, collision_model

model, collision_model = model_define()

camera = Camera.instance(width=ModelSettingValue.width.value, height=ModelSettingValue.height.value)


def detection_center(detection):
    """Computes the center x, y coordinates of the object"""
    bbox = detection['bbox']
    center_x = (bbox[0] + bbox[2]) / 2.0 - 0.5
    center_y = (bbox[1] + bbox[3]) / 2.0 - 0.5
    return (center_x, center_y)


def norm(vec):
    """Computes the length of the 2D vector"""
    return np.sqrt(vec[0] ** 2 + vec[1] ** 2)


def closest_detection(detections):
    """Finds the detection closest to the image center"""
    closest_detection = None
    for det in detections:
        print('det {}'.format(det))
        if closest_detection is None:
            closest_detection = det
        elif norm(detection_center(det)) < norm(detection_center(closest_detection)):
            closest_detection = det
    print('closest_detection {}'.format(closest_detection))
    return closest_detection

print('model execute')


def execute(change):
    image = change['new']
    print(image.shape)

    width = ModelSettingValue.width.value
    height = ModelSettingValue.height.value

    # compute all detected objects
    detections = model(image)

    # draw all detections on image
    for det in detections[0]:
        bbox = det['bbox']
        cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])),
                      (int(width * bbox[2]), int(height * bbox[3])), (255, 0, 0), 2)

    print('detections {}'.format(detections[0]))
    # select detections that match selected class label
    matching_detections = [d for d in detections[0] if d['label'] == int(ModelSettingValue.detect_label.value)]
    print('matching_detections {}'.format(matching_detections))

    # get detection closest to center of field of view and draw it
    det = closest_detection(matching_detections)
    print('det {}'.format(det))
    if det is not None:
        bbox = det['bbox']
        cv2.rectangle(image, (int(width * bbox[0]), int(height * bbox[1])),
                      (int(width * bbox[2]), int(height * bbox[3])), (0, 255, 0), 5)

    print('detections {}'.format(detections))
    cv2.imshow("Frame", image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        camera.unobserve_all()
        return

camera.unobserve_all()
print('camera observe start')
camera.observe(execute, names='value')
cv2.destroyAllWindows()
