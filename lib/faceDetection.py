import cv2
import dlib
import numpy as np
from PIL import Image


fast_detector = dlib.get_frontal_face_detector()
medium_detector = dlib.cnn_face_detection_model_v1("lib/model/mmod_human_face_detector.dat")
accurate_detector = dlib.cnn_face_detection_model_v1("lib/model/mmod_human_face_detector.dat")
shape_predictor = dlib.shape_predictor("lib/model/shape_predictor_68_face_landmarks.dat")


def convert_to_array(image):
    return np.array(image)


def face_detection_base(image, chain=False, humandetection=False):
    image_np = convert_to_array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    faces = fast_detector(gray)
    heads = []
    for face in faces:
        shape = shape_predictor(gray, face)
        nose_tip = shape.part(33)
        chin = shape.part(8)
        left_side = shape.part(0)
        right_side = shape.part(16)
        forehead_y = int(nose_tip.y - 2.5 * (chin.y - nose_tip.y))
        top = forehead_y
        bottom = chin.y
        left = left_side.x
        right = right_side.x
        head_box = dlib.rectangle(left, top, right, bottom)
        heads.append(head_box)

    return heads


def face_detection_exprt(image,humandetection=False):
    image_np = convert_to_array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    detected_faces = medium_detector(gray)
    faces = []
    for face in detected_faces:
        face_rect = face.rect  
        shape = shape_predictor(gray, face_rect)
        nose_tip = shape.part(33)
        chin = shape.part(8)
        left_side = shape.part(0)
        right_side = shape.part(16)
        forehead_y = int(nose_tip.y - 2.5 * (chin.y - nose_tip.y))
        top = forehead_y
        bottom = chin.y
        left = left_side.x
        right = right_side.x
        head_box = dlib.rectangle(left, top, right, bottom)
        faces.append(head_box)
    

    if not faces:
        return face_detection_most_accurate(image,humandetection)
    return faces


def face_detection_most_accurate(image,humandetection=False):
    image_np = convert_to_array(image)
    gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)
    
    detected_faces = accurate_detector(gray)
    faces = []
    for face in detected_faces:
        face_rect = face.rect 
        shape = shape_predictor(gray, face_rect)
        nose_tip = shape.part(33)
        chin = shape.part(8)
        left_side = shape.part(0)
        right_side = shape.part(16)
        forehead_y = int(nose_tip.y - 2.5 * (chin.y - nose_tip.y))
        top = forehead_y
        bottom = chin.y
        left = left_side.x
        right = right_side.x
        head_box = dlib.rectangle(left, top, right, bottom)
        faces.append(head_box)
    

    
    return faces


def humanFinder(image):
    image_np = convert_to_array(image)
    humans = detect_humans(image_np)
    if not humans:
        return []

    for human in humans:
        x, y, w, h = human
        human_region = image_np[y:y+h, x:x+w]
        faces = fast_detector(human_region)
        if faces:
            return faces
    
    return []


def detect_humans(image):
    hog = cv2.HOGDescriptor()
    hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    boxes, weights = hog.detectMultiScale(gray, winStride=(8,8))
    bounding_boxes = [(x, y, w, h) for (x, y, w, h) in boxes]

    return bounding_boxes
