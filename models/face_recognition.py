import numpy as np
import cv2
from mtcnn import MTCNN
from keras.models import load_model
from numpy.linalg import norm

# Initialize MTCNN for face detection
detector = MTCNN()

# Load the pre-trained FaceNet model.
FACENET_MODEL_PATH = 'models/facenet_keras.h5'
facenet_model = load_model(FACENET_MODEL_PATH)

def detect_faces(image, min_confidence=0.95):
    # Assumes input image is in RGB.
    faces = detector.detect_faces(image)
    faces = [face for face in faces if face.get('confidence', 0) >= min_confidence]
    return faces

def iou(box1, box2):
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    xi1 = max(x1, x2)
    yi1 = max(y1, y2)
    xi2 = min(x1 + w1, x2 + w2)
    yi2 = min(y1 + h1, y2 + h2)
    inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - inter_area
    if union_area == 0:
        return 0
    return inter_area / union_area

def nms_faces(detections, iou_threshold=0.7):
    if not detections:
        return []
    detections = sorted(detections, key=lambda x: x.get('confidence', 0), reverse=True)
    nms = []
    while detections:
        best = detections.pop(0)
        nms.append(best)
        detections = [d for d in detections if iou(best['box'], d['box']) < iou_threshold]
    return nms

def extract_face(image, box, required_size=(160, 160)):
    x, y, width, height = box
    x, y = max(0, x), max(0, y)
    face = image[y:y+height, x:x+width]
    face = cv2.resize(face, required_size)
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    return face

def get_embedding(face_pixels):
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = facenet_model.predict(face_pixels)
    embedding = embedding / norm(embedding)
    return embedding[0]

def cosine_similarity(emb1, emb2):
    return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
