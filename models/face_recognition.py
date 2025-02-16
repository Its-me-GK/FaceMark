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

def detect_faces(image):
    """Detect faces using MTCNN and return a list of face dictionaries."""
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(rgb_image)
    return faces

def extract_face(image, box, required_size=(160, 160)):
    """Extract and resize the face from the image using the provided bounding box."""
    x, y, width, height = box
    x, y = max(0, x), max(0, y)
    face = image[y:y+height, x:x+width]
    face = cv2.resize(face, required_size)
    face = face.astype('float32')
    mean, std = face.mean(), face.std()
    face = (face - mean) / std
    return face

def get_embedding(face_pixels):
    """Compute and return a 128-d embedding vector for the given face."""
    face_pixels = np.expand_dims(face_pixels, axis=0)
    embedding = facenet_model.predict(face_pixels)
    embedding = embedding / norm(embedding)
    return embedding[0]

def cosine_similarity(emb1, emb2):
    """Calculate cosine similarity between two embedding vectors."""
    return np.dot(emb1, emb2) / (norm(emb1) * norm(emb2))
