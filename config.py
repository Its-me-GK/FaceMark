import os

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_secret_key_here')
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.environ.get('MYSQL_USER', 'root')
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', '')
    MYSQL_DB = os.environ.get('MYSQL_DB', 'face_attendance16')
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
