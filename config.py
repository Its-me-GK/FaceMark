import os
import json

with open('config.json', 'r') as c:
    params = json.load(c)["params"]

class Config:
    SECRET_KEY = os.environ.get('SECRET_KEY', 'your_secret_key_here')
    MYSQL_HOST = os.environ.get('MYSQL_HOST', 'localhost')
    MYSQL_USER = os.environ.get('MYSQL_USER', params['db_user'])
    MYSQL_PASSWORD = os.environ.get('MYSQL_PASSWORD', params['db_pass'])
    MYSQL_DB = os.environ.get('MYSQL_DB', params['db_name'])
    UPLOAD_FOLDER = os.path.join(os.getcwd(), 'static', 'uploads')
