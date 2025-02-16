import os
import base64
import io
import json
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
import mysql.connector
from config import Config
from models.face_recognition import detect_faces, nms_faces, extract_face, get_embedding, cosine_similarity
from utils.image_processing import apply_filters
from PIL import Image
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import cv2
import concurrent.futures
from werkzeug.utils import secure_filename

# Configure TensorFlow GPU memory growth for better GPU utilization.
import tensorflow as tf
gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

app = Flask(__name__)
app.config.from_object(Config)
app.secret_key = app.config['SECRET_KEY']

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def get_db_connection():
    conn = mysql.connector.connect(
        host=app.config['MYSQL_HOST'],
        user=app.config['MYSQL_USER'],
        password=app.config['MYSQL_PASSWORD'],
        database=app.config['MYSQL_DB']
    )
    conn.autocommit = True
    return conn

# ------------------- Authentication -------------------

@app.route('/')
def root():
    return redirect(url_for('welcome'))

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method=='POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT * FROM users WHERE username=%s AND password=%s", (username, password))
        user = cursor.fetchone()
        cursor.close()
        conn.close()
        if user:
            session['user'] = {'username': user['username'], 'role': user['role']}
            flash("Logged in successfully.", "success")
            if user['role'] == 'admin':
                return redirect(url_for('admin_index'))
            else:
                return redirect(url_for('teacher_index'))
        else:
            flash("Invalid credentials.", "danger")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('user', None)
    flash("Logged out successfully.", "success")
    return redirect(url_for('welcome'))

# ------------------- Admin Routes -------------------

@app.route('/admin', endpoint='admin_index')
def admin_index():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    return render_template('admin_index.html')

# Admin - Teacher CRUD
@app.route('/admin/teachers')
def list_teachers():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM teachers")
    teachers = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('list_teachers.html', teachers=teachers)

@app.route('/admin/teachers/add', methods=['GET','POST'])
def add_teacher():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    if request.method=='POST':
        name = request.form.get('name')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        photo_file = request.files.get('photo')
        photo_url = ""
        if photo_file and photo_file.filename!="":
            filename = f"teacher_{datetime.now().timestamp()}_{secure_filename(photo_file.filename)}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            Image.open(photo_file.stream).save(path)
            photo_url = url_for('static', filename='uploads/' + filename)
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO teachers (name, email, username, photo) VALUES (%s, %s, %s, %s)",
                           (name, email, username, photo_url))
            conn.commit()
        except mysql.connector.errors.IntegrityError:
            flash("Teacher with that username may already exist.", "danger")
            return redirect(url_for('add_teacher'))
        try:
            cursor.execute("INSERT INTO users (username, password, role) VALUES (%s, %s, 'teacher')", (username, password))
            conn.commit()
        except mysql.connector.errors.IntegrityError:
            flash("Duplicate username in users table.", "danger")
            return redirect(url_for('add_teacher'))
        cursor.close()
        conn.close()
        flash("Teacher added successfully.", "success")
        return redirect(url_for('list_teachers'))
    return render_template('add_teacher.html')

@app.route('/admin/teachers/edit/<int:teacher_id>', methods=['GET','POST'])
def edit_teacher(teacher_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    if request.method=='POST':
        name = request.form.get('name')
        email = request.form.get('email')
        photo_file = request.files.get('photo')
        photo_url = None
        if photo_file and photo_file.filename!="":
            filename = f"teacher_{datetime.now().timestamp()}_{secure_filename(photo_file.filename)}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            Image.open(photo_file.stream).save(path)
            photo_url = url_for('static', filename='uploads/' + filename)
        if photo_url:
            cursor.execute("UPDATE teachers SET name=%s, email=%s, photo=%s WHERE teacher_id=%s", (name, email, photo_url, teacher_id))
        else:
            cursor.execute("UPDATE teachers SET name=%s, email=%s WHERE teacher_id=%s", (name, email, teacher_id))
        conn.commit()
        cursor.close()
        conn.close()
        flash("Teacher updated successfully.", "success")
        return redirect(url_for('list_teachers'))
    else:
        cursor.execute("SELECT * FROM teachers WHERE teacher_id=%s", (teacher_id,))
        teacher = cursor.fetchone()
        cursor.close()
        conn.close()
        return render_template('edit_teacher.html', teacher=teacher)

@app.route('/admin/teachers/delete/<int:teacher_id>')
def delete_teacher(teacher_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM teachers WHERE teacher_id=%s", (teacher_id,))
    conn.commit()
    cursor.close()
    conn.close()
    flash("Teacher deleted successfully.", "success")
    return redirect(url_for('list_teachers'))

# Admin - Student CRUD
@app.route('/admin/students')
def list_students():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM students")
    students = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('list_students.html', students=students)

@app.route('/admin/students/add', methods=['GET','POST'])
def add_student():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    if request.method=='POST':
        student_id = request.form.get('student_id')
        name = request.form.get('name')
        branch = request.form.get('branch')
        _class = request.form.get('class')
        roll_number = request.form.get('roll_number')
        files = [request.files.get('face_photo1'), request.files.get('face_photo2'), request.files.get('face_photo3')]
        if not all(files) or any(f.filename=="" for f in files):
            flash("Please upload exactly three photos.", "warning")
            return redirect(url_for('add_student'))
        conn = get_db_connection()
        cursor = conn.cursor()
        try:
            cursor.execute("INSERT INTO students (student_id, name, branch, class, roll_number) VALUES (%s, %s, %s, %s, %s)",
                           (student_id, name, branch, _class, roll_number))
            conn.commit()
        except mysql.connector.errors.IntegrityError:
            flash("Student ID already exists.", "danger")
            return redirect(url_for('add_student'))
        for file in files:
            image = np.array(Image.open(file.stream).convert('RGB'))
            image = apply_filters(image)
            faces = detect_faces(image)
            faces = nms_faces(faces, iou_threshold=0.5)
            if len(faces) != 1:
                flash("Each student registration photo must contain exactly one face.", "danger")
                return redirect(url_for('add_student'))
            box = faces[0]['box']
            face_img = extract_face(image, box)
            embedding = get_embedding(face_img)
            emb_str = ",".join(map(str, embedding.tolist()))
            filename = f"{student_id}_{datetime.now().timestamp()}_{secure_filename(file.filename)}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            Image.fromarray(image).save(path)
            cursor.execute("INSERT INTO student_faces (student_id, embedding, image_path) VALUES (%s, %s, %s)",
                           (student_id, emb_str, url_for('static', filename='uploads/' + filename)))
            conn.commit()
        cursor.close()
        conn.close()
        flash("Student added successfully.", "success")
        return redirect(url_for('list_students'))
    return render_template('add_student.html')

@app.route('/admin/students/edit/<student_id>', methods=['GET','POST'])
def edit_student(student_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    if request.method=='POST':
        name = request.form.get('name')
        branch = request.form.get('branch')
        _class = request.form.get('class')
        roll_number = request.form.get('roll_number')
        cursor.execute("UPDATE students SET name=%s, branch=%s, class=%s, roll_number=%s WHERE student_id=%s",
                       (name, branch, _class, roll_number, student_id))
        conn.commit()
        files = [request.files.get('face_photo1'), request.files.get('face_photo2'), request.files.get('face_photo3')]
        if all(files) and not any(f.filename=="" for f in files):
            cursor.execute("DELETE FROM student_faces WHERE student_id=%s", (student_id,))
            conn.commit()
            for file in files:
                image = np.array(Image.open(file.stream).convert('RGB'))
                image = apply_filters(image)
                faces = detect_faces(image)
                faces = nms_faces(faces, iou_threshold=0.5)
                if len(faces) != 1:
                    flash("Each student photo must contain exactly one face.", "danger")
                    return redirect(url_for('edit_student', student_id=student_id))
                box = faces[0]['box']
                face_img = extract_face(image, box)
                embedding = get_embedding(face_img)
                emb_str = ",".join(map(str, embedding.tolist()))
                filename = f"{student_id}_{datetime.now().timestamp()}_{secure_filename(file.filename)}"
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                Image.fromarray(image).save(path)
                cursor.execute("INSERT INTO student_faces (student_id, embedding, image_path) VALUES (%s, %s, %s)",
                               (student_id, emb_str, url_for('static', filename='uploads/' + filename)))
                conn.commit()
        cursor.close()
        conn.close()
        flash("Student updated successfully.", "success")
        return redirect(url_for('list_students'))
    else:
        cursor.execute("SELECT * FROM students WHERE student_id=%s", (student_id,))
        student = cursor.fetchone()
        cursor.close()
        conn.close()
        return render_template('edit_student.html', student=student)

@app.route('/admin/students/delete/<student_id>')
def delete_student(student_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor()
    try:
        cursor.execute("DELETE FROM students WHERE student_id=%s", (student_id,))
        conn.commit()
    except mysql.connector.errors.IntegrityError as e:
        flash("Cannot delete student due to related attendance records.", "danger")
        return redirect(url_for('list_students'))
    cursor.close()
    conn.close()
    flash("Student deleted successfully.", "success")
    return redirect(url_for('list_students'))

@app.route('/admin/attendance')
def admin_attendance():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    today = datetime.now().strftime('%Y-%m-%d')
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute(
        "SELECT s.roll_number, s.name, s.branch, a.status, a.timestamp FROM attendance a JOIN students s ON a.student_id = s.student_id WHERE DATE(a.timestamp)=%s ORDER BY s.roll_number",
        (today,))
    records = cursor.fetchall()
    for r in records:
        r['date'] = r['timestamp'].strftime('%Y-%m-%d')
    cursor.execute("SELECT COUNT(*) as total FROM students")
    total = cursor.fetchone()['total']
    present = sum(1 for r in records if r['status']=='present')
    absent = total - present
    summary = {'total': total, 'present': present, 'absent': absent}
    cursor.close()
    conn.close()
    return render_template('show_attendance.html', attendance_records=records, summary=summary)

@app.route('/admin/attendance/edit/<int:attendance_id>', methods=['GET','POST'])
def edit_attendance(attendance_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    if request.method=='POST':
        status = request.form.get('status')
        cursor.execute("UPDATE attendance SET status=%s WHERE id=%s", (status, attendance_id))
        conn.commit()
        cursor.close()
        conn.close()
        flash("Attendance record updated.", "success")
        return redirect(url_for('admin_attendance'))
    else:
        cursor.execute("SELECT * FROM attendance WHERE id=%s", (attendance_id,))
        record = cursor.fetchone()
        cursor.close()
        conn.close()
        return render_template('edit_attendance.html', record=record)

@app.route('/admin/attendance/delete/<int:attendance_id>')
def delete_attendance(attendance_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM attendance WHERE id=%s", (attendance_id,))
    conn.commit()
    cursor.close()
    conn.close()
    flash("Attendance record deleted.", "success")
    return redirect(url_for('admin_attendance'))

@app.route('/admin/manage_attendance', methods=['GET','POST'])
def manage_attendance():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    if request.method=='POST':
        student_id = request.form.get('student_id')
        date_str = request.form.get('date')
        status = request.form.get('status')
        cursor.execute("INSERT INTO attendance (student_id, timestamp, status) VALUES (%s, %s, %s)", (student_id, date_str, status))
        conn.commit()
        flash("Attendance record added.", "success")
    cursor.execute("SELECT * FROM attendance")
    records = cursor.fetchall()
    for r in records:
        r['date'] = r['timestamp'].strftime('%Y-%m-%d')
    cursor.close()
    conn.close()
    return render_template('manage_attendance.html', records=records)

# Admin - List Registration Requests and Process Them
@app.route('/admin/requests', endpoint='list_requests')
def list_requests():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM student_requests")
    requests = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('list_requests.html', requests=requests)

@app.route('/admin/action', methods=['POST'])
def admin_action():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    request_id = request.form.get('request_id')
    action = request.form.get('action')
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM student_requests WHERE request_id=%s", (request_id,))
    req_data = cursor.fetchone()
    if not req_data:
        flash("Request not found.", "danger")
        return redirect(url_for('list_requests'))
    if req_data['status'] != 'pending':
        flash("This request has already been processed.", "warning")
        return redirect(url_for('list_requests'))
    if action == 'approved':
        # Pre-check each photo for exactly one face
        photos = [req_data['photo1'], req_data['photo2'], req_data['photo3']]
        for photo_url in photos:
            filename = secure_filename(photo_url.split('/')[-1])
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            try:
                image = np.array(Image.open(image_path))
            except FileNotFoundError:
                flash(f"File not found: {filename}", "danger")
                cursor.close()
                conn.close()
                return redirect(url_for('list_requests'))
            image = apply_filters(image)
            faces = detect_faces(image)
            faces = nms_faces(faces, iou_threshold=0.5)
            if len(faces) != 1:
                flash("Approval failed: one of the photos did not contain exactly one face.", "danger")
                cursor.close()
                conn.close()
                return redirect(url_for('list_requests'))
        # All photos passed, now register student
        student_id = req_data['student_id']
        try:
            cursor.execute("INSERT INTO students (student_id, name, branch, class, roll_number) VALUES (%s, %s, %s, %s, %s)",
                           (student_id, req_data['student_name'], req_data['branch'], req_data['class'], req_data['roll_number']))
            conn.commit()
        except mysql.connector.errors.IntegrityError:
            flash("Student already registered.", "danger")
            cursor.close()
            conn.close()
            return redirect(url_for('list_requests'))
        for photo_url in photos:
            filename = secure_filename(photo_url.split('/')[-1])
            image_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            image = np.array(Image.open(image_path))
            image = apply_filters(image)
            faces = detect_faces(image)
            faces = nms_faces(faces, iou_threshold=0.5)
            box = faces[0]['box']
            face_img = extract_face(image, box)
            embedding = get_embedding(face_img)
            emb_str = ",".join(map(str, embedding.tolist()))
            cursor.execute("INSERT INTO student_faces (student_id, embedding, image_path) VALUES (%s, %s, %s)",
                           (student_id, emb_str, photo_url))
            conn.commit()
    cursor.execute("UPDATE student_requests SET status=%s WHERE request_id=%s", (action, request_id))
    conn.commit()
    cursor.close()
    conn.close()
    flash("Request updated.", "success")
    return redirect(url_for('list_requests'))

# ------------------- Teacher Routes -------------------

@app.route('/teacher')
def teacher_index():
    if 'user' not in session or session['user']['role'] != 'teacher':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    return render_template('teacher_index.html')

@app.route('/teacher/attendance_live', methods=['POST'])
def attendance_live():
    if 'user' not in session or session['user']['role'] != 'teacher':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    photo_data_json = request.form.get('photoData')
    if not photo_data_json:
        flash("No captured photos found.", "warning")
        return redirect(url_for('teacher_index'))
    try:
        data_urls = json.loads(photo_data_json)
    except Exception as e:
        flash("Error parsing captured photos.", "danger")
        return redirect(url_for('teacher_index'))
    all_annotated = []
    recognized = set()
    for data_url in data_urls:
        header, encoded = data_url.split(',', 1)
        data = base64.b64decode(encoded)
        image = np.array(Image.open(io.BytesIO(data)))
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = apply_filters(image)
        faces = detect_faces(image)
        if not faces:
            continue
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT student_id, embedding FROM student_faces")
        face_db = cursor.fetchall()
        cursor.close()
        conn.close()
        for face in faces:
            box = face['box']
            x, y, w, h = box
            face_img = extract_face(image, box)
            emb = get_embedding(face_img)
            best, best_score = None, -1
            for rec in face_db:
                stored = np.array(list(map(float, rec['embedding'].split(','))))
                score = cosine_similarity(emb, stored)
                if score > best_score:
                    best_score = score
                    best = rec['student_id']
            if best_score > 0.7:
                recognized.add(best)
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(image, best, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            else:
                cv2.rectangle(image, (x, y), (x+w, y+h), (0, 0, 255), 2)
                cv2.putText(image, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        all_annotated.append(image)
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT student_id FROM students")
    all_students = {row['student_id'] for row in cursor.fetchall()}
    for student in all_students:
        status = 'present' if student in recognized else 'absent'
        try:
            cursor.execute("INSERT INTO attendance (student_id, status) VALUES (%s, %s)", (student, status))
            conn.commit()
        except mysql.connector.Error as err:
            flash(f"Error marking attendance: {err}", "danger")
            return redirect(url_for('teacher_index'))
    cursor.close()
    conn.close()
    filenames = []
    for idx, img in enumerate(all_annotated):
        fname = f"attendance_{datetime.now().timestamp()}_{idx}.jpg"
        path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
        cv2.imwrite(path, img)
        filenames.append(fname)
    flash(f"Attendance processed from live capture. Recognized: {len(recognized)} students.", "success")
    image_urls = [url_for('static', filename='uploads/' + f) for f in filenames]
    return render_template('attendance_result.html', image_urls=image_urls)

@app.route('/teacher/attendance', methods=['GET','POST'])
def teacher_attendance():
    if 'user' not in session or session['user']['role'] != 'teacher':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    if request.method=='POST':
        photos = request.files.getlist('attendance_photos')
        if not photos or all(photo.filename=="" for photo in photos):
            flash("Please upload at least one photo.", "warning")
            return redirect(url_for('teacher_attendance'))
        recognized = set()
        annotated_imgs = []
        def process_photo(file):
            img = np.array(Image.open(file.stream).convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            img = apply_filters(img)
            faces = detect_faces(img)
            conn = get_db_connection()
            cursor = conn.cursor(dictionary=True)
            cursor.execute("SELECT student_id, embedding FROM student_faces")
            face_db = cursor.fetchall()
            cursor.close()
            conn.close()
            for face in faces:
                box = face['box']
                x, y, w, h = box
                face_img = extract_face(img, box)
                emb = get_embedding(face_img)
                best, best_score = None, -1
                for rec in face_db:
                    stored = np.array(list(map(float, rec['embedding'].split(','))))
                    score = cosine_similarity(emb, stored)
                    if score > best_score:
                        best_score = score
                        best = rec['student_id']
                if best_score > 0.7:
                    recognized.add(best)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, best, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return img
        with concurrent.futures.ThreadPoolExecutor() as executor:
            annotated_imgs = list(executor.map(process_photo, photos))
        conn = get_db_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT student_id FROM students")
        all_students = {row['student_id'] for row in cursor.fetchall()}
        for student in all_students:
            status = 'present' if student in recognized else 'absent'
            try:
                cursor.execute("INSERT INTO attendance (student_id, status) VALUES (%s, %s)", (student, status))
                conn.commit()
            except mysql.connector.Error as err:
                flash(f"Error marking attendance: {err}", "danger")
                return redirect(url_for('teacher_attendance'))
        cursor.close()
        conn.close()
        filenames = []
        for idx, img in enumerate(annotated_imgs):
            fname = f"attendance_{datetime.now().timestamp()}_{idx}.jpg"
            path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
            cv2.imwrite(path, img)
            filenames.append(fname)
        flash(f"Attendance processed from uploaded photos. Recognized: {len(recognized)} students.", "success")
        image_urls = [url_for('static', filename='uploads/' + f) for f in filenames]
        return render_template('attendance_result.html', image_urls=image_urls)
    return render_template('teacher_attendance.html')

# ------------------- Attendance Result -------------------

@app.route('/attendance_result')
def attendance_result():
    filename = request.args.get('filename')
    if filename:
        image_urls = [url_for('static', filename='uploads/' + filename)]
    else:
        image_urls = None
    return render_template('attendance_result.html', image_urls=image_urls)

# ------------------- Show Attendance -------------------

@app.route('/show_attendance', methods=['GET'])
def show_attendance():
    view_all = request.args.get('view_all')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT s.roll_number, s.name, s.branch, a.status, a.timestamp
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
    """
    conditions = []
    params = []
    if start_date:
        conditions.append("DATE(a.timestamp) >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(a.timestamp) <= %s")
        params.append(end_date)
    if start_time:
        conditions.append("TIME(a.timestamp) >= %s")
        params.append(start_time)
    if end_time:
        conditions.append("TIME(a.timestamp) <= %s")
        params.append(end_time)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY s.roll_number"
    cursor.execute(query, params)
    records = cursor.fetchall()
    for r in records:
        r['date'] = r['timestamp'].strftime('%Y-%m-%d')
    if not view_all:
        today = datetime.now().strftime('%Y-%m-%d')
        records = [r for r in records if r['date'] == today and r['status'] == 'present']
    cursor.execute("SELECT COUNT(*) as total FROM students")
    total = cursor.fetchone()['total']
    present = sum(1 for r in records if r['status'] == 'present')
    absent = total - present
    summary = {'total': total, 'present': present, 'absent': absent}
    cursor.close()
    conn.close()
    return render_template('show_attendance.html', attendance_records=records, summary=summary)

# ------------------- Download Attendance -------------------

@app.route('/download_attendance')
def download_attendance():
    file_format = request.args.get('file_format', 'excel')
    start_date = request.args.get('start_date')
    end_date = request.args.get('end_date')
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    conn = get_db_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT s.roll_number, s.name, s.branch, a.status, a.timestamp
        FROM attendance a
        JOIN students s ON a.student_id = s.student_id
    """
    conditions = []
    params = []
    if start_date:
        conditions.append("DATE(a.timestamp) >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(a.timestamp) <= %s")
        params.append(end_date)
    if start_time:
        conditions.append("TIME(a.timestamp) >= %s")
        params.append(start_time)
    if end_time:
        conditions.append("TIME(a.timestamp) <= %s")
        params.append(end_time)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY s.roll_number"
    cursor.execute(query, params)
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    df = pd.DataFrame(records)
    if file_format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
        output.seek(0)
        return send_file(output, download_name="attendance.xlsx", as_attachment=True)
    elif file_format == 'pdf':
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        data = [['Roll Number', 'Name', 'Branch', 'Status', 'Timestamp']]
        for index, row in df.iterrows():
            data.append([row['roll_number'], row['name'], row['branch'], row['status'], str(row['timestamp'])])
        table = Table(data)
        style = TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
            ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
            ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
            ('GRID', (0, 0), (-1, -1), 1, colors.black),
        ])
        table.setStyle(style)
        elements = [table]
        doc.build(elements)
        output.seek(0)
        return send_file(output, download_name="attendance.pdf", as_attachment=True)
    else:
        flash("Invalid file format requested.", "danger")
        return redirect(url_for('show_attendance'))

# ------------------- Student Registration Request (Teacher) -------------------

@app.route('/teacher/request_registration', methods=['GET','POST'])
def request_registration():
    if 'user' not in session or session['user']['role'] != 'teacher':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        student_name = request.form.get('student_name')
        branch = request.form.get('branch')
        _class = request.form.get('class')
        roll_number = request.form.get('roll_number')
        files = [request.files.get('face_photo1'), request.files.get('face_photo2'), request.files.get('face_photo3')]
        if not all(files) or any(f.filename == "" for f in files):
            flash("Please upload exactly three photos for registration request.", "warning")
            return redirect(url_for('request_registration'))
        photo_urls = []
        for file in files:
            image = np.array(Image.open(file.stream).convert('RGB'))
            image = apply_filters(image)
            faces = detect_faces(image)
            faces = nms_faces(faces, iou_threshold=0.5)
            if len(faces) != 1:
                flash("Each photo must contain exactly one face.", "danger")
                return redirect(url_for('request_registration'))
            filename = f"request_{datetime.now().timestamp()}_{secure_filename(file.filename)}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            Image.fromarray(image).save(path)
            photo_urls.append(url_for('static', filename='uploads/' + filename))
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute("INSERT INTO student_requests (student_id, student_name, branch, class, roll_number, photo1, photo2, photo3) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)",
                       (student_id, student_name, branch, _class, roll_number, photo_urls[0], photo_urls[1], photo_urls[2]))
        conn.commit()
        cursor.close()
        conn.close()
        flash("Student registration request submitted.", "success")
        return redirect(url_for('teacher_index'))
    return render_template('request_registration.html')

if __name__ == '__main__':
    app.run(debug=True)
