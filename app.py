import os
import base64
import io
import json
from datetime import datetime, date
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
import mysql.connector
from config import Config
from models.face_recognition import detect_faces, nms_faces, extract_face, get_embedding, cosine_similarity
from utils.image_processing import apply_clahe_filter, apply_bluish_filter_v2, apply_hist_eq_filter,apply_night_vision_filter, correct_orientation, apply_light_filter, apply_sharpening_filter, apply_bluish_filter,enhance_facial_features
from PIL import Image
import numpy as np
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle
import cv2
import concurrent.futures
from werkzeug.utils import secure_filename
import tensorflow as tf

# Configure TensorFlow GPU memory growth
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

# ================= Database Helper =================
class DatabaseHelper:
    @staticmethod
    def get_connection():
        conn = mysql.connector.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB']
        )
        conn.autocommit = True
        return conn

# ================= Attendance Manager =================
# class AttendanceManager:
#     @staticmethod
#     def update_attendance(student_id, new_status):
#         today_str = date.today().strftime('%Y-%m-%d')
#         conn = DatabaseHelper.get_connection()
#         cursor = conn.cursor(dictionary=True)
#         cursor.execute("SELECT * FROM attendance WHERE student_id=%s AND DATE(timestamp)=%s", (student_id, today_str))
#         record = cursor.fetchone()
#         if record:
#             if record['status'] == 'absent' and new_status == 'present':
#                 cursor.execute("UPDATE attendance SET status=%s WHERE id=%s", ('present', record['id']))
#                 conn.commit()
#         else:
#             cursor.execute("INSERT INTO attendance (student_id, status) VALUES (%s, %s)", (student_id, new_status))
#             conn.commit()
#         cursor.close()
#         conn.close()

class AttendanceManager:
    @staticmethod
    def update_attendance(student_id, new_status):
        today_str = date.today().strftime('%Y-%m-%d')
        conn = DatabaseHelper.get_connection()
        cursor = conn.cursor(dictionary=True)
        # Execute the SELECT query.
        cursor.execute(
            "SELECT * FROM attendance WHERE student_id=%s AND DATE(timestamp)=%s", 
            (student_id, today_str)
        )
        # Fetch one row (or all rows if needed) to ensure the result is fully read.
        record = cursor.fetchone()
        # Clear any unread results
        while cursor.nextset():
            pass

        if record:
            if record['status'] == 'absent' and new_status == 'present':
                cursor.execute(
                    "UPDATE attendance SET status=%s WHERE id=%s", 
                    ('present', record['id'])
                )
                conn.commit()
        else:
            cursor.execute(
                "INSERT INTO attendance (student_id, status) VALUES (%s, %s)", 
                (student_id, new_status)
            )
            conn.commit()
        cursor.close()
        conn.close()

    @staticmethod
    def get_records(start_date, end_date, start_time=None, end_time=None):
        conn = DatabaseHelper.get_connection()
        cursor = conn.cursor(dictionary=True)
        query = """
            SELECT s.student_id, s.roll_number, s.name, s.branch, a.status, a.timestamp
            FROM students s
            LEFT JOIN attendance a ON s.student_id = a.student_id
              AND DATE(a.timestamp) BETWEEN %s AND %s
        """
        params = [start_date, end_date]
        if start_time and end_time:
            query += " AND TIME(a.timestamp) BETWEEN %s AND %s"
            params.extend([start_time, end_time])
        query += " ORDER BY DATE(a.timestamp) ASC, CAST(s.roll_number AS UNSIGNED) ASC"
        cursor.execute(query, params)
        records = cursor.fetchall()
        for rec in records:
            if rec['timestamp'] is None:
                rec['timestamp'] = datetime.strptime(start_date, "%Y-%m-%d")
        cursor.close()
        conn.close()
        return records

# ================= Registration Manager =================
class RegistrationManager:
    @staticmethod
    def choose_filter_for_registration(image):
        # Compute brightness on grayscale
        # image = apply_clahe_filter(image) # added filter
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 80:
            image = apply_light_filter(image, gamma=0.8)
            return apply_clahe_filter(image)
        elif mean_brightness > 180: #need to verify manually
            image = apply_bluish_filter(image)
            # return apply_hist_eq_filter(image)
            return image
        else:
            # return apply_clahe_filter(image)
            return apply_bluish_filter_v2(image)
    
    @staticmethod
    def validate_single_face(image, min_confidence=0.95, iou_threshold=0.8):
        processed_image = RegistrationManager.choose_filter_for_registration(image)
        faces = detect_faces(processed_image, min_confidence=min_confidence)
        faces = nms_faces(faces, iou_threshold=iou_threshold)
        print(len(faces))
        return (len(faces) == 1), processed_image

# ================= Authentication Routes =================
@app.route('/')
def root():
    return redirect(url_for('welcome'))

@app.route('/welcome')
def welcome():
    return render_template('welcome.html')

@app.route('/login', methods=['GET','POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        conn = DatabaseHelper.get_connection()
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

# ================= Admin Routes =================
@app.route('/admin', endpoint='admin_index')
def admin_index():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    return render_template('admin_index.html')

# --- Teacher Management ---
@app.route('/admin/teachers')
def list_teachers():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = DatabaseHelper.get_connection()
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
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        username = request.form.get('username')
        password = request.form.get('password')
        photo_file = request.files.get('photo')
        photo_url = ""
        if photo_file and photo_file.filename != "":
            filename = f"teacher_{datetime.now().timestamp()}_{secure_filename(photo_file.filename)}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            Image.open(photo_file.stream).save(path)
            photo_url = url_for('static', filename='uploads/' + filename)
        conn = DatabaseHelper.get_connection()
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
    conn = DatabaseHelper.get_connection()
    cursor = conn.cursor(dictionary=True)
    if request.method == 'POST':
        name = request.form.get('name')
        email = request.form.get('email')
        photo_file = request.files.get('photo')
        photo_url = None
        if photo_file and photo_file.filename != "":
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
    conn = DatabaseHelper.get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM teachers WHERE teacher_id=%s", (teacher_id,))
    conn.commit()
    cursor.close()
    conn.close()
    flash("Teacher deleted successfully.", "success")
    return redirect(url_for('list_teachers'))

# --- Student Management ---
@app.route('/admin/students')
def list_students():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = DatabaseHelper.get_connection()
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
    if request.method == 'POST':
        student_id = request.form.get('student_id')
        name = request.form.get('name')
        branch = request.form.get('branch')
        _class = request.form.get('class')
        roll_number = request.form.get('roll_number')
        files = [request.files.get('face_photo1'), request.files.get('face_photo2'), request.files.get('face_photo3')]
        if not all(files) or any(f.filename == "" for f in files):
            flash("Please upload exactly three photos.", "warning")
            return redirect(url_for('add_student'))
        conn = DatabaseHelper.get_connection()
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
            valid, proc_image = RegistrationManager.validate_single_face(image, min_confidence=0.95, iou_threshold=0.8)
            if not valid:
                flash("Each student registration photo must contain exactly one clear face.", "danger")
                return redirect(url_for('add_student'))
            faces = detect_faces(proc_image, min_confidence=0.90)
            faces = nms_faces(faces, iou_threshold=0.7)
            box = faces[0]['box']
            # proc_image = RegistrationManager.choose_filter_for_registration(proc_image) # filter - adding student ++ /blue
            face_img = extract_face(proc_image, box)
            embedding = get_embedding(face_img)
            emb_str = ",".join(map(str, embedding.tolist()))
            filename = f"{student_id}_{datetime.now().timestamp()}_{secure_filename(file.filename)}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            Image.fromarray(proc_image).save(path)
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
    conn = DatabaseHelper.get_connection()
    cursor = conn.cursor(dictionary=True)
    if request.method == 'POST':
        name = request.form.get('name')
        branch = request.form.get('branch')
        _class = request.form.get('class')
        roll_number = request.form.get('roll_number')
        cursor.execute("UPDATE students SET name=%s, branch=%s, class=%s, roll_number=%s WHERE student_id=%s",
                       (name, branch, _class, roll_number, student_id))
        conn.commit()
        files = [request.files.get('face_photo1'), request.files.get('face_photo2'), request.files.get('face_photo3')]
        if all(files) and not any(f.filename == "" for f in files):
            cursor.execute("DELETE FROM student_faces WHERE student_id=%s", (student_id,))
            conn.commit()
            for file in files:
                image = np.array(Image.open(file.stream).convert('RGB'))
                valid, proc_image = RegistrationManager.validate_single_face(image, min_confidence=0.95, iou_threshold=0.85)
                if not valid:
                    flash("Each student photo must contain exactly one clear face.", "danger")
                    return redirect(url_for('edit_student', student_id=student_id))
                faces = detect_faces(proc_image, min_confidence=0.95)
                faces = nms_faces(faces, iou_threshold=0.85)
                box = faces[0]['box']
                # proc_image = RegistrationManager.choose_filter_for_registration(proc_image) # filter - updating student ++ /blue
                face_img = extract_face(proc_image, box)
                embedding = get_embedding(face_img)
                emb_str = ",".join(map(str, embedding.tolist()))
                filename = f"{student_id}_{datetime.now().timestamp()}_{secure_filename(file.filename)}"
                path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                Image.fromarray(proc_image).save(path)
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
    conn = DatabaseHelper.get_connection()
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

# ---------------- Admin Attendance & Management ----------------
@app.route('/admin/attendance')
def admin_attendance():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    start_date = request.args.get('start_date') or date.today().strftime("%Y-%m-%d")
    end_date = request.args.get('end_date') or date.today().strftime("%Y-%m-%d")
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    current_time = datetime.now().strftime("%H:%M:%S")
    if start_time and not end_time:
        end_time = current_time
    conn = DatabaseHelper.get_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT s.student_id, s.roll_number, s.name, s.branch, a.status, a.timestamp
        FROM attendance a JOIN students s ON a.student_id = s.student_id
    """
    conditions = []
    params = []
    if start_date:
        conditions.append("DATE(a.timestamp) >= %s")
        params.append(start_date)
    if end_date:
        conditions.append("DATE(a.timestamp) <= %s")
        params.append(end_date)
    if start_time and end_time:
        conditions.append("TIME(a.timestamp) BETWEEN %s AND %s")
        params.extend([start_time, end_time])
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY DATE(a.timestamp) ASC, CAST(s.roll_number AS UNSIGNED) ASC" #casted roll number into string to number
    cursor.execute(query, params)
    records = cursor.fetchall()
    cursor.execute("SELECT COUNT(*) as total FROM students")
    total = cursor.fetchone()['total']
    present = sum(1 for r in records if r['status'] == 'present')
    absent = total - present
    if absent < 0:
        absent = 0
    summary = {'total': total, 'present': present, 'absent': absent}
    grouped = {}
    for rec in records:
        rec_date = rec['timestamp'].strftime("%d-%m-%Y") if rec['timestamp'] else "No Date"
        if rec_date not in grouped:
            grouped[rec_date] = []
        if not any(r['student_id'] == rec['student_id'] for r in grouped[rec_date]):
            grouped[rec_date].append(rec)
        else:
            existing = next(r for r in grouped[rec_date] if r['student_id'] == rec['student_id'])
            if existing['status'] != 'present' and rec['status'] == 'present':
                grouped[rec_date].remove(existing)
                grouped[rec_date].append(rec)
    cursor.close()
    conn.close()
    return render_template('show_attendance.html', grouped_records=grouped, summary=summary)

@app.route('/admin/attendance/edit/<int:attendance_id>', methods=['GET','POST'])
def edit_attendance(attendance_id):
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = DatabaseHelper.get_connection()
    cursor = conn.cursor(dictionary=True)
    if request.method == 'POST':
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
    conn = DatabaseHelper.get_connection()
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
    conn = DatabaseHelper.get_connection()
    cursor = conn.cursor(dictionary=True)
    if request.method == 'POST':
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

# ---------------- Admin Registration Requests ----------------
@app.route('/admin/requests', endpoint='list_requests')
def list_requests():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    conn = DatabaseHelper.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM student_requests")
    requests_data = cursor.fetchall()
    cursor.close()
    conn.close()
    return render_template('list_requests.html', requests=requests_data)

@app.route('/admin/action', methods=['POST'])
def admin_action():
    if 'user' not in session or session['user']['role'] != 'admin':
        flash("Access denied.", "danger")
        return redirect(url_for('login'))
    request_id = request.form.get('request_id')
    action = request.form.get('action')
    conn = DatabaseHelper.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT * FROM student_requests WHERE request_id=%s", (request_id,))
    req_data = cursor.fetchone()
    if not req_data:
        flash("Request not found.", "danger")
        cursor.close()
        conn.close()
        return redirect(url_for('list_requests'))
    if req_data['status'] != 'pending':
        flash("This request has already been processed.", "warning")
        cursor.close()
        conn.close()
        return redirect(url_for('list_requests'))
    if action == 'approved':
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
            # image = apply_clahe_filter(image)
            image = RegistrationManager.choose_filter_for_registration(image) # filter - request approval
            if not RegistrationManager.validate_single_face(image, min_confidence=0.90, iou_threshold=0.7)[0]:
                flash("Approval failed: one of the photos did not contain exactly one clear face.", "danger")
                cursor.close()
                conn.close()
                return redirect(url_for('list_requests'))
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
            # image = apply_clahe_filter(image) -- add below enhance feature then blue
            image = RegistrationManager.choose_filter_for_registration(image) # filter - storing in db -admin add
            faces = detect_faces(image, min_confidence=0.90)
            faces = nms_faces(faces, iou_threshold=0.7)
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

# ---------------- Teacher Routes ----------------
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
        # image = RegistrationManager.choose_filter_for_registration(image) #filter - live
        faces = detect_faces(image, min_confidence=0.85)
        if not faces:
            continue
        conn = DatabaseHelper.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT student_id, embedding FROM student_faces")
        face_db = cursor.fetchall()
        cursor.close()
        conn.close()
        image=apply_clahe_filter(image) #added
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
        all_annotated.append(image) #returning image for live
    conn = DatabaseHelper.get_connection()
    cursor = conn.cursor(dictionary=True)
    cursor.execute("SELECT student_id FROM students")
    all_students = {row['student_id'] for row in cursor.fetchall()}
    cursor.close()
    conn.close()
    for student in all_students:
        if student in recognized:
            AttendanceManager.update_attendance(student, 'present')
        else:
            AttendanceManager.update_attendance(student, 'absent')
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
    if request.method == 'POST':
        photos = request.files.getlist('attendance_photos')
        if not photos or all(photo.filename == "" for photo in photos):
            flash("Please upload at least one photo.", "warning")
            return redirect(url_for('teacher_attendance'))
        recognized = set()
        annotated_imgs = []
        def process_photo(file):
            img = np.array(Image.open(file.stream).convert('RGB'))
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            # img = apply_clahe_filter(img)
            img = apply_bluish_filter(img)
            # img = RegistrationManager.choose_filter_for_registration(img) # filter - manual
            faces = detect_faces(img, min_confidence=0.90)
            conn = DatabaseHelper.get_connection()
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
                if(best_score>0.76):
                    print(best_score)
                if best_score > 0.77: #matching score .62
                    recognized.add(best)
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(img, best, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                else:
                    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(img, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            return img
        with concurrent.futures.ThreadPoolExecutor() as executor:
            annotated_imgs = list(executor.map(process_photo, photos))
        conn = DatabaseHelper.get_connection()
        cursor = conn.cursor(dictionary=True)
        cursor.execute("SELECT student_id FROM students")
        all_students = {row['student_id'] for row in cursor.fetchall()}
        cursor.close()
        conn.close()
        for student in all_students:
            if student in recognized:
                AttendanceManager.update_attendance(student, 'present')
            else:
                AttendanceManager.update_attendance(student, 'absent')
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

# ---------------- Attendance Result ----------------
@app.route('/attendance_result')
def attendance_result():
    filename = request.args.get('filename')
    if filename:
        image_urls = [url_for('static', filename='uploads/' + filename)]
    else:
        image_urls = None
    return render_template('attendance_result.html', image_urls=image_urls)

# ---------------- Show Attendance ----------------
@app.route('/show_attendance', methods=['GET'])
def show_attendance_view():
    start_date = request.args.get('start_date') or date.today().strftime("%Y-%m-%d")
    end_date = request.args.get('end_date') or date.today().strftime("%Y-%m-%d")
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    current_time = datetime.now().strftime("%H:%M:%S")
    if start_time and not end_time:
        end_time = current_time
    records = AttendanceManager.get_records(start_date, end_date, start_time, end_time)
    grouped = {}
    for rec in records:
        rec_date = rec['timestamp'].strftime("%d-%m-%Y")
        if rec_date not in grouped:
            grouped[rec_date] = []
        if not any(r['student_id'] == rec['student_id'] for r in grouped[rec_date]):
            grouped[rec_date].append(rec)
        else:
            existing = next(r for r in grouped[rec_date] if r['student_id'] == rec['student_id'])
            if existing['status'] != 'present' and rec['status'] == 'present':
                grouped[rec_date].remove(existing)
                grouped[rec_date].append(rec)
    total = len(records)
    present = sum(1 for rec in records if rec['status'] == 'present')
    absent = total - present
    if absent < 0:
        absent = 0
    summary = {'total': total, 'present': present, 'absent': absent}
    return render_template('show_attendance.html', grouped_records=grouped, summary=summary)

# ---------------- Download Attendance ----------------
@app.route('/download_attendance', methods=['GET'])
def download_attendance_view():
    file_format = request.args.get('file_format', 'excel')
    start_date = request.args.get('start_date') or date.today().strftime("%Y-%m-%d")
    end_date = request.args.get('end_date') or date.today().strftime("%Y-%m-%d")
    start_time = request.args.get('start_time')
    end_time = request.args.get('end_time')
    current_time = datetime.now().strftime("%H:%M:%S")
    if start_time and not end_time:
        end_time = current_time
    conn = DatabaseHelper.get_connection()
    cursor = conn.cursor(dictionary=True)
    query = """
        SELECT s.roll_number, s.name, s.branch,
               COALESCE(a.status, 'absent') as status,
               a.timestamp
        FROM students s
        LEFT JOIN attendance a ON s.student_id = a.student_id
          AND DATE(a.timestamp) BETWEEN %s AND %s
    """
    params = [start_date, end_date]
    if start_time and end_time:
        query += " AND TIME(a.timestamp) BETWEEN %s AND %s"
        params.extend([start_time, end_time])
    query += " ORDER BY DATE(a.timestamp) ASC, CAST(s.roll_number AS UNSIGNED) ASC"
    cursor.execute(query, params)
    records = cursor.fetchall()
    cursor.close()
    conn.close()
    for rec in records:
        if rec['timestamp'] is None:
            rec['timestamp'] = datetime.strptime(start_date, "%Y-%m-%d")
    df = pd.DataFrame(records)
    if not df.empty:
        df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime("%d-%m-%Y %I:%M %p")
    if file_format == 'excel':
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='xlsxwriter') as writer:
            df.to_excel(writer, index=False, sheet_name='Attendance')
            workbook  = writer.book
            worksheet = writer.sheets['Attendance']
            header_format = workbook.add_format({
                'bold': True,
                'text_wrap': True,
                'valign': 'middle',
                'fg_color': '#D7E4BC',
                'border': 1})
            for col_num, value in enumerate(df.columns.values):
                worksheet.write(0, col_num, value, header_format)
        output.seek(0)
        return send_file(output, download_name="attendance.xlsx", as_attachment=True)
    elif file_format == 'pdf':
        output = io.BytesIO()
        doc = SimpleDocTemplate(output, pagesize=letter)
        data = [df.columns.tolist()] + df.values.tolist()
        table = Table(data)
        style = TableStyle([
            ('BACKGROUND', (0,0), (-1,0), colors.grey),
            ('TEXTCOLOR',(0,0),(-1,0),colors.whitesmoke),
            ('ALIGN',(0,0),(-1,-1),'CENTER'),
            ('FONTNAME',(0,0),(-1,0),'Helvetica-Bold'),
            ('BOTTOMPADDING',(0,0),(-1,0),12),
            ('BACKGROUND',(0,1),(-1,-1),colors.beige),
            ('GRID',(0,0),(-1,-1),1,colors.black),
        ])
        table.setStyle(style)
        elements = [table]
        doc.build(elements)
        output.seek(0)
        return send_file(output, download_name="attendance.pdf", as_attachment=True)
    else:
        flash("Invalid file format requested.", "danger")
        return redirect(url_for('show_attendance_view'))

# ---------------- Student Registration Request (Teacher) ----------------
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
            valid, proc_image = RegistrationManager.validate_single_face(image, min_confidence=0.90, iou_threshold=0.7)
            if not valid:
                flash("Each photo must contain exactly one clear face.", "danger")
                return redirect(url_for('request_registration'))
            filename = f"request_{datetime.now().timestamp()}_{secure_filename(file.filename)}"
            path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            Image.fromarray(proc_image).save(path)
            photo_urls.append(url_for('static', filename='uploads/' + filename))
        conn = DatabaseHelper.get_connection()
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
    # app.run(debug=True, host='0.0.0.0' , ssl_context=('Deploy/cert.pem', 'Deploy/key.pem'))
