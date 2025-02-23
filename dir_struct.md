face_recognition_attendance/
├── app.py
├── config.py
├── database.sql
├── requirements.txt
├── models/
│   ├── face_recognition.py
│   └── facenet_keras.h5        (your pre-trained FaceNet model)
├── utils/
│   └── image_processing.py
├── static/
│   ├── css/
│   │   └── style.css
│   ├── js/
│   │   └── main.js
│   └── uploads/                (this folder must exist)
└── templates/
    ├── base.html
    ├── welcome.html
    ├── login.html
    ├── admin_index.html
    ├── list_teachers.html
    ├── add_teacher.html
    ├── edit_teacher.html
    ├── list_students.html
    ├── add_student.html
    ├── edit_student.html
    ├── teacher_index.html
    ├── teacher_attendance.html
    ├── attendance_result.html
    ├── show_attendance.html
    ├── manage_attendance.html
    ├── list_requests.html
    ├── edit_attendance.html
    └── request_registration.html
