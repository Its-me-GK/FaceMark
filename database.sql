-- Create the main database
CREATE DATABASE IF NOT EXISTS face_attendance16;
USE face_attendance16;

-- Users table for authentication
CREATE TABLE IF NOT EXISTS users (
  user_id INT AUTO_INCREMENT PRIMARY KEY,
  username VARCHAR(50) UNIQUE NOT NULL,
  password VARCHAR(100) NOT NULL,
  role ENUM('admin','teacher') NOT NULL
);

-- Teachers table (for teacher details)
CREATE TABLE IF NOT EXISTS teachers (
  teacher_id INT AUTO_INCREMENT PRIMARY KEY,
  name VARCHAR(100) NOT NULL,
  email VARCHAR(100),
  username VARCHAR(50) UNIQUE,
  photo VARCHAR(255)
);

-- Students table
CREATE TABLE IF NOT EXISTS students (
    student_id VARCHAR(20) PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    branch VARCHAR(50),
    class VARCHAR(50),
    roll_number VARCHAR(20)
);

-- Student faces table (each student registration requires exactly three photos)
CREATE TABLE IF NOT EXISTS student_faces (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(20),
    embedding TEXT,
    image_path VARCHAR(255),
    FOREIGN KEY (student_id) REFERENCES students(student_id) ON DELETE CASCADE
);

-- Attendance table
CREATE TABLE IF NOT EXISTS attendance (
    id INT AUTO_INCREMENT PRIMARY KEY,
    student_id VARCHAR(20),
    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
    status VARCHAR(20),
    FOREIGN KEY (student_id) REFERENCES students(student_id)
);

-- Student registration requests (submitted by teachers)
CREATE TABLE IF NOT EXISTS student_requests (
  request_id INT AUTO_INCREMENT PRIMARY KEY,
  student_name VARCHAR(100),
  branch VARCHAR(50),
  class VARCHAR(50),
  roll_number VARCHAR(20),
  photo1 VARCHAR(255),
  photo2 VARCHAR(255),
  photo3 VARCHAR(255),
  status ENUM('pending','approved','rejected') DEFAULT 'pending'
);
