# FaceMark - Group Face Recognition Attendance & Management System

FaceMark is a comprehensive attendance management system that leverages advanced face detection and recognition techniques to automatically mark attendance from group photos and live camera feeds. Built using Python, Flask, MySQL, and OpenCV, the system employs MTCNN for robust face detection and a pre-trained FaceNet model for generating 128-dimensional face embeddings. It supports multi-image enrollment (three images per user) to capture natural variations in appearance, ensuring improved recognition accuracy.

## Features
- **Group Face Recognition:** Detects and recognizes multiple faces in a single image.
- **Adaptive Image Processing:** Applies dynamic filters (CLAHE, gamma correction, and a custom bluish filter) based on image brightness to enhance facial features.
- **Multi-image Enrollment:** Registers each student or teacher using three images, leading to more stable embedding extraction.
- **Real-time Attendance:** Processes both live camera feeds and uploaded group photos for instant attendance marking.
- **Responsive Dashboard:** Provides a user-friendly web interface with support for record management and data export (Excel/PDF).

## Technologies
- **Language:** Python 3.x  
- **Web Framework:** Flask  
- **Database:** MySQL  
- **Computer Vision:** MTCNN, FaceNet, OpenCV, Pillow  
- **Frontend:** HTML5, CSS3 (Bootstrap), JavaScript  

## Installation & Setup
1. **Clone the Repository**  
   ```bash
   git clone https://github.com/yourusername/facemark.git
   cd facemark
   ```
2. **Create and Activate a Virtual Environment**  
   ```bash
   python -m venv venv
   # On Windows:
   venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```
3. **Install Dependencies**  
   ```bash
   pip install -r requirements.txt
   ```
4. **Configure the Database**  
   - Run the provided `database.sql` script to create the database schema.  
   - Update `config.py` with your MySQL credentials.
5. **Run the Application**  
   ```bash
   python app.py
   ```

## Usage
Users can register by uploading three images, mark attendance through live or group photos, and manage records via a responsive dashboard. The system dynamically processes images to enhance detection and recognition accuracy.

## Contributing
Contributions are welcome. Please fork the repository and submit a pull request for any fixes or enhancements.
