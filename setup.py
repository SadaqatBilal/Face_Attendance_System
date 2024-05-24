from cx_Freeze import setup, Executable
import sys
from mtcnn import MTCNN

# Create an MTCNN detector to determine the model path
detector = MTCNN()

base = None
if sys.platform == 'win32':
    base = "Win32GUI"

executables = [Executable("train.py", base=base)]

options = {
    'build_exe': {
        'packages': ["cv2", "os", "shutil", "csv", "numpy", "PIL", "pandas", "datetime", "time", "tkinter"],
        'include_files': [
            (detector.model_path, "mtcnn_models"),  # Include the MTCNN models
            "haarcascade_frontalface_default.xml",  # Include the Haar Cascade XML file
            "StudentDetails\\StudentDetails.csv"    # Include the student details CSV
        ],
        'include_msvcr': True,
    }
}

setup(
    name="FaceAttendanceSystem",
    options=options,
    version="0.0.1",
    description='Face Attendance System',
    executables=executables
)
