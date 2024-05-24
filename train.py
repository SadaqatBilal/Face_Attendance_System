import cv2
import os
import shutil
import csv
import numpy as np
from PIL import Image
import pandas as pd
import datetime
import time
import tkinter as tk
from tkinter import Message, Text
import tkinter.ttk as ttk
import tkinter.font as font

# Initialize tkinter window
window = tk.Tk()
window.title("Face Attendance System")
window.geometry('1280x720')
window.resizable(True, False)
window.configure(background='Blue')

# Create GUI elements
message = tk.Label(window, text="Face Attendance System", bg="Blue", fg="Black", width=50, height=3, font=('times', 30, 'bold'))
message.place(x=100, y=10)

lbl = tk.Label(window, text="Enter ID", width=20, height=2, fg="black", bg="Blue", font=('times', 15, 'bold'))
lbl.place(x=100, y=150)
txt = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, 'bold'))
txt.place(x=300, y=160)

lbl2 = tk.Label(window, text="Enter Name", width=20, fg="black", bg="Blue", height=2, font=('times', 15, 'bold'))
lbl2.place(x=100, y=270)
txt2 = tk.Entry(window, width=20, bg="white", fg="black", font=('times', 15, 'bold'))
txt2.place(x=300, y=280)

lbl3 = tk.Label(window, text="Notification:", width=20, fg="black", bg="Blue", height=2, font=('times', 15, 'bold underline'))
lbl3.place(x=400, y=340)
message = tk.Label(window, text="", bg="Blue", fg="black", width=30, height=2, activebackground="Deep Sky Blue", font=('times', 15, 'bold'))
message.place(x=600, y=340)

lbl3 = tk.Label(window, text="Attendance:", width=20, fg="black", bg="Blue", height=2, font=('times', 15, 'bold underline'))
lbl3.place(x=300, y=500)
message2 = tk.Label(window, text="", fg="black", bg="white", activeforeground="green", width=50, height=8, font=('times', 15, 'bold'))
message2.place(x=500, y=500)

# Function to clear ID input field
def clear():
    txt.delete(0, 'end')
    message.configure(text="")

# Function to clear Name input field
def clear2():
    txt2.delete(0, 'end')
    message.configure(text="")

# Function to check if a string is a number
def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False

# Function to capture images and save with ID and Name
def TakeImages():
    Id = txt.get()
    name = txt2.get()
    if is_number(Id) and name.isalpha():
        cam = cv2.VideoCapture(0)
        harcascadePath = "haarcascade_frontalface_default.xml"
        detector = cv2.CascadeClassifier(harcascadePath)
        sampleNum = 0
        while True:
            ret, img = cam.read()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            faces = detector.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
                sampleNum += 1
                cv2.imwrite("TrainingImage\\" + name + "." + Id + '.' + str(sampleNum) + ".jpg", gray[y:y+h, x:x+w])
                cv2.imshow('frame', img)
            if cv2.waitKey(100) & 0xFF == ord('q'):
                break
            elif sampleNum > 60:
                break
        cam.release()
        cv2.destroyAllWindows()
        res = "Images Saved for ID : " + Id + " Name : " + name
        row = [Id, name]
        with open('StudentDetails\\StudentDetails.csv', 'a+') as csvFile:
            writer = csv.writer(csvFile)
            writer.writerow(row)
        csvFile.close()
        message.configure(text=res)
    else:
        if is_number(Id):
            message.configure(text="Enter Alphabetical Name")
        if name.isalpha():
            message.configure(text="Enter Numeric Id")

# Function to train images and create a recognizer model
def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    harcascadePath = "haarcascade_frontalface_default.xml"
    detector = cv2.CascadeClassifier(harcascadePath)
    faces, Id = getImagesAndLabels("TrainingImage")
    recognizer.train(faces, np.array(Id))
    recognizer.save("TrainingImageLabel\\Trainner.yml")
    message.configure(text="Image Trained")

# Helper function to get images and labels
def getImagesAndLabels(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    Ids = []
    for imagePath in imagePaths:
        pilImage = Image.open(imagePath).convert('L')
        imageNp = np.array(pilImage, 'uint8')
        Id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces.append(imageNp)
        Ids.append(Id)
    return faces, Ids

# Function to track images and mark attendance
def TrackImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("TrainingImageLabel\\Trainner.yml")
    harcascadePath = "haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(harcascadePath)
    df = pd.read_csv("StudentDetails\\StudentDetails.csv")
    cam = cv2.VideoCapture(0)
    font = cv2.FONT_HERSHEY_SIMPLEX
    col_names = ['Id', 'Name', 'Date', 'Time']
    attendance = pd.DataFrame(columns=col_names)
    while True:
        ret, im = cam.read()
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
        faces = faceCascade.detectMultiScale(gray, 1.2, 5)
        for (x, y, w, h) in faces:
            cv2.rectangle(im, (x, y), (x + w, y + h), (225, 0, 0), 2)
            Id, conf = recognizer.predict(gray[y:y + h, x:x + w])
            if conf < 50:
                ts = time.time()
                date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
                timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
                aa = df.loc[df['Id'] == Id]['Name'].values
                tt = str(Id) + "-" + aa[0]
                attendance.loc[len(attendance)] = [Id, aa[0], date, timeStamp]
            else:
                Id = 'Unknown'
                tt = str(Id)
            if conf > 75:
                noOfFile = len(os.listdir("ImagesUnknown")) + 1
                cv2.imwrite("ImagesUnknown\\Image" + str(noOfFile) + ".jpg", im[y:y + h, x:x + w])
            cv2.putText(im, str(tt), (x, y + h), font, 1, (255, 255, 255), 2)
        attendance = attendance.drop_duplicates(subset=['Id'], keep='first')
        cv2.imshow('im', im)
        if cv2.waitKey(1) == ord('q'):
            break
    ts = time.time()
    date = datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d')
    timeStamp = datetime.datetime.fromtimestamp(ts).strftime('%H:%M:%S')
    Hour, Minute, Second = timeStamp.split(":")
    fileName = "Attendance\\Attendance_" + date + "_" + Hour + "-" + Minute + "-" + Second + ".csv"
    attendance.to_csv(fileName, index=False)
    cam.release()
    cv2.destroyAllWindows()
    message2.configure(text=attendance)

# Create GUI buttons
clearButton = tk.Button(window, text="Clear", command=clear, fg="white", bg="Deep Sky Blue", width=20, height=2, activebackground="Red", font=('times', 15, 'bold'))
clearButton.place(x=950, y=150)
clearButton2 = tk.Button(window, text="Clear", command=clear2, fg="white", bg="Deep Sky Blue", width=20, height=2, activebackground="Red", font=('times', 15, 'bold'))
clearButton2.place(x=950, y=270)
takeImg = tk.Button(window, text="Take Images", command=TakeImages, fg="white", bg="Deep Sky Blue", width=20, height=3, activebackground="Red", font=('times', 15, 'bold'))
takeImg.place(x=200, y=400)
trainImg = tk.Button(window, text="Train Images", command=TrainImages, fg="white", bg="Deep Sky Blue", width=20, height=3, activebackground="Red", font=('times', 15, 'bold'))
trainImg.place(x=500, y=400)
trackImg = tk.Button(window, text="Track Images", command=TrackImages, fg="white", bg="Deep Sky Blue", width=20, height=3, activebackground="Red", font=('times', 15, 'bold'))
trackImg.place(x=800, y=400)
quitWindow = tk.Button(window, text="Quit", command=window.destroy, fg="white", bg="Deep Sky Blue", width=20, height=3, activebackground="Red", font=('times', 15, 'bold'))
quitWindow.place(x=1100, y=400)

# Run the tkinter main loop
window.mainloop()

