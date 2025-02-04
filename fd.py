import cv2
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import threading

# Initialize Tkinter
root = tk.Tk()
root.title("Face Detection - Upload Image/Video")

# Load Face Detection Model (Haarcascade)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Tkinter Label for Displaying Image/Video
label = Label(root)
label.pack()

def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if not file_path:
        return  # No file selected

    # Read Image
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect Faces
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Convert to Tkinter Image
    img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(img)
    imgtk = ImageTk.PhotoImage(image=img)
    label.imgtk = imgtk
    label.configure(image=imgtk)

def upload_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])
    if not file_path:
        return  # No file selected

    cap = cv2.VideoCapture(file_path)

    def process_video():
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break  # End of video

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)
            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Convert frame to Tkinter Image
            img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)
            label.imgtk = imgtk
            label.configure(image=imgtk)
            root.update_idletasks()
            root.update()

        cap.release()

    # Run video processing in a separate thread to prevent freezing
    threading.Thread(target=process_video).start()

# UI Elements
btn_upload_image = Button(root, text="Upload Image", command=upload_image)
btn_upload_image.pack()

btn_upload_video = Button(root, text="Upload Video", command=upload_video)
btn_upload_video.pack()

btn_quit = Button(root, text="Quit", command=root.quit)
btn_quit.pack()

# Run the GUI
root.mainloop()
