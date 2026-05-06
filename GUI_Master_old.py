import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
import cv2
import json
import numpy as np
import time
from keras.models import load_model 

import os
global fn
fn = ""

# Valid image check
VALID_IMAGE_EXTENSIONS = {'.png', '.jpg', '.jpeg', '.bmp', '.gif'}

def is_valid_image(file_path):
    if not os.path.isfile(file_path):
        return False
    ext = os.path.splitext(file_path)[1].lower()
    if ext not in VALID_IMAGE_EXTENSIONS:
        return False
    try:
        Image.open(file_path).verify()
        return True
    except Exception:
        return False

# Initialize the main window
root = tk.Tk()
root.configure(background="seashell2")
root.geometry("1300x700")
w, h = root.winfo_screenwidth(), root.winfo_screenheight()
root.geometry("%dx%d+0+0" % (w, h))
root.title("Seed Detection and Fertilizer Detection")

# Add background image
image2 = Image.open('bg.JPG')
image2 = image2.resize((1300, 800), Image.Resampling.LANCZOS)
background_image = ImageTk.PhotoImage(image2)
background_label = tk.Label(root, image=background_image)
background_label.image = background_image
background_label.place(x=250, y=1)

# Title and welcome message
lbl = tk.Label(
    root, text="Seed Detection and Fertilizer Detection",
    font=('times', 30, 'bold'),
    width=65, height=1, bg="black", fg="white"
)
lbl.place(x=0, y=0)

wlcm = tk.Label(
    root,
    text="......Welcome to the Seed Detection and Fertilizer Detection System......",
    width=110, height=1,
    background="black", foreground="white",
    font=("Times new roman", 19, "bold")
)
wlcm.place(x=0, y=655)

# Frame for processes
frame_alpr = tk.LabelFrame(
    root, text=" --Process-- ",
    width=262, height=743, bd=5,
    font=('times', 14, 'bold'),
    bg="black", fg="white"
)
frame_alpr.grid(row=0, column=0, sticky='nw')
frame_alpr.place(x=0, y=50)

# Label for prediction result
result_label = tk.Label( root, text="",width=60,height=5,font=("bold", 20), bg='Plum', fg='black', wraplength=800,justify=LEFT,)
result_label.place(x=300, y=450)

# Functions
def update_label(text):
    """Update the result label with the given text."""
    result_label.config(text=text)

def test_model_proc(fn):
    IMAGE_SIZE = 64
    if fn != "":
        # Load the pre-trained model
        model = load_model("model.h5", compile=False)

        # Load config for dynamic thresholds
        with open('config.json', 'r') as f:
            config = json.load(f)
        thresholds = config['quality_thresholds']
        qualities = config['qualities']

        # Preprocess the image
        img = Image.open(fn).convert('RGB')
        img = img.resize((IMAGE_SIZE, IMAGE_SIZE))
        img = np.array(img)
        img = img.reshape(1, IMAGE_SIZE, IMAGE_SIZE, 3)
        img = img.astype('float32') / 255.0

        # Predict the class
        prediction = model.predict(img)
        index = np.argmax(prediction)
        confidence_score = float(prediction[0][index])

        class_mapping = {
            0: "Corn seed",
            1: "Maize seed",
            2: "Soybean seed"
        }

        # Quality mapping based on thresholds from config.json
        if confidence_score >= thresholds.get("excellent", 0.9):
            predicted_quality = qualities.get("excellent", "Excellent")
        elif confidence_score >= thresholds.get("good", 0.8):
            predicted_quality = qualities.get("good", "Good")
        elif confidence_score >= thresholds.get("average", 0.6):
            predicted_quality = qualities.get("average", "Average")
        elif confidence_score >= thresholds.get("poor", 0.4):
            predicted_quality = qualities.get("poor", "Poor")
        else:
            predicted_quality = qualities.get("bad", "Bad")

        seed_type = class_mapping.get(index, "Unknown seed type")
        return f"Seed Type: {seed_type}, Predicted Quality: {predicted_quality} (Confident:{confidence_score:.1f})"
    else:
        return "No file selected for processing"

def test_model():
    global fn, valid_image, image_preprocessed
    if fn != "":
        if not valid_image:
            update_label("Invalid image. Please select a valid dataset image.")
            return
        if not image_preprocessed:
            update_label("Please preprocess the image first before prediction.")
            return
        update_label("Model Testing Start...............")
        start = time.time()
        X = test_model_proc(fn)
        X1 = "{0}".format(X)
        end = time.time()
        ET = "Execution Time: {0:.4} seconds".format(end - start)
        msg = "Image Testing Completed..\n" + X1 + '\n' + ET
        fn = ""
        image_preprocessed = False  # reset after testing
    else:
        msg = "Please Select Image For Prediction...."
    update_label(msg)

def openimage():
    global fn, image_preprocessed, valid_image
    fileName = askopenfilename(title='Select image for analysis', filetypes=[("Image files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif"), ("All files", "*.*")])
    if fileName:
        fn = fileName
        valid_image = is_valid_image(fileName)
        image_preprocessed = False
        if not valid_image:
            update_label("Invalid image. Please select a valid dataset image.")
            fn = ""
            return
        img = Image.open(fn).resize((200, 200))
        imgtk = ImageTk.PhotoImage(img)
        display = tk.Label(root, image=imgtk, height=250, width=250)
        display.image = imgtk
        display.place(x=300, y=100)

def convert_grey():
    global fn, image_preprocessed
    if fn == "":
        update_label("Please select image first!")
        return
    img = Image.open(fn).resize((200, 200))
    img_array = np.array(img)
    gs = cv2.cvtColor(cv2.imread(fn, 1), cv2.COLOR_RGB2GRAY)
    gs = cv2.resize(gs, (img_array.shape[0], img_array.shape[1]))
    retval, threshold = cv2.threshold(gs, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    im = Image.fromarray(gs)
    imgtk = ImageTk.PhotoImage(image=im)
    img2 = tk.Label(root, image=imgtk, height=250, width=250, bg='white')
    img2.image = imgtk
    img2.place(x=580, y=100)

    im_th = Image.fromarray(threshold)
    imgtk_th = ImageTk.PhotoImage(image=im_th)
    img3 = tk.Label(root, image=imgtk_th, height=250, width=250)
    img3.image = imgtk_th
    img3.place(x=880, y=100)

    image_preprocessed = True

def window():
    root.quit()

# Buttons
button1 = tk.Button(frame_alpr, text="Select Image", command=openimage, bd=20, width=15, height=1, font=('times', 15, 'bold'), bg="Plum", fg="black")
button1.place(x=30, y=60)

button2 = tk.Button(frame_alpr, text="Image Preprocess", command=convert_grey, bd=20, width=15, height=1, font=('times', 15, 'bold'), bg="Plum", fg="black")
button2.place(x=30, y=140)

button4 = tk.Button(frame_alpr, text="CNN Prediction", command=test_model, width=15, bd=20, height=1, bg="Plum", fg="black", font=('times', 15, 'bold'))
button4.place(x=30, y=220)

exit_button = tk.Button(frame_alpr, text="Exit", command=window, bd=20, width=15, height=1, bg="Plum", font=('times', 15, 'bold'), fg="black")
exit_button.place(x=30, y=500)

root.mainloop()
