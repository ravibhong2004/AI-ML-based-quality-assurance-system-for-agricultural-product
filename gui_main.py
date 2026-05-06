 
import logging
import tkinter as tk
from tkinter import ttk, LEFT, END
from PIL import Image, ImageTk
from tkinter.filedialog import askopenfilename
from tkinter import messagebox as ms
import cv2
import sqlite3
import os
import numpy as np
import time
from tkvideo import tkvideo

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
'''import detection_emotion_practice as validate'''

root=tk.Tk()

root.title("seed quality and fertilizer detection")
w,h = root.winfo_screenwidth(),root.winfo_screenheight()

# bg = Image.open(r"E:/carrier_choice_prediction/y9.jpg")
# bg.resize((1366,500),Image.ANTIALIAS)
# print(w,h)
# bg_img = ImageTk.PhotoImage(bg)
# bg_lbl = tk.Label(root,image=bg_img)
# bg_lbl.place(x=0,y=93)
# #, relwidth=1, relheight=1)

video_label =tk.Label(root)
video_label.pack()
# read video to display on label
player = tkvideo("Seed.F.V.mp4", video_label,loop = 1, size = (w, h))
player.play()

w = tk.Label(root, text="Seed quality and fertilizer detection using machine learning",width=110,background="paleturquoise",foreground="black",height=2,font=("Times new roman",19,"bold"))
w.place(x=0,y=0)



w,h = root.winfo_screenwidth(),root.winfo_screenheight()
root.geometry("%dx%d+0+0"%(w,h))
root.configure(background="#800517")


from tkinter import messagebox as ms


def Login():
    logger.info("Main GUI: open login window")
    from subprocess import call
    call(["python","login.py"])
def Register():
    logger.info("Main GUI: open registration window")
    from subprocess import call
    call(["python","registration.py"])


wlcm=tk.Label(root,text="......Welcome to seed quality and fertilizer detection System ......",width=110,height=3,background="paleturquoise",foreground="black",font=("Times new roman",19,"bold"))
wlcm.place(x=0,y=620)




d2=tk.Button(root,text="Login",command=Login,width=20,height=2,bd=20,background="plum",foreground="black",font=("times new roman",17,"bold"))
d2.place(x=0,y=400)


d3=tk.Button(root,text="Register",command=Register,width=20,height=2,bd=20,background="Plum",foreground="black",font=("times new roman",17,"bold"))
d3.place(x=0,y=250)



root.mainloop()
