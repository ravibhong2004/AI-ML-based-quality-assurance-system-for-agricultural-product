
import logging
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tkinter import *
def Train():
    """GUI"""
from tkinter import *
from tkinter import ttk 
import tkinter as tk
import numpy as np
import pandas as pd

logging.basicConfig(filename='app.log', level=logging.INFO,
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

from sklearn.decomposition import PCA
from sklearn.preprocessing import LabelEncoder

root = tk.Tk()

root.geometry("800x850+250+5")
root.title("seed quality and fertilizer detection")
root.configure(background="crimson")

State = tk.StringVar()
Year = tk.IntVar()
Season = tk.StringVar()
Crop = tk.StringVar()
Area = tk.IntVar()
Production = tk.IntVar()
Rainfall = tk.IntVar()
avg_temp = tk.IntVar()
PH_Value_of_Soil = tk.IntVar()
Type_of_soil= tk.StringVar()
#Suitable_Fertilizer = tk.IntVar()

    
    #===================================================================================================================

 
'''def Detect1():
    e1=State.get()
    print(e1)
    if e1=="Andaman and Nicobar Islands":
        e1=0
    elif e1=="Arunachal Pradesh":
        e1=1
    else :
        e1=2
    e2=Year.get()
    print(e2)
   
    e3=Season.get()
    print(e3)
    if e3=="Autumn":
        e3=0
    elif e3=="Kharif":
        e3=1
    elif e3=="Rabi":
        e3=2
    elif e3=="Whole Year":
        e3=3
    else :
        e3=4
    
    e4=Crop.get()
    print(e4)
    if e4=="Banana":
       e4=0
    elif e4=="Cashewnut":
        e4=1
    elif e4=="Coconut":
        e4=2
    elif e4=="Coriander":
        e4=3
    elif e4=="Dry chillies":
        e4=4
    elif e4=="Dry ginger":
        e4=5
    elif e4=="Garlic":
        e4=6
    elif e4=="Onion":
        e4=7
    elif e4=="Potato":
        e4=8
    elif e4=="Rice":
        e4=9
    elif e4=="Sugarcane":
        e4=10
    elif e4=="Turmeric":
        e4=11
    else :
        e4=12
    e5=Area.get()
    print(e5)
    e6=Production.get()
    print(e6)
    e7=Rainfall.get()
    print(e7)
    e8=avg_temp.get()
    print(e8)
    e9=PH_Value_of_Soil.get()
    print(e9)
    e10=Type_of_soil.get()
    print(e10)
    if e10=="Clayey soil":
       e10=0
    elif e10=="Loamy soil":
        e10=1
    else :
        e10=2
    #e11= Suitable_Fertilizer.get()
    #print(e11)
    
     
        
        
#########################################################################################
    from joblib import load

    # Load the trained model
    model = load('crop_Model1.joblib')

    # Example — get values from the form
    v1 = model.predict([[e1, e2, e3, e4, e5, e6, e7, e8, e9, e10]])
    print("Predicted Fertilizer:", v1)
    print(v1)

    yes = tk.Label(root,text=" Detect Suitable Fertilizer:"  +'\n'+ str(v1),background="blue",foreground="white",font=('times', 20, ' bold '),width=30)
    yes.place(x=800,y=650)
    
    import requests 
    api_address = "http://api.openweathermap.org/data/2.5/weather?appid=0c42f7f6b53b244c78a418f4f181282a&q="
    
    import json
    location_req_url='http://api.ipstack.com/103.51.95.183?access_key=4b0f43bbd8b7a123d8b9d6cde128e6ca'
    r = requests.get(location_req_url)
    location_obj = json.loads(r.text)
    
    lat = location_obj['latitude']
    lon = location_obj['longitude']
    latitude = lat
    longitude = lon
    
    
    location2 = "%s, %s" % (location_obj['city'], location_obj['region_code'])
    location2 = location2.replace(',','')
    print(location2.split()[0])
    city =location2.split()[0] 
    
    location_label = tk.Label(root,text="At location  "+str(location2),font=('Times New Roman',35,'italic'),bg="cyan3",fg="white")
    location_label.place(x=850,y=580)'''

import tkinter as tk
from joblib import load
import requests
import json
import warnings

warnings.filterwarnings("ignore")  # Hide sklearn feature name warnings

def Detect1():
    logger.info("Fertilizer prediction triggered: State=%s Year=%s Season=%s Crop=%s Area=%s Production=%s Rainfall=%s avg_temp=%s PH=%s Type=%s",
                State.get(), Year.get(), Season.get(), Crop.get(), Area.get(), Production.get(), Rainfall.get(), avg_temp.get(), PH_Value_of_Soil.get(), Type_of_soil.get())
    from joblib import load

    # Load model & encoders
    model = load('crop_Model1.joblib')
    label_encoders = load("label_encoders.joblib")
    
    logger.info("Model loaded, type: %s", type(model))
    logger.info("Fertilizer encoder classes: %s", label_encoders["Suitable Fertilizer"].classes_)

    # Input encoding
    try:
        e1 = label_encoders["State"].transform([State.get()])[0]
    except:
        logger.warning("Unknown State: %s, using default", State.get())
        e1 = 0
    e2 = int(Year.get())

    # ✅ FIXED TRY-EXCEPT
    try:
        e3 = label_encoders["Season"].transform([Season.get()])[0]
    except:
        logger.warning("Unknown Season: %s, using default", Season.get())
        e3 = 0

    try:
        e4 = label_encoders["Crop"].transform([Crop.get()])[0]
    except:
        logger.warning("Unknown Crop: %s, using default", Crop.get())
        e4 = 0

    e5 = float(Area.get())
    e6 = float(Production.get())
    e7 = float(Rainfall.get())
    e8 = float(avg_temp.get())
    e9 = float(PH_Value_of_Soil.get())

    try:
        e10 = label_encoders["Type of soil"].transform([Type_of_soil.get()])[0]
    except:
        logger.warning("Unknown Type of soil: %s, using default", Type_of_soil.get())
        e10 = 0

   

    # --------------------- INPUT COLLECTION ---------------------
    #e1 = State.get()
   # print(e1)
   # if e1 == "Andaman and Nicobar Islands":
   #     e1 = 0
   # elif e1 == "Arunachal Pradesh":
    #    e1 = 1
    #else:
      #  e1 = 2

   # e2 = int(Year.get())
   # e3 = Season.get()
   # print(e3)
   # if e3 == "Autumn":
   #     e3 = 0
   # elif e3 == "Kharif":
   #     e3 = 1
   # elif e3 == "Rabi":
    #    e3 = 2
    #elif e3 == "Whole Year":
  #      e3 = 3
  #  else:
   #     e3 = 4

   # e4 = Crop.get()
   # print(e4)
   # crop_map = {
       # "Banana": 0, "Cashewnut": 1, "Coconut": 2, "Coriander": 3,
       # "Dry chillies": 4, "Dry ginger": 5, "Garlic": 6, "Onion": 7,
       # "Potato": 8, "Rice": 9, "Sugarcane": 10, "Turmeric": 11, "Wheat": 12
  #  }
    #e4 = crop_map.get(e4, 0)

    #e5 = float(Area.get())
    #e6 = float(Production.get())
    #e7 = float(Rainfall.get())
   # e8 = float(avg_temp.get())
   # e9 = float(PH_Value_of_Soil.get())

    #e10 = Type_of_soil.get()
   # print(e10)
    #if e10 == "Clayey soil":
    #e10 = 0
   # elif e10 == "Loamy soil":
    #    e10 = 1
   # else:
      #  e10 = 2

   # --------------------- MODEL PREDICTION ---------------------
       
    from joblib import load

    model = load('crop_Model1.joblib')
    label_encoders = load("label_encoders.joblib")

    v1 = model.predict([[e1, e2, e3, e4, e5, e6, e7, e8, e9, e10]])
    print("v1 raw:", v1)
    logger.info("Model input: [%s, %s, %s, %s, %s, %s, %s, %s, %s, %s]", e1, e2, e3, e4, e5, e6, e7, e8, e9, e10)
    logger.info("Model prediction raw: %s", v1.tolist() if hasattr(v1, 'tolist') else v1)

    # Decode fertilizer
    try:
        fertilizer_encoder = label_encoders["Suitable Fertilizer"]
        fert_name = fertilizer_encoder.inverse_transform(v1)[0]
        logger.info("Fertilizer encoder classes: %s", fertilizer_encoder.classes_)
        logger.info("Prediction result: %s", fert_name)
    except Exception as e:
        logger.error("Error decoding fertilizer: %s", str(e))
        fert_name = "Error in prediction"

    logger.info("Prediction raw=%s, fertilizer=%s", v1.tolist() if hasattr(v1, 'tolist') else v1, fert_name)
    print("Final Fertilizer:", fert_name)

    #fertilizer_encoder = label_encoders["Suitable Fertilizer"]
    #fert_name = fertilizer_encoder.inverse_transform(v1)[0]

    #print("Correct Fertilizer:", fert_name)

    # Decode fertilizer (you can adjust labels according to your dataset)
   # fertilizer_map = {
    #    0:"Nitrogen-based Fertilizer",
     #   1:"Phosphorus-based Fertilizer",
      #  2:"Potassium-based Fertilizer",
      #  3:"Compost",
       # 4:"Organic Manure"
   # }
   # fert_name = fertilizer_map.get(int(v1[0]), f"Type {v1[0]}")


    # --------------------- WEATHER LOCATION INFO ---------------------
    try:
        location_req_url = 'http://api.ipstack.com/103.51.95.183?access_key=4b0f43bbd8b7a123d8b9d6cde128e6ca'
        r = requests.get(location_req_url)
        location_obj = json.loads(r.text)
        city = location_obj.get('city', 'Unknown City')
        region = location_obj.get('region_code', '')
        location2 = f"{city}, {region}"
        logger.info("Weather lookup successful: %s", location2)
    except Exception as ex:
        location2 = "Unknown Location"
        logger.warning("Weather lookup failed: %s", ex)

    # --------------------- DISPLAY ON GUI ---------------------
    # Clear previous labels (optional)
    for widget in root.place_slaves():
        if isinstance(widget, tk.Label) and "Result" in str(widget):
            widget.destroy()

    # Result Frame
    result_frame = tk.LabelFrame(root, text="Fertilizer Prediction Result", bg="blue", fg="white",
                                 font=('Times New Roman', 20, 'bold'), width=700, height=200)
    result_frame.place(x=700, y=580)

    # Fertilizer Label
    fert_label = tk.Label(result_frame,
                          text=f"🌾 Suitable Fertilizer:\n{fert_name}",
                          bg="blue", fg="white",
                          font=('Times New Roman', 22, 'bold'))
    fert_label.place(relx=0.5, rely=0.4, anchor='center')

    # Location Label
    loc_label = tk.Label(result_frame,
                         text=f"📍 Location: {location2}",
                         bg="blue", fg="white",
                         font=('Times New Roman', 16))
    loc_label.place(relx=0.5, rely=0.8, anchor='center')

def call_file():
            import location
            location.Train()
 
l1=tk.Label(root,text="State",background="#33fff3",font=('times', 20, ' bold '),width=25)
l1.place(x=5,y=1)

monthchoosen = ttk.Combobox(root, width =15,font=20,textvariable =State)
    
    # Adding combobox drop down list
monthchoosen['values'] = ('Andaman and Nicobar Islands',
    					'Arunachal Pradesh',
    					'Bihar')
monthchoosen.place(x=500,y=1)
monthchoosen.current()


l2=tk.Label(root,text="Year",background="#33fff3",font=('times', 20, ' bold '),width=25)
l2.place(x=5,y=50)
monthchoosen = ttk.Combobox(root, width =15,font=20,textvariable =Year)
    
    # Adding combobox drop down list
monthchoosen['values'] = ('2000',
    					'2001',
                        '2002',
                        '2003',
                        '2004',
                        '2005',
                        '2006',
                        '2007')
monthchoosen.place(x=500,y=50)
monthchoosen.current()

l3=tk.Label(root,text="Season",background="#33fff3",font=('times', 20, ' bold '),width=25)
l3.place(x=5,y=100)
monthchoosen = ttk.Combobox(root, width =15,font=20,textvariable =Season)
    
    # Adding combobox drop down list
monthchoosen['values'] = ('Autumn',
    					'Kharif',
    					'Rabi',
                        'Whole Year',
                            'Winter')
monthchoosen.place(x=500,y=100)
monthchoosen.current()

l4=tk.Label(root,text="Crop",background="#33fff3",font=('times', 20, ' bold '),width=25)
l4.place(x=5,y=150)
monthchoosen = ttk.Combobox(root, width =15,font=20,textvariable =Crop)
    
    # Adding combobox drop down list
monthchoosen['values'] = ('Banana',
    					'Cashewnut',
    					'Coconut',
                        'Coriander',
                        'Dry chillies',
                        'Dry ginger',
                        'Garlic',
                        'Onion'
                        'Potato',
                        'Rice',
                        'Sugarcane',
                        'Turmeric',
                        'Wheat')
monthchoosen.place(x=500,y=150)
monthchoosen.current()

l5=tk.Label(root,text="Area",background="#33fff3",font=('times', 20, ' bold '),width=25)
l5.place(x=5,y=200)
Area=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Area)
Area.place(x=500,y=200)



l6=tk.Label(root,text="Production",background="#33fff3",font=('times', 20, ' bold '),width=25)
l6.place(x=5,y=250)
Production=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Production)
Production.place(x=500,y=250)

l7=tk.Label(root,text="Rainfall",background="#33fff3",font=('times', 20, ' bold '),width=25)
l7.place(x=5,y=300)
Rainfall=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=Rainfall)
Rainfall.place(x=500,y=300)

l8=tk.Label(root,text="avg_temp",background="#33fff3",font=('times', 20, ' bold '),width=25)
l8.place(x=5,y=350)
avg_temp=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=avg_temp)
avg_temp.place(x=500,y=350)

l9=tk.Label(root,text="PH_Value_of_Soil",background="#33fff3",font=('times', 20, ' bold '),width=25)
l9.place(x=5,y=400)
PH_Value_of_Soil=tk.Entry(root,bd=2,width=5,font=("TkDefaultFont", 20),textvar=PH_Value_of_Soil)
PH_Value_of_Soil.place(x=500,y=400)



 

l10=tk.Label(root,text="Type_of_soil",background="#33fff3",font=('times', 20, ' bold '),width=25)
l10.place(x=5,y=450)
monthchoosen = ttk.Combobox(root, width =15,font=20,textvariable =Type_of_soil)
    
    # Adding combobox drop down list
monthchoosen['values'] = ('Clayey soil',
    						'Loamy soil',
    						'Sandy soil')
monthchoosen.place(x=500,y=450)
monthchoosen.current()

# l10=tk.Label(root,text="Suitable_Fertilizer",background="darkolivegreen1",font=('times', 20, ' bold '),width=25)
# l10.place(x=5,y=500)




button1 = tk.Button(root,text="Suitable Fertilizer",command=Detect1,background="#33fff3",bd=20,font=('times', 20, ' bold '),width=15)
button1.place(x=500,y=600)


root.mainloop()
            
Train()

        
       
          



    
    
    