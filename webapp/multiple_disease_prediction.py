import streamlit as st
import pandas as pd
import pickle
from streamlit_option_menu import option_menu

import json
from PIL import Image
import requests  # pip install requests
from streamlit_lottie import st_lottie  # pip install streamlit-lottie
import streamlit as st
import numpy as np
import cv2
from keras.models import load_model
import tensorflow as tf

# GitHub: https://github.com/andfanilo/streamlit-lottie
# Lottie Files: https://lottiefiles.com/

# loading the saved models

diabetes_model = pickle.load(open('F:/python code/temporary/model/diabetes_trained_model.sav', 'rb'))

heart_disease_model = pickle.load(open('F:/python code/temporary/model/heart_disease_model.sav','rb'))

parkinsons_model = pickle.load(open('F:/python code/temporary/model/parkinsons_model.sav', 'rb'))

lung_cancer_model = pickle.load(open('F:/python code/temporary/model/lungCancer_trained_model.sav', 'rb'))

model0= pickle.load(open('F:/python code/temporary/model/disease_predict_by_symptoms.sav', 'rb'))

# Loading the Model
model = load_model('F:/python code/temporary/model/tomato_disease_prediction_model.h5')

model1 = load_model('F:/python code/temporary/model/rice_disease_prediction_model.h5')

model2 = load_model('F:/python code/temporary/model/wheat_disease_prediction_model.h5')

model3 = load_model('F:/python code/temporary/model/apple_disease_prediction_model.h5')
model4 = load_model('F:/python code/temporary/model/PNEUMONIA_prediction_model.h5')
# Security
#passlib,hashlib,bcrypt,scrypt
import hashlib
def make_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_hashes(password,hashed_text):
    if make_hashes(password) == hashed_text:
        return hashed_text
    return False
# DB Management
import sqlite3 
conn = sqlite3.connect('data.db')
c = conn.cursor()
# DB  Functions
def create_usertable():
    c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')


def add_userdata(username,password):
    c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
    conn.commit()
    

def login_user(username,password):
    c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
    data = c.fetchall()
    return data


def view_all_users():
    c.execute('SELECT * FROM userstable')
    data = c.fetchall()
    return data

def view_user_data(username):
    c.execute('SELECT * FROM userstable WHERE username = ?', (username,))
    data = c.fetchall()
    return data

def main():
    """Simple Login App"""

    st.title("Multiple Disease Prediction System")
    with st.sidebar:

        choice = option_menu('Multiple Disease Prediction System',
                              
                              ['Home',
                                'Login',
                               'SignUp',
                               ],
                              icons=['house','check-circle','plus-circle'],
                              default_index=0,)
        
    #menu = ["Home","Login","SignUp"]
    #choice = st.sidebar.selectbox("Menu",menu)

    if choice == "Home":
        def load_lottiefile(filepath: str):
            with open(filepath, "r") as f:
                return json.load(f)
        
        
    
            
        
        lottie_hello = load_lottiefile("F:/python code/temporary/lotti_files/hello.json")  # replace link to local lottie file
        #lottie_hello = load_lottieurl("https://assets9.lottiefiles.com/packages/lf20_M9p23l.json")
        
        
        st_lottie(
            lottie_hello,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=400,
            width=700,
            key=None,
        )

        st.markdown('We can build a multiple diseases prediction app using machine learning. The app would take input from the user, such as age, gender, symptoms, medical history, test results and images, and use machine learning algorithms to predict the probability of multiple diseases.')
        st.subheader("This web app can predict Human disease and crop disease")
        st.subheader("Human Disease Prediction System:")
        st.markdown("Human Disease Prediction System aims to improve healthcare outcomes by detecting diseases early and providing personalized treatment recommendations. This can be achieved by analyzing user input data and medical datasets to identify patterns that may indicate the presence of a disease or the risk of developing one.")
        
        lottie_health = load_lottiefile("F:/python code/temporary/lotti_files/health.json")
        
        
        
        lottie_agri = load_lottiefile("F:/python code/temporary/lotti_files/agriculture.json")
        st_lottie(
            lottie_health,
            
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=400,
            width=700,
            key=None,
        )
        
        st.subheader("Crop Disease Prediction System:")
        st.markdown("Crop Disease Prediction System aims to improve agricultural outcomes by detecting diseases early and providing personalized treatment recommendations. This can be achieved by analyzing user input image datasets to identify patterns that may indicate the presence of a disease or the risk of developing one.")
        
        
        st_lottie(
            lottie_agri,
            
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=400,
            width=700,
            key=None,
        )
        
        st.subheader('Thank you for choosing us.Praying to God for your good health.Please help us by giving an honest feedback so that we can improve more.')
        
        

    elif choice == "Login":
        st.subheader("Login Section")
        def load_lottiefile(filepath: str):
            with open(filepath, "r") as f:
                return json.load(f)
        
        
        
            
        
        lottie_login = load_lottiefile("F:/python code/temporary/lotti_files/login01.json")  # replace link to local lottie file
        
        
        
        st_lottie(
            lottie_login,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=500,
            width=700,
            key=None,
        )
        
        username = st.sidebar.text_input("User Name")
        password = st.sidebar.text_input("Password",type='password')
        if st.sidebar.checkbox("Login"):
            # if password == '12345':
            create_usertable()
            hashed_pswd = make_hashes(password)

            result = login_user(username,check_hashes(password,hashed_pswd))
            if result:
                lottie_login = load_lottiefile("F:/python code/temporary/lotti_files/welcome.json")  # replace link to local lottie file
                
                
                
                st_lottie(
                    lottie_login,
                    speed=1,
                    reverse=False,
                    loop=True,
                    quality="low",
                    height=400,
                    width=700,
                    key=None,
                )
                st.success("Logged In as {}".format(username))
                
                selected1 = option_menu('',
                                      
                                      [
                                        'Health Disease Prediction',
                                       'Crop Disease Prediction',
                                       'Profiles'],
                                      icons=['person-lines-fill','caret-right-square-fill','person-circle'],
                                      default_index=0,orientation="horizontal")
                
                
                if (selected1 == 'Health Disease Prediction'):
                    
                    selected = option_menu('',
                                          
                                          [
                                            'Diabetes Prediction',
                                           'Heart Disease Prediction',
                                           'Parkinsons Prediction',
                                           'Lung Cancer Prediction','Disease Predict By Symptoms',
                                           'PNEUMONIA Disease Predict By Chest-Xray'],
                                          icons=['activity','heart','person-fill','person-fill','person-fill','person-fill'],
                                          default_index=0,orientation="horizontal")
                    
                    
                   
                    # Diabetes Prediction Page
                    if (selected == 'Diabetes Prediction'):
                        
                        # page title
                        
                        img = Image.open("F:/python code/temporary/image/1.webp")
                        st.image(img)
                        st.title('Diabetes Prediction using ML')
                        # getting the input data from the user
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            Pregnancies = st.text_input('Number of Pregnancies')
                            
                        with col2:
                            Glucose = st.text_input('Glucose Level')
                        
                        with col3:
                            BloodPressure = st.text_input('Blood Pressure value')
                        
                        with col1:
                            SkinThickness = st.text_input('Skin Thickness value')
                        
                        with col2:
                            Insulin = st.text_input('Insulin Level')
                        
                        with col3:
                            BMI = st.text_input('BMI value')
                        
                        with col1:
                            DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function value')
                        
                        with col2:
                            Age = st.text_input('Age of the Person')
                        
                        
                        # code for Prediction
                        diab_diagnosis = ''
                        
                        # creating a button for Prediction
                        
                        if st.button('Diabetes Test Result'):
                            diab_prediction = diabetes_model.predict([[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age]])
                            
                            if (diab_prediction[0] == 1):
                              diab_diagnosis = 'The person is diabetic'
                            else:
                              diab_diagnosis = 'The person is not diabetic'
                            
                        st.success(diab_diagnosis)
    
    
    
    
                    # Heart Disease Prediction Page
                    if (selected == 'Heart Disease Prediction'):
                        img = Image.open("F:/python code/temporary/image/2.jpg")
                        st.image(img)
                        # page title
                        st.title('Heart Disease Prediction using ML')
                        
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            age = st.text_input('Age')
                            
                        with col2:
                            sex = st.text_input('Sex')
                            
                        with col3:
                            cp = st.text_input('Chest Pain types')
                            
                        with col1:
                            trestbps = st.text_input('Resting Blood Pressure')
                            
                        with col2:
                            chol = st.text_input('Serum Cholestoral in mg/dl')
                            
                        with col3:
                            fbs = st.text_input('Fasting Blood Sugar > 120 mg/dl')
                            
                        with col1:
                            restecg = st.text_input('Resting Electrocardiographic results')
                            
                        with col2:
                            thalach = st.text_input('Maximum Heart Rate achieved')
                            
                        with col3:
                            exang = st.text_input('Exercise Induced Angina')
                            
                        with col1:
                            oldpeak = st.text_input('ST depression induced by exercise')
                            
                        with col2:
                            slope = st.text_input('Slope of the peak exercise ST segment')
                            
                        with col3:
                            ca = st.text_input('Major vessels colored by flourosopy')
                            
                        with col1:
                            thal = st.text_input('thal: 0 = normal; 1 = fixed defect; 2 = reversable defect')
                            
                            
                         
                         
                        # code for Prediction
                        heart_diagnosis = ''
                        
                        # creating a button for Prediction
                        
                        if st.button('Heart Disease Test Result'):
                            # convert input data to float
                            #input_data = [float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]
                            heart_prediction = heart_disease_model.predict([[age, sex, cp, trestbps, chol, fbs, restecg,thalach,exang,oldpeak,slope,ca,thal]])                          
                            
                            # make prediction using model
                            #heart_prediction = heart_disease_model.predict_proba([[float(age), float(sex), float(cp), float(trestbps), float(chol), float(fbs), float(restecg), float(thalach), float(exang), float(oldpeak), float(slope), float(ca), float(thal)]])
                            
                            #heart_prediction is equal to 1. You can use the any() method for this. 
                            
                            
                            
                            if(heart_prediction[0] == 1):  
                              heart_diagnosis = 'The person is having heart disease'
                            else:
                              heart_diagnosis = 'The person does not have any heart disease'
                            
                        st.success(heart_diagnosis)
                            
                        
    
                    # parkinsons Prediction Page
                    if (selected == 'Parkinsons Prediction'):
                        img = Image.open("F:/python code/temporary/image/3.webp")
                        st.image(img)
                        # page title
                        st.title('Parkinsons Prediction using ML')
                        col1, col2, col3, col4, col5 = st.columns(5)  
                        
                        with col1:
                            fo = st.text_input('MDVP:Fo(Hz)')
                            
                        with col2:
                            fhi = st.text_input('MDVP:Fhi(Hz)')
                            
                        with col3:
                            flo = st.text_input('MDVP:Flo(Hz)')
                            
                        with col4:
                            Jitter_percent = st.text_input('MDVP:Jitter(%)')
                            
                        with col5:
                            Jitter_Abs = st.text_input('MDVP:Jitter(Abs)')
                            
                        with col1:
                            RAP = st.text_input('MDVP:RAP')
                            
                        with col2:
                            PPQ = st.text_input('MDVP:PPQ')
                            
                        with col3:
                            DDP = st.text_input('Jitter:DDP')
                            
                        with col4:
                            Shimmer = st.text_input('MDVP:Shimmer')
                            
                        with col5:
                            Shimmer_dB = st.text_input('MDVP:Shimmer(dB)')
                            
                        with col1:
                            APQ3 = st.text_input('Shimmer:APQ3')
                            
                        with col2:
                            APQ5 = st.text_input('Shimmer:APQ5')
                            
                        with col3:
                            APQ = st.text_input('MDVP:APQ')
                            
                        with col4:
                            DDA = st.text_input('Shimmer:DDA')
                            
                        with col5:
                            NHR = st.text_input('NHR')
                            
                        with col1:
                            HNR = st.text_input('HNR')
                            
                        with col2:
                            RPDE = st.text_input('RPDE')
                            
                        with col3:
                            DFA = st.text_input('DFA')
                            
                        with col4:
                            spread1 = st.text_input('spread1')
                            
                        with col5:
                            spread2 = st.text_input('spread2')
                            
                        with col1:
                            D2 = st.text_input('D2')
                            
                        with col2:
                            PPE = st.text_input('PPE')
                            
                        
                        
                        # code for Prediction
                        parkinsons_diagnosis = ''
                        
                        # creating a button for Prediction    
                        if st.button("Parkinson's Test Result"):
                            parkinsons_prediction = parkinsons_model.predict([[fo, fhi, flo, Jitter_percent, Jitter_Abs, RAP, PPQ,DDP,Shimmer,Shimmer_dB,APQ3,APQ5,APQ,DDA,NHR,HNR,RPDE,DFA,spread1,spread2,D2,PPE]])                          
                            
                            if (parkinsons_prediction[0] == 1):
                              parkinsons_diagnosis = "The person has Parkinson's disease"
                            else:
                              parkinsons_diagnosis = "The person does not have Parkinson's disease"
                            
                        st.success(parkinsons_diagnosis)
    
    
    
                    # lung cancer Prediction Page
                    if (selected == 'Lung Cancer Prediction'):
                        img = Image.open("F:/python code/temporary/image/4.webp")
                        st.image(img)
                        # page title
                        st.title('Lung Cancer Prediction using ML') 
                        st.markdown('Yes = 2 And No = 1')
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            AGE = st.text_input('Age')
                            
                        with col2:
                            SMOKING = st.text_input('SMOKING')
                            
                        with col3:
                            YELLOW_FINGERS = st.text_input('YELLOW_FINGERS')
                            
                        with col1:
                            ANXIETY = st.text_input('ANXIETY')
                            
                        with col2:
                            PEER_PRESSURE = st.text_input('PEER_PRESSURE')
                            
                        with col3:
                            CHRONIC_DISEASE = st.text_input('CHRONIC DISEASE')
                            
                        with col1:
                            FATIGUE = st.text_input('FATIGUE')
                            
                        with col2:
                            ALLERGY = st.text_input('ALLERGY')
                            
                        with col3:
                            WHEEZING = st.text_input('WHEEZING')
                            
                        with col1:
                            ALCOHOL_CONSUMING = st.text_input('ALCOHOL CONSUMING')
                            
                        with col2:
                            COUGHING = st.text_input('COUGHING')
                            
                        with col3:
                            SHORTNESS_OF_BREATH = st.text_input('SHORTNESS OF BREATH')
                            
                        with col1:
                            SWALLOWING_DIFFICULTY = st.text_input('SWALLOWING DIFFICULTY')
                        with col2:
                            CHEST_PAIN = st.text_input('CHEST PAIN')
                        
                        
                        # code for Prediction
                        lung_cancer_diagnosis = ''
                        
                        # creating a button for Prediction    
                        if st.button("lung cancer Test Result"):
                             lung_cancer_prediction = lung_cancer_model.predict([[AGE,SMOKING,YELLOW_FINGERS,ANXIETY,PEER_PRESSURE,CHRONIC_DISEASE,FATIGUE,ALLERGY,WHEEZING,ALCOHOL_CONSUMING,COUGHING,SHORTNESS_OF_BREATH,SWALLOWING_DIFFICULTY,CHEST_PAIN]])                          
                             
                             if(lung_cancer_prediction[0]==2):
                                 
                                 lung_cancer_diagnosis ="The person has lung cancer"
                             else:
                                 
                                 lung_cancer_diagnosis ="The person has no lung cancer"
                                 
                            
                        st.success(lung_cancer_diagnosis)
                        
                    if (selected == 'Disease Predict By Symptoms'):
                            img = Image.open("F:/python code/temporary/image/1.jpg")
                            st.image(img)
                            # Define the symptoms and diseases
                            l1 = ['itching','skin_rash','nodal_skin_eruptions','continuous_sneezing','shivering','chills','joint_pain',
                                  'stomach_pain','acidity','ulcers_on_tongue','muscle_wasting','vomiting','burning_micturition',
                                  'spotting_ urination','fatigue','weight_gain','anxiety','cold_hands_and_feets','mood_swings','weight_loss',
                                  'restlessness','lethargy','patches_in_throat','irregular_sugar_level','cough','high_fever','sunken_eyes',
                                  'breathlessness','sweating','dehydration','indigestion','headache','yellowish_skin','dark_urine','nausea',
                                  'loss_of_appetite','pain_behind_the_eyes','back_pain','constipation','abdominal_pain','diarrhoea',
                                  'mild_fever','yellow_urine','yellowing_of_eyes','acute_liver_failure','fluid_overload',
                                  'swelling_of_stomach','swelled_lymph_nodes','malaise','blurred_and_distorted_vision','phlegm',
                                  'throat_irritation','redness_of_eyes','sinus_pressure','runny_nose','congestion','chest_pain',
                                  'weakness_in_limbs','fast_heart_rate','pain_during_bowel_movements','pain_in_anal_region','bloody_stool',
                                  'irritation_in_anus','neck_pain','dizziness','cramps','bruising','obesity','swollen_legs',
                                  'swollen_blood_vessels','puffy_face_and_eyes','enlarged_thyroid','brittle_nails','swollen_extremeties',
                                  'excessive_hunger','extra_marital_contacts','drying_and_tingling_lips','slurred_speech','knee_pain',
                                  'hip_joint_pain','muscle_weakness','stiff_neck','swelling_joints','movement_stiffness','spinning_movements',
                                  'loss_of_balance','unsteadiness','weakness_of_one_body_side','loss_of_smell','bladder_discomfort',
                                  'foul_smell_of urine','continuous_feel_of_urine','passage_of_gases','internal_itching','toxic_look_(typhos)',
                                  'depression','irritability','muscle_pain','altered_sensorium','red_spots_over_body','belly_pain',
                                  'abnormal_menstruation','dischromic _patches','watering_from_eyes','increased_appetite','polyuria',
                                  'family_history','mucoid_sputum','rusty_sputum','lack_of_concentration','visual_disturbances',
                                  'receiving_blood_transfusion','receiving_unsterile_injections','coma','stomach_bleeding',
                                  'distention_of_abdomen','history_of_alcohol_consumption','fluid_overload.1','blood_in_sputum',
                                  'prominent_veins_on_calf','palpitations','painful_walking','pus_filled_pimples','blackheads',
                                  'scurring','skin_peeling','silver_like_dusting','small_dents_in_nails','inflammatory_nails','blister',
                                  'red_sore_around_nose','yellow_crust_ooze']
                            
                            disease = ['Fungal infection', 'Allergy', 'GERD', 'Chronic cholestasis', 'Drug Reaction', 'Peptic ulcer disease',
                                       'AIDS', 'Diabetes', 'Gastroenteritis', 'Bronchial Asthma', 'Hypertension', 'Migraine',
                                       'Cervical spondylosis', 'Paralysis (brain hemorrhage)', 'Jaundice', 'Malaria', 'Chicken pox', 'Dengue',
                                       'Typhoid', 'hepatitis A', 'Hepatitis B', 'Hepatitis C', 'Hepatitis D', 'Hepatitis E',
                                       'Alcoholic hepatitis', 'Tuberculosis', 'Common Cold', 'Pneumonia', 'Dimorphic hemorrhoids(piles)',
                                       'Heart attack', 'Varicose veins', 'Hypothyroidism', 'Hyperthyroidism', 'Hypoglycemia', 'Osteoarthritis',
                                       'Arthritis', '(vertigo) Paroxysmal Positional Vertigo', 'Acne', 'Urinary tract infection', 'Psoriasis',
                                       'Impetigo']
                            
                            l2 = [0] * len(l1)

                          
                            st.title("Multiple Disease Prediction Based On Symptoms")
                            
                        
                            
                            name = st.text_input("Name of the Patient")
                            OPTIONS = l1
                            symptoms = [st.selectbox(f"Symptom {i}",OPTIONS) for i in range(1, 6)]
                            
                            if st.button("Predict Disease"):
                                # Convert symptoms to binary input vector
                                for i, symptom in enumerate(symptoms):
                                    if symptom in l1:
                                        l2[l1.index(symptom)] = 1
                            
                                input_test = [l2]
                            
                                # Decision Tree
                                
                                y_pred = model0.predict(input_test)
                                #predicted = y_pred[0]
                                
                                st.subheader(f"The person suffering from {y_pred[0]} disease.")
                        
                        
                    if selected == 'PNEUMONIA Disease Predict By Chest-Xray':
                        img = Image.open("F:/python code/temporary/image/10.jpg")
                        st.image(img)
                        
                        CLASS_NAMES = ('NORMAL', 'PNEUMONIA')
                        
                        # Setting Title of App
                        st.title("PNEUMONIA Disease Predict By Chest-Xray")
                        st.markdown("Upload an image of the Chest-Xray")
                        
                        # Uploading the dog image
                        plant_image = st.file_uploader("Choose an image...", type = "jpeg")
                        submit = st.button('predict Disease')
                        
                        # On predict button click
                        if submit:
                            if plant_image is not None:
                                # Convert the file to an opencv image.
                                file_bytes = np.asarray (bytearray(plant_image.read()), dtype = np.uint8)
                                opencv_image = cv2.imdecode(file_bytes, 1)
                                
                                # Displaying the image
                                st.image(opencv_image, channels="BGR")
                                st.write(opencv_image.shape)
                                
                                # Resizing the image
                                opencv_image = cv2.resize(opencv_image, (256, 256))
                                
                                # Convert image to 4 Dimension
                                opencv_image.shape = (1, 256, 256, 3)
                                
                                #Make Prediction
                                Y_pred = model4.predict(opencv_image)
                                result = CLASS_NAMES[np.argmax(Y_pred)]
                                st.subheader(str("This is image shows a "+result +" Chest-Xray")) 
                                
                                if result=="PNEUMONIA":
                                    
                                    st.subheader("Treatment:")
                                    st.markdown(" Pneumonia can be serious so it's important to get treatment quickly. The main treatment for pneumonia is antibiotics.The best initial antibiotic choice is thought to be a macrolide.You should also rest and drink plenty of water. If you're diagnosed with bacterial pneumonia, your doctor should give you antibiotics to take within four hours.")
                                    
    
                            
                            
                            
                        
                if selected1 == 'Crop Disease Prediction':
                    selected2 = option_menu('',
                                          
                                          [
                                            'Rice Disease Prediction',
                                           'Tomato Disease Prediction',
                                           'Wheat Disease Prediction',
                                           'Apple Disease Prediction'],
                                          icons=['caret-right-square-fill','caret-right-square-fill','caret-right-square-fill','caret-right-square-fill'],
                                          default_index=0,orientation="horizontal")
                    if selected2 == 'Rice Disease Prediction':
                        img = Image.open("F:/python code/temporary/image/rice.jpg")
                        st.image(img)
                        img = Image.open("F:/python code/temporary/image/farmer.jpg")
                        st.image(img)
                        
                        
                        
                        CLASS_NAMES = ('Rice-Bacterial leaf blight', 'Rice-Brown spot', 'Rice-Leaf smut')
                        
                        # Title and Description
                        st.title('Rice Diesease Detection')
                        st.write("Just Upload your rice's Leaf Image and get predictions if the plant is healthy or not")
                        # Uploading the dog image
                        plant_image = st.file_uploader("Choose an image...", type = "jpg")
                        submit = st.button('predict Disease')
                        
                        # On predict button click
                        if submit:
                            if plant_image is not None:
                                # Convert the file to an opencv image.
                                file_bytes = np.asarray (bytearray(plant_image.read()), dtype = np.uint8)
                                opencv_image = cv2.imdecode(file_bytes, 1)
                                
                                # Displaying the image
                                st.image(opencv_image, channels="BGR")
                                st.write(opencv_image.shape)
                                
                                # Resizing the image
                                opencv_image = cv2.resize(opencv_image, (256, 256))
                                
                                # Convert image to 4 Dimension
                                opencv_image.shape = (1, 256, 256, 3)
                                
                                #Make Prediction
                                Y_pred = model1.predict(opencv_image)
                                result = CLASS_NAMES[np.argmax(Y_pred)]
                                st.subheader(str("This is "+result.split('-')[0]+ " leaf with " +  result.split('-')[1]))
                                
                                if result=="Rice-Bacterial leaf blight":
                                    st.subheader("Rice-Bacterial Leaf Blight (caused by bacteria):")
                                    st.subheader("Treatment:")
                                    st.markdown(" Apply copper-based bactericides (e.g., Copper Hydroxide) early in the disease.")
                                    st.subheader("Prevention:")
                                    st.markdown(" Plant resistant varieties, practice proper water management, and avoid consecutive rice planting.")
                                    
                                if result=="Rice-Brown spot":
                                    st.subheader("Rice-Brown Spot (caused by a fungus):")
                                    st.subheader("Treatment:")
                                    st.markdown(" Apply appropriate fungicides (e.g., Trifloxystrobin or Propiconazole) when spotting begins.")
                                    st.subheader("Prevention:")
                                    st.markdown(" Plant resistant varieties, maintain proper spacing, and remove affected leaves.")
                                    
                                if result=="Rice-Leaf smut":
                                    st.subheader("Rice-Leaf Smut (caused by a fungus):")
                                    st.subheader("Treatment:")
                                    st.markdown("Treatment: Treat seeds with suitable fungicides (e.g., Thiram or Carbendazim), or bactericides designed for seeds.")
                                    st.subheader("Prevention:")
                                    st.markdown(" Plant resistant varieties, practice crop rotation, and maintain field sanitation.")
                                        
                                    
                                    
                        
                            
                    if selected2 == 'Tomato Disease Prediction':
                        img = Image.open("F:/python code/temporary/image/7.jpg")
                        st.image(img)
                        
                        CLASS_NAMES = ('Tomato - Early_blight', 'Tomato - Healthy', 'Tomato - Septoria_leaf_spot', 'Tomato - Tomato_Yellow_Leaf_Curl_Virus')
                        
                        # Setting Title of App
                        st.title("Tomato Disease Detection")
                        st.markdown("Upload an image of the tomato leaf")
                        
                        # Uploading the dog image
                        plant_image = st.file_uploader("Choose an image...", type = "jpg")
                        submit = st.button('predict Disease')
                        
                        # On predict button click
                        if submit:
                            if plant_image is not None:
                                # Convert the file to an opencv image.
                                file_bytes = np.asarray (bytearray(plant_image.read()), dtype = np.uint8)
                                opencv_image = cv2.imdecode(file_bytes, 1)
                                
                                # Displaying the image
                                st.image(opencv_image, channels="BGR")
                                st.write(opencv_image.shape)
                                
                                # Resizing the image
                                opencv_image = cv2.resize(opencv_image, (256, 256))
                                
                                # Convert image to 4 Dimension
                                opencv_image.shape = (1, 256, 256, 3)
                                
                                #Make Prediction
                                Y_pred = model.predict(opencv_image)
                                result = CLASS_NAMES[np.argmax(Y_pred)]
                                st.subheader(str("This is "+result.split('-')[0]+ " leaf with " +  result.split('-')[1]))
                                
                                if result=="Tomato - Early_blight":
                                    
                                    st.subheader("Treatment:")
                                    st.markdown(" Use a copper-based fungicide or one with the active ingredient chlorothalonil to treat early blight. Follow the product label for proper usage.")
                                    st.subheader("Prevention:")
                                    st.markdown(" Practice crop rotation and avoid planting tomatoes in the same area for at least two years. No specific medicine is involved in this prevention method.")
                                    
                                if result=="Tomato - Septoria_leaf_spot":
                                   
                                    st.subheader("Treatment:")
                                    st.markdown("  Apply a fungicide like chlorothalonil or copper-based fungicides when you notice leaf spots. Follow label instructions carefully.")
                                    st.subheader("Prevention:")
                                    st.markdown(" Prune lower branches and leaves for better airflow. No specific medicine is involved in this prevention method.")
                                    
                                if result=="Tomato - Tomato_Yellow_Leaf_Curl_Virus":
                                    
                                    st.subheader("Treatment:")
                                    st.markdown("Use insecticides like neem oil or insecticidal soap to control whiteflies, which transmit TYLCV. Follow product instructions for proper usage.")
                                    st.subheader("Prevention:")
                                    st.markdown(" Choose tomato varieties bred for resistance to Tomato Yellow Leaf Curl Virus (TYLCV). No specific medicine is involved in this prevention method.")
                                
                                
                                
                                
                    
                    
                    if selected2 == 'Wheat Disease Prediction':
                        img = Image.open("F:/python code/temporary/image/8.jpg")
                        st.image(img)
                        
                        CLASS_NAMES = ('Wheat___Brown_Rust', 'Wheat___Healthy', 'Wheat___Yellow_Rust')
                        
                        # Setting Title of App
                        st.title("Wheat Disease Detection")
                        st.markdown("Upload an image of the tomato leaf")
                        
                        # Uploading the dog image
                        plant_image = st.file_uploader("Choose an image...", type = "jpg")
                        submit = st.button('predict Disease')
                        
                        # On predict button click
                        if submit:
                            if plant_image is not None:
                                # Convert the file to an opencv image.
                                file_bytes = np.asarray (bytearray(plant_image.read()), dtype = np.uint8)
                                opencv_image = cv2.imdecode(file_bytes, 1)
                                
                                # Displaying the image
                                st.image(opencv_image, channels="BGR")
                                st.write(opencv_image.shape)
                                
                                # Resizing the image
                                opencv_image = cv2.resize(opencv_image, (256, 256))
                                
                                # Convert image to 4 Dimension
                                opencv_image.shape = (1, 256, 256, 3)
                                
                                #Make Prediction
                                Y_pred = model2.predict(opencv_image)
                                result = CLASS_NAMES[np.argmax(Y_pred)]
                                st.subheader(str("This is "+result.split('___')[0]+ " leaf with " +  result.split('___')[1])) 
                                
                                if result=="Wheat___Brown_Rust":
                                    
                                    st.subheader("Treatment:")
                                    st.markdown(" Apply fungicides like Triadimefon or Propiconazole when you observe early signs of brown rust on wheat plants. Follow recommended application rates and timings.")
                                    st.subheader("Prevention:")
                                    st.markdown("Choose wheat varieties that are resistant to brown rust. Planting resistant varieties can significantly reduce the risk of infection.")
                                    
                                
                                if result=="Wheat___Yellow_Rust":
                                    
                                    st.subheader("Treatment:")
                                    st.markdown("Apply fungicides like Tebuconazole or Flutriafol if you notice yellow rust symptoms on wheat leaves. Early treatment is crucial for effective control.")
                                    st.subheader("Prevention:")
                                    st.markdown(" Opt for wheat varieties that have built-in resistance to yellow rust. Resistant varieties are a proactive way to prevent the disease.")
                                
                                
                                
                                
                                
                    
                    
                    if selected2 == 'Apple Disease Prediction':
                        img = Image.open("F:/python code/temporary/image/9.jpg")
                        st.image(img)
                        
                        CLASS_NAMES = ('Apple-apple_scab', 'Apple-black_rot', 'Apple-cedar_apple_rust', 'Apple-healthy')
                        
                        # Setting Title of App
                        st.title("Apple Leaf Disease Detection")
                        st.markdown("Upload an image of the tomato leaf")
                        
                        # Uploading the dog image
                        plant_image = st.file_uploader("Choose an image...", type = "jpg")
                        submit = st.button('predict Disease')
                        
                        # On predict button click
                        if submit:
                            if plant_image is not None:
                                # Convert the file to an opencv image.
                                file_bytes = np.asarray (bytearray(plant_image.read()), dtype = np.uint8)
                                opencv_image = cv2.imdecode(file_bytes, 1)
                                
                                # Displaying the image
                                st.image(opencv_image, channels="BGR")
                                st.write(opencv_image.shape)
                                
                                # Resizing the image
                                opencv_image = cv2.resize(opencv_image, (256, 256))
                                
                                # Convert image to 4 Dimension
                                opencv_image.shape = (1, 256, 256, 3)
                                
                                #Make Prediction
                                Y_pred = model3.predict(opencv_image)
                                result = CLASS_NAMES[np.argmax(Y_pred)]
                                
                                st.subheader(str("This is "+result.split('-')[0]+ " leaf with " +  result.split('-')[1]))
                                
                                if result=="Apple-apple_scab":
                                    
                                    st.subheader("Treatment:")
                                    st.markdown(" Apply fungicides such as Mancozeb or Chlorothalonil during the spring before apple tree buds open and continue at regular intervals throughout the growing season. These fungicides help prevent and control apple scab.")
                                    st.subheader("Prevention:")
                                    st.markdown(" Prune and remove infected leaves and branches from the tree. Dispose of these materials away from the orchard to reduce the source of infection for the next season.")
                                    
                                if result=="Apple-black_rot":
                                   
                                    st.subheader("Treatment:")
                                    st.markdown(" Apply fungicides like Captan or Thiophanate-methyl when apple trees are in the green-tip stage and continue at regular intervals through the growing season. These fungicides help prevent and control black rot.")
                                    st.subheader("Prevention:")
                                    st.markdown(" Remove and destroy any fallen or mummified apples. These can harbor the black rot fungus and serve as a source of infection.")
                                    
                                if result=="Apple-cedar_apple_rust":
                                    
                                    st.subheader("Treatment:")
                                    st.markdown("Apply fungicides such as Myclobutanil or Tebuconazole during the spring when cedar apple rust spores are active. This helps protect apple trees from infection.")
                                    st.subheader("Prevention:")
                                    st.markdown(" If possible, remove cedar trees near your apple orchard because they serve as an alternate host for cedar apple rust. Reducing cedar trees can decrease disease pressure.")
                                
                                
                                
                                
                        
                        
                elif selected1 == "Profiles":
                    
                    def load_lottiefile(filepath: str):
                        with open(filepath, "r") as f:
                            return json.load(f)
                                                                           
                    
                    lottie_profile = load_lottiefile("F:/python code/temporary/lotti_files/profile 2.json")  # replace link to local lottie file
                    
                    
                    
                    st_lottie(
                        lottie_profile,
                        speed=1,
                        reverse=False,
                        loop=True,
                        quality="low",
                        height=600,
                        width=800,
                        key=None,
                    )
                    st.subheader("User Profiles")
                    #user_result = view_all_users()
                    user_result = view_user_data(username)
                    clean_db = pd.DataFrame(user_result,columns=["Username","Password"])
                    st.dataframe(clean_db)
            else:
                st.warning("Incorrect Username/Password")





    elif choice == "SignUp":
        def load_lottiefile(filepath: str):
            with open(filepath, "r") as f:
                return json.load(f)
        
        
        
            
        
        lottie_signup = load_lottiefile("F:/python code/temporary/lotti_files/signup.json")  # replace link to local lottie file
        
        
        
        st_lottie(
            lottie_signup,
            speed=1,
            reverse=False,
            loop=True,
            quality="low",
            height=400,
            width=700,
            key=None,
        )
        st.subheader("Create New Account")
        new_user = st.text_input("Username")
        new_password = st.text_input("Password",type='password')

        if st.button("Signup"):
            create_usertable()
            add_userdata(new_user,make_hashes(new_password))
            st.success("You have successfully created a valid Account")
            st.info("Go to Login Menu to login")



if __name__ == '__main__':
    main()