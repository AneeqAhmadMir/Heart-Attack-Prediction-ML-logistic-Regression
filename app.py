import numpy as np
import pandas as pd
import pickle as pk
import streamlit as st

model = pk.load(open('model.pkl', 'rb'))

st.header('heart attack Prediction ML Model')

heart_data = pd.read_csv('heart.csv')

# Age = st.selectbox('Select your Age',heart_data['Age'].unique())
# Age = st.number_input('Enter your Age', min_value=0, max_value=100, step=1)
Age = st.slider('Select your Age', min_value=1, max_value=120, value=30)
Sex = st.selectbox('Select your gender',heart_data['Sex'].unique())
ChestPainType = st.selectbox('select Chest Pain Type',heart_data['ChestPainType'].unique())
RestingBP = st.slider('Select Resting BP',60,220)
Cholesterol = st.slider('Select Cholesterol',1,700)
FastingBS = st.selectbox('select Fasting BS ',heart_data['FastingBS'].unique())
RestingECG = st.selectbox('select Resting ECG',heart_data['RestingECG'].unique())
MaxHR = st.slider('select MaxHR	',40,220)
ExerciseAngina = st.selectbox('select Exercise Angina',heart_data['ExerciseAngina'].unique())
Oldpeak = st.selectbox('Oldpeak',heart_data['Oldpeak'].unique())
ST_Slope = st.selectbox('ST_Slope',heart_data['ST_Slope'].unique())


if st.button("Predict"):
    input_data_model = pd.DataFrame(
    [[Age,Sex,ChestPainType,RestingBP,Cholesterol,FastingBS,RestingECG,MaxHR,ExerciseAngina,Oldpeak,ST_Slope]],
    columns=['Age','Sex','ChestPainType','RestingBP','Cholesterol','FastingBS','RestingECG','MaxHR','ExerciseAngina','Oldpeak','ST_Slope'])

    input_data_model['Sex'].replace(['M','F'],[0,1],inplace = True)
    input_data_model['ChestPainType'].replace(['ATA','NAP','ASY','TA'],[0,1,2,3],inplace = True)
    input_data_model['RestingECG'].replace(['Normal','ST','LVH'],[0,1,2],inplace = True)
    input_data_model['ExerciseAngina'].replace(['N' ,'Y'],[0,1],inplace = True)
    input_data_model['ST_Slope'].replace(['Up','Flat','Down'],[0,1,2],inplace = True)


    prediction = model.predict(input_data_model) 

    if prediction == 1:
        st.warning('⚠️ High risk of heart attack! Please consult a doctor immediately.')
    else:
        st.success('✅ Low risk of heart attack! Keep maintaining a healthy lifestyle.')