import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from tensorflow.keras.models import load_model

model = load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file)


st.title("Customer Churn Model")

##User Input
geography = st.selectbox('Enter your location', one_hot_encoder_geo.categories_[0])
gender = st.radio('Select Gender', label_encoder_gender.classes_)
age = st.slider("Enter your age", 18, 92)
balance = st.number_input("Enter your balance")
estimated_salary = st.number_input('Estimated Salary')
credit_score = st.number_input("Enter your credit score")
tenure = st.slider('Tenure', 0, 10)
num_of_products = st.radio("Number of Products", [1, 2, 3, 4])
has_cr_card = st.selectbox("Do you have a credit card ?", [0, 1])
is_active_member = st.radio("Are you an active member?", [0, 1])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

##onehotencoding the geography
geo_encoded = one_hot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns = one_hot_encoder_geo.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

##scaling input data
input_data_scaled = scaler.transform(input_data)


##prediction
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.write(f'Churn Probability:{prediction_probability : .2f}')

if prediction_probability > 0.5 :
    st.warning("The customer will likely churn")
else:
    st.success("The customer will likely not churn")
