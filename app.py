import Streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import pickle
import sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoding
from tensorflow.keras.model import load_model

model = tf.load_model('model.h5')

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_encoder_geo.pkl', 'rb') as file:
    one_hot_encoder_geo = pickle.load(file)

with open('scaler.pkl', 'rb') as file:
    scalar = pickle.load(file)





