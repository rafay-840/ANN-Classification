import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import tensorflow as tf
import pickle
from keras.models import load_model


model = tf.keras.models.load_model('ann_model.h5')


with open('onehot_encoder_geo.pkl', 'rb') as f:
    encoder = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder = pickle.load(f)


## streamlit app
st.title('Churn Prediction App')

geography = st.selectbox('Geography',encoder.categories_[0])
gender = st.selectbox('Gender',label_encoder.classes_)
age = st.number_input('Age', min_value=18, max_value=100, value=30)
balance = st.number_input('Balance', min_value=0.0, value=0.0)
credit_score = st.number_input('Credit Score', min_value=300, max_value=850, value=600)
estimated_salary = st.number_input('Estimated Salary', min_value=0.0, value=0.0)
tenure = st.number_input('Tenure', min_value=0, max_value=10, value=0)
num_of_products = st.number_input('Number of Products', min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox('Has Credit Card', ['Yes', 'No'])
is_active_member = st.selectbox('Is Active Member', ['Yes', 'No'])

input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [1 if has_cr_card == 'Yes' else 0],
    'IsActiveMember': [1 if is_active_member == 'Yes' else 0],
    'EstimatedSalary': [estimated_salary],
})

# One-hot encode the 'Geography' feature
geography_encoded = encoder.transform([[geography]]).toarray()
geography_encoded_df = pd.DataFrame(geography_encoded, columns=encoder.get_feature_names_out(['Geography']))

input_data = pd.concat([input_data.reset_index(drop=True), geography_encoded_df], axis=1)

#scale the input data
input_df_scaled = scaler.transform(input_data)


#predict the output using the trained model
prediction = model.predict(input_df_scaled)
prediction_proba = prediction[0][0]
if prediction_proba > 0.5:
    st.write(f"The customer is likely to leave the bank with a probability of {prediction_proba:.2f}.")
else:
    st.write(f"The customer is likely to stay with the bank with a probability of {1 - prediction_proba:.2f}.")