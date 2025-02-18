import streamlit as st
import pandas as pd
import numpy as np
import pickle
import base64
import time

encoder = pickle.load(open('encoder', 'rb'))
scaler = pickle.load(open('scaler', 'rb'))
model = pickle.load(open('xgb_vanilla', 'rb'))

st.write("""
         # Welcome to the Ultimate Car Price Predictor Machine!
         
         Let's play with the widgets in the slidebar 
         and then click on the button below to get the price of your dream car!
         """)

st.sidebar.header('This is where your inputs go!')

def user_input_features():
    make_model = st.sidebar.selectbox('Make and Model', ('Audi A1', 'Audi A2', 'Audi A3', 'Opel Astra', 'Opel Corsa', 'Opel Insignia',
 'Renault Clio', 'Renault Duster', 'Renault Espace'))
    body_type = st.sidebar.selectbox('Body Type', ('Sedans', 'Station wagon', 'Compact', 'Coupe', 'Van', 'Off-Road', 'Convertible',
 'Transporter'))
    km = st.sidebar.slider('Kilometers', 0, 300000, 125000)
    Type = st.sidebar.selectbox('Type', ('Used', "Employee's car", 'New', 'Demonstration', 'Pre-registered'))
    Fuel = st.sidebar.selectbox('Fuel', ('Diesel', 'Benzine', 'LPG/CNG', 'Electric'))
    Age = st.sidebar.slider('Age', 0, 3, 1)
    Previous_Owners = st.sidebar.slider('Previous Owners', 0, 4, 1)
    hp_kW = st.sidebar.slider('Horsepower', 40, 300, 100)
    Paint_Type = st.sidebar.radio('Paint Type', ('Metallic', 'Uni/basic', 'Perl effect'))
    Upholstery_type = st.sidebar.radio('Upholstery Type', ('Cloth', 'Part/Full Leather'))
    Gearing_Type = st.sidebar.radio('Gearing Type', ('Automatic', 'Manual', 'Semi-automatic'))
    Weight_kg = st.sidebar.slider('Weight', 600, 2500, 1250)
    Drive_chain = st.sidebar.radio('Drive Chain', ('front', 'rear', '4WD'))
    cons_comb = st.sidebar.slider('Consumption', 3.0, 9.5, 5.5)

    data = {'make_model': make_model, 'body_type': body_type, 'km': km, 
            'Type': Type, 'Fuel': Fuel, 'age': Age, 'Previous_Owners': Previous_Owners, 
            'hp_kW': hp_kW, 'Paint_Type': Paint_Type, 'Upholstery_type': Upholstery_type, 
            'Gearing_Type': Gearing_Type, 'Weight_kg': Weight_kg, 
            'Drive_chain': Drive_chain, 'cons_comb': cons_comb}
    
    features = pd.DataFrame(data, index=[0])
    return features

df = user_input_features()

file_ = open("aladin.gif", "rb")
contents = file_.read()
data_url = base64.b64encode(contents).decode("utf-8")
file_.close()

st.markdown(
    f'<img src="data:image/gif;base64,{data_url}" alt="the jinni gif">',
    unsafe_allow_html=True
)

st.markdown('###')

st.markdown('## Are you sure these are the features?')
st.table(df)

encoded_df = encoder.transform(df)
scaled_df = scaler.transform(encoded_df)
prediction = model.predict(scaled_df)

st.markdown('###')

st.subheader('Now, you mortal, click the button below!')

if st.button('Me, me, me!'):
    time.sleep(1)
    st.write('### Hmmmmmm...')
    time.sleep(3)
    st.write('### Magic takes time...')
    time.sleep(2)
    st.success(f'## The price of your dream car is: €{round(prediction[0])}')
    st.balloons()