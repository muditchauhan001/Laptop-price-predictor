import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load your machine learning model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")


company = st.selectbox('Brand', df['Company'].unique())

type_ = st.selectbox('Type', df['TypeName'].unique())

ram = st.selectbox('RAM (in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])

weight = st.number_input('Enter the weight of the laptop (in kg):')

touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])

ips = st.selectbox('IPS', ['No', 'Yes'])

screen_size = st.number_input('Screen size (in inches)')

resolution = st.selectbox('Screen Resolution',
                          ['1920x1080', '1366x768', '1600x900', '3840x2160',
                           '3200x1800', '2560x1600', '2560x1440'])

cpu = st.selectbox('CPU', df['Cpu brand'].unique())

hdd = st.selectbox('HDD (in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD (in GB)', [0, 8, 128, 256, 512, 1024])

gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())


if st.button('Predict Price'):
    ppi = None
    if touchscreen == 'Yes':
        touchscreen = 1
    else:
        touchscreen = 0

    if ips == 'Yes':
        ips = 1
    else:
        ips = 0

    X_res, Y_res = map(int, resolution.split('x'))
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    query = pd.DataFrame({
        'Company': [company],
        'TypeName': [type_],
        'Ram': [ram],
        'Weight': [weight],
        'Touchscreen': [touchscreen],
        'Ips': [ips],
        'ppi': [ppi],
        'Cpu brand': [cpu],
        'HDD': [hdd],
        'SSD': [ssd],
        'Gpu brand': [gpu],
        'os': [os]
    })

    try:
        prediction = pipe.predict(query)
        st.title(f"Predicted Price: Rs.{int(np.exp(prediction[0]))}")
    except Exception as e:
        st.error(f"Error in prediction: {e}")
