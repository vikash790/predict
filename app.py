import streamlit as st
import pickle
import numpy as np

# Load the model and data
pipe = pickle.load(open('pipe.pkl', 'rb'))
df = pickle.load(open('df.pkl', 'rb'))

st.title("Laptop Price Predictor")

# Get user input
company = st.selectbox('Brand', df['Company'].unique())
type_laptop = st.selectbox('Type', df['TypeName'].unique())
ram = st.selectbox('RAM(in GB)', [2, 4, 6, 8, 12, 16, 24, 32, 64])
weight = st.number_input('Weight of the Laptop')
touchscreen = st.selectbox('Touchscreen', ['No', 'Yes'])
ips = st.selectbox('IPS', ['No', 'Yes'])
screen_size = st.slider('Screen size in inches', 10.0, 18.0, 13.0)
resolution = st.selectbox('Screen Resolution', ['1920x1080', '1366x768', '1600x900', '3840x2160', '3200x1800', '2880x1800', '2560x1600', '2560x1440', '2304x1440'])
cpu = st.selectbox('CPU', df['Cpu brand'].unique())
hdd = st.selectbox('HDD(in GB)', [0, 128, 256, 512, 1024, 2048])
ssd = st.selectbox('SSD(in GB)', [0, 8, 128, 256, 512, 1024])
gpu = st.selectbox('GPU', df['Gpu brand'].unique())
os = st.selectbox('OS', df['os'].unique())

if st.button('Predict Price'):
    # Convert categorical inputs to numeric format (as required)
    touchscreen = 1 if touchscreen == 'Yes' else 0
    ips = 1 if ips == 'Yes' else 0

    # Extract screen resolution values
    X_res = int(resolution.split('x')[0])
    Y_res = int(resolution.split('x')[1])

    # Calculate PPI (Pixels Per Inch)
    ppi = ((X_res**2) + (Y_res**2))**0.5 / screen_size

    # Create the input array (let the pipeline handle categorical encoding)
    query = np.array([company, type_laptop, ram, weight, touchscreen, ips, ppi, cpu, hdd, ssd, gpu, os], dtype=object)

    # Reshape the query to match the model's expected input shape
    query = query.reshape(1, -1)

    try:
        # Let the pipeline handle transformations and make a prediction
        predicted_price = pipe.predict(query)

        # Display the result
        st.title(f"The predicted price of this configuration is ₹ {int(np.exp(predicted_price[0]))}")
    except ValueError as e:
        st.error(f"Error in prediction: {e}")
