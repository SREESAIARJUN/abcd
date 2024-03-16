import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np

# Load the model and weights
model_path = "Pneumonia_detection_using_CNN.h5"
weights_path = "Pneumonia_detection_using_CNN.weights.h5"

model = load_model(model_path)
model.load_weights(weights_path)

# Streamlit app
st.title('Pneumonia Detection App')

# File uploader for image input
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    st.image(uploaded_file, caption='Uploaded Image', use_column_width=True)
    
    # Check if predict button is clicked
    if st.button('Predict'):
        # Load and preprocess the image
        img = image.load_img(uploaded_file, target_size=(224, 224))
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0)

        # Make a prediction
        prediction = model.predict(img_array)

        # Display the prediction with confidence level in large highlighted text
        class_names = ['Normal ', 'Pneumonia']
        predicted_class = class_names[np.argmax(prediction)]
        confidence_level = np.max(prediction) * 100  # Convert probability to percentage
        
        # Set text color based on prediction
        if predicted_class == 'Normal':
            text_color = 'green'
        else:
            text_color = 'red'
        
        # Display prediction and confidence level in large highlighted text
        st.markdown(f'<p style="font-size:32px; color:{text_color};">Prediction: {predicted_class}</p>', unsafe_allow_html=True)
        st.markdown(f'<p style="font-size:32px; color:{text_color};">Possibility of {predicted_class}: {confidence_level:.2f}%</p>', unsafe_allow_html=True)