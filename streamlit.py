import streamlit as st
st.set_page_config(page_title="Covid19 Detection Tool", page_icon="covid19.jpeg", layout='centered', initial_sidebar_state='auto')
import os
import time
import cv2
from PIL import Image,ImageEnhance
import numpy as np 
import tensorflow as tf


def main():
    image='covid2.png'
    st.image(image,width=None)
    """Simple Tool for Covid-19 Detection from Chest X-Ray"""
    html_templ = """
    <h1 style="color:black">Covid-19 Detection Tool</h1>
    </div>
    """
    image2='facts.jpg'
    st.imgae(image,width=None)
    st.markdown(html_templ,unsafe_allow_html=True)
    st.write("A simple proposal for Covid-19 Diagnosis powered by Deep Learning and Streamlit")
    st.sidebar.image("covid.png",width=300)
    image_file = st.sidebar.file_uploader("Upload an X-Ray Image (jpg, png or jpeg)",type=['jpg','png','jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        if st.sidebar.button("Image Preview"):
            st.sidebar.image(our_image,width=300)
        activities = ["Image Enhancement","Diagnosis", "Disclaimer and Info"]
        choice = st.sidebar.selectbox("Select Activty",activities)
        if choice == 'Image Enhancement':
            st.subheader("Image Enhancement")
            enhance_type = st.sidebar.radio("Enhance Type",["Original","Contrast","Brightness"])
            if enhance_type == 'Contrast':
                c_rate = st.slider("Contrast",0.5,5.0)
                enhancer = ImageEnhance.Contrast(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output,use_column_width=True)
            elif enhance_type == 'Brightness':
                c_rate = st.slider("Brightness",0.5,5.0)
                enhancer = ImageEnhance.Brightness(our_image)
                img_output = enhancer.enhance(c_rate)
                st.image(img_output,width=600,use_column_width=True)
            else:
                st.text("Original Image")
                st.image(our_image,width=600,use_column_width=True)
        elif choice == 'Diagnosis':
            if st.sidebar.button("Diagnosis"):
                # Image to Black and White
                new_img = np.array(our_image.convert('RGB')) 
                #our image is binary we have to convert it in array
                new_img = cv2.cvtColor(new_img,1) 
                st.text("Chest X-Ray")
                st.image(new_img,use_column_width=True)
                # PX-Ray (Image) Preprocessing
                IMG_SIZE = (224,224)
                img = cv2.resize(new_img,IMG_SIZE)
                img=img/255.
                
                X_Ray = img.reshape(1,224,224,3)
                # Pre-Trained CNN Model Importing
                model = tf.keras.models.load_model("model.h5")
                # Diagnosis (Prevision=Binary Classification)
                diagnosis = model.predict_classes(X_Ray)
                my_bar = st.sidebar.progress(0)
                for percent_complete in range(100):
                    time.sleep(0.05)
                    my_bar.progress(percent_complete + 1)
                # Diagnosis Cases: Covid=0, No-Covid=1
                if diagnosis == 0:
                    st.sidebar.error("DIAGNOSIS: COVID-19 ")
                else:
                    st.sidebar.success("DIAGNOSIS: NO-COVID-19")
                st.warning("This Web App is just a DEMO about Artificial Neural Networks so there is no clinical value in its diagnosis and the author is not a Doctor!")
        else:
            st.subheader("Disclaimer and Info")
            st.subheader("Disclaimer")
            st.write("**This Tool is just a DEMO about Artificial Neural Networks so there is no clinical value in its diagnosis and the author is not a Doctor!**")
            st.write("**Please don't take the diagnosis outcome seriously and NEVER consider it valid!!!**")
            st.subheader("Info")
            st.write("This Tool gets inspiration from the following works:")
            st.write("- [Detecting COVID-19 in X-ray images with Keras, TensorFlow, and Deep Learning](https://www.pyimagesearch.com/2020/03/16/detecting-covid-19-in-x-ray-images-with-keras-tensorflow-and-deep-learning/)") 
            st.write("- [Covid 19 Detection Using CNN](https://www.youtube.com/watch?v=nHQDDAAzIsI)")
            st.write("- [Covid 19 Detector tool](https://github.com/rosariomoscato/covid19-detection-tool)")
            
        if st.sidebar.button("About the Author"):
            st.sidebar.subheader("Covid-19 Detection Tool")
        st.sidebar.text("All Rights Reserved (2020)")


if __name__ == '__main__':
    main()
 

    
    





    
