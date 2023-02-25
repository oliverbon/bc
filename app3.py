import streamlit as st
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import numpy as np
from tensorflow import keras
import tensorflow as tf

HEIGHT = 512 # pixels
WIDTH = 512 # pixels
CLASS_NAMES = ["cancer", "no cancer"]

def main():
    st.write("# Deteccion de Cancer")
    with st.form("my-form",clear_on_submit=True):
        uploaded_file = st.file_uploader("Choose a DCM file", type=['png','jpg','dcm'])
        submitted = st.form_submit_button("Submit")
        if submitted:
            if uploaded_file is not None:
                if uploaded_file.type == 'application/dicom' or uploaded_file.type == 'application/octet-stream':
                    dcm = pydicom.dcmread(uploaded_file)
                    img = dcm.pixel_array
                    img = img - np.min(img)
                    img = img / np.max(img)
                    img = (img * 255).astype(np.uint8)
                    img = Image.fromarray(img).resize((WIDTH, HEIGHT), Image.LANCZOS).convert('L')
                    
                if uploaded_file.type == 'image/png' or uploaded_file.type == 'image/jpeg':
                    img = Image.open(uploaded_file)
                    if img.size != (WIDTH, HEIGHT):
                        img = img.resize((WIDTH, HEIGHT), Image.LANCZOS).convert('L')
                        
                pix = np.asarray(img)
                fig, ax = plt.subplots(1, 1, figsize=(20, 5))
                plt.imshow(pix, cmap='gray')
                plt.title(f'img {uploaded_file}')
                plt.colorbar()
                st.pyplot(fig)
                score = predecir(pix)
                st.write("Predicted class : %s" % (CLASS_NAMES[np.argmax(score)]))
                st.write("Score : %f" % (100 * np.max(score)))


def predecir(imgMat):
    imgMat = np.repeat(imgMat[:, :, np.newaxis], 3, axis=2)
    imgMat = imgMat / 255  ## los calculos son numeros reales entre 0 <-> 1 
    #imgMat = imgMat.reshape(-1, HEIGHT, WIDTH, 1)
    imgMat = np.expand_dims(imgMat, axis=0)
    d=tf.convert_to_tensor(imgMat)
    learn = keras.models.load_model('models/Modelo1OAZ.h5')
    #learn.load_weights.load_weights('models/modeWeights1OAZ.h5')
    predictions = learn.predict(d)
    print (predictions)
    score = tf.nn.softmax(predictions)
    return score


if __name__ == "__main__":
    st.set_page_config(
        page_title="Breast Cancer Detetion",
        page_icon="🩻",
        layout="centered",
        menu_items={
        'About': "# Breas Cancer Detection. Uses jpg, png, *dicom* images!"
        }
    )
    main()
