
import streamlit as st
import pandas as pd
import pydicom
import matplotlib.pyplot as plt
import seaborn as sns

with st.form("my-form",clear_on_submit=True):
    uploaded_files = st.file_uploader("Choose a DCM file", type='dcm', accept_multiple_files=True)
    submitted = st.form_submit_button("Submit")
    if submitted:
        for uploaded_file in uploaded_files:
            img = pydicom.dcmread(uploaded_file)
            fig, ax = plt.subplots(1, 1, figsize=(20, 5))
            plt.imshow(img.pixel_array, cmap='gray')
            #plt.title(f'img {uploaded_file}')
            plt.colorbar()
            st.pyplot(fig)