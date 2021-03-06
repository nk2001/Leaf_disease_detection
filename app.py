import numpy as np
import streamlit as st 
from PIL import Image
from classify import identify
from tempfile import NamedTemporaryFile
from keras.preprocessing.image import load_img


st.set_option('deprecation.showfileUploaderEncoding', False)

st.title("Identifying if a leaf is healthy")
uploaded_file = st.file_uploader("Upload an image of a leaf in jpg format ...", type="jpg")
temp_file = NamedTemporaryFile(delete=False)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image.', use_column_width=True)
	#Use this trick to overcome streamit file upload problem discussed here: https://discuss.streamlit.io/t/image-upload-problem/4810
    temp_file.write(uploaded_file.getvalue())
	
    st.write("")
    st.subheader("Checking the Image...")
    label = identify(temp_file.name)
    st.subheader(label)