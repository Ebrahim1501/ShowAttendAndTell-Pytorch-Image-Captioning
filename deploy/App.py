
import torch
import torchvision.transforms as T
import matplotlib.pyplot as plt
import cv2
from utils import set_cuda,image_transformation,plot_attention
import streamlit as st
from Architecture import Encoder_Decoder_Model,Vocabulary,Image_encoder,Attention_Based_Decoder,AttentionLayer
import numpy as np 
from PIL import Image



device=set_cuda()

#model=torch.load('captioning_model.pt', map_location=device)
model=torch.load('captioning_model.pt', map_location=device)
vocab=torch.load("vocab.pth",map_location=device)


st.set_page_config(page_title="Show-Attend-And-Tell", page_icon="icon.png")


#css styles for titles,texts
st.markdown("""                                                     
    <style>
        .title {
            text-align: center;
            font-size: 36px;
            color: #4CAF50;
            font-weight: bold;
        }
        .header {
            text-align: center;
            font-size: 18px;
            color: #777;
        }
    </style>
""", unsafe_allow_html=True)





st.markdown('<div class="title">Image Captioning App</div>', unsafe_allow_html=True)


st.markdown('<div class="header">Upload an image and see The captioning model in action!!</div>', unsafe_allow_html=True)


uploaded_image = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if "caption" not in st.session_state:
    st.session_state.caption = None
if "attentions" not in st.session_state:
    st.session_state.attentions = None



if uploaded_image is not None:
    
    
    image=Image.open(uploaded_image)
    image=np.array(image)
    image=cv2.resize(image,(480,480))
    st.image(image,caption="Uploaded Image")
    
    pressed=False
    display_att=False
    pressed=st.button("Generate Caption")
    
    if pressed:
        with st.spinner("Generating caption..."):
            #image=cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
            attentions, caption = model.predict(image, vocab)
            st.success(" Caption Generated:")
            st.write(f" {' '.join(caption[1:-1])}")