#Author Helal Chowdhury
# Version:1

from __future__ import print_function, division
import pandas as pd
from matplotlib.pyplot import imshow
import numpy as np
#import plotly.express as px
from PIL import Image
import torchvision
from torchvision import models, transforms

import torch
import torch.nn as nn
import torch.nn.functional as F
import streamlit as st
# Setting Manual Seed for Recreation of results
torch.manual_seed(42)
np.random.seed(0)

# Title
st.title("Image Classification")
# streamlit button 
m = st.markdown("""
<style>
div.stButton > button:first-child {
    background-color: #0099ff;
    color:#ffffff;
}
div.stButton > button:hover {
    background-color: #00ff00;
    color:#ff0000;
    }
</style>""", unsafe_allow_html=True)
# streamlit info
st.info('Models: Resnet18, Densenet161', icon="ℹ️")
st.info('The Models are trained for five image classes: Mango, Banana, Jackfruit, Apple, Orange', icon="ℹ️")

st.info('Action flow: Select Model-> Upload image -> Finally,click Predict button', icon="ℹ️")
#--------------------------------
# Highlight the predicted result

def highlight_max(data, color="yellow"):
    """highlight the maximum in a Series or DataFrame"""
    attr = "background-color: {}".format(color)
    if data.ndim == 1:  # Series from .apply(axis=0) or axis=1
    	
        is_max = data == data.max()

        return [attr if v else "" for v in is_max]
    else:  # from .apply(axis=None)
        is_max = data == data.max().max()
        return pd.DataFrame(
            np.where(is_max, attr, ""), index=data.index, columns=data.columns
        )

#----------------
def color_negative_red(val):
    """
    Takes a scalar and returns a string with
    the css property `'color: red'` for negative
    strings, black otherwise.
    """
    color = "red" if val < 0 else "black"
    return "color: %s" % color

#--------------------
# Model Path
ResnetPATH = "Resnet_model.pt"
#DensenetPATH="FINALDensenet109_model.pt"
#DensenetPATH="Densenet161.pt"
# normalize the input image with mean and std
mean=[0.485, 0.456, 0.406]
mean=torch.tensor(mean)
std=[0.229, 0.224, 0.225]
std=torch.tensor(std)

image_transform = transforms.Compose([
                                         #transforms.Resize((240,240)),
                                         transforms.ToTensor(),
                                         transforms.Normalize(mean=mean,std=std)
                                        ])
                                        
#Selection model
#model_name=st.sidebar.selectbox("Select Model,( "Resnet18","Densnet161") )   
#model_name=st.sidebar.selectbox("Select Model, ("Resnet18",)                                
model_name="Resnet18"

# function call to retrive the model		
model=torch.load(ResnetPATH,map_location=torch.device('cpu'))
# function definition to inference the image to classify
def classify(image_data,model):
    model=model.eval()
    #image=Image.open(image_path)
    image=image_transform(image_data).float()
    #image=image[np.newaxis,...]
    image=torch.unsqueeze(image, 0)
    output=model(image)
    probs = torch.nn.functional.softmax(output, dim=1)
    _,predicted=torch.max(output.data,1)
    np.set_printoptions(precision=2)
    probs=probs.detach().numpy()
    probs = np.around(probs, 2)
    st.subheader("Image is predicted as {}".format(klas[predicted.item()])   )
    df= pd.DataFrame(probs, columns=['Apple', 'Banana', 'Jackfruit', 'Mango', 'Orange'])
    st.dataframe(
    df.style.applymap(color_negative_red).apply(highlight_max, color="darkorange", axis=1).format("{:.2%}")
    )
    #st.dataframe(df.style.format("{:.2%}"))
    
 #st.dataframe(
  #  df.style.applymap(color_negative_red).apply(highlight_max, color="darkorange", axis=0).format("{:.2%}")
#)
          		
klas=['Apple', 'Banana', 'Jackfruit', 'Mango', 'Orange']

uploaded_file=st.sidebar.file_uploader("Upload image",type="jpg")
generate_pred=st.sidebar.button("Predict")
if generate_pred:
	 model=torch.load(ResnetPATH,map_location=torch.device('cpu'))
	 image=Image.open(uploaded_file)
	 with st.expander("image",expanded=True):
	 	st.image(image,width=150)
	 pred=classify(image,model)



