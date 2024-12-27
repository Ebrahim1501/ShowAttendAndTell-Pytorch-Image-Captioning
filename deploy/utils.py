import torch
import matplotlib.pyplot as plt
import torchvision.transforms as T
import streamlit as st
import cv2
import re
import string
import random


image_input_size=(224,224)
mean_normalization_vec=[0.485, 0.456, 0.406]
std_normalization_vec=[0.229, 0.224, 0.225]

image_transformation=T.Compose([T.Resize(image_input_size),T.ToTensor(),T.Normalize(mean=mean_normalization_vec,std=std_normalization_vec)])


def set_cuda():
    """
    set torch destination device to cuda if it's available and prints it.

    Returns:
        torch device : The device in use, either 'cuda' or 'cpu'.

    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device in use : {}".format(device))
    return device

        

def Detensorize_image(image_tensor, mean_vector, std_vector, denormalize=True): 
    
    
    """
    Converts a normalized image tensor back to a denormalized format for visualization.

    Args:
        image_tensor(torch tensor) : The input image tensor with shape (C, H, W).
        mean_vector(torch tensor) : The mean values used during normalization, usually of length 3 for RGB channels.
        std_vector(torch tensor) : The standard deviation values used during normalization, usually of length 3 for RGB channels.
        denormalize(boolean value) : If True, denormalizes the tensor using the provided mean and std vectors.

    Returns:
        detensorized_image (numpy array): The detensorized image array with shape (H, W, C).
    """
    
    
    
    
    #reshape mean and std vectors to tensors with appropriate shape
    mean_tensor = torch.tensor(mean_vector).view(3, 1, 1)
    std_tensor = torch.tensor(std_vector).view(3, 1, 1)
    
    if denormalize:
        
        detensorized_image = (image_tensor * std_tensor) + mean_tensor
    else:
        detensorized_image = image_tensor
    
    
    detensorized_image = detensorized_image.permute(1, 2, 0)#convert image's dimensions from (C, H, W) to (H, W, C) 
    
    detensorized_image = detensorized_image.numpy()#convert the image tensor to a numpy array for visualization

    
    return detensorized_image




def standarize_text(text):
    """
    convert captions into a unified format by removing punctuation, trimming whitespace, and converting to lowercase.

    Args:
        text : a single input text string to be standardized.

    Returns:
        text : a standardized text string, stripped of punctuation and converted to lowercase.
    """
    text=text.strip()
    text = re.sub(f'[{re.escape(string.punctuation)}]', '',text)#remove punctuation 
    text=text.lower()
    # text="<start> "+text+" <end>"
    return text



def plot_attention(img, caption, attention_vectors,is_streamlit=False):
    """
        displays attention heatmaps for each word in the caption.

    Args:
        img (numpy array): The input image in the shape (H, W, C) for visualization.
        caption (list[str]): The list of words in the caption corresponding to the attention vectors.
        attention_vectors (list[numpy arrays]): The attention scores for each word, each with a flattened shape of (49,).
        is_streamlit (boolean): If True, displays the plot using Streamlit's st.pyplot.

    Returns:
        None

    """



    temp_image = img

    fig, axes = plt.subplots(len(caption) - 1, 1, figsize=(8, 8 * len(caption)))

    assert len(caption)>1,f" image has caption of {len(caption)}words only "

    for l, ax in enumerate(axes):
        temp_att = attention_vectors[l].reshape(7, 7)  # convert the 49 attention vector to 7x7 attention map
        
        
        att_resized = cv2.resize(temp_att, (temp_image.shape[1], temp_image.shape[0])) #resizingg attention map to match the image size


        ax.imshow(temp_image)
        ax.imshow(att_resized, cmap='jet', alpha=0.4)  # heat map over the original image for attended regions
        ax.set_title(caption[l])
        ax.axis("off")

    plt.tight_layout()
    plt.show()
    if streamlit==True:
        st.pyplot(fig)#for compatibility with in streamlit's pyplot




def pick_random_image(directory):
    """
    Selects a random image file from a selected directory.

    Args:
        directory (string): The directory path containing the images.

    Returns:
        random_image (string): The file name of the (randomly) selected image.
.
    """
    files = os.listdir(directory)
    

    images = [f for f in files if f.lower().endswith(('png', 'jpg', 'jpeg', 'gif', 'bmp'))]
    
    assert images, "no images  found in the directory."
    
    random_image = random.choice(images)
    
    return random_image



