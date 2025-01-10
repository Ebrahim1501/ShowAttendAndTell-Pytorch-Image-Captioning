# Show, Attend, and Tell: Image Captioning with Attention Mechanism

## Project Overview  
This project is my personal PyTorch implementation of the [*Show, Attend, and Tell*](https://arxiv.org/pdf/1502.03044) paper, which utilizes the attention mechanism for image captioning. The model is trained on the Flickr30k image captioning dataset using Kaggle Cloud Notebooks, and the final web app is deployed on *Streamlit Community Cloud*.





![captioning_sample1](https://github.com/user-attachments/assets/d0ad8f76-2ad7-494e-87e8-1803a564693f)








### Repository Contents:  

**Root Directory**  
- `note_book_requirements.txt`  
  - Contains the requirements for running the notebook locally.  
- `show-attend-and-tell-image-captioning-final.ipynb`  
  - Jupyter notebook used for training and evaluating the model.  

---

**Deploy** *(Files related to app deployment and backend functionality)*  
- `App.py`  
  - Streamlit app for creating the GUI and handling captions generation.  
- `Architecture.py`  
  - Contains the classes architecture for the model and vocabulary classes.  
- `captioning_model.pt`  
  - The seralized trained captioning model file.  
- `Dockerfile`  
  - Used to build the Docker image for deployment.  
- `requirements.txt`  
  - Dependencies for running the Streamlit application backend.  
- `utils.py`  
  - Contains all the utility functions for processing the image's input and generating captions.  
- `vocab.pth`  
  - Serialized vocabulary that contains all the word (token) used during runtime by the model for generating captions.  

---

**Images Folder** *(Contains visual outputs and metrics)*  
- Sample outputs:  
  - `captioning_sample1.gif`
  - `captioning_sample2.gif`
  - `Result.png` (Overall results)  

- Evaluation Metrics:  
  - `ErrorCurves.png` (Training error trends)
  - `Bleu.png` (Bleu score curve during training)
- Attention Visualizations:  
  - `Result_with_attention.jpg`
  - `Result_with_attention1.jpg`,  
    `Result_with_attention2.jpg`
    `Result_with_Attention3.jpg`  

---
## Dataset  
The dataset used for this project is [*Flickr30k*](https://www.kaggle.com/datasets/abhinavbenagi/flickr30k), which consists of:  
- Around ~31,000 unique images.  
- Each image paired with 5 captions, resulting in ~151,000 image-caption pairs.  

### Dataset Structure  

- **Images Folder**  
  - Contains all the images used for training and evaluation.  

- **Captions File**  
  - `captions.txt`: A text file mapping each image to its corresponding caption.  
  - Each line follows the format: `(image_name, caption)`  

---
## Model Architecture  
The model follows an encoder-decoder based architecture consisting of three main components:

### 1. CNN Image Encoder  
- A pre-trained CNN *(ResNet101)* in our case is used as the feature extractor.  
- The final fully connected and classification layers are removed to obtain a 2048-dimensional feature map for each image these features will be the input for the next attention layer.  

### 2. Attention Layer 
- Improves the capability to describe multiple objects within a single image, overcoming the limitations of traditional RNN-based captioning models which usually tends to give more generic description for the image.  
- Computes a *context vector* as a weighted sum of the image feature map using the attention mechanism.
- This project employs the *Bahdanau soft attention mechanism* for architecture's simplicity
### 3. LSTM Decoder  
- A single-layer *LSTM* predicts the next token in the sequence based on the context vector given from the previous the attention layer.  
- Implements *teacher forcing* during training as:  
  - The ground truth token is fed as input to the next timestep instead of the predicted token.  
  - This technique reduces the error propagation during training and ensures faster learning.  

---

## Model Summary diagram:
![lstm-with-attention-diagram](https://github.com/user-attachments/assets/bd2faa93-e358-4c84-8782-e5815b819547)

---

 


## Cloning the repo    
   ```bash

   git clone https://github.com/Ebrahim1501/ShowAttendAndTell-Pytorch-ImageCaptioning-Implementation
   cd <your-dir-name>
  
   ```
##  to run Using The Docker file

   ```bash
#using docker terminal
   docker build -t <any name for your image >   
   ```
then
   ```bash

   docker run <the name of ur image>   
   ```
and the web app would be available throught port 8501 in your local host!!!
 
