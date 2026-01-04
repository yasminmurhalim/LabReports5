import streamlit as st
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import pandas as pd
import urllib.request
import json

# Configuration and Imports 
st.set_page_config(page_title="Lab 5: CV Classifier", layout="centered")
st.title("Image Classification with ResNet18")
st.write("Upload an image to classify it using a pre-trained ResNet18 model.")

# Configure CPU Settings 
device = torch.device('cpu')

# Load Pre-trained ResNet18 
@st.cache_resource
def load_model():
    # Load the model with default weights (pre-trained on ImageNet)
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    # Move to CPU and set to evaluation mode
    model.to(device)
    model.eval()
    return model

model = load_model()

# Helper: Download ImageNet labels (so we see "Goldfish" instead of "Class 1")
@st.cache_data
def load_labels():
    url = "https://raw.githubusercontent.com/anishathalye/imagenet-simple-labels/master/imagenet-simple-labels.json"
    with urllib.request.urlopen(url) as f:
        labels = json.load(f)
    return labels

labels = load_labels()

# Preprocessing Transformations
# Standard transforms for ResNet18
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                         std=[0.229, 0.224, 0.225]),
])

# User Interface for Upload
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    st.write("Classifying...")

    #Convert to Tensor and Inference
    input_tensor = preprocess(image)
    # Add batch dimension (1, 3, 224, 224)
    input_batch = input_tensor.unsqueeze(0).to(device) 

    with torch.no_grad():
        output = model(input_batch)

    # Softmax and Top-5 Predictions
    probabilities = torch.nn.functional.softmax(output[0], dim=0)
    
    # Get top 5 probabilities and indices
    top5_prob, top5_catid = torch.topk(probabilities, 5)
    
    # Prepare data for visualization
    results = []
    for i in range(top5_prob.size(0)):
        results.append({
            "Class": labels[top5_catid[i]],
            "Probability": top5_prob[i].item()
        })
    
    df = pd.DataFrame(results)

    # Visualize with Bar Chart 
    st.subheader("Top 5 Predictions")
    # Display table
    st.table(df)
    
    # Display bar chart (Set Class as index for better chart labeling)
    st.bar_chart(df.set_index("Class"))