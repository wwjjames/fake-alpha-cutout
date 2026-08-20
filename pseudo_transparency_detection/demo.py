import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import os

class GridNet2depth(nn.Module):
    def __init__(self):
        super(GridNet2depth, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.fc1 = nn.Linear(64 * 56 * 56, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GridNetBaseline(nn.Module):
    def __init__(self):
        super(GridNetBaseline, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.fc1 = nn.Linear(128 * 28 * 28, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class GridNet4depth(nn.Module):
    def __init__(self):
        super(GridNet4depth, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128, 256, 3, padding=1)
        self.bn4 = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(256 * 14 * 14, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = self.pool(self.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x

@st.cache_resource 
def load_model(model_path, architecture_type):
    if architecture_type == "2-Layer":
        model = GridNet2depth()
    elif architecture_type == "4-Layer":
        model = GridNet4depth()
    else:
        model = GridNetBaseline() 
    
    try:
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
    except FileNotFoundError:
        return None
    except Exception as e:
        st.error(f"failure: {e}")
        return None

    model.eval() 
    return model


def process_image(image):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(image).unsqueeze(0) 


st.set_page_config(page_title="Fake Transparent Image Detector Demo", layout="wide")

st.title("🕵️‍♀️Fake Transparent Image Detector")
st.markdown("Upload a picture，and examine if it is a Fake Transparent Image ")

st.sidebar.header("Select a model")

model_options = {
    "3-Layer CNN trained by dataset 400+400 (baseline)": {"path": "models/400_3depth(baseline).pth", "arch": "3-Layer"},
    "2-Layer CNN trained by dataset 400+400 ":           {"path": "models/400_2depth.pth", "arch": "2-Layer"},
    "4-Layer CNN trained by dataset 400+400 ":           {"path": "models/400_4depth.pth", "arch": "4-Layer"},
    "SmallData (3-Layer CNN trained by dataset 100+100)":      {"path": "models/100_3depth.pth", "arch": "3-Layer"},
    "SmallData+25Epoch (3-Layer CNN trained by dataset 100+100)":  {"path": "models/100_3depth_25epochs.pth", "arch": "3-Layer"},
    "SmallData+Aug+25Epoch (3-Layer CNN trained by dataset 100+100)":  {"path": "models/100_3depth_25epochs(A).pth", "arch": "3-Layer"},
    "SmallData+Aug+40Epoch (3-Layer CNN trained by dataset 100+100)":  {"path": "models/100_3depth_40epochs(A).pth", "arch": "3-Layer"},
}

selected_model_name = st.sidebar.selectbox("Choose a model", list(model_options.keys()))
selected_info = model_options[selected_model_name]


model = load_model(selected_info["path"], selected_info["arch"])

if model is None:
    st.sidebar.error(f"Model file not found: {selected_info['path']}，check the path！")
else:
    st.sidebar.success(f"✅ model loaded: {selected_model_name}")


uploaded_file = st.file_uploader("Select a picture", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    col1, col2 = st.columns(2)
    
    image = Image.open(uploaded_file).convert('RGB') 
    with col1:
        st.image(image, caption='uploaded picture', use_container_width=True)

    if model:
        img_tensor = process_image(image)

        with torch.no_grad():
            outputs = model(img_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            
            prob_fake = probabilities[0][1].item() 
            prob_real = probabilities[0][0].item() 
        
            
            prediction = "Fake" if prob_fake > prob_real else "Ordinary"
            confidence = max(prob_fake, prob_real)

    with col2:
        st.subheader("Result")
        
        if prob_fake > prob_real:
            st.error(f"🚨 **{prediction}**") 
        else:
            st.success(f"✅ **{prediction}**") 
            
        st.metric("Confidence", f"{confidence*100:.2f}%")
        
        # 绘制概率条
        st.write("Detail:")
        st.progress(prob_fake, text=f"Probability of fake: {prob_fake*100:.2f}%")
        st.progress(prob_real, text=f"Probability of ordinary: {prob_real*100:.2f}%")