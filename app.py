import sys
import os
import base64
import torch
import streamlit as st
from PIL import Image, ImageDraw
from torchvision import transforms
from scripts.ocr_model import LicensePlateOCR
from scripts.resnet_model import get_resnet18

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Streamlit Configuration
st.set_page_config(
    page_title="Vehicle Monitoring System",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Load OCR Model
ocr_model_path = "models/license_plate_ocr.pth"
ocr_model = LicensePlateOCR(num_classes=37)
ocr_model.idx_to_char = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789 "
ocr_model.idx_to_char = {i: char for i, char in enumerate(ocr_model.idx_to_char)}
ocr_model.load_state_dict(torch.load(ocr_model_path, map_location=device))
ocr_model.to(device)
ocr_model.eval()

# Load Driver Behavior Model
driver_behavior_model_path = "models/resnet18_driver_behavior.pth"
driver_behavior_model = get_resnet18(num_classes=5)
driver_behavior_model.load_state_dict(torch.load(driver_behavior_model_path, map_location=device))
driver_behavior_model.to(device)
driver_behavior_model.eval()

# Preprocessing
ocr_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
])

driver_behavior_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# License Plate Detection
def detect_plate_region(image):
    width, height = image.size
    xmin, ymin = int(width * 0.3), int(height * 0.7)
    xmax, ymax = int(width * 0.7), int(height * 0.85)
    return xmin, ymin, xmax, ymax

# Recognize License Plate Text
def recognize_plate_text(image, bbox):
    xmin, ymin, xmax, ymax = bbox
    cropped_plate = image.crop((xmin, ymin, xmax, ymax))
    processed_plate = ocr_transform(cropped_plate).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = ocr_model(processed_plate)
        _, preds = torch.max(outputs, dim=2)
        pred_text = "".join([ocr_model.idx_to_char[idx.item()] for idx in preds[0] if idx.item() != 0])
    return cropped_plate, pred_text

# Predict Driver Behavior
def detect_driver_behavior(image):
    processed_image = driver_behavior_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        outputs = driver_behavior_model(processed_image)
        _, pred = torch.max(outputs, dim=1)
        behavior_labels = ["Drinking", "Playing Radio", "Regular Conductor", "Seeing Behind", "Using Phone"]
        return behavior_labels[pred.item()]

# Add Background Image
def set_png_as_page_bg(png_file):
    with open(png_file, 'rb') as f:
        bin_str = base64.b64encode(f.read()).decode()
    page_bg_img = f'''
    <style>
    .stApp {{
        background: url("data:image/png;base64,{bin_str}");
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    .block-container {{
        background: rgba(0, 0, 0, 0.7); /* Dark overlay */
        border-radius: 10px;
        padding: 20px;
    }}
    .sidebar .sidebar-content {{
        background: rgba(50, 50, 50, 0.8); /* Sidebar background */
    }}
    h1, h2, h3, h4, h5, h6, p, div, label {{
        color: #FFFFFF !important; /* Text color */
    }}
    </style>
    '''
    st.markdown(page_bg_img, unsafe_allow_html=True)

# Set the PNG file as the background
set_png_as_page_bg('background.jpg')

# Navigation
st.sidebar.title("Navigation")
selected_page = st.sidebar.radio("Go to", ["Welcome", "License Plate Recognition", "Driver Behavior Detection", "GIF Output", "About"])

# Welcome Page
if selected_page == "Welcome":
    st.title("🚘 Vehicle Monitoring System")
    st.markdown("### **PLEASE DRIVE SAFE** 🚦")
    st.markdown("Welcome to the **Vehicle Monitoring System**, an advanced application combining **License Plate Recognition** and **Driver Behavior Detection** to ensure safety and compliance on the road.")
    st.subheader("📋 Features")
    st.markdown("""
    - Detects and recognizes license plates from vehicle images.
    - Identifies specific driver behaviors like:
      - Drinking
      - Using a phone
      - Playing the radio
      - Looking behind
      - Regular behavior.
    """)
    st.image("me.jpeg", caption="Developed by Badis Sdiri", width=200)

# License Plate Recognition Page
elif selected_page == "License Plate Recognition":
    st.title("🚗 License Plate Recognition")
    uploaded_file = st.file_uploader("Upload an image of the vehicle", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        bbox = detect_plate_region(image)
        cropped_plate, plate_text = recognize_plate_text(image, bbox)
        draw = ImageDraw.Draw(image)
        xmin, ymin, xmax, ymax = bbox
        draw.rectangle([xmin, ymin, xmax, ymax], outline="red", width=3)
        draw.text((xmin, ymin - 20), plate_text, fill="red")
        st.image(image, caption="Processed Image with License Plate", use_column_width=True)
        st.write(f"**Recognized License Plate Text:** {plate_text}")

# Driver Behavior Detection Page
elif selected_page == "Driver Behavior Detection":
    st.title("🧑‍✈️ Driver Behavior Detection")
    uploaded_file = st.file_uploader("Upload an image of the driver", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="Uploaded Image", use_column_width=True)
        predicted_behavior = detect_driver_behavior(image)
        draw = ImageDraw.Draw(image)
        draw.text((10, 10), predicted_behavior, fill="red")
        st.image(image, caption="Processed Image with Driver Behavior", use_column_width=True)
        st.write(f"**Predicted Driver Behavior:** {predicted_behavior}")

# GIF Output Page
elif selected_page == "GIF Output":
    st.title("🎞️ GIF Output")
    st.markdown("**Here is the output in GIF format:**")
    gif_path = os.path.join(os.getcwd(), "output.gif")
    with open(gif_path, "rb") as f:
        gif_data = f.read()
        encoded_gif = base64.b64encode(gif_data).decode("utf-8")
        st.markdown(
            f'<img src="data:image/gif;base64,{encoded_gif}" alt="Output GIF" style="width:30%;height:auto;">',
            unsafe_allow_html=True,
        )

# About Page
elif selected_page == "About":
    st.title("👨‍💻 Developed By")
    st.image("me.jpeg", caption="Badis Sdiri", width=300)
    st.markdown("""
    **Name:** Badis Sdiri  
    **GitHub:** [https://github.com/BadisSdiri](https://github.com/BadisSdiri)  
    **LinkedIn:** [www.linkedin.com/in/badis-sdiri](https://www.linkedin.com/in/badis-sdiri)  
    """)
