import streamlit as st
import os
import shutil
import time
import cv2
import torch
from PIL import Image
from utils.logger import logger
from facenet_pytorch import MTCNN

st.set_page_config(page_title="Dataset Management", page_icon="📁", layout="wide")

st.title("📁 Dataset Management")

DATASETS_DIR = "datasets"

@st.cache_resource
def load_mtcnn():
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    return MTCNN(keep_all=True, device=device)

mtcnn = load_mtcnn()

st.subheader("1. Create / Select Dataset Project")
col1, col2 = st.columns(2)

with col1:
    new_dataset_name = st.text_input("New Dataset Name")
    if st.button("Create Dataset"):
        if new_dataset_name:
            dataset_path = os.path.join(DATASETS_DIR, new_dataset_name, "images")
            os.makedirs(dataset_path, exist_ok=True)
            logger.info(f"Created new dataset: {new_dataset_name}")
            st.success(f"Dataset '{new_dataset_name}' created successfully!")
        else:
            st.error("Please enter a valid name.")

with col2:
    existing_datasets = [d for d in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, d))]
    selected_dataset = st.selectbox("Select Existing Dataset", existing_datasets if existing_datasets else ["No datasets available"])

st.divider()

if selected_dataset and selected_dataset != "No datasets available":
    dataset_path = os.path.join(DATASETS_DIR, selected_dataset, "images")
    
    st.subheader(f"2. Upload Images to '{selected_dataset}'")
    
    uploaded_files = st.file_uploader("Upload Images (JPG, PNG)", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing images and extracting faces..."):
            extracted_count = 0
            for uploaded_file in uploaded_files:
                try:
                    img_pil = Image.open(uploaded_file).convert("RGB")
                    boxes, probs = mtcnn.detect(img_pil)
                    
                    base_name = os.path.splitext(uploaded_file.name)[0]
                    if boxes is not None:
                        for i, box in enumerate(boxes):
                            if probs[i] is None or probs[i] < 0.90:
                                continue
                            x1, y1, x2, y2 = [int(b) for b in box]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(img_pil.width, x2), min(img_pil.height, y2)
                            
                            face_img = img_pil.crop((x1, y1, x2, y2))
                            face_filename = f"{base_name}_face_{i}_{int(time.time()*1000)}.jpg"
                            file_path = os.path.join(dataset_path, face_filename)
                            face_img.save(file_path, "JPEG")
                            extracted_count += 1
                    else:
                        st.warning(f"No face detected in {uploaded_file.name}")
                except Exception as e:
                    logger.error(f"Error processing {uploaded_file.name}: {e}")
                    st.error(f"Error processing {uploaded_file.name}")
                    
        st.success(f"Successfully processed images and extracted {extracted_count} valid faces!")
        logger.info(f"Extracted {extracted_count} faces from uploads to '{selected_dataset}'")
        
    st.markdown("### 📷 Camera Capture Tools")
    
    col_cam1, col_cam2 = st.columns(2)
    with col_cam1:
        base_name = st.text_input("Person Name / File Prefix", value="person")
    with col_cam2:
        num_burst = st.number_input("Burst Shots amount (multiple at once)", min_value=1, max_value=20, value=5)
    
    # Standard single Streamlit picture
    camera_image = st.camera_input("Take a standard single picture")
    if camera_image:
        file_path = os.path.join(dataset_path, f"{base_name}_{int(time.time())}.jpg")
        with open(file_path, "wb") as f:
            f.write(camera_image.getbuffer())
        st.success(f"Image saved as '{os.path.basename(file_path)}'!")
        logger.info(f"Saved camera capture '{file_path}'")
        
    # Burst shots via backend OpenCV
    if st.button(f"Take {num_burst} Burst Shots via Local Camera"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open local webcam for burst.")
        else:
            with st.spinner(f"Taking {num_burst} shots quickly... Please look at the camera!"):
                for i in range(num_burst):
                    ret, frame = cap.read()
                    if ret:
                        file_path = os.path.join(dataset_path, f"{base_name}_burst_{int(time.time())}_{i}.jpg")
                        cv2.imwrite(file_path, frame)
                        time.sleep(0.2) # small delay between frames for variation
                cap.release()
            st.success(f"Successfully saved {num_burst} burst shots for '{base_name}'!")
            logger.info(f"Saved {num_burst} burst shots for '{base_name}' in dataset '{selected_dataset}'")
            st.rerun()

    st.divider()
    st.subheader("3. Image Gallery")
    
    images = [f for f in os.listdir(dataset_path) if f.endswith(('jpg', 'jpeg', 'png'))]
    if images:
        st.write(f"Total Images: **{len(images)}**")
        
        cols = st.columns(4)
        for i, img_name in enumerate(images):
            img_path = os.path.join(dataset_path, img_name)
            with cols[i % 4]:
                st.image(img_path, use_container_width=True, caption=img_name)
                if st.button(f"Delete", key=f"del_{img_name}"):
                    os.remove(img_path)
                    st.rerun()
    else:
        st.info("No images currently in this dataset.")
