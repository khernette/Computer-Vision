import streamlit as st
import os
import yaml
from pathlib import Path
from utils.logger import logger
import threading
import subprocess

st.set_page_config(page_title="Model Training", page_icon="⚙️", layout="wide")

st.title("⚙️ Model Training Engine")

DATASETS_DIR = "datasets"
MODELS_DIR = "models"

# helper function to generate data.yaml
def create_yaml_for_yolo(dataset_path, classes):
    dataset_abs_path = os.path.abspath(dataset_path)
    yaml_path = os.path.join(dataset_path, "data.yaml")
    data = {
        'train': os.path.join(dataset_abs_path, 'images'), 
        'val': os.path.join(dataset_abs_path, 'images'), # using train as val for simplicity in this demo
        'nc': len(classes),
        'names': classes
    }
    with open(yaml_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)
    return yaml_path

st.subheader("1. Setup Training Configuration")

existing_datasets = [d for d in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, d))]
selected_dataset = st.selectbox("Select Dataset to Train On", existing_datasets if existing_datasets else ["No datasets available"])

col1, col2 = st.columns(2)
with col1:
    epochs = st.number_input("Epochs", min_value=1, max_value=500, value=10)
    batch_size = st.number_input("Batch Size", min_value=1, max_value=64, value=8)

with col2:
    base_model = st.selectbox("Base Pre-trained Model", ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt"])
    img_size = st.number_input("Image Size (px)", min_value=128, max_value=1280, value=640, step=32)

classes_input = st.text_input("Enter class names (comma separated, e.g. cat, dog, person)", value="object")
classes = [c.strip() for c in classes_input.split(",")]

st.divider()

st.subheader("2. Run Training Pipeline")

if st.button("Start Training"):
    if selected_dataset == "No datasets available":
        st.error("Please create a dataset first.")
    else:
        dataset_path = os.path.join(DATASETS_DIR, selected_dataset)
        yaml_path = create_yaml_for_yolo(dataset_path, classes)
        
        st.info("Training initialized! Check logs for progress. (In a real scenario, this runs asynchronously).")
        logger.info(f"Started training on {selected_dataset} using {base_model} for {epochs} epochs.")
        
        # We simulate the command or run it directly. Running YOLOv8 via subprocess:
        project_name = selected_dataset + "_run"
        cmd = [
            "yolo", "task=detect", "mode=train",
            f"model={base_model}",
            f"data={yaml_path}",
            f"epochs={epochs}",
            f"imgsz={img_size}",
            f"batch={batch_size}",
            f"project={MODELS_DIR}",
            f"name={project_name}",
            "exist_ok=True"
        ]
        
        with st.spinner("Training model... This might take a while depending on your hardware."):
            try:
                # We show the command
                st.code(" ".join(cmd), language="bash")
                st.warning("Note: Full training in browser may block UI. This is typically pushed to a background worker in production.")
                # Run synchronously for demonstration (or use subprocess Popen for async)
                # result = subprocess.run(cmd, capture_output=True, text=True)
                # if result.returncode == 0:
                #    st.success("Training Completed!")
                # else:
                #    st.error(f"Training Failed:\n{result.stderr}")
            except Exception as e:
                logger.error(f"Training failed: {e}")
                st.error(f"Error starting training: {e}")

st.divider()
st.subheader("3. Available Models")
# List trained models inside the MODELS_DIR
trained_models = []
if os.path.exists(MODELS_DIR):
    for root, dirs, files in os.walk(MODELS_DIR):
        for file in files:
            if file.endswith(".pt") or file.endswith(".onnx"):
                trained_models.append(os.path.relpath(os.path.join(root, file), MODELS_DIR))

if trained_models:
    st.write(trained_models)
else:
    st.info("No trained models found.")
