import streamlit as st
from utils.logger import logger

st.set_page_config(
    page_title="AI Vision Studio",
    page_icon="👁️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("👁️ AI Vision Studio")
st.markdown("""
Welcome to **AI Vision Studio**, a comprehensive tool for end-to-end computer vision workflows.

### 🌟 Features:
1. **Dataset Management**: Upload, manage, and label your image datasets.
2. **Model Training Engine**: Fine-tune state-of-the-art YOLOv8 models on your custom datasets.
3. **Live Inference**: Deploy your trained models for real-time object detection via webcam.

---

👈 Select a module from the sidebar to get started!
""")

# Ensure directory structure
import os
os.makedirs("datasets", exist_ok=True)
os.makedirs("models", exist_ok=True)
os.makedirs("logs", exist_ok=True)

logger.info("Application started.")
