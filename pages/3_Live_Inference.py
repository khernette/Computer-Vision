import streamlit as st
import cv2
import os
import time
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils.logger import logger

st.set_page_config(page_title="Face Identification", page_icon="👤", layout="wide")

st.title("👤 Live Video Face Identification")
st.markdown("Real-time **Face Detection and Recognition**. The model maps human faces and cross-references them against your uploaded dataset.")

DATASETS_DIR = "datasets"

@st.cache_resource
def load_face_models():
    """Loads FaceNet MTCNN for detection and InceptionResnetV1 for recognition."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn_detector = MTCNN(keep_all=True, device=device)
    mtcnn_extractor = MTCNN(keep_all=False, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn_detector, mtcnn_extractor, resnet, device

mtcnn, mtcnn_dataset, resnet, device = load_face_models()

existing_datasets = ["All Datasets"]
if os.path.exists(DATASETS_DIR):
    for d in os.listdir(DATASETS_DIR):
        if os.path.isdir(os.path.join(DATASETS_DIR, d)):
            existing_datasets.append(d)

col1, col2 = st.columns([1, 3])

with col1:
    st.subheader("Configuration")
    selected_dataset = st.selectbox("Select Face Dataset to Match Against", existing_datasets)
    sim_threshold = st.slider("Similarity Threshold", 0.0, 1.0, 0.65, 0.05)
    
    col_start, col_stop = st.columns(2)
    with col_start:
        start_cam = st.button("Start Camera", use_container_width=True)
    with col_stop:
        stop_cam = st.button("Stop Camera", use_container_width=True)
        
    if "run_camera" not in st.session_state:
        st.session_state.run_camera = False
        
    if start_cam:
        st.session_state.run_camera = True
    if stop_cam:
        st.session_state.run_camera = False
        
    run_camera = st.session_state.run_camera
    
    st.markdown("---")
    st.subheader("Live Activity Log")
    log_placeholder = st.empty()

with col2:
    st.subheader("Live Feed")
    feed_placeholder = st.empty()

dataset_embeddings = {}
if run_camera:
    datasets_to_load = []
    if selected_dataset == "All Datasets":
        if os.path.exists(DATASETS_DIR):
            datasets_to_load = [os.path.join(DATASETS_DIR, d, "images") for d in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, d))]
    else:
        datasets_to_load = [os.path.join(DATASETS_DIR, selected_dataset, "images")]

    for dataset_path in datasets_to_load:
        if os.path.exists(dataset_path):
            dataset_name = os.path.basename(os.path.dirname(dataset_path))
            with st.spinner(f"Indexing faces from {dataset_name} dataset..."):
                for img_name in os.listdir(dataset_path):
                    if img_name.lower().endswith(('jpg', 'jpeg', 'png')):
                        img_path = os.path.join(dataset_path, img_name)
                        try:
                            img_pil = Image.open(img_path).convert("RGB")
                            face_tensor = mtcnn_dataset(img_pil)
                            if face_tensor is not None:
                                emb = resnet(face_tensor.unsqueeze(0).to(device)).detach()
                                # Keep base name logic mapping
                                base_name = img_name.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                                if "_face_" in base_name:
                                    base_name = base_name.split("_face_")[0]
                                elif "_burst_" in base_name:
                                    base_name = base_name.split("_burst_")[0]
                                else:
                                    base_name = base_name.split("_burst")[0].split("_")[0]

                                dataset_embeddings[img_name] = {"emb": emb, "name": base_name, "dataset": dataset_name}
                        except Exception as e:
                            pass

if run_camera:
    cap = None
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Error: Could not open the specified webcam feed.")
            st.session_state.run_camera = False
        else:
            st.success("Face Scanning is Active... Click 'Stop Camera' to end.")
            logger.info("Started live FaceNet inference.")
            
            last_log_time = 0
            log_messages = []
            
            while st.session_state.run_camera:
                ret, frame = cap.read()
                if not ret: break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame_pil = Image.fromarray(frame_rgb)
                
                boxes, probs = mtcnn.detect(frame_pil)
                if boxes is not None:
                    faces_tensor = mtcnn.extract(frame_pil, boxes, save_path=None)
                    if faces_tensor is not None:
                        embeddings = resnet(faces_tensor.to(device)).detach()
                        
                        for i in range(len(boxes)):
                            box, prob, emb = boxes[i], probs[i], embeddings[i]
                            if prob is None or prob < 0.90: continue
                                
                            x1, y1, x2, y2 = map(int, box)
                            label, color = "Unknown Face", (0, 165, 255)
                            
                            if dataset_embeddings:
                                best_sim, best_match_name, best_match_dataset = -1.0, None, None
                                
                                for fname, data in dataset_embeddings.items():
                                    sim = F.cosine_similarity(emb.unsqueeze(0), data["emb"]).item()
                                    if sim > best_sim:
                                        best_sim = sim
                                        best_match_name = data["name"]
                                        best_match_dataset = data.get("dataset", "Unknown Dataset")
                                
                                # Throttled Logging to prevent flooding
                                current_time = time.time()
                                if best_sim >= sim_threshold:
                                    label = f"{best_match_name} ({best_sim:.2f})"
                                    color = (0, 255, 0)
                                    if current_time - last_log_time > 2.0:
                                        msg = f"✅ MATCH: Detect '{best_match_name}' from dataset '{best_match_dataset}' correctly (sim: {best_sim:.2f})"
                                        logger.info(msg)
                                        log_messages.insert(0, msg)
                                        last_log_time = current_time
                                else:
                                    label = f"Unknown ({best_sim:.2f})"
                                    if current_time - last_log_time > 3.0:
                                        msg = f"❌ UNVERIFIED: Detected face. Best guess '{best_match_name}' from '{best_match_dataset}' but sim too low ({best_sim:.2f})"
                                        logger.info(msg)
                                        log_messages.insert(0, msg)
                                        last_log_time = current_time
                                        
                            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(frame, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                            cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                annotated_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                feed_placeholder.image(annotated_frame, channels="RGB")
                
                # Update UI Log
                if log_messages:
                    log_placeholder.markdown("\n".join([f"- {m}" for m in log_messages[:5]]))
                
    except Exception as e:
        if type(e).__name__ != 'RerunException' and type(e).__name__ != 'StopException':
            logger.error(f"Inference error: {e}")
            st.error(f"An error occurred during inference: {e}")
    finally:
        if cap is not None:
            cap.release()
else:
    st.info("System stand-by. Click 'Start Camera' above to launch detection.")
