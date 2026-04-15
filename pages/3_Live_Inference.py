import streamlit as st
import cv2
import os
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from utils.logger import logger
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, RTCConfiguration, WebRtcMode

# Page Config
st.set_page_config(page_title="Cloud Face ID", page_icon="👤", layout="wide")

st.title("👤 Online Live Face Identification")
st.markdown("This version uses **WebRTC** to stream your local camera safely to the cloud server.")

DATASETS_DIR = "datasets"

# STUN Servers for reliable connection on Cloud
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

@st.cache_resource
def load_face_models():
    """Loads FaceNet MTCNN for detection and InceptionResnetV1 for recognition."""
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    mtcnn_detector = MTCNN(keep_all=True, device=device)
    resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)
    return mtcnn_detector, resnet, device

mtcnn, resnet, device = load_face_models()

@st.cache_data(show_spinner=False)
def get_dataset_embeddings(dataset_name, _mtcnn_dataset, _resnet, _device):
    """Computes and caches embeddings for the selected dataset."""
    embeddings = {}
    datasets_to_load = []
    
    if dataset_name == "All Datasets":
        if os.path.exists(DATASETS_DIR):
            datasets_to_load = [os.path.join(DATASETS_DIR, d, "images") for d in os.listdir(DATASETS_DIR) if os.path.isdir(os.path.join(DATASETS_DIR, d))]
    else:
        datasets_to_load = [os.path.join(DATASETS_DIR, dataset_name, "images")]

    for dataset_path in datasets_to_load:
        if os.path.exists(dataset_path):
            curr_dataset_name = os.path.basename(os.path.dirname(dataset_path))
            for img_name in os.listdir(dataset_path):
                if img_name.lower().endswith(('jpg', 'jpeg', 'png')):
                    img_path = os.path.join(dataset_path, img_name)
                    try:
                        img_pil = Image.open(img_path).convert("RGB")
                        # Use a smaller version for quick indexing
                        face_tensor = _mtcnn_dataset(img_pil)
                        if face_tensor is not None:
                            # We take the first face detected for indexing
                            emb = _resnet(face_tensor[0].unsqueeze(0).to(_device)).detach()
                            base_name = img_name.replace('.jpg', '').replace('.jpeg', '').replace('.png', '')
                            if "_face_" in base_name:
                                base_name = base_name.split("_face_")[0]
                            elif "_burst_" in base_name:
                                base_name = base_name.split("_burst_")[0]
                            else:
                                base_name = base_name.split("_burst")[0].split("_")[0]

                            embeddings[img_name] = {"emb": emb, "name": base_name, "dataset": curr_dataset_name}
                    except:
                        pass
    return embeddings

class FaceIDTransformer(VideoTransformerBase):
    def __init__(self, mtcnn, resnet, device, embeddings, threshold):
        self.mtcnn = mtcnn
        self.resnet = resnet
        self.device = device
        self.embeddings = embeddings
        self.threshold = threshold

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to RGB for MTCNN
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        # Detect faces
        boxes, probs = self.mtcnn.detect(img_pil)
        
        if boxes is not None:
            # Re-extract for embeddings
            faces_tensor = self.mtcnn.extract(img_pil, boxes, save_path=None)
            if faces_tensor is not None:
                face_embeddings = self.resnet(faces_tensor.to(self.device)).detach()
                
                for i in range(len(boxes)):
                    box, prob, emb = boxes[i], probs[i], face_embeddings[i]
                    if prob is None or prob < 0.90: continue
                        
                    x1, y1, x2, y2 = map(int, box)
                    label, color = "Unknown", (0, 165, 255)
                    
                    if self.embeddings:
                        best_sim, best_match_name = -1.0, None
                        
                        for fname, data in self.embeddings.items():
                            sim = F.cosine_similarity(emb.unsqueeze(0), data["emb"]).item()
                            if sim > best_sim:
                                best_sim = sim
                                best_match_name = data["name"]
                        
                        if best_sim >= self.threshold:
                            label = f"{best_match_name} ({best_sim:.2f})"
                            color = (0, 255, 0)
                        else:
                            label = f"Unknown ({best_sim:.2f})"
                            
                    # Draw on frame
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(img, (x1, y1 - th - 5), (x1 + tw, y1), color, -1)
                    cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return img

# --- UI Setup ---
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
    
    # Pre-index based on selection
    with st.spinner(f"Indexing '{selected_dataset}'..."):
        dataset_embeddings = get_dataset_embeddings(selected_dataset, mtcnn, resnet, device)
    
    st.success(f"Indexed {len(dataset_embeddings)} face patterns.")
    st.info("The Activity Log is disabled in WebRTC mode to ensure smooth performance.")

with col2:
    st.subheader("WebRTC Live Feed")
    webrtc_streamer(
        key="face-id",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIGURATION,
        video_processor_factory=lambda: FaceIDTransformer(mtcnn, resnet, device, dataset_embeddings, sim_threshold),
        async_processing=True,
    )
