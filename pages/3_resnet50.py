import streamlit as st
import numpy as np
import faiss
import matplotlib.pyplot as plt
from PIL import Image
import torchvision.transforms as transforms
import torchvision.models as models
import base64
import torch
import os


def get_image_base64(path):
    # guard: some np arrays hold bytes; convert path to str
    path = str(path)
    mime = "image/png" if path.lower().endswith(".png") else "image/jpeg"
    with open(path, "rb") as img_file:
        b64 = base64.b64encode(img_file.read()).decode()
    return mime, b64


# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
FEATURE_PATH = "features/"
st.set_page_config(
        page_title="ResNet50",  # Page title
        layout="wide" 
    )

st.markdown("""
    <style>
    .zoom-img {
        transition: transform 0.3s ease;
        border-radius: 10px;
        cursor: pointer;
    }
    .zoom-img:hover {
        transform: scale(1.5);
        z-index: 999;
        position: relative;
    }
    </style>
""", unsafe_allow_html=True)

# --- LOAD MODEL & FEATURES ---
resnet = models.resnet50(pretrained=True)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1]).to(DEVICE)
resnet.eval()

res_faiss = FEATURE_PATH + "resnet50_index.faiss"
resnet_index = faiss.read_index(res_faiss)
resnet_labels =  FEATURE_PATH + "resnet50_labels.npy"
resnet_paths  = np.load(FEATURE_PATH + "resnet50_paths.npy")
labels = np.load(resnet_labels)
paths = np.array([
    p.replace("../../dataset/train", "../images-retrieval/data/train")
    for p in resnet_paths
])
paths = np.array([os.path.normpath(p) for p in paths])

# --- PAGE UI ---
st.title("Landmark Image Retrieval (ResNet50)")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
top_k = st.slider("Top-K results", 1, 100, 5)


    
if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    cols = st.columns([0.2, 2.6, 0.2])
    with cols[1]:
        st.image(query_img, caption="Query Image", use_container_width=True)

    # --- Extract global descriptor ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def extract_feature(model, pil_img):
        tensor = transform(pil_img).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            vec = model(tensor).squeeze().cpu().numpy()
        faiss.normalize_L2(vec.reshape(1, -1))
        return vec.reshape(1, -1)

    res_vec = extract_feature(resnet, query_img)

    # --- Compute FAISS ---
    D_r, I_r = resnet_index.search(res_vec, top_k)  # D_r shape: (1, top_k), I_r shape: (1, top_k)

    # top match (if any)
    if I_r.size and I_r[0][0] != -1:
        top_idx = int(I_r[0][0])
        top_match_path = paths[top_idx]
        
        # if you have labels, show label; otherwise derive from filename
        try:
            landmark_name = str(labels[top_idx])
        except Exception:
            landmark_name = os.path.basename(top_match_path).split("_")[1] if "_" in top_match_path else os.path.basename(top_match_path)
        st.markdown(f"### Landmark: **{landmark_name}**")
    else:
        st.markdown("### Landmark: **Unknown**")

    # --- Display top results (Grid + Hover Zoom + Base64 images) ---

    st.markdown("### Top Retrieved Images")

    cols_per_row = 4

    matches = []
    for idx, (match_idx, score) in enumerate(zip(I_r[0], D_r[0])):
        if int(match_idx) == -1:
            continue
        match_idx = int(match_idx)
        match_path = paths[match_idx]
        matches.append((match_path, float(score)))

    if not matches:
        st.info("No matches found.")
    else:
        row_cols = None
        for i, (mpath, score) in enumerate(matches):
            if i % cols_per_row == 0:
                row_cols = st.columns(cols_per_row)
            col = row_cols[i % cols_per_row]
            with col:
                try:
                    mime, b64 = get_image_base64(mpath)
                    img_html = f"""
                    <div style='text-align:center;'>
                        <img src='data:{mime};base64,{b64}' class='zoom-img' width='100%'/>
                        <div style='font-size:0.8rem; color:gray;'>
                            {os.path.basename(mpath)}<br>Sim: {score:.4f}
                        </div>
                    </div>
                    """
                    st.markdown(img_html, unsafe_allow_html=True)
                except FileNotFoundError:
                    st.warning(f"File not found: {mpath}")


else:
    st.info("Upload an image above to retrieve similar landmarks.")