import streamlit as st
from PIL import Image
import torch
import numpy as np
from torchvision import transforms
from sklearn.metrics.pairwise import cosine_similarity
import os
from DELG_Class import DELG
from Retrieval_Class import ImageRetrieval 

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "allModel/delg_global_finetune_local_pretrained.pth"
FEATURE_PATH = "features/delg_global_finetune_local_pretrained_features.npz"

# --- LOAD MODEL & FEATURES ---
@st.cache_resource
def load_retriever():
    retriever = ImageRetrieval(
        model_class=DELG,
        model_path=MODEL_PATH,
        device=DEVICE,
        use_global=True,
        use_local=True
    )
    retriever.load_features(FEATURE_PATH)
    return retriever

retriever = load_retriever()

# --- PAGE UI ---
st.title("Landmark Image Retrieval (DELG)")

uploaded_file = st.file_uploader("📸 Upload an image", type=["jpg", "jpeg", "png"])
top_k = st.slider("Top-K results", 1, 20, 5)

if uploaded_file:
    query_img = Image.open(uploaded_file).convert("RGB")
    st.image(query_img, caption="Query Image", use_container_width=True)

    # --- Extract global descriptor ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(query_img).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = retriever.model(img_tensor)
        if isinstance(outputs, dict) and "global" in outputs:
            query_feat = outputs["global"][0].cpu().numpy()
        elif isinstance(outputs, dict) and "global_descriptor" in outputs:
            query_feat = outputs["global_descriptor"][0].cpu().numpy()
        else:
            query_feat = outputs.cpu().numpy()

    # --- Compute cosine similarities ---
    all_paths = list(retriever.global_features.keys())
    all_feats = np.array(list(retriever.global_features.values()))
    sims = cosine_similarity(query_feat[None, :], all_feats)[0]
    top_idx = np.argsort(-sims)[:top_k]
    results = [(all_paths[i], sims[i]) for i in top_idx]

    # --- Extract landmark name from best match path ---
    top_match_path = results[0][0]
    # Example: dataset/train_Stonehenge_2320.jpg -> "Stonehenge"
    landmark_name = os.path.basename(top_match_path).split("_")[1] if "_" in top_match_path else "Unknown"

    st.markdown(f"### Landmark: **{landmark_name}**")

    # --- Display top results ---
    st.markdown("### Top Retrieved Images")
    cols = st.columns(top_k)
    for i, (path, score) in enumerate(results):
        with cols[i % top_k]:
            st.image(path, caption=f"{os.path.basename(path)}\nSim: {score:.4f}", use_container_width=True)
