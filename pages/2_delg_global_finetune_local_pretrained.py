import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import os
from DELG_Class import DELG
from Retrieval_Class import ImageRetrieval 
import base64

def get_image_base64(path):
    with open(path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

# --- CONFIG ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_PATH = "allModel/delg_global_finetune_local_pretrained.pth"
FEATURE_PATH = "features/delg_global_finetune_local_pretrained_features.npz"

st.set_page_config(
    page_title="DELG",
    layout="wide"
)

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
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
top_k = st.slider("Top-K results", 1, 100, 5)

if uploaded_file:
    # --- Display Query Image ---
    query_img = Image.open(uploaded_file).convert("RGB")
    cols = st.columns([0.2, 2.6, 0.2])
    with cols[1]:
        st.image(query_img, caption="Query Image", use_container_width=True)

    # --- Transform Query Image ---
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(query_img).unsqueeze(0)

    # --- Use Class Query Function ---
    results = retriever.query(img_tensor, top_k=top_k)

    # --- Retrieve paths and compute similarity scores (optional: skip scores) ---
    result_paths = results if isinstance(results[0], str) else [r[0] for r in results]

    # --- Get landmark name from top result ---
    top_match_path = result_paths[0]
    landmark_name = os.path.basename(top_match_path).split("_")[1] if "_" in top_match_path else "Unknown"
    st.markdown(f"### Landmark: **{landmark_name}**")

    # --- Display results with Hover Zoom ---
    st.markdown("### Top Retrieved Images")
    st.markdown("""
    <style>
    .image-container img {
        width: 100%;
        height: auto;
        transition: transform 0.3s ease-in-out;
        border-radius: 10px;
        cursor: pointer;
    }
    .image-container:hover img {
        transform: scale(1.5);
        z-index: 10;
        position: relative;
    }
    .caption {
        font-size: 12px;
        text-align: center;
        margin-top: 4px;
    }
    </style>
    """, unsafe_allow_html=True)

    cols_per_row = 4
    for idx, path in enumerate(result_paths):
        if idx % cols_per_row == 0:
            cols = st.columns(cols_per_row)
        with cols[idx % cols_per_row]:
            img_base64 = get_image_base64(path)
            st.markdown(
                f"""
                <div class="image-container">
                    <img src="data:image/jpeg;base64,{img_base64}">
                </div>
                <div class="caption">
                    {os.path.basename(path)}
                </div>
                """,
                unsafe_allow_html=True
            )
else:
    st.info("Upload an image above to retrieve similar landmarks.")
