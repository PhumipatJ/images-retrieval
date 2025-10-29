import streamlit as st

st.set_page_config(
    page_title="Home",  # Page title
    layout="wide" 
)
st.title("Image Retrieval System")

# Introduction
st.write("""
## About This Project
This project aims to build an image search system capable of searching and retrieving images 
from a digital repository with high precision and efficiency by using a training dataset. 
Traditional keyword-based search often fails to capture the semantic meaning of visual content, 
leading to inaccurate or incomplete results. This project leverages machine learning (ML) techniques to enable 
content-based image retrieval (CBIR) and semantic search, allowing users to find images
not only by textual metadata but also by visual similarity.
""")

st.write("""
## Models Implemented
The following models have been implemented and can be explored through the navigation buttons below:
- **DELG** 
- **ResNet50**
""")

st.divider()
example_query = "data/test/180.jpg"
example_results = [
    "dataset/train_Pyramids Of Giza - Egypt_1736.jpg",
    "dataset/train_Pyramids Of Giza - Egypt_1849.jpg",
    "dataset/train_Pyramids Of Giza - Egypt_1706.jpg",
    "dataset/train_Pyramids Of Giza - Egypt_1733.jpg",
    "dataset/train_Pyramids Of Giza - Egypt_1779.jpg",
]

# Layout: Query | Arrow | Retrievals
cols = st.columns([1, 0.2, 5])

with cols[0]:
    st.markdown("<center><b>User’s Query</b></center>", unsafe_allow_html=True)
    st.image(example_query, use_container_width=True)

with cols[1]:
    st.header("&emsp;  ->")  

with cols[2]:
    st.markdown("<center><b>Top-5 Retrieval Results</b></center>", unsafe_allow_html=True)
    result_cols = st.columns(5)
    for i, img_path in enumerate(example_results):
        with result_cols[i]:
            st.image(img_path, use_container_width=True)

st.divider()
# Navigation buttons
st.write("### Models DEMO:")
col1, col2 = st.columns(2)

with col1:
    if st.button("DELG Details"):
        st.switch_page("pages/1_delg_details.py")
    elif st.button("DELG with finetune Global  & Pretrained Local "):
        st.switch_page("pages/2_delg_global_finetune_local_pretrained.py")

with col2:
    if st.button("ResNet50 Pretrained"):
        st.switch_page("pages/3_resnet50.py")

# Footer
st.write("---")
st.write("6604062630439 Phumipat Jitreephot Sec 1")
st.write("6604062630501 Vorapon Witheethum Sec 1")