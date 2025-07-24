import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os

st.set_page_config(layout="wide")
st.title("🖊️ Handwriting to Font (Mobile Preview)")

uploaded_file = st.file_uploader("📷 Upload a handwriting scan", type=["png", "jpg", "jpeg"])

def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for cnt in sorted(contours, key=lambda x: cv2.boundingRect(x)[0]):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100:
            char_img = thresh[y:y+h, x:x+w]
            chars.append(char_img)
    return chars

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="📄 Uploaded Scan", use_column_width=True)

    st.header("🔍 Segmenting Characters...")
    chars = segment_image(image_np)

    if chars:
        st.success(f"✅ Found {len(chars)} characters")
        cols = st.columns(min(5, len(chars)))
        for i, char in enumerate(chars):
            with cols[i % len(cols)]:
                st.image(char, width=60, caption=f"{i+1}")
        if st.button("💾 Save Characters as PNGs"):
            os.makedirs("chars", exist_ok=True)
            for i, char in enumerate(chars):
                Image.fromarray(255 - char).save(f"chars/char_{i}.png")
            st.success("Saved to /chars folder")
    else:
        st.warning("⚠️ No characters found. Try a clearer image.")