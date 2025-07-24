import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables import _g_l_y_f, _h_m_t_x, _c_m_a_p, _m_a_x_p, _h_h_e_a
import io
import os

TEMPLATE_FONT_PATH = "blank_template.ttf"

st.set_page_config(layout="wide")
st.title("✍️ Handwriting to Font — Label & Generate")

# --- Character Segmentation ---
def segment_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    chars = []
    for cnt in sorted(contours, key=lambda x: cv2.boundingRect(x)[0]):
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 100:
            char_img = thresh[y:y+h, x:x+w]
            chars.append((char_img, (x, y, w, h)))
    return chars

def pil_image_from_np(np_img):
    return Image.fromarray(255 - np_img)

# --- Glyph Creation ---
def create_glyph(width, height):
    pen = TTGlyphPen(None)
    pen.moveTo((0, 0))
    pen.lineTo((width, 0))
    pen.lineTo((width, height))
    pen.lineTo((0, height))
    pen.closePath()
    return pen.glyph()

# --- Font Generation ---
def make_font_from_template(char_images, char_labels):
    font = TTFont(TEMPLATE_FONT_PATH)
    font.setGlyphOrder(['.notdef'] + char_labels)

    # Add missing tables
    if 'glyf' not in font:
        font['glyf'] = _g_l_y_f.table__glyf()
    if 'hmtx' not in font:
        font['hmtx'] = _h_m_t_x.table__hmtx()
    if 'cmap' not in font:
        font['cmap'] = _c_m_a_p.table__cmap()
        font['cmap'].tables = [_c_m_a_p.cmap_format_4(platformID=3, platEncID=1, language=0, cmap={})]
    if 'maxp' not in font:
        font['maxp'] = _m_a_x_p.table__maxp()
    if 'hhea' not in font:
        font['hhea'] = _h_h_e_a.table__hhea()

    glyf = font['glyf']
    hmtx = font['hmtx']
    cmap_table = font['cmap'].tables[0]

    for label, img in zip(char_labels, char_images):
        pil_img = pil_image_from_np(img)
        width, height = pil_img.size
        glyph = create_glyph(width, height)
        glyf.glyphs[label] = glyph
        hmtx.metrics[label] = (width, 0)
        cmap_table.cmap[ord(label)] = label

    font['maxp'].numGlyphs = len(char_labels) + 1
    font['hhea'].numberOfHMetrics = len(char_labels) + 1

    return font

# --- Streamlit UI ---
uploaded_file = st.file_uploader("Upload a handwriting scan (PNG/JPG)")

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    chars = segment_image(image_np)

    if not chars:
        st.warning("No characters detected. Try a clearer or higher-contrast image.")
    else:
        st.success(f"Detected {len(chars)} characters.")
        char_images = [c[0] for c in chars]
        char_labels = []

        st.write("### Label each character:")
        cols = st.columns(min(5, len(char_images)))

        for i, img in enumerate(char_images):
            with cols[i % 5]:
                st.image(img, width=60)
                label = st.text_input(f"Char #{i+1}", max_chars=1, key=f"label_{i}")
                char_labels.append(label)

        if st.button("🎉 Generate Font"):
            if any(len(l) != 1 for l in char_labels):
                st.error("All characters must be labeled with exactly one character.")
            elif not os.path.exists(TEMPLATE_FONT_PATH):
                st.error("Template font missing. Please upload `blank_template.ttf`.")
            else:
                font = make_font_from_template(char_images, char_labels)
                buf = io.BytesIO()
                font.save(buf)
                buf.seek(0)
                st.download_button("Download Your Font (.ttf)", buf, file_name="custom_font.ttf", mime="font/ttf")
