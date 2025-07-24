import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
import io

st.set_page_config(layout="wide")
st.title("✍️ Handwriting to Font — Label & Generate")

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

def create_glyph(width, height):
    pen = TTGlyphPen(None)
    pen.moveTo((0, 0))
    pen.lineTo((width, 0))
    pen.lineTo((width, height))
    pen.lineTo((0, height))
    pen.closePath()
    return pen.glyph()

def make_font(char_images, char_labels):
    font = TTFont()
    glyph_order = ['.notdef'] + char_labels
    font.setGlyphOrder(glyph_order)

    # Create required tables
    font['head'] = font.newTable('head')
    font['head'].unitsPerEm = 1000
    font['head'].xMin = 0
    font['head'].yMin = 0
    font['head'].xMax = 1000
    font['head'].yMax = 1000
    font['head'].indexToLocFormat = 0
    font['head'].glyphDataFormat = 0

    font['hhea'] = font.newTable('hhea')
    font['hhea'].ascent = 800
    font['hhea'].descent = -200
    font['hhea'].lineGap = 0
    font['hhea'].numberOfHMetrics = len(glyph_order)

    font['maxp'] = font.newTable('maxp')
    font['maxp'].numGlyphs = len(glyph_order)

    font['OS/2'] = font.newTable('OS/2')
    font['OS/2'].usFirstCharIndex = min(ord(l) for l in char_labels)
    font['OS/2'].usLastCharIndex = max(ord(l) for l in char_labels)
    font['OS/2'].sTypoAscender = 800
    font['OS/2'].sTypoDescender = -200
    font['OS/2'].usWinAscent = 800
    font['OS/2'].usWinDescent = 200

    font['post'] = font.newTable('post')
    font['post'].formatType = 3.0

    glyf_table = font.newTable('glyf')
    glyf_table.glyphs = {}
    font['glyf'] = glyf_table

    font['hmtx'] = font.newTable('hmtx')
    font['hmtx'].metrics = {}

    cmap = {}
    for label, np_img in zip(char_labels, char_images):
        img = pil_image_from_np(np_img)
        width, height = img.size
        glyph = create_glyph(width, height)
        font['glyf'].glyphs[label] = glyph
        font['hmtx'].metrics[label] = (width, 0)
        cmap[ord(label)] = label

    from fontTools.ttLib.tables._c_m_a_p import cmap_format_4
    cmap_table = font.newTable('cmap')
    cmap_table.tableVersion = 0
    cmap_format = cmap_format_4(4)
    cmap_format.platformID = 3
    cmap_format.platEncID = 1
    cmap_format.language = 0
    cmap_format.cmap = cmap
    cmap_table.tables = [cmap_format]
    font['cmap'] = cmap_table

    font['glyf'].glyphs['.notdef'] = create_glyph(500, 500)

    return font

# --- UI START ---

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
            else:
                font = make_font(char_images, char_labels)
                buf = io.BytesIO()
                font.save(buf)
                buf.seek(0)
                st.download_button("Download Your Font (.ttf)", buf, file_name="custom_font.ttf", mime="font/ttf")
