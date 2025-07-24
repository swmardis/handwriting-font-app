import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from fontTools.ttLib import TTFont, newTable
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables._g_l_y_f import Glyph
import os

st.set_page_config(layout="wide")
st.title("🖊️ Handwriting to Font — Label & Generate")

uploaded_file = st.file_uploader("Upload handwriting scan (png/jpg)")

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

def create_glyph_from_bitmap(bitmap):
    # Convert monochrome bitmap (PIL Image) to glyph
    bitmap = bitmap.convert("L")
    bitmap = bitmap.point(lambda x: 0 if x < 128 else 255, '1')
    width, height = bitmap.size
    # Create a simple glyph pen (bitmap font)
    pen = TTGlyphPen(None)
    # We will create a rectangle around bitmap to mimic glyph box
    pen.moveTo((0, 0))
    pen.lineTo((width, 0))
    pen.lineTo((width, height))
    pen.lineTo((0, height))
    pen.closePath()
    glyph = pen.glyph()
    glyph.width = width
    return glyph, width, height

def make_font(char_images, char_labels):
    font = TTFont()
    font.setGlyphOrder(['.notdef'] + char_labels)
    font['head'] = newTable('head')
    font['head'].setDefaults()
    font['hhea'] = newTable('hhea')
    font['hhea'].setDefaults()
    font['maxp'] = newTable('maxp')
    font['maxp'].setDefaults()
    font['OS/2'] = newTable('OS/2')
    font['OS/2'].setDefaults()
    font['post'] = newTable('post')
    font['post'].setDefaults()

    glyf = newTable('glyf')
    glyf.glyphs = {}
    advanceWidth = 600
    hmtx = {}
    cmap = {}

    for label, img in zip(char_labels, char_images):
        pil_img = pil_image_from_np(img)
        glyph, width, height = create_glyph_from_bitmap(pil_img)
        glyf.glyphs[label] = glyph
        hmtx[label] = (advanceWidth, 0)
        cmap[ord(label)] = label

    glyf.glyphs['.notdef'] = Glyph()
    glyf.glyphOrder = ['.notdef'] + char_labels
    font['glyf'] = glyf
    font['hmtx'] = newTable('hmtx')
    font['hmtx'].metrics = hmtx
    font['cmap'] = newTable('cmap')
    font['cmap'].tableVersion = 0
    cmap_subtable = {
        'platformID': 3,
        'platEncID': 1,
        'language': 0,
        'cmap': cmap
    }
    font['cmap'].tables = [type('CmapSubtable', (), cmap_subtable)()]

    font['name'] = newTable('name')
    font['name'].setName('HandwritingFont', 1, 3, 1, 0x409)  # Font Family name
    font['name'].setName('HandwritingFont Regular', 4, 3, 1, 0x409)  # Full name

    font['post'].formatType = 3.0

    font['OS/2'].usWinAscent = 800
    font['OS/2'].usWinDescent = 200
    font['hhea'].ascent = 800
    font['hhea'].descent = -200

    return font

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Uploaded Scan", use_column_width=True)

    chars = segment_image(image_np)

    if not chars:
        st.warning("No characters detected. Try a clearer image.")
    else:
        st.success(f"Detected {len(chars)} characters!")

        char_images = [c[0] for c in chars]
        char_labels = []

        st.write("### Label each character:")

        cols = st.columns(min(5, len(char_images)))
        for i, img in enumerate(char_images):
            with cols[i % 5]:
                st.image(img, width=60)
                label = st.text_input(f"Character #{i+1}", max_chars=1, key=f"label{i}")
                char_labels.append(label)

        if st.button("Generate & Download Font"):
            # Validate labels
            if any(len(l) != 1 for l in char_labels):
                st.error("Please label every character with exactly one letter/symbol.")
            else:
                font = make_font(char_images, char_labels)
                buf = io.BytesIO()
                font.save(buf)
                buf.seek(0)
                st.success("Font generated! Download below:")
                st.download_button("Download .TTF Font", buf, file_name="handwriting_font.ttf", mime="font/ttf")
