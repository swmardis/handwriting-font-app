import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from streamlit_drawable_canvas import st_canvas
import io

st.set_page_config(layout="wide")
st.title("✍️ Handwriting to Font")

EM_SIZE = 1000
SCALE_MULTIPLIER = 1.5

def create_font_from_template(template_path="template_font.ttf"):
    font = TTFont(template_path)

    # Remove all glyphs except .notdef
    glyphs = font['glyf'].glyphs
    for g in list(glyphs.keys()):
        if g != '.notdef':
            del glyphs[g]

    # Clear cmap subtables
    for table in font['cmap'].tables:
        table.cmap.clear()

    # Reset hmtx metrics except .notdef
    hmtx_metrics = font['hmtx'].metrics
    keys_to_remove = [k for k in hmtx_metrics if k != '.notdef']
    for key in keys_to_remove:
        del hmtx_metrics[key]

    return font

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

def glyph_from_image(img):
    contours, _ = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    pen = TTGlyphPen(None)

    img_height = img.shape[0]
    scale = EM_SIZE / img_height * SCALE_MULTIPLIER

    for cnt in contours:
        points = cnt[:, 0]
        if len(points) < 2:
            continue
        pen.moveTo((int(points[0][0] * scale), int((img_height - points[0][1]) * scale)))
        for pt in points[1:]:
            x = int(pt[0] * scale)
            y = int((img_height - pt[1]) * scale)
            pen.lineTo((x, y))
        pen.closePath()
    return pen.glyph()

def make_font(char_images, char_labels, template_path="template_font.ttf"):
    font = create_font_from_template(template_path)
    font.setGlyphOrder(['.notdef'] + char_labels)

    glyf = font['glyf']
    hmtx = font['hmtx']
    cmap_table = font['cmap'].tables[0]

    for label, img in zip(char_labels, char_images):
        glyph = glyph_from_image(img)
        glyf.glyphs[label] = glyph

        width = img.shape[1]
        scale = EM_SIZE / img.shape[0] * SCALE_MULTIPLIER
        advance_width = int(width * scale) + 80
        hmtx.metrics[label] = (advance_width, 0)
        cmap_table.cmap[ord(label)] = label

    font['maxp'].numGlyphs = len(char_labels) + 1
    font['hhea'].numberOfHMetrics = len(char_labels) + 1
    font['hhea'].ascent = int(EM_SIZE * SCALE_MULTIPLIER * 1.1)
    font['hhea'].descent = int(-EM_SIZE * SCALE_MULTIPLIER * 0.3)

    return font

uploaded_file = st.file_uploader("Upload a handwriting scan (PNG/JPG)")

char_images = []
char_labels = []

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

        st.write("### Label each character:")
        cols = st.columns(min(5, len(char_images)))
        for i, img in enumerate(char_images):
            with cols[i % 5]:
                st.image(img, width=60)
                label = st.text_input(f"Char #{i+1}", max_chars=1, key=f"label_{i}")
                char_labels.append(label)

# --- DRAWING CANVAS ---
st.write("### Or draw your own character")

canvas_result = st_canvas(
    fill_color="white",
    stroke_width=10,
    stroke_color="black",
    background_color="white",
    height=200,
    width=200,
    drawing_mode="freedraw",
    key="canvas",
)

draw_label = st.text_input("Label for drawn character", max_chars=1, key="draw_label")

if canvas_result.image_data is not None and draw_label:
    drawn_image = canvas_result.image_data.astype(np.uint8)
    gray = cv2.cvtColor(drawn_image, cv2.COLOR_RGBA2GRAY)
    _, binary = cv2.threshold(gray, 180, 255, cv2.THRESH_BINARY_INV)
    char_images.append(binary)
    char_labels.append(draw_label)

# --- GENERATE FONT ---
if char_images and char_labels and st.button("🎉 Generate Font"):
    if any(len(l) != 1 for l in char_labels):
        st.error("All characters must be labeled with exactly one character.")
    else:
        font = make_font(char_images, char_labels)
        buf = io.BytesIO()
        font.save(buf)
        buf.seek(0)
        st.download_button("Download Your Font (.ttf)", buf, file_name="custom_font.ttf", mime="font/ttf")
