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

# ------------------ Session State ------------------
if "drawn_images" not in st.session_state:
    st.session_state.drawn_images = []
if "drawn_labels" not in st.session_state:
    st.session_state.drawn_labels = []

# ------------------ Font Tools ------------------

def create_font_from_template(template_path="template_font.ttf"):
    font = TTFont(template_path)
    glyphs = font['glyf'].glyphs
    for g in list(glyphs.keys()):
        if g != '.notdef':
            del glyphs[g]
    for table in font['cmap'].tables:
        table.cmap.clear()
    hmtx_metrics = font['hmtx'].metrics
    for k in list(hmtx_metrics.keys()):
        if k != ".notdef":
            del hmtx_metrics[k]
    return font

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
        if img is None or img.shape[0] == 0 or img.shape[1] == 0:
            continue  # Skip invalid
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

# ------------------ Crop (Safe) ------------------

def safe_crop(img):
    """Crop black/white canvas to bounding box, return grayscale mask."""
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    _, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
    coords = cv2.findNonZero(thresh)
    if coords is None:
        return None  # completely blank
    x, y, w, h = cv2.boundingRect(coords)
    cropped = thresh[y:y+h, x:x+w]
    return cropped

# ------------------ Draw Section ------------------

st.subheader("✏️ Draw a Character")

draw_label = st.text_input("Character Label (e.g. A, b, 3)", max_chars=1)

canvas_result = st_canvas(
    fill_color="rgba(255, 255, 255, 0)",  # Transparent fill
    stroke_width=5,
    stroke_color="#FFFFFF",              # White ink
    background_color="#000000",          # Black background
    width=200,
    height=200,
    drawing_mode="freedraw",
    key="canvas"
)

if st.button("➕ Add Drawn Character"):
    if canvas_result.image_data is not None and draw_label:
        raw = canvas_result.image_data.astype(np.uint8)
        img_rgb = cv2.cvtColor(raw, cv2.COLOR_RGBA2RGB)
        cropped = safe_crop(img_rgb)

        if cropped is None:
            st.warning("Canvas is empty. Draw something before adding.")
        else:
            st.session_state.drawn_images.append(cropped)
            st.session_state.drawn_labels.append(draw_label)
            st.success(f"✅ Added drawn character: '{draw_label}'")

# ------------------ Display Drawn ------------------

if st.session_state.drawn_images:
    st.write("### 🖼️ Drawn Characters")
    cols = st.columns(min(5, len(st.session_state.drawn_images)))
    for i, (img, label) in enumerate(zip(st.session_state.drawn_images, st.session_state.drawn_labels)):
        with cols[i % 5]:
            st.image(img, width=60)
            st.caption(f"'{label}'")

# ------------------ Upload Image ------------------

uploaded_file = st.file_uploader("📷 Or Upload Handwriting Image (PNG/JPG)")

char_images = []
char_labels = []

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    st.image(image, caption="Uploaded Image", use_column_width=True)

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

    chars = segment_image(image_np)

    if not chars:
        st.warning("No characters detected.")
    else:
        st.success(f"Detected {len(chars)} characters.")
        char_images = [c[0] for c in chars]

        st.write("### 🏷️ Label Each Character")
        cols = st.columns(min(5, len(char_images)))
        for i, img in enumerate(char_images):
            with cols[i % 5]:
                st.image(img, width=60)
                label = st.text_input(f"Char #{i+1}", max_chars=1, key=f"label_{i}")
                char_labels.append(label)

# ------------------ Generate Font ------------------

if st.button("🎉 Generate Font"):
    all_images = char_images + st.session_state.drawn_images
    all_labels = char_labels + st.session_state.drawn_labels

    if any(len(l) != 1 for l in all_labels):
        st.error("All characters must be labeled with exactly one character.")
    elif len(set(all_labels)) < len(all_labels):
        st.error("Duplicate character labels detected.")
    else:
        font = make_font(all_images, all_labels)
        buf = io.BytesIO()
        font.save(buf)
        buf.seek(0)
        st.download_button("⬇️ Download Your Font (.ttf)", buf, file_name="custom_font.ttf", mime="font/ttf")
