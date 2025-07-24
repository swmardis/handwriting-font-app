import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fontTools.ttLib import TTFont
from fontTools.pens.ttGlyphPen import TTGlyphPen
from streamlit_drawable_canvas import st_canvas
import io

st.set_page_config(layout="wide")
st.title("✍️ Full Font Creator — Draw or Upload Each Character")

EM_SIZE = 1000
SCALE_MULTIPLIER = 1.5

# Full character set: uppercase, lowercase, digits, punctuation
CHAR_SET = list(
    "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    "abcdefghijklmnopqrstuvwxyz"
    "0123456789"
    ".,!?;:-_'\"@#$%&*()[]{}<>/\\|+=~`^"
)

# Initialize session state dicts to store images for characters
if "char_images" not in st.session_state:
    st.session_state.char_images = {c: None for c in CHAR_SET}

def create_font_from_template(template_path="template_font.ttf"):
    font = TTFont(template_path)

    glyphs = font['glyf'].glyphs
    for g in list(glyphs.keys()):
        if g != '.notdef':
            del glyphs[g]

    for table in font['cmap'].tables:
        table.cmap.clear()

    hmtx_metrics = font['hmtx'].metrics
    keys_to_remove = [k for k in hmtx_metrics if k != '.notdef']
    for key in keys_to_remove:
        del hmtx_metrics[key]

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

def process_image_for_font(img):
    # Convert RGBA or RGB image (PIL) to binary numpy array for glyph generation
    img = img.convert("L")  # grayscale
    np_img = np.array(img)
    _, thresh = cv2.threshold(np_img, 200, 255, cv2.THRESH_BINARY_INV)
    return thresh

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

st.write("### Input your characters below: For each character you can either upload an image or draw it.")

# Layout: split characters into columns for compactness
cols_per_row = 5
for i in range(0, len(CHAR_SET), cols_per_row):
    cols = st.columns(cols_per_row)
    for j, c in enumerate(CHAR_SET[i:i+cols_per_row]):
        with cols[j]:
            st.markdown(f"**'{c}'**")
            # Upload input
            uploaded = st.file_uploader(f"Upload '{c}'", type=['png','jpg','jpeg'], key=f"upload_{c}", help="Upload an image of this character")
            # Draw canvas input
            canvas_result = st_canvas(
                fill_color="white",
                stroke_width=10,
                stroke_color="black",
                background_color="white",
                height=150,
                width=150,
                drawing_mode="freedraw",
                key=f"canvas_{c}",
            )
            # Decide which input to use: Upload preferred over canvas if both exist
            img_for_char = None
            if uploaded is not None:
                try:
                    pil_img = Image.open(uploaded).convert("L")
                    img_for_char = process_image_for_font(pil_img)
                except:
                    st.error("Invalid image")
            elif canvas_result.image_data is not None:
                raw = canvas_result.image_data.astype(np.uint8)
                pil_img = Image.fromarray(cv2.cvtColor(raw, cv2.COLOR_RGBA2RGB))
                img_for_char = process_image_for_font(pil_img)

            # Save processed image or None
            st.session_state.char_images[c] = img_for_char

if st.button("🎉 Generate Font"):
    # Filter out empty inputs
    filled_chars = [c for c, img in st.session_state.char_images.items() if img is not None]
    filled_images = [img for img in st.session_state.char_images.values() if img is not None]

    if not filled_chars:
        st.error("Please provide at least one character (draw or upload).")
    else:
        font = make_font(filled_images, filled_chars)
        buf = io.BytesIO()
        font.save(buf)
        buf.seek(0)
        st.download_button("⬇️ Download Your Font (.ttf)", buf, file_name="custom_font.ttf", mime="font/ttf")
