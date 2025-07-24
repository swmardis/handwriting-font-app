import streamlit as st
import numpy as np
import cv2
from PIL import Image
from fontTools.ttLib import TTFont, newTable
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables import _m_a_x_p
import io

st.set_page_config(layout="wide")
st.title("✍️ Handwriting to Font — Label & Generate")

def create_minimal_blank_font():
    font = TTFont()

    font.setGlyphOrder(['.notdef'])

    # Initialize tables except maxp
    for tag in ['head', 'hhea', 'name', 'OS/2', 'post', 'cmap', 'glyf', 'hmtx']:
        font[tag] = newTable(tag)

    # Properly initialize maxp table from its class
    maxp = _m_a_x_p.table__m_a_x_p()
    maxp.tableVersion = 1.0
    maxp.numGlyphs = 1
    maxp.maxPoints = 0
    maxp.maxContours = 0
    maxp.maxCompositePoints = 0
    maxp.maxCompositeContours = 0
    maxp.maxZones = 1
    maxp.maxTwilightPoints = 0
    maxp.maxStorage = 0
    maxp.maxFunctionDefs = 0
    maxp.maxInstructionDefs = 0
    maxp.maxStackElements = 0
    maxp.maxSizeOfInstructions = 0
    maxp.maxComponentElements = 0
    maxp.maxComponentDepth = 0
    font['maxp'] = maxp

    # head table minimal required fields
    font['head'].unitsPerEm = 1000
    font['head'].xMin = 0
    font['head'].yMin = 0
    font['head'].xMax = 1000
    font['head'].yMax = 1000
    font['head'].flags = 0
    font['head'].created = 0
    font['head'].modified = 0
    font['head'].macStyle = 0
    font['head'].lowestRecPPEM = 8
    font['head'].fontDirectionHint = 2
    font['head'].magicNumber = 0x5F0F3CF5
    font['head'].indexToLocFormat = 0
    font['head'].glyphDataFormat = 0

    # hhea table
    font['hhea'].ascent = 800
    font['hhea'].descent = -200
    font['hhea'].lineGap = 0
    font['hhea'].numberOfHMetrics = 1
    font['hhea'].advanceWidthMax = 1000
    font['hhea'].minLeftSideBearing = 0
    font['hhea'].minRightSideBearing = 0
    font['hhea'].xMaxExtent = 1000
    font['hhea'].caretSlopeRise = 1
    font['hhea'].caretSlopeRun = 0
    font['hhea'].caretOffset = 0
    font['hhea'].reserved0 = 0
    font['hhea'].reserved1 = 0
    font['hhea'].reserved2 = 0
    font['hhea'].reserved3 = 0

    # name table empty list (you can add metadata here later)
    font['name'].names = []

    # OS/2 minimal
    font['OS/2'].usFirstCharIndex = 0
    font['OS/2'].usLastCharIndex = 0
    font['OS/2'].sTypoAscender = 800
    font['OS/2'].sTypoDescender = -200
    font['OS/2'].usWinAscent = 800
    font['OS/2'].usWinDescent = 200
    font['OS/2'].fsSelection = 0
    font['OS/2'].panose = b'\x00'*10
    font['OS/2'].achVendID = 'NONE'

    # post table minimal
    font['post'].formatType = 3.0
    font['post'].underlinePosition = 0
    font['post'].underlineThickness = 0
    font['post'].isFixedPitch = 0
    font['post'].minMemType42 = 0
    font['post'].maxMemType42 = 0
    font['post'].minMemType1 = 0
    font['post'].maxMemType1 = 0

    # cmap table with valid format 4 subtable
    from fontTools.ttLib.tables._c_m_a_p import cmap_format_4
    cmap_subtable = cmap_format_4(4)
    cmap_subtable.platformID = 3
    cmap_subtable.platEncID = 1
    cmap_subtable.language = 0
    cmap_subtable.cmap = {}
    font['cmap'].tables = [cmap_subtable]

    # Initialize glyf and hmtx tables
    font['glyf'].glyphs = {}
    font['hmtx'].metrics = {}

    # Add empty .notdef glyph
    pen = TTGlyphPen(None)
    pen.moveTo((0, 0))
    pen.lineTo((0, 0))
    pen.closePath()
    glyph = pen.glyph()
    font['glyf'].glyphs['.notdef'] = glyph

    font['hmtx'].metrics['.notdef'] = (0, 0)

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
    font = create_minimal_blank_font()
    font.setGlyphOrder(['.notdef'] + char_labels)

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
