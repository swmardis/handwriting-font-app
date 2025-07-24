import streamlit as st
import cv2
import numpy as np
from PIL import Image
import io
from fontTools.ttLib import TTFont, newTable
from fontTools.pens.ttGlyphPen import TTGlyphPen
from fontTools.ttLib.tables._g_l_y_f import Glyph

from fontTools.ttLib.tables._h_e_a_d import table__h_e_a_d
from fontTools.ttLib.tables._h_h_e_a import table__h_h_e_a
from fontTools.ttLib.tables._m_a_x_p import table__m_a_x_p
from fontTools.ttLib.tables._o_s_2f_2 import table__O_S_2f_2
from fontTools.ttLib.tables._p_o_s_t import table__p_o_s_t

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
    bitmap = bitmap.convert("L")
    bitmap = bitmap.point(lambda x: 0 if x < 128 else 255, '1')
    width, height = bitmap.size
    pen = TTGlyphPen(None)
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

    # Initialize tables without setDefaults
    font['head'] = table__h_e_a_d()
    font['head'].tableVersion = 1.0
    font['head'].fontRevision = 1.0
    font['head'].flags = 0
    font['head'].unitsPerEm = 1000
    font['head'].created = 0
    font['head'].modified = 0
    font['head'].xMin = 0
    font['head'].yMin = 0
    font['head'].xMax = 0
    font['head'].yMax = 0
    font['head'].macStyle = 0
    font['head'].lowestRecPPEM = 8
    font['head'].fontDirectionHint = 2
    font['head'].indexToLocFormat = 0
    font['head'].glyphDataFormat = 0

    font['hhea'] = table__h_h_e_a()
    font['hhea'].tableVersion = 0x00010000
    font['hhea'].ascent = 800
    font['hhea'].descent = -200
    font['hhea'].lineGap = 0
    font['hhea'].advanceWidthMax = 600
    font['hhea'].minLeftSideBearing = 0
    font['hhea'].minRightSideBearing = 0
    font['hhea'].xMaxExtent = 0
    font['hhea'].caretSlopeRise = 1
    font['hhea'].caretSlopeRun = 0
    font['hhea'].caretOffset = 0
    font['hhea'].reserved0 = 0
    font['hhea'].reserved1 = 0
    font['hhea'].reserved2 = 0
    font['hhea'].reserved3 = 0
    font['hhea'].metricDataFormat = 0
    font['hhea'].numberOfHMetrics = len(char_labels) + 1

    font['maxp'] = table__m_a_x_p()
    font['maxp'].tableVersion = 0x00010000
    font['maxp'].numGlyphs = len(char_labels) + 1

    font['OS/2'] = table__O_S_2f_2()
    font['OS/2'].version = 3
    font['OS/2'].xAvgCharWidth = 500
    font['OS/2'].usWeightClass = 400
    font['OS/2'].usWidthClass = 5
    font['OS/2'].fsType = 0
    font['OS/2'].ySubscriptXSize = 650
    font['OS/2'].ySubscriptYSize = 600
    font['OS/2'].ySubscriptXOffset = 0
    font['OS/2'].ySubscriptYOffset = 75
    font['OS/2'].ySuperscriptXSize = 650
    font['OS/2'].ySuperscriptYSize = 600
    font['OS/2'].ySuperscriptXOffset = 0
    font['OS/2'].ySuperscriptYOffset = 350
    font['OS/2'].yStrikeoutSize = 50
    font['OS/2'].yStrikeoutPosition = 250
    font['OS/2'].sFamilyClass = 0
    font['OS/2'].panose = b"\\x00" * 10
    font['OS/2'].ulUnicodeRange1 = 0xFFFFFFFF
    font['OS/2'].ulUnicodeRange2 = 0x0
    font['OS/2'].ulUnicodeRange3 = 0x0
    font['OS/2'].ulUnicodeRange4 = 0x0
    font['OS/2'].achVendID = "PYFT"
    font['OS/2'].fsSelection = 0x40
    font['OS/2'].usFirstCharIndex = min(ord(l) for l in char_labels)
    font['OS/2'].usLastCharIndex = max(ord(l) for l in char_labels)
    font['OS/2'].sTypoAscender = 800
    font['OS/2'].sTypoDescender = -200
    font['OS/2'].sTypoLineGap = 0
    font['OS/2'].usWinAscent = 800
    font['OS/2'].usWinDescent = 200

    font['post'] = table__p_o_s_t()
    font['post'].formatType = 3.0
    font['post'].italicAngle = 0
    font['post'].underlinePosition = 0
    font['post'].underlineThickness = 0
    font['post'].isFixedPitch = 0
    font['post'].minMemType42 = 0
    font['post'].maxMemType42 = 0
    font['post'].minMemType1 = 0
    font['post'].maxMemType1 = 0

    glyf = newTable('glyf')
    glyf.glyphs = {}
    hmtx = {}
    cmap = {}

    advanceWidth = 600

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

    return font

def pil_image_from_np(np_img):
    return Image.fromarray(255 - np_img)

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

uploaded_file = st.file_uploader("Upload handwriting scan (png/jpg)")

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
            if any(len(l) != 1 for l in char_labels):
                st.error("Please label every character with exactly one letter/symbol.")
            else:
                font = make_font(char_images, char_labels)
                buf = io.BytesIO()
                font.save(buf)
                buf.seek(0)
                st.success("Font generated! Download below:")
                st.download_button("Download .TTF Font", buf, file_name="handwriting_font.ttf", mime="font/ttf")
