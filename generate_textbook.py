#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SamOpenCVWeb 한글 PDF 교재 생성 스크립트
실행: python generate_textbook.py
출력: SamOpenCVWeb_교재.pdf
"""
import os, sys, math, io, textwrap, json
from collections import OrderedDict
from fpdf import FPDF
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_PATH = os.path.join(BASE_DIR, 'SamOpenCVWeb_교재.pdf')
FONT_REG = r'C:\Windows\Fonts\malgun.ttf'
FONT_BOLD = r'C:\Windows\Fonts\malgunbd.ttf'

CATEGORIES = OrderedDict([
    ('io',           ('#4CAF50', '입출력 (I/O)')),
    ('color',        ('#9C27B0', '색상 (Color)')),
    ('filter',       ('#FF9800', '필터/블러 (Filter)')),
    ('edge',         ('#F44336', '에지 검출 (Edge)')),
    ('threshold',    ('#795548', '임계값 (Threshold)')),
    ('morph',        ('#607D8B', '형태학 (Morphology)')),
    ('contour',      ('#009688', '컨투어 (Contour)')),
    ('transform',    ('#00BCD4', '변환 (Transform)')),
    ('histogram',    ('#673AB7', '히스토그램 (Histogram)')),
    ('feature',      ('#CDDC39', '특징 검출 (Feature)')),
    ('drawing',      ('#E91E63', '드로잉 (Drawing)')),
    ('arithmetic',   ('#3F51B5', '산술 연산 (Arithmetic)')),
    ('detection',    ('#FF5722', '검출 (Detection)')),
    ('segmentation', ('#795548', '분할 (Segmentation)')),
    ('value',        ('#78909C', '값 (Value)')),
    ('control',      ('#26A69A', '제어 흐름 (Control)')),
    ('script',       ('#FFC107', '스크립트 (Script)')),
])

def hex_to_rgb(h):
    h = h.lstrip('#')
    return tuple(int(h[i:i+2], 16) for i in (0, 2, 4))

def render_pipeline(nodes, width=760, node_h=54):
    """Render pipeline diagram. nodes=[(label, color_hex, sub_label), ...]"""
    n = len(nodes)
    if n == 0:
        return None
    pad = 30
    node_w = 120
    gap = 50
    total_w = n * node_w + (n - 1) * gap + pad * 2
    if total_w > width:
        node_w = max(80, (width - pad*2 - (n-1)*gap) // n)
        total_w = n * node_w + (n-1)*gap + pad*2
    img_w = max(width, total_w)
    img_h = node_h + pad * 2 + 20
    img = Image.new('RGB', (img_w, img_h), (30, 30, 46))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_REG, 13)
        font_sm = ImageFont.truetype(FONT_REG, 10)
    except:
        font = ImageFont.load_default()
        font_sm = font
    start_x = (img_w - (n*node_w + (n-1)*gap)) // 2
    centers = []
    for i, (label, color, sub) in enumerate(nodes):
        x = start_x + i * (node_w + gap)
        y = pad
        rgb = hex_to_rgb(color)
        # Node body
        draw.rounded_rectangle([x, y, x+node_w, y+node_h], radius=8, fill=(45, 45, 60), outline=rgb, width=2)
        # Top color bar
        draw.rounded_rectangle([x, y, x+node_w, y+16], radius=8, fill=rgb)
        draw.rectangle([x, y+8, x+node_w, y+16], fill=rgb)
        # Label
        bbox = draw.textbbox((0,0), label, font=font)
        tw = bbox[2] - bbox[0]
        tx = x + (node_w - tw) // 2
        draw.text((tx, y+20), label, fill=(220,220,220), font=font)
        # Sub label
        if sub:
            bbox2 = draw.textbbox((0,0), sub, font=font_sm)
            tw2 = bbox2[2] - bbox2[0]
            draw.text((x+(node_w-tw2)//2, y+36), sub, fill=(160,160,180), font=font_sm)
        # Ports
        port_y = y + node_h // 2
        draw.ellipse([x-5, port_y-5, x+5, port_y+5], fill=(100,100,120), outline=(180,180,200))
        draw.ellipse([x+node_w-5, port_y-5, x+node_w+5, port_y+5], fill=(100,100,120), outline=(180,180,200))
        centers.append((x, x+node_w, port_y))
    # Connection lines
    for i in range(len(centers)-1):
        x1 = centers[i][1] + 5
        x2 = centers[i+1][0] - 5
        y = centers[i][2]
        # Bezier-like curve
        mid = (x1+x2)//2
        for t_i in range(20):
            t = t_i / 20.0
            t2 = (t_i+1) / 20.0
            px1 = int((1-t)**3*x1 + 3*(1-t)**2*t*mid + 3*(1-t)*t**2*mid + t**3*x2)
            py1 = int((1-t)**3*y + 3*(1-t)**2*t*(y-10) + 3*(1-t)*t**2*(y+10) + t**3*y)
            px2 = int((1-t2)**3*x1 + 3*(1-t2)**2*t2*mid + 3*(1-t2)*t2**2*mid + t2**3*x2)
            py2 = int((1-t2)**3*y + 3*(1-t2)**2*t2*(y-10) + 3*(1-t2)*t2**2*(y+10) + t2**3*y)
            draw.line([(px1,py1),(px2,py2)], fill=(180,180,200), width=2)
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


# ═══════════════════════════════════════════════════════════════
# Node type → thumbnail key mapping
# ═══════════════════════════════════════════════════════════════
THUMB_MAP = {
    'image_read':'original','video_read':'original','camera_capture':'original',
    'image_show':None,'image_write':None,
    'cvt_color':'gray','in_range':'binary','histogram_eq':'enhanced',
    'split_channels':'channel','merge_channels':'original',
    'gaussian_blur':'blurred','median_blur':'blurred','bilateral_filter':'blurred',
    'box_filter':'blurred','sharpen':'sharpened','filter2d':'filtered',
    'canny':'edges','sobel':'edges','laplacian':'edges','scharr':'edges',
    'threshold':'binary','adaptive_threshold':'binary','otsu_threshold':'binary',
    'morphology':'morphed','dilate':'dilated','erode':'eroded','structuring_element':None,
    'find_contours':'contours','draw_contours':'contours','bounding_rect':'bboxes',
    'min_enclosing_circle':'circles','convex_hull':'contours','approx_poly':'contours',
    'contour_area':'contours','contour_properties':'contours',
    'resize':'original','rotate':'rotated','flip':'flipped','crop':'cropped',
    'warp_affine':'warped','warp_perspective':'warped','remap':'warped',
    'harris_corner':'corners','good_features':'corners','orb_features':'features',
    'fast_features':'features','match_features':'matched','hough_lines':'lines',
    'draw_line':'drawn','draw_rectangle':'drawn','draw_circle':'drawn',
    'draw_ellipse':'drawn','draw_text':'text_drawn','draw_polylines':'drawn',
    'add_weighted':'blended','bitwise_and':'masked','bitwise_or':'combined',
    'bitwise_not':'inverted','bitwise_xor':'combined',
    'add':'brightened','subtract':'darkened','multiply':'original','absdiff':'diff',
    'haar_cascade':'face_detect','hough_circles':'circles_detect',
    'template_match':'matched','image_extract':'cropped',
    'flood_fill':'filled','grabcut':'segmented','watershed':'segmented',
    'calc_histogram':'histogram',
    'python_script':'custom',
    'control_for':'original','control_while':'original','control_if':'original','control_switch':'original',
    'val_integer':None,'val_float':None,'val_boolean':None,
    'val_point':None,'val_scalar':None,'val_math':None,'val_list':None,
}

_THUMBS_CACHE = None

def generate_thumbnails(tw=48, th=36):
    """Generate sample processing result thumbnails using OpenCV."""
    global _THUMBS_CACHE
    if _THUMBS_CACHE is not None:
        return _THUMBS_CACHE
    # Create a sample image with shapes
    img = np.zeros((120, 160, 3), dtype=np.uint8)
    img[:] = (200, 180, 140)
    cv2.rectangle(img, (20, 15), (75, 75), (40, 40, 200), -1)
    cv2.circle(img, (120, 55), 28, (200, 50, 30), -1)
    cv2.line(img, (10, 100), (150, 100), (60, 60, 60), 2)
    cv2.putText(img, 'Abc', (35, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 140, 0), 2)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, binary = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    k5 = np.ones((5, 5), np.uint8)
    t = {}
    def r(i): return cv2.resize(i, (tw, th))
    def g2b(i): return cv2.cvtColor(i, cv2.COLOR_GRAY2BGR)
    t['original'] = r(img)
    t['gray'] = r(g2b(gray))
    t['blurred'] = r(cv2.GaussianBlur(img, (11, 11), 0))
    t['edges'] = r(g2b(cv2.Canny(gray, 50, 150)))
    t['binary'] = r(g2b(binary))
    t['dilated'] = r(g2b(cv2.dilate(binary, k5, iterations=2)))
    t['eroded'] = r(g2b(cv2.erode(binary, k5, iterations=2)))
    t['morphed'] = r(g2b(cv2.morphologyEx(binary, cv2.MORPH_OPEN, k5)))
    t['sharpened'] = r(cv2.filter2D(img, -1, np.array([[-1,-1,-1],[-1,9,-1],[-1,-1,-1]])))
    c_img = img.copy()
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(c_img, cnts, -1, (0, 255, 0), 2)
    t['contours'] = r(c_img)
    bb_img = img.copy()
    for c in cnts:
        x2, y2, w2, h2 = cv2.boundingRect(c)
        cv2.rectangle(bb_img, (x2, y2), (x2+w2, y2+h2), (0, 255, 0), 2)
    t['bboxes'] = r(bb_img)
    ci_img = img.copy()
    for c in cnts:
        (cx, cy), rad = cv2.minEnclosingCircle(c)
        cv2.circle(ci_img, (int(cx), int(cy)), int(rad), (0, 255, 0), 2)
    t['circles'] = r(ci_img)
    t['inverted'] = r(cv2.bitwise_not(img))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    t['enhanced'] = r(g2b(clahe.apply(gray)))
    M = cv2.getRotationMatrix2D((80, 60), 25, 1.0)
    t['rotated'] = r(cv2.warpAffine(img, M, (160, 120)))
    t['flipped'] = r(cv2.flip(img, 1))
    t['cropped'] = r(img[15:75, 20:80])
    t['blended'] = r(cv2.addWeighted(img, 0.6, np.full_like(img, 100), 0.4, 0))
    t['masked'] = r(cv2.bitwise_and(img, g2b(binary)))
    t['combined'] = r(cv2.bitwise_or(g2b(cv2.Canny(gray, 50, 150)), g2b(cv2.Sobel(gray, cv2.CV_8U, 1, 1, ksize=3))))
    t['diff'] = r(cv2.absdiff(img, np.roll(img, 15, axis=1)))
    fc = img.copy()
    cv2.rectangle(fc, (35, 10), (125, 90), (0, 255, 0), 2)
    t['face_detect'] = r(fc)
    ln = img.copy()
    cv2.line(ln, (5, 10), (155, 115), (0, 0, 255), 2)
    cv2.line(ln, (5, 60), (155, 60), (0, 0, 255), 2)
    t['lines'] = r(ln)
    cd = img.copy()
    cv2.circle(cd, (50, 50), 28, (0, 0, 255), 2)
    cv2.circle(cd, (120, 60), 22, (0, 0, 255), 2)
    t['circles_detect'] = r(cd)
    t['filled'] = r(cv2.rectangle(img.copy(), (20, 15), (75, 75), (0, 255, 255), -1))
    seg = img.copy()
    seg[binary > 0] = [60, 200, 60]
    t['segmented'] = r(seg)
    h_img = np.ones((120, 160, 3), dtype=np.uint8) * 240
    for ch, col in enumerate([(200, 60, 60), (60, 200, 60), (60, 60, 200)]):
        hist = cv2.calcHist([img], [ch], None, [32], [0, 256])
        cv2.normalize(hist, hist, 0, 100, cv2.NORM_MINMAX)
        for i in range(32):
            cv2.rectangle(h_img, (i*5, 120-int(hist[i][0])), ((i+1)*5-1, 119), col, -1)
    t['histogram'] = r(h_img)
    pts1 = np.float32([[10, 10], [150, 20], [0, 110], [155, 100]])
    pts2 = np.float32([[0, 0], [160, 0], [0, 120], [160, 120]])
    t['warped'] = r(cv2.warpPerspective(img, cv2.getPerspectiveTransform(pts1, pts2), (160, 120)))
    td = img.copy()
    cv2.putText(td, '(C)', (45, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    t['text_drawn'] = r(td)
    dr = img.copy()
    cv2.rectangle(dr, (10, 10), (60, 60), (255, 255, 0), 2)
    cv2.circle(dr, (120, 80), 18, (255, 0, 255), 2)
    t['drawn'] = r(dr)
    inv_g = 255 - gray
    bl_inv = cv2.GaussianBlur(inv_g, (21, 21), 0)
    sk = cv2.divide(gray, 255 - bl_inv, scale=256)
    t['custom'] = r(g2b(sk))
    ft = img.copy()
    pts = cv2.goodFeaturesToTrack(gray, 30, 0.01, 10)
    if pts is not None:
        for p in pts:
            px, py = p.ravel()
            cv2.circle(ft, (int(px), int(py)), 3, (0, 0, 255), -1)
    t['features'] = r(ft)
    t['corners'] = t['features']
    mt = np.zeros((120, 160, 3), dtype=np.uint8)
    mt[:, :80] = cv2.resize(img, (80, 120))
    mt[:, 80:] = cv2.resize(img, (80, 120))
    for i in range(5):
        yl = 15 + i * 22
        cv2.line(mt, (35+i*4, yl), (115-i*4, yl), (0, 255, 0), 1)
    t['matched'] = r(mt)
    t['brightened'] = r(np.clip(img.astype(np.int16) + 60, 0, 255).astype(np.uint8))
    t['darkened'] = r(np.clip(img.astype(np.int16) - 60, 0, 255).astype(np.uint8))
    t['channel'] = r(g2b(gray))
    ek = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]])
    t['filtered'] = r(g2b(np.clip(np.abs(cv2.filter2D(gray.astype(np.float32), -1, ek)), 0, 255).astype(np.uint8)))
    # Convert all BGR to RGB for Pillow
    for k in t:
        t[k] = cv2.cvtColor(t[k], cv2.COLOR_BGR2RGB)
    _THUMBS_CACHE = t
    return t


def render_graph_from_json(json_path, max_w=760, max_h=340):
    """Render a 2D node graph from a JSON example file with image preview thumbnails."""
    thumbs = generate_thumbnails()
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    nodes = data.get('nodes', [])
    connections = data.get('connections', [])
    if not nodes:
        return None
    # Node dimensions (in graph space)
    NW, NH = 110, 88
    THUMB_W, THUMB_H = 48, 36
    PAD = 30
    # Bounding box
    xs = [n['x'] for n in nodes]
    ys = [n['y'] for n in nodes]
    min_x, min_y = min(xs), min(ys)
    max_x, max_y = max(xs), max(ys)
    graph_w = max_x - min_x + NW + PAD * 2
    graph_h = max_y - min_y + NH + PAD * 2
    # Scale to fit
    sc = min(max_w / graph_w, max_h / graph_h, 1.0)
    img_w = max(int(graph_w * sc), 200)
    img_h = max(int(graph_h * sc), 100)
    snw = int(NW * sc)
    snh = int(NH * sc)
    stw = int(THUMB_W * sc)
    sth = int(THUMB_H * sc)
    img = Image.new('RGB', (img_w, img_h), (30, 30, 46))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype(FONT_REG, max(int(11 * sc), 8))
        font_sm = ImageFont.truetype(FONT_REG, max(int(9 * sc), 7))
    except:
        font = ImageFont.load_default()
        font_sm = font
    # Map node id to center position
    node_pos = {}
    node_map = {}
    for n in nodes:
        nx = int((n['x'] - min_x + PAD) * sc)
        ny = int((n['y'] - min_y + PAD) * sc)
        node_pos[n['id']] = (nx, ny, nx + snw, ny + snh)
        node_map[n['id']] = n
    # Draw connections first (behind nodes)
    for conn in connections:
        src = node_pos.get(conn['sourceNode'])
        tgt = node_pos.get(conn['targetNode'])
        if not src or not tgt:
            continue
        x1 = src[2]  # right edge of source
        y1 = (src[1] + src[3]) // 2
        x2 = tgt[0]  # left edge of target
        y2 = (tgt[1] + tgt[3]) // 2
        # Bezier curve
        mx = (x1 + x2) // 2
        steps = 25
        for i in range(steps):
            t1 = i / steps
            t2 = (i + 1) / steps
            def bx(t): return int((1-t)**3*x1 + 3*(1-t)**2*t*mx + 3*(1-t)*t**2*mx + t**3*x2)
            def by(t): return int((1-t)**3*y1 + 3*(1-t)**2*t*(y1-8) + 3*(1-t)*t**2*(y2+8) + t**3*y2)
            draw.line([(bx(t1), by(t1)), (bx(t2), by(t2))], fill=(140, 180, 200), width=max(int(2*sc), 1))
    # Draw nodes
    for n in nodes:
        nx, ny, nx2, ny2 = node_pos[n['id']]
        ntype = n.get('type', '')
        label = n.get('label', '') or ntype.replace('_', ' ').title()
        if len(label) > 14:
            label = label[:13] + '..'
        # Get category color
        color = (80, 80, 100)
        for nd in NODES:
            if nd['t'] == ntype:
                color = hex_to_rgb(nd['c'])
                break
        # Node body
        draw.rounded_rectangle([nx, ny, nx2, ny2], radius=max(int(6*sc), 3),
                               fill=(40, 40, 55), outline=(70, 70, 90), width=1)
        # Top color bar
        bar_h = max(int(14 * sc), 8)
        draw.rounded_rectangle([nx, ny, nx2, ny + bar_h], radius=max(int(6*sc), 3), fill=color)
        draw.rectangle([nx, ny + bar_h//2, nx2, ny + bar_h], fill=color)
        # Label
        bbox = draw.textbbox((0, 0), label, font=font)
        tw_text = bbox[2] - bbox[0]
        tx = nx + (snw - tw_text) // 2
        draw.text((tx, ny + bar_h + max(int(2*sc), 1)), label, fill=(210, 210, 220), font=font)
        # Thumbnail
        thumb_key = THUMB_MAP.get(ntype)
        if thumb_key and thumb_key in thumbs and stw > 10 and sth > 8:
            thumb_data = thumbs[thumb_key]
            thumb_pil = Image.fromarray(thumb_data).resize((stw, sth), Image.LANCZOS)
            thumb_x = nx + (snw - stw) // 2
            thumb_y = ny2 - sth - max(int(4*sc), 2)
            # Border
            draw.rectangle([thumb_x-1, thumb_y-1, thumb_x+stw, thumb_y+sth],
                          outline=(60, 60, 80), width=1)
            img.paste(thumb_pil, (thumb_x, thumb_y))
        # Input port (left)
        port_r = max(int(4*sc), 2)
        py_port = (ny + ny2) // 2
        draw.ellipse([nx-port_r, py_port-port_r, nx+port_r, py_port+port_r],
                    fill=(80, 80, 110), outline=(160, 160, 190))
        # Output port (right)
        draw.ellipse([nx2-port_r, py_port-port_r, nx2+port_r, py_port+port_r],
                    fill=(80, 80, 110), outline=(160, 160, 190))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


NODES = [
    # --- IO (5) ---
    {'t':'image_read','en':'Image Read','kr':'이미지 읽기','cat':'io','c':'#4CAF50',
     'sig':'cv2.imread(filename, flags)','desc':'파일에서 이미지를 읽어옵니다. BMP, JPEG, PNG, TIFF 등을 지원합니다.',
     'i':[],'o':[('image','image')],'p':[('filepath','File Path','text','','파일 경로')]},
    {'t':'image_show','en':'Image Show','kr':'이미지 표시','cat':'io','c':'#2196F3',
     'sig':'cv2.imshow(winname, mat)','desc':'이미지를 미리보기 패널에 표시합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('windowName','Window Name','text','Output','윈도우 이름')]},
    {'t':'image_write','en':'Image Write','kr':'이미지 저장','cat':'io','c':'#8BC34A',
     'sig':'cv2.imwrite(filename, img)','desc':'이미지를 파일로 저장합니다. 비디오 출력 모드도 지원합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('filepath','File Path','text','output.png','저장 경로'),('format','Format','select','PNG','형식'),('quality','Quality','number','95','품질'),('videoOutput','Video Output','checkbox','false','비디오 출력'),('videoCodec','Codec','select','mp4v','코덱'),('videoFps','FPS','number','30','프레임율')]},
    {'t':'video_read','en':'Video Read','kr':'비디오 읽기','cat':'io','c':'#FF5722',
     'sig':'cv2.VideoCapture(filename)','desc':'비디오 파일에서 프레임을 읽습니다. 단일/루프 모드 지원.',
     'i':[],'o':[('image','frame')],'p':[('filepath','Video Path','text','','경로'),('mode','Mode','select','single','모드'),('frameIndex','Frame Index','number','0','프레임'),('startFrame','Start','number','0','시작'),('endFrame','End','number','-1','끝'),('step','Step','number','1','스텝')]},
    {'t':'camera_capture','en':'Camera Capture','kr':'카메라 캡처','cat':'io','c':'#FF5722',
     'sig':'cv2.VideoCapture(index)','desc':'웹캠에서 한 프레임을 캡처합니다.',
     'i':[],'o':[('image','frame')],'p':[('cameraIndex','Camera Index','number','0','카메라 인덱스')]},
    # --- Color (5) ---
    {'t':'cvt_color','en':'CvtColor','kr':'색상 변환','cat':'color','c':'#9C27B0',
     'sig':'cv2.cvtColor(src, code)','desc':'이미지의 색상 공간을 변환합니다. BGR↔Gray, BGR↔HSV 등.',
     'i':[('image','image')],'o':[('image','image')],'p':[('code','Color Code','select','COLOR_BGR2GRAY','변환 코드')]},
    {'t':'in_range','en':'InRange','kr':'범위 필터','cat':'color','c':'#E91E63',
     'sig':'cv2.inRange(src, lowerb, upperb)','desc':'지정 범위 내 픽셀만 선택하는 이진 마스크를 생성합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('lowerB','Lower B/H','number','0','하한'),('lowerG','Lower G/S','number','0',''),('lowerR','Lower R/V','number','0',''),('upperB','Upper B/H','number','255','상한'),('upperG','Upper G/S','number','255',''),('upperR','Upper R/V','number','255','')]},
    {'t':'histogram_eq','en':'Histogram EQ','kr':'히스토그램 평활화','cat':'color','c':'#673AB7',
     'sig':'cv2.equalizeHist / cv2.createCLAHE','desc':'히스토그램을 균일하게 분포시켜 대비를 개선합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('useCLAHE','Use CLAHE','checkbox','false','CLAHE 사용'),('clipLimit','Clip Limit','number','2.0','클립 제한'),('tileGridSize','Tile Size','number','8','타일 크기')]},
    {'t':'split_channels','en':'Split Channels','kr':'채널 분리','cat':'color','c':'#9C27B0',
     'sig':'cv2.split(m)','desc':'다채널 이미지를 개별 채널로 분리합니다.',
     'i':[('image','image')],'o':[('ch0','ch 0'),('ch1','ch 1'),('ch2','ch 2')],'p':[]},
    {'t':'merge_channels','en':'Merge Channels','kr':'채널 병합','cat':'color','c':'#9C27B0',
     'sig':'cv2.merge(mv)','desc':'단일 채널들을 합쳐 다채널 이미지를 만듭니다.',
     'i':[('ch0','ch 0'),('ch1','ch 1'),('ch2','ch 2')],'o':[('image','image')],'p':[]},
    # --- Filter (6) ---
    {'t':'gaussian_blur','en':'Gaussian Blur','kr':'가우시안 블러','cat':'filter','c':'#FF9800',
     'sig':'cv2.GaussianBlur(src, ksize, sigmaX)','desc':'가우시안 필터로 이미지를 부드럽게 합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('ksize','Kernel Size','number','5','커널 크기 (홀수)'),('sigmaX','Sigma X','number','0','표준편차')]},
    {'t':'median_blur','en':'Median Blur','kr':'미디언 블러','cat':'filter','c':'#FF9800',
     'sig':'cv2.medianBlur(src, ksize)','desc':'미디언 필터로 소금-후추 노이즈를 제거합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('ksize','Kernel Size','number','5','커널 크기')]},
    {'t':'bilateral_filter','en':'Bilateral Filter','kr':'양방향 필터','cat':'filter','c':'#FF9800',
     'sig':'cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace)','desc':'에지를 보존하면서 노이즈를 제거합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('d','Diameter','number','9','직경'),('sigmaColor','Sigma Color','number','75','색상 시그마'),('sigmaSpace','Sigma Space','number','75','공간 시그마')]},
    {'t':'box_filter','en':'Box Filter','kr':'박스 필터','cat':'filter','c':'#FF9800',
     'sig':'cv2.boxFilter(src, ddepth, ksize)','desc':'평균값 필터로 이미지를 부드럽게 합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('kwidth','Width','number','5','너비'),('kheight','Height','number','5','높이'),('normalize','Normalize','checkbox','true','정규화')]},
    {'t':'sharpen','en':'Sharpen','kr':'샤프닝','cat':'filter','c':'#FF9800',
     'sig':'Unsharp Mask','desc':'언샤프 마스크로 이미지를 선명하게 만듭니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('strength','Strength','number','1.5','강도')]},
    {'t':'filter2d','en':'Filter2D','kr':'2D 필터','cat':'filter','c':'#FF9800',
     'sig':'cv2.filter2D(src, ddepth, kernel)','desc':'커스텀 커널로 컨볼루션을 수행합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('kernelSize','Kernel Size','number','3','커널 크기'),('preset','Preset','select','sharpen','프리셋')]},
    # --- Edge (4) ---
    {'t':'canny','en':'Canny','kr':'캐니 에지','cat':'edge','c':'#F44336',
     'sig':'cv2.Canny(image, threshold1, threshold2)','desc':'캐니 알고리즘으로 에지를 검출합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('threshold1','Threshold 1','number','100','하위 임계값'),('threshold2','Threshold 2','number','200','상위 임계값')]},
    {'t':'sobel','en':'Sobel','kr':'소벨','cat':'edge','c':'#F44336',
     'sig':'cv2.Sobel(src, ddepth, dx, dy, ksize)','desc':'소벨 연산자로 이미지 미분을 계산합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('dx','dx','number','1','X 미분 차수'),('dy','dy','number','0','Y 미분 차수'),('ksize','Kernel Size','number','3','커널')]},
    {'t':'laplacian','en':'Laplacian','kr':'라플라시안','cat':'edge','c':'#F44336',
     'sig':'cv2.Laplacian(src, ddepth, ksize)','desc':'라플라시안으로 2차 미분 에지를 검출합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('ksize','Kernel Size','number','3','커널')]},
    {'t':'scharr','en':'Scharr','kr':'샤르','cat':'edge','c':'#F44336',
     'sig':'cv2.Scharr(src, ddepth, dx, dy)','desc':'소벨보다 정밀한 3x3 1차 미분을 계산합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('dx','dx','number','1','X'),('dy','dy','number','0','Y')]},
    # --- Threshold (3) ---
    {'t':'threshold','en':'Threshold','kr':'임계값','cat':'threshold','c':'#795548',
     'sig':'cv2.threshold(src, thresh, maxval, type)','desc':'고정 임계값으로 이진 이미지를 생성합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('thresh','Threshold','number','127','임계값'),('maxval','Max Value','number','255','최대값'),('type','Type','select','THRESH_BINARY','유형')]},
    {'t':'adaptive_threshold','en':'Adaptive Threshold','kr':'적응형 임계값','cat':'threshold','c':'#795548',
     'sig':'cv2.adaptiveThreshold(src, maxValue, method, type, blockSize, C)','desc':'영역별 적응적 임계값을 적용합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('maxval','Max Value','number','255','최대값'),('adaptiveMethod','Method','select','ADAPTIVE_THRESH_GAUSSIAN_C','방법'),('thresholdType','Type','select','THRESH_BINARY','유형'),('blockSize','Block Size','number','11','블록'),('C','C','number','2','상수')]},
    {'t':'otsu_threshold','en':'Otsu Threshold','kr':'오츠 임계값','cat':'threshold','c':'#795548',
     'sig':'cv2.threshold(src, 0, maxval, THRESH_OTSU)','desc':'오츠 알고리즘으로 최적 임계값을 자동 결정합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('maxval','Max Value','number','255','최대값')]},
    # --- Morphology (4) ---
    {'t':'morphology','en':'Morphology Ex','kr':'형태학 연산','cat':'morph','c':'#607D8B',
     'sig':'cv2.morphologyEx(src, op, kernel)','desc':'열기, 닫기, 그래디언트 등 형태학 변환을 수행합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('operation','Operation','select','MORPH_OPEN','연산'),('ksize','Kernel Size','number','5','커널'),('shape','Shape','select','MORPH_RECT','모양'),('iterations','Iterations','number','1','반복')]},
    {'t':'dilate','en':'Dilate','kr':'팽창','cat':'morph','c':'#607D8B',
     'sig':'cv2.dilate(src, kernel, iterations)','desc':'밝은 영역을 확장합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('ksize','Kernel Size','number','5','커널'),('iterations','Iterations','number','1','반복')]},
    {'t':'erode','en':'Erode','kr':'침식','cat':'morph','c':'#607D8B',
     'sig':'cv2.erode(src, kernel, iterations)','desc':'밝은 영역을 축소합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('ksize','Kernel Size','number','5','커널'),('iterations','Iterations','number','1','반복')]},
    {'t':'structuring_element','en':'Structuring Element','kr':'구조 요소','cat':'morph','c':'#607D8B',
     'sig':'cv2.getStructuringElement(shape, ksize)','desc':'형태학 연산용 커널을 생성합니다.',
     'i':[],'o':[('image','element')],'p':[('shape','Shape','select','MORPH_RECT','모양'),('width','Width','number','5','너비'),('height','Height','number','5','높이')]},
    # --- Contour (7) ---
    {'t':'draw_contours','en':'Draw Contours','kr':'컨투어 그리기','cat':'contour','c':'#009688',
     'sig':'cv2.findContours + cv2.drawContours','desc':'이진 이미지에서 윤곽선을 찾아 그립니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('mode','Mode','select','RETR_EXTERNAL','검색'),('method','Approx','select','CHAIN_APPROX_SIMPLE','근사'),('contourIdx','Index','number','-1','인덱스'),('thickness','Thickness','number','2','두께'),('colorR','R','number','0',''),('colorG','G','number','255',''),('colorB','B','number','0','')]},
    {'t':'bounding_rect','en':'Bounding Rect','kr':'바운딩 박스','cat':'contour','c':'#009688',
     'sig':'cv2.boundingRect + cv2.rectangle','desc':'컨투어의 외접 사각형을 그립니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('mode','Mode','select','RETR_EXTERNAL',''),('method','Approx','select','CHAIN_APPROX_SIMPLE',''),('thickness','Thickness','number','2',''),('colorR','R','number','0',''),('colorG','G','number','255',''),('colorB','B','number','0','')]},
    {'t':'min_enclosing_circle','en':'Min Enclosing Circle','kr':'최소 외접원','cat':'contour','c':'#009688',
     'sig':'cv2.minEnclosingCircle + cv2.circle','desc':'컨투어의 최소 외접원을 그립니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('mode','Mode','select','RETR_EXTERNAL',''),('method','Approx','select','CHAIN_APPROX_SIMPLE',''),('thickness','Thickness','number','2',''),('colorR','R','number','0',''),('colorG','G','number','255',''),('colorB','B','number','0','')]},
    {'t':'convex_hull','en':'Convex Hull','kr':'볼록 껍질','cat':'contour','c':'#009688',
     'sig':'cv2.convexHull + cv2.drawContours','desc':'컨투어의 볼록 껍질을 그립니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('mode','Mode','select','RETR_EXTERNAL',''),('method','Approx','select','CHAIN_APPROX_SIMPLE',''),('thickness','Thickness','number','2',''),('colorR','R','number','0',''),('colorG','G','number','255',''),('colorB','B','number','0','')]},
    {'t':'approx_poly','en':'Approx Poly','kr':'다각형 근사','cat':'contour','c':'#009688',
     'sig':'cv2.approxPolyDP','desc':'컨투어를 다각형으로 근사합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('epsilon','Epsilon','number','0.02','정밀도'),('closed','Closed','checkbox','true','닫힘'),('mode','Mode','select','RETR_EXTERNAL',''),('method','Approx','select','CHAIN_APPROX_SIMPLE',''),('thickness','Thickness','number','2',''),('colorR','R','number','0',''),('colorG','G','number','255',''),('colorB','B','number','0','')]},
    {'t':'contour_area','en':'Contour Area','kr':'면적 필터','cat':'contour','c':'#009688',
     'sig':'cv2.contourArea + cv2.drawContours','desc':'면적 범위로 컨투어를 필터링합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('mode','Mode','select','RETR_EXTERNAL',''),('method','Approx','select','CHAIN_APPROX_SIMPLE',''),('minArea','Min','number','100','최소'),('maxArea','Max','number','100000','최대'),('thickness','Thickness','number','2',''),('colorR','R','number','0',''),('colorG','G','number','255',''),('colorB','B','number','0','')]},
    {'t':'contour_properties','en':'Contour Properties','kr':'컨투어 속성','cat':'contour','c':'#009688',
     'sig':'cv2.contourArea + cv2.arcLength + cv2.moments','desc':'면적, 둘레, 중심점 등의 속성을 표시합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('mode','Mode','select','RETR_EXTERNAL',''),('method','Approx','select','CHAIN_APPROX_SIMPLE',''),('showArea','Area','checkbox','true','면적'),('showPerimeter','Perimeter','checkbox','true','둘레'),('showCenter','Center','checkbox','true','중심')]},
    # --- Transform (7) ---
    {'t':'resize','en':'Resize','kr':'리사이즈','cat':'transform','c':'#00BCD4',
     'sig':'cv2.resize(src, dsize, fx, fy)','desc':'이미지 크기를 변경합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('width','Width','number','0','너비'),('height','Height','number','0','높이'),('fx','Scale X','number','0.5','X비율'),('fy','Scale Y','number','0.5','Y비율'),('interpolation','Interp','select','INTER_LINEAR','보간')]},
    {'t':'rotate','en':'Rotate','kr':'회전','cat':'transform','c':'#00BCD4',
     'sig':'cv2.getRotationMatrix2D + cv2.warpAffine','desc':'이미지를 중심 기준으로 회전합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('angle','Angle','number','90','각도')]},
    {'t':'flip','en':'Flip','kr':'뒤집기','cat':'transform','c':'#00BCD4',
     'sig':'cv2.flip(src, flipCode)','desc':'이미지를 수평/수직으로 뒤집습니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('flipCode','Mode','select','Horizontal (1)','방향')]},
    {'t':'crop','en':'Crop','kr':'자르기','cat':'transform','c':'#00BCD4',
     'sig':'img[y:y+h, x:x+w]','desc':'이미지의 관심 영역(ROI)을 잘라냅니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('x','X','number','0','X'),('y','Y','number','0','Y'),('width','Width','number','100','너비'),('height','Height','number','100','높이')]},
    {'t':'warp_affine','en':'Warp Affine','kr':'아핀 변환','cat':'transform','c':'#00BCD4',
     'sig':'cv2.warpAffine(src, M, dsize)','desc':'2x3 변환 행렬로 아핀 변환을 적용합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('m00','M[0,0]','number','1',''),('m01','M[0,1]','number','0',''),('m02','tx','number','0','X이동'),('m10','M[1,0]','number','0',''),('m11','M[1,1]','number','1',''),('m12','ty','number','0','Y이동')]},
    {'t':'warp_perspective','en':'Warp Perspective','kr':'투시 변환','cat':'transform','c':'#00BCD4',
     'sig':'cv2.getPerspectiveTransform + cv2.warpPerspective','desc':'4꼭짓점 대응으로 투시 변환을 적용합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('srcX1','SrcX1','number','0',''),('srcY1','SrcY1','number','0',''),('srcX2','SrcX2','number','300',''),('srcY2','SrcY2','number','0',''),('srcX3','SrcX3','number','300',''),('srcY3','SrcY3','number','300',''),('srcX4','SrcX4','number','0',''),('srcY4','SrcY4','number','300',''),('dstX1','DstX1','number','0',''),('dstY1','DstY1','number','0',''),('dstX2','DstX2','number','300',''),('dstY2','DstY2','number','0',''),('dstX3','DstX3','number','300',''),('dstY3','DstY3','number','300',''),('dstX4','DstX4','number','0',''),('dstY4','DstY4','number','300','')]},
    {'t':'remap','en':'Remap','kr':'리맵','cat':'transform','c':'#00BCD4',
     'sig':'cv2.remap(src, map1, map2, interpolation)','desc':'배럴/핀쿠션 왜곡을 보정합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('distortionK','K','number','0.5','왜곡계수'),('interpolation','Interp','select','INTER_LINEAR','보간')]},
    # --- Histogram (1) ---
    {'t':'calc_histogram','en':'Calc Histogram','kr':'히스토그램 계산','cat':'histogram','c':'#673AB7',
     'sig':'cv2.calcHist','desc':'이미지 히스토그램을 계산하고 시각화합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('histSize','Bins','number','256','빈'),('normalize','Normalize','checkbox','true','정규화')]},
    # --- Feature (7) ---
    {'t':'find_contours','en':'Find Contours','kr':'컨투어 찾기','cat':'feature','c':'#CDDC39',
     'sig':'cv2.findContours(image, mode, method)','desc':'이진 이미지에서 윤곽선을 검출합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('mode','Mode','select','RETR_EXTERNAL','검색'),('method','Approx','select','CHAIN_APPROX_SIMPLE','근사')]},
    {'t':'hough_lines','en':'Hough Lines','kr':'허프 직선','cat':'feature','c':'#CDDC39',
     'sig':'cv2.HoughLines(image, rho, theta, threshold)','desc':'허프 변환으로 직선을 검출합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('rho','Rho','number','1','거리'),('theta_divisor','Theta (π/N)','number','180','각도'),('threshold','Threshold','number','100','임계')]},
    {'t':'harris_corner','en':'Harris Corner','kr':'해리스 코너','cat':'feature','c':'#CDDC39',
     'sig':'cv2.cornerHarris(src, blockSize, ksize, k)','desc':'해리스 코너 검출기로 모서리를 찾습니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('blockSize','Block','number','2','블록'),('ksize','Sobel K','number','3','커널'),('k','K','number','0.04','파라미터')]},
    {'t':'good_features','en':'Good Features','kr':'좋은 특징점','cat':'feature','c':'#CDDC39',
     'sig':'cv2.goodFeaturesToTrack','desc':'Shi-Tomasi 알고리즘으로 강한 코너를 검출합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('maxCorners','Max','number','100','최대'),('qualityLevel','Quality','number','0.01','품질'),('minDistance','MinDist','number','10','거리')]},
    {'t':'orb_features','en':'ORB','kr':'ORB 특징점','cat':'feature','c':'#CDDC39',
     'sig':'cv2.ORB_create(nfeatures)','desc':'ORB 특징점을 검출합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('nFeatures','Features','number','500','특징수')]},
    {'t':'fast_features','en':'FAST','kr':'FAST 코너','cat':'feature','c':'#CDDC39',
     'sig':'cv2.FastFeatureDetector_create','desc':'FAST 알고리즘으로 빠르게 코너를 검출합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('threshold','Threshold','number','25','임계값'),('nonmaxSuppression','NMS','checkbox','true','비극대')]},
    {'t':'match_features','en':'Match Features','kr':'특징 매칭','cat':'feature','c':'#CDDC39',
     'sig':'cv2.ORB_create + cv2.BFMatcher','desc':'ORB로 두 이미지의 특징점을 매칭합니다.',
     'i':[('image','image'),('image2','image2')],'o':[('image','image')],'p':[('nFeatures','Features','number','500','수'),('matchRatio','Ratio','number','0.75','비율')]},
    # --- Drawing (6) ---
    {'t':'draw_line','en':'Draw Line','kr':'직선 그리기','cat':'drawing','c':'#E91E63',
     'sig':'cv2.line(img, pt1, pt2, color, thickness)','desc':'이미지에 직선을 그립니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('x1','X1','number','10',''),('y1','Y1','number','10',''),('x2','X2','number','200',''),('y2','Y2','number','200',''),('colorR','R','number','0',''),('colorG','G','number','255',''),('colorB','B','number','0',''),('thickness','Thickness','number','2','')]},
    {'t':'draw_rectangle','en':'Draw Rectangle','kr':'사각형 그리기','cat':'drawing','c':'#E91E63',
     'sig':'cv2.rectangle(img, pt1, pt2, color, thickness)','desc':'이미지에 사각형을 그립니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('x','X','number','10',''),('y','Y','number','10',''),('width','W','number','100',''),('height','H','number','100',''),('colorR','R','number','0',''),('colorG','G','number','255',''),('colorB','B','number','0',''),('thickness','Thickness','number','2','')]},
    {'t':'draw_circle','en':'Draw Circle','kr':'원 그리기','cat':'drawing','c':'#E91E63',
     'sig':'cv2.circle(img, center, radius, color, thickness)','desc':'이미지에 원을 그립니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('centerX','CX','number','100',''),('centerY','CY','number','100',''),('radius','R','number','50',''),('colorR','R','number','0',''),('colorG','G','number','255',''),('colorB','B','number','0',''),('thickness','Thickness','number','2','')]},
    {'t':'draw_ellipse','en':'Draw Ellipse','kr':'타원 그리기','cat':'drawing','c':'#E91E63',
     'sig':'cv2.ellipse(img, center, axes, angle, ...)','desc':'이미지에 타원을 그립니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('centerX','CX','number','100',''),('centerY','CY','number','100',''),('axesW','W','number','80',''),('axesH','H','number','40',''),('angle','Angle','number','0',''),('colorR','R','number','0',''),('colorG','G','number','255',''),('colorB','B','number','0',''),('thickness','Thickness','number','2','')]},
    {'t':'draw_text','en':'Draw Text','kr':'텍스트 그리기','cat':'drawing','c':'#E91E63',
     'sig':'cv2.putText(img, text, org, font, scale, color, thickness)','desc':'이미지에 텍스트를 표시합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('text','Text','text','Hello','텍스트'),('x','X','number','50',''),('y','Y','number','50',''),('fontScale','Scale','number','1.0','크기'),('colorR','R','number','255',''),('colorG','G','number','255',''),('colorB','B','number','255',''),('thickness','Thickness','number','2','')]},
    {'t':'draw_polylines','en':'Draw Polylines','kr':'다각선 그리기','cat':'drawing','c':'#E91E63',
     'sig':'cv2.polylines(img, [pts], isClosed, color, thickness)','desc':'다각형 또는 다각선을 그립니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('points','Points','text','10,10;100,50;50,100','좌표'),('isClosed','Closed','checkbox','true','닫힘'),('colorR','R','number','0',''),('colorG','G','number','255',''),('colorB','B','number','0',''),('thickness','Thickness','number','2','')]},
    # --- Arithmetic (9) ---
    {'t':'add_weighted','en':'Add Weighted','kr':'가중 합성','cat':'arithmetic','c':'#3F51B5',
     'sig':'cv2.addWeighted(src1, alpha, src2, beta, gamma)','desc':'두 이미지를 가중치로 합성합니다.',
     'i':[('image','image'),('image2','image2')],'o':[('image','image')],'p':[('alpha','Alpha','number','0.5',''),('beta','Beta','number','0.5',''),('gamma','Gamma','number','0','')]},
    {'t':'bitwise_and','en':'Bitwise AND','kr':'비트 AND','cat':'arithmetic','c':'#3F51B5',
     'sig':'cv2.bitwise_and(src1, src2)','desc':'비트 AND 연산. 마스크 적용에 사용합니다.',
     'i':[('image','image'),('image2','image2')],'o':[('image','image')],'p':[]},
    {'t':'bitwise_or','en':'Bitwise OR','kr':'비트 OR','cat':'arithmetic','c':'#3F51B5',
     'sig':'cv2.bitwise_or(src1, src2)','desc':'비트 OR 연산을 수행합니다.',
     'i':[('image','image'),('image2','image2')],'o':[('image','image')],'p':[]},
    {'t':'bitwise_not','en':'Bitwise NOT','kr':'비트 NOT','cat':'arithmetic','c':'#3F51B5',
     'sig':'cv2.bitwise_not(src)','desc':'모든 비트를 반전합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[]},
    {'t':'add','en':'Add','kr':'덧셈','cat':'arithmetic','c':'#3F51B5',
     'sig':'cv2.add(src1, src2)','desc':'픽셀별 덧셈 (포화).',
     'i':[('image','image'),('image2','image2')],'o':[('image','image')],'p':[]},
    {'t':'subtract','en':'Subtract','kr':'뺄셈','cat':'arithmetic','c':'#3F51B5',
     'sig':'cv2.subtract(src1, src2)','desc':'픽셀별 뺄셈 (포화).',
     'i':[('image','image'),('image2','image2')],'o':[('image','image')],'p':[]},
    {'t':'multiply','en':'Multiply','kr':'곱셈','cat':'arithmetic','c':'#3F51B5',
     'sig':'cv2.multiply(src1, src2, scale)','desc':'픽셀별 곱셈.',
     'i':[('image','image'),('image2','image2')],'o':[('image','image')],'p':[('scale','Scale','number','1.0','스케일')]},
    {'t':'absdiff','en':'AbsDiff','kr':'절대 차이','cat':'arithmetic','c':'#3F51B5',
     'sig':'cv2.absdiff(src1, src2)','desc':'절대 차이. 변화 감지에 사용합니다.',
     'i':[('image','image'),('image2','image2')],'o':[('image','image')],'p':[]},
    {'t':'bitwise_xor','en':'Bitwise XOR','kr':'비트 XOR','cat':'arithmetic','c':'#3F51B5',
     'sig':'cv2.bitwise_xor(src1, src2)','desc':'비트 XOR 연산.',
     'i':[('image','image'),('image2','image2')],'o':[('image','image')],'p':[]},
    # --- Detection (4) ---
    {'t':'haar_cascade','en':'Haar Cascade','kr':'하르 캐스케이드','cat':'detection','c':'#FF5722',
     'sig':'cv2.CascadeClassifier.detectMultiScale()','desc':'얼굴, 눈, 미소 등을 검출합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('cascadeType','Type','select','face','유형'),('scaleFactor','Scale','number','1.1','스케일'),('minNeighbors','Neighbors','number','5','이웃'),('minWidth','MinW','number','30','최소폭'),('minHeight','MinH','number','30','최소높이')]},
    {'t':'hough_circles','en':'Hough Circles','kr':'허프 원','cat':'detection','c':'#FF5722',
     'sig':'cv2.HoughCircles(image, method, dp, minDist, ...)','desc':'허프 변환으로 원을 검출합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('dp','dp','number','1.2','역비율'),('minDist','MinDist','number','50','최소간격'),('param1','P1','number','100','캐니'),('param2','P2','number','30','축적기'),('minRadius','MinR','number','0','최소R'),('maxRadius','MaxR','number','0','최대R')]},
    {'t':'template_match','en':'Template Match','kr':'템플릿 매칭','cat':'detection','c':'#FF5722',
     'sig':'cv2.matchTemplate(image, templ, method)','desc':'템플릿으로 일치 영역을 찾습니다. matches 포트로 좌표 리스트 출력.',
     'i':[('image','image'),('image2','template')],'o':[('image','image'),('matches','matches')],'p':[('method','Method','select','TM_CCOEFF_NORMED','방법'),('threshold','Threshold','number','0.8','임계')]},
    {'t':'image_extract','en':'Image Extract','kr':'이미지 추출','cat':'detection','c':'#FF5722',
     'sig':'Image Extract Node','desc':'좌표 [x1,y1,x2,y2]로 이미지 영역을 추출합니다.',
     'i':[('image','image'),('coords','coords')],'o':[('image','image')],'p':[('padding','Padding','number','0','패딩')]},
    # --- Segmentation (3) ---
    {'t':'flood_fill','en':'Flood Fill','kr':'플러드 필','cat':'segmentation','c':'#795548',
     'sig':'cv2.floodFill(image, mask, seedPoint, newVal, ...)','desc':'시드 점에서 연결 영역을 채웁니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('seedX','SeedX','number','0','X'),('seedY','SeedY','number','0','Y'),('colorR','R','number','255',''),('colorG','G','number','0',''),('colorB','B','number','0',''),('loDiff','LoDiff','number','20','하한'),('upDiff','UpDiff','number','20','상한')]},
    {'t':'grabcut','en':'GrabCut','kr':'그랩컷','cat':'segmentation','c':'#795548',
     'sig':'cv2.grabCut(img, mask, rect, ...)','desc':'GrabCut으로 전경/배경을 분리합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('x','X','number','10','X'),('y','Y','number','10','Y'),('width','W','number','200','너비'),('height','H','number','200','높이'),('iterations','Iter','number','5','반복')]},
    {'t':'watershed','en':'Watershed','kr':'워터쉐드','cat':'segmentation','c':'#795548',
     'sig':'cv2.watershed(image, markers)','desc':'워터쉐드로 이미지를 영역 분할합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('markerSize','Marker K','number','10','커널')]},
    # --- Value (7) ---
    {'t':'val_integer','en':'Integer','kr':'정수','cat':'value','c':'#78909C',
     'sig':'Integer Value','desc':'정수 상수를 출력합니다.',
     'i':[],'o':[('value','value')],'p':[('value','Value','number','0','값'),('min','Min','number','0','최소'),('max','Max','number','255','최대')]},
    {'t':'val_float','en':'Float','kr':'실수','cat':'value','c':'#78909C',
     'sig':'Float Value','desc':'실수 상수를 출력합니다.',
     'i':[],'o':[('value','value')],'p':[('value','Value','number','0.0','값')]},
    {'t':'val_boolean','en':'Boolean','kr':'불리언','cat':'value','c':'#78909C',
     'sig':'Boolean Value','desc':'True/False 값을 출력합니다.',
     'i':[],'o':[('value','value')],'p':[('value','Value','checkbox','false','값')]},
    {'t':'val_point','en':'Point','kr':'점 좌표','cat':'value','c':'#78909C',
     'sig':'Point Value','desc':'(x, y) 좌표를 출력합니다.',
     'i':[],'o':[('value','point')],'p':[('x','X','number','0','X'),('y','Y','number','0','Y')]},
    {'t':'val_scalar','en':'Scalar','kr':'스칼라','cat':'value','c':'#78909C',
     'sig':'Scalar Value','desc':'4원소 스칼라 값을 출력합니다.',
     'i':[],'o':[('value','scalar')],'p':[('v0','V0','number','0',''),('v1','V1','number','0',''),('v2','V2','number','0',''),('v3','V3','number','0','')]},
    {'t':'val_math','en':'Math Operation','kr':'수학 연산','cat':'value','c':'#78909C',
     'sig':'Math Operation','desc':'기본 수학 연산을 수행합니다.',
     'i':[('a','A'),('b','B')],'o':[('value','result')],'p':[('operation','Op','select','add','연산')]},
    {'t':'val_list','en':'List Index/Slice','kr':'리스트 인덱싱','cat':'value','c':'#78909C',
     'sig':'List Index/Slice','desc':'리스트에서 인덱싱 또는 슬라이싱합니다.',
     'i':[('list','list')],'o':[('value','result')],'p':[('mode','Mode','select','index','모드'),('index','Index','number','0','인덱스'),('start','Start','number','0','시작'),('stop','Stop','number','-1','끝'),('step','Step','number','1','스텝')]},
    # --- Control (4) ---
    {'t':'control_if','en':'If','kr':'조건 분기','cat':'control','c':'#26A69A',
     'sig':'If (condition) -> True / False','desc':'조건에 따라 True/False 경로로 분기합니다.',
     'i':[('image','image')],'o':[('true','true'),('false','false')],'p':[('condition','Condition','select','not_empty','조건'),('value','Value','number','100','비교값'),('customExpr','Custom','text','img.shape[0]>100','식')]},
    {'t':'control_for','en':'For Loop','kr':'For 반복','cat':'control','c':'#26A69A',
     'sig':'for i in range(N): operation','desc':'연산을 N번 반복 적용합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('iterations','N','number','3','반복'),('operation','Op','select','gaussian_blur','연산'),('ksize','K','number','3','커널'),('customCode','Code','text','','코드')]},
    {'t':'control_while','en':'While Loop','kr':'While 반복','cat':'control','c':'#26A69A',
     'sig':'while (cond): operation','desc':'조건이 참인 동안 반복합니다.',
     'i':[('image','image')],'o':[('image','image')],'p':[('condition','Cond','select','mean_gt','조건'),('value','Val','number','128','임계'),('operation','Op','select','gaussian_blur','연산'),('ksize','K','number','3',''),('maxIter','Max','number','50','제한'),('customCond','CCond','text','','조건식'),('customCode','CCode','text','','코드')]},
    {'t':'control_switch','en':'Switch-Case','kr':'스위치-케이스','cat':'control','c':'#26A69A',
     'sig':'switch -> case 0/1/2','desc':'조건에 따라 3개 출력 중 하나로 분기합니다.',
     'i':[('image','image')],'o':[('case0','case 0'),('case1','case 1'),('case2','case 2')],'p':[('switchOn','Switch','select','channels','기준'),('customExpr','Custom','text','','식')]},
    # --- Script (1) ---
    {'t':'python_script','en':'Python Script','kr':'파이썬 스크립트','cat':'script','c':'#FFC107',
     'sig':'Custom Python Script','desc':'사용자 정의 Python 코드를 실행합니다. img_input/img_output 사용.',
     'i':[('image','image')],'o':[('image','image')],'p':[('script','Code','textarea','# img_input → img_output','코드')]},
]


TUTORIALS = [
    {'n':1,'title':'첫 이미지 처리 — 그레이스케일 & 에지 검출',
     'goal':'CvtColor와 Canny를 사용하여 컬러 이미지에서 에지를 검출합니다.',
     'pipe':[('Image Read','#4CAF50',''),('CvtColor','#9C27B0','BGR2GRAY'),('Canny','#F44336',''),('Image Show','#4CAF50','')],
     'steps':['팔레트 → I/O에서 Image Read를 캔버스에 드래그합니다.','Image Read를 더블클릭하여 이미지를 업로드합니다.','팔레트 → Color에서 CvtColor를 추가하고 Color Code를 COLOR_BGR2GRAY로 설정합니다.','Image Read의 출력 포트를 CvtColor의 입력 포트로 드래그하여 연결합니다.','팔레트 → Edge에서 Canny를 추가합니다. Threshold1=100, Threshold2=200.','CvtColor → Canny를 연결합니다.','Image Show를 추가하고 Canny → Image Show를 연결합니다.','Execute 버튼을 클릭하면 에지 검출 결과가 표시됩니다.'],
     'extra':'Threshold 값을 변경하여 에지 감도를 조절해 보세요.'},
    {'n':2,'title':'블러 & 임계값 — 이진 마스크 생성',
     'goal':'가우시안 블러로 노이즈를 제거한 후 임계값으로 이진 이미지를 만듭니다.',
     'pipe':[('Image Read','#4CAF50',''),('CvtColor','#9C27B0','GRAY'),('Gaussian Blur','#FF9800','k=5'),('Threshold','#795548',''),('Image Show','#4CAF50','')],
     'steps':['Image Read → CvtColor(BGR2GRAY)를 구성합니다.','Gaussian Blur를 추가합니다. Kernel Size=5.','CvtColor → Gaussian Blur를 연결합니다.','Threshold를 추가합니다. Thresh=127, MaxVal=255, Type=THRESH_BINARY.','Gaussian Blur → Threshold → Image Show를 연결합니다.','Execute하면 흰/검 이진 마스크가 표시됩니다.'],
     'extra':'THRESH_OTSU를 사용하면 자동으로 최적 임계값을 찾습니다.'},
    {'n':3,'title':'형태학 연산 — 이진 이미지 정리',
     'goal':'Opening 연산으로 이진 이미지의 노이즈를 제거합니다.',
     'pipe':[('Image Read','#4CAF50',''),('CvtColor','#9C27B0','GRAY'),('Threshold','#795548',''),('Morphology Ex','#607D8B','OPEN'),('Image Show','#4CAF50','')],
     'steps':['Tutorial 2의 파이프라인(Image Read → CvtColor → Threshold)을 구성합니다.','Morphology Ex를 추가합니다. Operation=MORPH_OPEN, Kernel=5, Shape=MORPH_RECT.','Threshold → Morphology Ex → Image Show를 연결합니다.','Execute하면 작은 노이즈가 제거된 깨끗한 이진 이미지가 표시됩니다.'],
     'extra':'MORPH_CLOSE로 변경하면 작은 구멍이 메워집니다.'},
    {'n':4,'title':'컨투어 검출 — 물체 외곽선 찾기',
     'goal':'이진 이미지에서 컨투어를 찾아 원본 이미지 위에 그립니다.',
     'pipe':[('Image Read','#4CAF50',''),('CvtColor','#9C27B0','GRAY'),('Threshold','#795548',''),('Draw Contours','#009688',''),('Image Show','#4CAF50','')],
     'steps':['Image Read → CvtColor → Threshold로 이진 이미지를 준비합니다.','Draw Contours를 추가합니다. Mode=RETR_EXTERNAL, Thickness=2.','Threshold → Draw Contours → Image Show를 연결합니다.','Execute하면 윤곽선이 녹색으로 그려집니다.'],
     'extra':'Bounding Rect나 Min Enclosing Circle로 다른 표현을 시도해 보세요.'},
    {'n':5,'title':'이미지 합성 — 두 이미지 블렌딩',
     'goal':'Add Weighted로 두 이미지를 투명도를 조절하여 합성합니다.',
     'pipe':[('Image Read','#4CAF50','A'),('Image Read','#4CAF50','B'),('Add Weighted','#3F51B5','α=0.7'),('Image Show','#4CAF50','')],
     'steps':['Image Read 2개를 추가하고 각각 다른 이미지를 업로드합니다.','두 이미지의 크기가 같아야 합니다 (Resize로 맞출 수 있습니다).','Add Weighted를 추가합니다. Alpha=0.7, Beta=0.3, Gamma=0.','첫 번째 Image Read → image 포트, 두 번째 → image2 포트에 연결합니다.','Add Weighted → Image Show를 연결하고 Execute합니다.'],
     'extra':'Alpha+Beta=1이면 전체 밝기가 유지됩니다.'},
    {'n':6,'title':'템플릿 매칭 & 이미지 추출',
     'goal':'Template Match로 패턴을 찾고 Image Extract로 해당 영역을 추출합니다.',
     'pipe':[('Image Read','#4CAF50','원본'),('Image Read','#4CAF50','템플릿'),('Template Match','#FF5722',''),('Image Show','#4CAF50','')],
     'steps':['Image Read 2개: 원본 이미지와 찾을 템플릿 이미지.','Template Match를 추가합니다. Method=TM_CCOEFF_NORMED, Threshold=0.8.','원본 → image 포트, 템플릿 → template 포트에 연결합니다.','Image Extract를 추가합니다.','원본 → Image Extract의 image, Template Match의 matches → coords에 연결합니다.','Image Extract → Image Show를 연결하고 Execute합니다.'],
     'extra':'Threshold를 낮추면 더 많은 매칭이 검출됩니다.'},
    {'n':7,'title':'리스트 인덱싱과 템플릿 매칭 결합',
     'goal':'matches 리스트에서 특정 인덱스의 좌표를 선택합니다.',
     'pipe':[('Template Match','#FF5722',''),('List Index','#78909C','[1]'),('Image Extract','#FF5722',''),('Image Show','#4CAF50','')],
     'steps':['Tutorial 6의 Template Match 파이프라인을 구성합니다.','List Index/Slice 노드를 추가합니다. Mode=index, Index=1.','matches → List Index의 list 입력에 연결합니다.','result → Image Extract의 coords에 연결합니다.','Execute하면 두 번째 매칭 영역이 추출됩니다.'],
     'extra':'slice 모드로 여러 매칭을 한번에 처리할 수 있습니다.'},
    {'n':8,'title':'비디오 프레임별 처리',
     'goal':'Video Read 루프 모드로 모든 프레임에 에지 검출을 적용합니다.',
     'pipe':[('Video Read','#FF5722','loop'),('CvtColor','#9C27B0','GRAY'),('Canny','#F44336',''),('Image Write','#8BC34A','video')],
     'steps':['Video Read를 추가하고 비디오를 업로드합니다.','Mode=loop, Start Frame=0, End Frame=-1, Step=1로 설정합니다.','CvtColor(BGR2GRAY) → Canny(100,200)를 연결합니다.','Image Write를 추가합니다. Video Output 체크, 경로=output.mp4.','Canny → Image Write를 연결하고 Execute합니다.'],
     'extra':'Step을 변경하면 프레임을 건너뛸 수 있습니다.'},
    {'n':9,'title':'얼굴 검출 (Haar Cascade)',
     'goal':'Haar Cascade 분류기로 이미지에서 얼굴을 검출합니다.',
     'pipe':[('Image Read','#4CAF50',''),('Haar Cascade','#FF5722','face'),('Image Show','#4CAF50','')],
     'steps':['Image Read에 사람 얼굴이 있는 사진을 업로드합니다.','Haar Cascade를 추가합니다. Type=face, Scale=1.1, Neighbors=5.','Image Read → Haar Cascade → Image Show를 연결합니다.','Execute하면 검출된 얼굴에 사각형이 그려집니다.'],
     'extra':'Type을 eye나 smile로 변경해 보세요.'},
    {'n':10,'title':'커스텀 Python 스크립트',
     'goal':'Python Script 노드로 직접 코드를 작성하여 실행합니다.',
     'pipe':[('Image Read','#4CAF50',''),('Python Script','#FFC107',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 이미지를 업로드합니다.','Python Script를 추가합니다.','코드: img_output = cv2.Canny(img_input, 50, 150)','Image Read → Python Script → Image Show를 연결합니다.','Execute하면 커스텀 코드 결과가 표시됩니다.'],
     'extra':'cv2, np를 자유롭게 사용할 수 있습니다.'},
    {'n':11,'title':'HSV 색상 필터링 (InRange)',
     'goal':'HSV 공간에서 특정 색상 범위만 추출합니다.',
     'pipe':[('Image Read','#4CAF50',''),('CvtColor','#9C27B0','BGR2HSV'),('InRange','#E91E63',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 컬러 이미지를 업로드합니다.','CvtColor를 추가하고 COLOR_BGR2HSV로 설정합니다.','InRange를 추가합니다. 예: 빨간색 → Lower=(0,100,100), Upper=(10,255,255).','Image Read → CvtColor → InRange → Image Show를 연결합니다.','Execute하면 해당 색상만 흰색인 마스크가 표시됩니다.'],
     'extra':'결과 마스크를 Bitwise AND에 사용하면 원본에서 해당 색상만 추출합니다.'},
    {'n':12,'title':'채널 분리 & 병합',
     'goal':'Split/Merge Channels로 BGR 채널을 분리하고 재조합합니다.',
     'pipe':[('Image Read','#4CAF50',''),('Split Channels','#9C27B0',''),('Merge Channels','#9C27B0',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 컬러 이미지를 업로드합니다.','Split Channels를 추가하고 연결합니다. ch0=B, ch1=G, ch2=R.','Merge Channels를 추가합니다.','Split의 ch0→Merge의 ch2, ch2→ch0으로 교차 연결합니다 (R↔B 교환).','Merge → Image Show를 연결하고 Execute합니다.'],
     'extra':'한 채널만 Image Show에 연결하면 해당 채널의 그레이스케일 이미지를 볼 수 있습니다.'},
    {'n':13,'title':'적응형 임계값 처리',
     'goal':'Adaptive Threshold로 조명이 불균일한 이미지를 이진화합니다.',
     'pipe':[('Image Read','#4CAF50',''),('CvtColor','#9C27B0','GRAY'),('Adaptive Threshold','#795548',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 문서/텍스트 이미지를 업로드합니다.','CvtColor(BGR2GRAY) → Adaptive Threshold를 연결합니다.','Method=GAUSSIAN_C, Block Size=11, C=2로 설정합니다.','Execute하면 조명 차이에 관계없이 깨끗하게 이진화됩니다.'],
     'extra':'Block Size를 키우면 더 넓은 영역을 참조합니다.'},
    {'n':14,'title':'이미지 샤프닝',
     'goal':'Sharpen 노드로 흐릿한 이미지를 선명하게 합니다.',
     'pipe':[('Image Read','#4CAF50',''),('Sharpen','#FF9800',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 흐릿한 이미지를 업로드합니다.','Sharpen을 추가합니다. Strength=1.5.','Image Read → Sharpen → Image Show를 연결합니다.','Execute하면 선명해진 이미지가 표시됩니다.'],
     'extra':'Gaussian Blur → Sharpen 조합으로 노이즈 제거+선명화를 할 수 있습니다.'},
    {'n':15,'title':'소벨 에지 그래디언트',
     'goal':'Sobel로 X, Y 방향 에지 그래디언트를 시각화합니다.',
     'pipe':[('Image Read','#4CAF50',''),('CvtColor','#9C27B0','GRAY'),('Sobel','#F44336','dx=1'),('Image Show','#4CAF50','')],
     'steps':['Image Read → CvtColor(BGR2GRAY)를 구성합니다.','Sobel 2개 추가: 하나는 dx=1,dy=0(수직 에지), 다른 하나는 dx=0,dy=1(수평 에지).','CvtColor에서 두 Sobel로 각각 연결합니다.','Add Weighted(α=0.5,β=0.5)로 합성합니다.','Execute하면 합쳐진 에지 이미지가 표시됩니다.'],
     'extra':'X 방향은 수직 에지, Y 방향은 수평 에지를 검출합니다.'},
    {'n':16,'title':'팽창 & 침식',
     'goal':'Dilate와 Erode로 이진 이미지 형태를 변형합니다.',
     'pipe':[('Image Read','#4CAF50',''),('CvtColor','#9C27B0','GRAY'),('Threshold','#795548',''),('Dilate','#607D8B',''),('Erode','#607D8B',''),('Image Show','#4CAF50','')],
     'steps':['Image Read → CvtColor → Threshold로 이진 이미지를 만듭니다.','Dilate를 추가합니다. Kernel=5, Iterations=1.','Erode를 추가합니다. 같은 설정.','Threshold → Dilate → Erode → Image Show를 연결합니다.','Execute하면 팽창 후 침식 결과가 표시됩니다.'],
     'extra':'순서를 Erode → Dilate로 바꾸면 닫기 연산이 됩니다.'},
    {'n':17,'title':'허프 직선 검출',
     'goal':'Canny 에지에서 Hough Lines로 직선을 검출합니다.',
     'pipe':[('Image Read','#4CAF50',''),('CvtColor','#9C27B0','GRAY'),('Canny','#F44336',''),('Hough Lines','#CDDC39',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 직선이 많은 이미지를 업로드합니다.','CvtColor(GRAY) → Canny를 연결합니다.','Hough Lines를 추가합니다. Rho=1, Theta=180, Threshold=100.','Canny → Hough Lines → Image Show를 연결합니다.','Execute하면 검출 직선이 빨간색으로 표시됩니다.'],
     'extra':'Threshold를 낮추면 더 많은 직선이 검출됩니다.'},
    {'n':18,'title':'허프 원 검출',
     'goal':'Hough Circles로 원형 물체를 검출합니다.',
     'pipe':[('Image Read','#4CAF50',''),('Hough Circles','#FF5722',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 원형 물체가 있는 이미지를 업로드합니다.','Hough Circles를 추가합니다. dp=1.2, MinDist=50, P1=100, P2=30.','Image Read → Hough Circles → Image Show를 연결합니다.','Execute하면 원이 녹색으로 표시됩니다.'],
     'extra':'Min/Max Radius로 검출할 원 크기를 제한할 수 있습니다.'},
    {'n':19,'title':'리사이즈 & 스케일',
     'goal':'Resize 노드로 이미지 크기를 변경합니다.',
     'pipe':[('Image Read','#4CAF50',''),('Resize','#00BCD4',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 이미지를 업로드합니다.','Resize를 추가합니다. Scale X=0.5, Scale Y=0.5.','Image Read → Resize → Image Show를 연결합니다.','Execute하면 절반 크기 이미지가 표시됩니다.'],
     'extra':'INTER_CUBIC은 확대에, INTER_AREA는 축소에 적합합니다.'},
    {'n':20,'title':'회전 & 뒤집기',
     'goal':'Rotate와 Flip으로 이미지 방향을 변경합니다.',
     'pipe':[('Image Read','#4CAF50',''),('Rotate','#00BCD4','45°'),('Flip','#00BCD4','H'),('Image Show','#4CAF50','')],
     'steps':['Image Read에 이미지를 업로드합니다.','Rotate를 추가합니다. Angle=45.','Flip을 추가합니다. Mode=Horizontal (1).','Image Read → Rotate → Flip → Image Show를 연결합니다.','Execute합니다.'],
     'extra':'Vertical (0)은 상하 반전, Both (-1)은 180도 회전과 같습니다.'},
    {'n':21,'title':'자르기 — 관심 영역 (ROI)',
     'goal':'Crop으로 이미지의 특정 영역만 잘라냅니다.',
     'pipe':[('Image Read','#4CAF50',''),('Crop','#00BCD4',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 이미지를 업로드합니다.','Crop을 추가합니다. X=50, Y=50, Width=200, Height=150.','Image Read → Crop → Image Show를 연결합니다.','Execute하면 지정 영역만 표시됩니다.'],
     'extra':'Template Match + Image Extract와 조합하면 자동 추출이 가능합니다.'},
    {'n':22,'title':'도형 그리기',
     'goal':'Draw 노드들로 이미지 위에 도형을 그립니다.',
     'pipe':[('Image Read','#4CAF50',''),('Draw Rect','#E91E63',''),('Draw Circle','#E91E63',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 이미지를 업로드합니다.','Draw Rectangle을 추가합니다. X=10, Y=10, W=100, H=80, Color=(0,255,0).','Draw Circle을 추가합니다. Center(200,150), R=40, Color=(0,0,255).','Image Read → Rectangle → Circle → Image Show를 연결합니다.','Execute하면 도형이 그려집니다.'],
     'extra':'Thickness=-1이면 채워진 도형이 됩니다.'},
    {'n':23,'title':'텍스트 오버레이',
     'goal':'Draw Text로 이미지에 텍스트를 표시합니다.',
     'pipe':[('Image Read','#4CAF50',''),('Draw Text','#E91E63','Hello'),('Image Show','#4CAF50','')],
     'steps':['Image Read에 이미지를 업로드합니다.','Draw Text를 추가합니다. Text="Hello OpenCV", (50,50), Scale=1.5, Color=(255,255,255).','Image Read → Draw Text → Image Show를 연결합니다.','Execute합니다.'],
     'extra':'여러 Draw Text를 연결하면 여러 줄을 추가할 수 있습니다.'},
    {'n':24,'title':'비트 마스크 연산',
     'goal':'InRange 마스크를 Bitwise AND로 적용하여 특정 색상만 추출합니다.',
     'pipe':[('Image Read','#4CAF50',''),('CvtColor','#9C27B0','HSV'),('InRange','#E91E63',''),('Bitwise AND','#3F51B5',''),('Image Show','#4CAF50','')],
     'steps':['Image Read → CvtColor(BGR2HSV) → InRange를 구성합니다.','InRange에서 원하는 색상의 HSV 범위를 설정합니다.','Bitwise AND를 추가합니다.','원본 Image Read → image, InRange 결과 → image2에 연결합니다.','Execute하면 해당 색상만 원본 색상으로 표시됩니다.'],
     'extra':'Bitwise NOT으로 마스크를 반전하면 반대 영역을 추출합니다.'},
    {'n':25,'title':'히스토그램 평활화 (CLAHE)',
     'goal':'CLAHE로 대비가 낮은 이미지를 개선합니다.',
     'pipe':[('Image Read','#4CAF50',''),('CvtColor','#9C27B0','GRAY'),('Histogram EQ','#673AB7','CLAHE'),('Image Show','#4CAF50','')],
     'steps':['Image Read에 대비가 낮은 이미지를 업로드합니다.','CvtColor(BGR2GRAY) → Histogram EQ를 연결합니다.','Use CLAHE 체크, Clip Limit=2.0, Tile Size=8.','Execute하면 대비가 개선된 이미지가 표시됩니다.'],
     'extra':'Clip Limit를 높이면 대비가 더 강해집니다.'},
    {'n':26,'title':'이미지 차이 — 변화 감지 (AbsDiff)',
     'goal':'AbsDiff로 두 이미지의 차이를 계산합니다.',
     'pipe':[('Image Read','#4CAF50','A'),('Image Read','#4CAF50','B'),('AbsDiff','#3F51B5',''),('Threshold','#795548',''),('Image Show','#4CAF50','')],
     'steps':['Image Read 2개에 같은 장면의 전후 이미지를 업로드합니다.','AbsDiff를 추가합니다. A → image, B → image2.','Threshold를 추가합니다. 임계값=30.','AbsDiff → Threshold → Image Show를 연결합니다.','Execute하면 변화 영역이 흰색으로 표시됩니다.'],
     'extra':'CvtColor(GRAY) 후 AbsDiff를 적용하면 더 정확합니다.'},
    {'n':27,'title':'GrabCut 배경 제거',
     'goal':'GrabCut으로 전경 물체를 분리합니다.',
     'pipe':[('Image Read','#4CAF50',''),('GrabCut','#795548',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 물체가 중앙에 있는 이미지를 업로드합니다.','GrabCut을 추가합니다. 물체를 포함하는 사각영역 설정.','Iterations=5로 설정합니다.','Image Read → GrabCut → Image Show를 연결합니다.','Execute하면 배경이 제거됩니다.'],
     'extra':'사각영역이 물체를 잘 감싸야 정확한 결과가 나옵니다.'},
    {'n':28,'title':'Watershed 이미지 분할',
     'goal':'Watershed로 겹치는 물체를 분리합니다.',
     'pipe':[('Image Read','#4CAF50',''),('Watershed','#795548',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 여러 물체가 겹친 이미지를 업로드합니다.','Watershed를 추가합니다. Marker Kernel=10.','Image Read → Watershed → Image Show를 연결합니다.','Execute하면 경계가 빨간 선으로 표시됩니다.'],
     'extra':'Marker Kernel이 클수록 분할 영역이 적어집니다.'},
    {'n':29,'title':'투시 변환 (Perspective Transform)',
     'goal':'Warp Perspective로 비스듬한 이미지를 정면으로 보정합니다.',
     'pipe':[('Image Read','#4CAF50',''),('Warp Perspective','#00BCD4',''),('Image Show','#4CAF50','')],
     'steps':['Image Read에 비스듬히 촬영된 문서 이미지를 업로드합니다.','Warp Perspective를 추가합니다.','Src 4꼭짓점을 문서 모서리로, Dst를 직사각형으로 설정합니다.','Image Read → Warp Perspective → Image Show를 연결합니다.','Execute하면 정면으로 보정됩니다.'],
     'extra':'명함, 간판 등의 기울어진 촬영물 보정에 활용하세요.'},
    {'n':30,'title':'For Loop — 반복 블러 처리',
     'goal':'For Loop으로 블러를 여러 번 반복 적용합니다.',
     'pipe':[('Image Read','#4CAF50',''),('For Loop','#26A69A','×5'),('Image Show','#4CAF50','')],
     'steps':['Image Read에 이미지를 업로드합니다.','For Loop을 추가합니다. N=5, Operation=gaussian_blur, K=3.','Image Read → For Loop → Image Show를 연결합니다.','Execute하면 블러가 5번 적용된 결과가 표시됩니다.'],
     'extra':'custom 모드에서 직접 코드를 작성하면 복잡한 반복도 가능합니다.'},
]


# ═══════════════════════════════════════════════════════════════
# Part 6: 응용 예제 (10개 복합 프로젝트)
# ═══════════════════════════════════════════════════════════════
EXAMPLES = [
    {'n':1, 'title':'손가락 개수 세기 (Finger Counter)', 'difficulty':'★★★',
     'goal':'비디오에서 피부색을 감지하고 컨투어/볼록 껍질 분석으로 손가락 개수를 추정합니다.',
     'concepts':['HSV 색상 공간', '피부색 범위 필터링', '침식/팽창 노이즈 제거', '컨투어 분석', '볼록 껍질(Convex Hull)'],
     'pipe':[('Video Read','#FF5722','비디오'), ('CvtColor','#9C27B0','HSV'), ('InRange','#E91E63','피부색'),
             ('Gaussian','#FF9800',''), ('Erode','#607D8B',''), ('Dilate','#607D8B',''),
             ('Contours','#009688',''), ('Convex Hull','#009688',''), ('Show','#4CAF50','')],
     'steps':[
         'Video Read 노드를 추가하고 손 영상 파일을 업로드합니다.',
         'CvtColor (BGR->HSV)을 연결합니다. HSV 색상 공간은 피부색 감지에 효과적입니다.',
         'InRange를 추가합니다. 피부색 HSV 범위: H=0~20, S=30~150, V=60~255.',
         'Gaussian Blur (ksize=5)로 노이즈를 줄입니다.',
         'Erode (ksize=3, iterations=2)로 잔여 노이즈를 제거합니다.',
         'Dilate (ksize=3, iterations=3)로 손 영역을 확장합니다.',
         'Find Contours (RETR_EXTERNAL)로 가장 큰 윤곽선(손)을 찾습니다.',
         'Convex Hull로 볼록 껍질을 그립니다. 손가락 사이 오목한 부분이 손가락 수를 결정합니다.',
         'Contour Properties를 추가하면 면적/중심점이 표시됩니다.',
         'Execute하면 손 위에 볼록 껍질이 그려집니다.'],
     'file':'01_finger_counter.json',
     'extra':'조명 조건에 따라 InRange의 HSV 범위를 조절하세요. 어두운 환경에서는 V 하한을 낮추세요.'},

    {'n':2, 'title':'이미지 페이드 전환 (Fade In/Out)', 'difficulty':'★★',
     'goal':'두 이미지를 Alpha 블렌딩으로 부드럽게 전환합니다.',
     'concepts':['Alpha 블렌딩', '가중치 합산', '이미지 크기 매칭'],
     'pipe':[('Image Read','#4CAF50','A'), ('Resize','#00BCD4','640x480'), ('Add Weighted','#3F51B5','a=0.5'), ('Show','#4CAF50','')],
     'pipe2':[('Image Read','#4CAF50','B'), ('Resize','#00BCD4','640x480')],
     'steps':[
         'Image Read 두 개를 추가합니다. 각각 Image A, Image B를 업로드합니다.',
         'Resize 두 개를 추가합니다. 두 이미지를 같은 크기(640x480)로 통일합니다.',
         'Add Weighted를 추가합니다. Alpha=0.5, Beta=0.5, Gamma=0.',
         'Image A의 Resize -> Add Weighted의 image 포트에 연결합니다.',
         'Image B의 Resize -> Add Weighted의 image2 포트에 연결합니다.',
         'Add Weighted -> Image Show를 연결하고 Execute합니다.',
         'Alpha를 0.0->1.0으로 변경하면 Fade-in/out 효과를 확인할 수 있습니다.'],
     'file':'02_fade_transition.json',
     'extra':'Alpha+Beta=1.0이면 원래 밝기가 유지됩니다. Alpha를 0.0에서 1.0까지 변경하며 전환 효과를 확인하세요.'},

    {'n':3, 'title':'움직임 감지 (Motion Detection)', 'difficulty':'★★★',
     'goal':'비디오의 연속 프레임 차이를 이용하여 움직이는 물체를 감지합니다.',
     'concepts':['프레임 차분', '절대값 차이(AbsDiff)', '임계값 처리', '바운딩 박스'],
     'pipe':[('Video T','#FF5722','0'), ('AbsDiff','#3F51B5',''), ('CvtColor','#9C27B0','GRAY'),
             ('Threshold','#795548','30'), ('Dilate','#607D8B',''), ('Contours','#009688',''),
             ('BBox','#009688',''), ('Show','#4CAF50','')],
     'pipe2':[('Video T+5','#FF5722','5')],
     'steps':[
         'Video Read 두 개를 추가합니다. 같은 비디오를 로드하되 Frame Index를 0과 5로 설정합니다.',
         'AbsDiff 노드를 추가합니다. 두 프레임의 차이를 계산합니다.',
         'Video(0) -> AbsDiff의 image, Video(5) -> AbsDiff의 image2에 연결합니다.',
         'CvtColor(BGR2GRAY)로 그레이스케일 변환합니다.',
         'Threshold (thresh=30, BINARY)로 유의미한 변화만 추출합니다.',
         'Dilate (ksize=5, iterations=3)로 감지 영역을 확장합니다.',
         'Find Contours -> Bounding Rect를 연결합니다.',
         'Execute하면 움직임이 있는 영역에 녹색 사각형이 표시됩니다.'],
     'file':'03_motion_detection.json',
     'extra':'Frame Index 차이를 크게 하면 큰 움직임, 작게 하면 미세한 움직임을 감지합니다.'},

    {'n':4, 'title':'색상 객체 추적 (Color Tracking)', 'difficulty':'★★★',
     'goal':'HSV 색상 범위로 특정 색상의 물체를 실시간 추적합니다.',
     'concepts':['HSV 필터링', '모폴로지 열기/닫기', '최소 외접원'],
     'pipe':[('Video Read','#FF5722','loop'), ('CvtColor','#9C27B0','HSV'), ('InRange','#E91E63','파란색'),
             ('Morph Open','#607D8B',''), ('Morph Close','#607D8B',''), ('Contours','#009688',''),
             ('Circle','#009688',''), ('Show','#4CAF50','')],
     'steps':[
         'Video Read를 추가하고 파란색 물체가 있는 영상을 업로드합니다. Mode=loop.',
         'CvtColor (BGR->HSV)을 연결합니다.',
         'InRange를 추가합니다. 파란색 HSV: H=100~130, S=50~255, V=50~255.',
         'Morphology (MORPH_OPEN, ksize=5, Ellipse)로 작은 노이즈를 제거합니다.',
         'Morphology (MORPH_CLOSE, ksize=5, Ellipse)로 내부 빈 공간을 채웁니다.',
         'Find Contours (RETR_EXTERNAL) -> Min Enclosing Circle을 연결합니다.',
         'Image Show를 연결하고 Execute합니다.',
         '파란색 물체 주위에 원이 그려집니다.'],
     'file':'04_color_tracking.json',
     'extra':'빨간색: H=0~10/170~180, 녹색: H=35~85. 환경 조명에 따라 S,V 범위 조절이 필요합니다.'},

    {'n':5, 'title':'얼굴 + 눈 검출 (Face & Eye Detection)', 'difficulty':'★★',
     'goal':'Haar Cascade 분류기를 이용하여 얼굴과 눈을 동시에 검출합니다.',
     'concepts':['Haar Cascade', '캐스케이드 분류기', '얼굴 검출', '눈 검출'],
     'pipe':[('Image Read','#4CAF50',''), ('Haar Cascade','#FF5722','face'), ('Haar Cascade','#FF5722','eye'), ('Show','#4CAF50','')],
     'steps':[
         'Image Read에 사람 얼굴이 포함된 이미지를 업로드합니다.',
         'Haar Cascade (Face)를 추가합니다. Type=frontalface, Scale=1.3, Neighbors=5.',
         'Image Read -> Haar Cascade (Face)를 연결합니다.',
         'Haar Cascade (Eye)를 추가합니다. Type=eye, Scale=1.1, Neighbors=5.',
         'Haar Cascade (Face) 출력 -> Haar Cascade (Eye) 입력에 연결합니다.',
         '이렇게 하면 얼굴 영역 안에서 눈을 더 정확하게 감지합니다.',
         'Haar Cascade (Eye) -> Image Show를 연결하고 Execute합니다.',
         '얼굴에 큰 사각형, 눈에 작은 사각형이 그려집니다.'],
     'file':'05_face_detection.json',
     'extra':'단체 사진에서 minNeighbors를 높이면 오검출이 줄어듭니다.'},

    {'n':6, 'title':'카툰 효과 (Cartoon Effect)', 'difficulty':'★★★★',
     'goal':'양방향 필터 + 에지 마스크를 결합하여 만화 스타일 효과를 만듭니다.',
     'concepts':['양방향 필터링', '적응형 임계값', '채널 병합', 'Bitwise AND 마스킹'],
     'pipe':[('Image Read','#4CAF50',''), ('Bilateral','#FF9800','d=9'), ('Bitwise AND','#3F51B5',''), ('Show','#4CAF50','')],
     'pipe2':[('CvtColor','#9C27B0','GRAY'), ('Median','#FF9800','k=7'), ('Adaptive TH','#795548',''), ('Merge Ch','#9C27B0','3ch')],
     'steps':[
         'Image Read에 이미지를 업로드합니다.',
         '[색상 경로] Bilateral Filter (d=9, sigmaColor=75, sigmaSpace=75)를 추가합니다.',
         'Image Read -> Bilateral Filter를 연결합니다. 에지를 보존하며 부드럽게 합니다.',
         '[에지 경로] CvtColor(BGR2GRAY) -> Median Blur(ksize=7) -> Adaptive Threshold를 연결합니다.',
         'Adaptive Threshold: Method=MEAN_C, Block Size=9, C=2. 에지가 검은 선으로 표현됩니다.',
         'Merge Channels를 추가합니다. Adaptive Threshold 출력을 ch0, ch1, ch2 모두에 연결합니다.',
         '이렇게 하면 1채널 에지 이미지가 3채널(BGR)로 변환됩니다.',
         '[합성] Bitwise AND를 추가합니다.',
         'Bilateral -> image, Merge Channels -> image2에 연결합니다.',
         'Bitwise AND -> Image Show를 연결하고 Execute합니다.',
         '만화처럼 색상은 부드럽고 에지가 선명한 이미지가 생성됩니다.'],
     'file':'06_cartoon_effect.json',
     'extra':'Bilateral Filter의 반복 횟수를 늘리면(d 값 증가) 더 강한 카툰 효과를 얻습니다.'},

    {'n':7, 'title':'문서 스캐너 (Document Scanner)', 'difficulty':'★★★',
     'goal':'에지 검출과 컨투어 분석으로 문서 윤곽을 감지합니다.',
     'concepts':['적응형 임계값', '모폴로지 닫기', '컨투어 검출', '다각형 근사'],
     'pipe':[('Image Read','#4CAF50',''), ('CvtColor','#9C27B0','GRAY'), ('Gaussian','#FF9800','k=5'),
             ('Adaptive TH','#795548',''), ('Morph Close','#607D8B',''), ('Contours','#009688',''),
             ('Approx Poly','#009688',''), ('Show','#4CAF50','')],
     'steps':[
         'Image Read에 문서 사진을 업로드합니다. 배경과 문서의 대비가 중요합니다.',
         'CvtColor (BGR->GRAY)으로 변환합니다.',
         'Gaussian Blur (ksize=5)로 노이즈를 줄입니다.',
         'Adaptive Threshold (GAUSSIAN_C, blockSize=11, C=2)를 적용합니다.',
         'Morphology (MORPH_CLOSE, ksize=3, iterations=2)로 에지 끊김을 연결합니다.',
         'Find Contours (RETR_EXTERNAL)로 외곽 윤곽을 찾습니다.',
         'Approx Poly (epsilon=0.02)를 연결합니다. 4개 꼭지점이 감지되면 문서 윤곽입니다.',
         'Execute하면 문서 영역에 녹색 다각형이 그려집니다.',
         '* Gaussian에서 분기하여 Canny -> Image Show로 에지를 보조 확인할 수 있습니다.'],
     'file':'07_document_scanner.json',
     'extra':'감지된 4꼭지점을 Warp Perspective에 적용하면 문서를 정면 뷰로 변환할 수 있습니다.'},

    {'n':8, 'title':'히스토그램 CLAHE 보정', 'difficulty':'★★',
     'goal':'CLAHE 알고리즘으로 이미지 대비를 개선하고 히스토그램으로 비교합니다.',
     'concepts':['CLAHE', '히스토그램 평활화', '대비 보정', '히스토그램 시각화'],
     'pipe':[('Image Read','#4CAF50',''), ('Histogram EQ','#673AB7','CLAHE'), ('Show','#4CAF50','보정후'),
             ('Calc Hist','#673AB7',''), ('Show','#4CAF50','히스토그램')],
     'steps':[
         'Image Read에 어둡거나 대비가 낮은 이미지를 업로드합니다.',
         'Histogram EQ를 추가합니다. CLAHE=true, Clip Limit=2.0, Tile Size=8.',
         'Image Read -> Histogram EQ -> Image Show(보정 후)를 연결합니다.',
         '원본 비교: Image Read -> 다른 Image Show(원본)에도 연결합니다.',
         'Calc Histogram 두 개를 추가합니다.',
         '하나는 Image Read -> Calc Histogram -> Image Show (원본 히스토그램).',
         '다른 하나는 Histogram EQ -> Calc Histogram -> Image Show (보정 후 히스토그램).',
         'Execute하면 원본/보정 이미지와 히스토그램을 비교할 수 있습니다.'],
     'file':'08_histogram_clahe.json',
     'extra':'Clip Limit을 높이면 대비가 강해지지만, 너무 높으면 노이즈가 증폭됩니다.'},

    {'n':9, 'title':'이미지 워터마크 합성', 'difficulty':'★★',
     'goal':'두 이미지를 블렌딩하고 텍스트 워터마크를 추가합니다.',
     'concepts':['Alpha 블렌딩', '텍스트 드로잉', '워터마크 합성'],
     'pipe':[('Image Read','#4CAF50','메인'), ('Resize','#00BCD4','640x480'), ('Add Weighted','#3F51B5','a=0.7'),
             ('Draw Text','#E91E63','(C)'), ('Show','#4CAF50','')],
     'pipe2':[('Image Read','#4CAF50','오버레이'), ('Resize','#00BCD4','640x480')],
     'steps':[
         'Image Read 두 개를 추가합니다. 메인 이미지와 워터마크/로고 이미지를 업로드합니다.',
         '각각 Resize로 동일한 크기(640x480)로 맞춥니다.',
         'Add Weighted를 추가합니다. Alpha=0.7(메인), Beta=0.3(오버레이).',
         '메인 Resize -> image, 오버레이 Resize -> image2에 연결합니다.',
         'Draw Text를 추가합니다. Text="(C) SamOpenCV", 위치=(20,460), 색상=흰색.',
         'Add Weighted -> Draw Text -> Image Show를 연결합니다.',
         'Execute하면 두 이미지가 합성되고 텍스트 워터마크가 추가됩니다.'],
     'file':'09_watermark_blend.json',
     'extra':'Alpha를 1.0에 가깝게 하면 메인 이미지가 더 선명하게 보입니다.'},

    {'n':10, 'title':'에지 아트 (Multi-Edge Art)', 'difficulty':'★★★',
     'goal':'다중 에지 검출기의 결과를 결합하여 예술적 효과를 만듭니다.',
     'concepts':['캐니 에지', '소벨 에지', 'Bitwise OR 결합', '비트 반전'],
     'pipe':[('Image Read','#4CAF50',''), ('CvtColor','#9C27B0','GRAY'), ('Gaussian','#FF9800','k=3'),
             ('Canny','#F44336',''), ('Bitwise OR','#3F51B5',''), ('NOT','#3F51B5',''), ('Show','#4CAF50','')],
     'pipe2':[('Sobel','#F44336','dx=1,dy=1')],
     'steps':[
         'Image Read에 풍경 또는 인물 이미지를 업로드합니다.',
         'CvtColor (BGR->GRAY)로 그레이스케일 변환합니다.',
         '[캐니 경로] Gaussian Blur(ksize=3) -> Canny(50, 150)을 연결합니다.',
         '[소벨 경로] CvtColor에서 Sobel(dx=1, dy=1)을 분기합니다.',
         'Bitwise OR를 추가합니다. Canny -> image, Sobel -> image2에 연결합니다.',
         '두 에지 검출 결과가 합성됩니다.',
         'Bitwise NOT을 추가합니다. 흰 배경에 검은 선으로 반전됩니다.',
         'Image Show를 연결하고 Execute합니다.',
         '마치 연필 스케치처럼 보이는 에지 아트가 생성됩니다.'],
     'file':'10_edge_art.json',
     'extra':'Laplacian이나 Scharr도 Bitwise OR에 추가하면 더 풍부한 에지 표현을 얻습니다.'},

    {'n':11, 'title':'노이즈 제거 비교 (Blur Comparison)', 'difficulty':'★★',
     'goal':'세 가지 블러 필터(Gaussian, Median, Bilateral)의 노이즈 제거 효과를 비교합니다.',
     'concepts':['가우시안 블러', '미디언 블러', '양방향 필터', '노이즈 제거'],
     'pipe':[('Image Read','#4CAF50',''), ('Gaussian','#FF9800','k=5'), ('Show','#4CAF50','Gaussian')],
     'pipe2':[('Median','#FF9800','k=5'), ('Bilateral','#FF9800','d=9')],
     'steps':[
         'Image Read에 노이즈가 있는 이미지를 업로드합니다.',
         'Gaussian Blur (ksize=5)를 연결합니다. 전체적으로 균일하게 흐려집니다.',
         'Median Blur (ksize=5)를 별도 연결합니다. 소금-후추 노이즈에 효과적입니다.',
         'Bilateral Filter (d=9, sigmaColor=75)를 별도 연결합니다. 에지를 보존합니다.',
         '각각 Image Show에 연결하고 Execute합니다.',
         '세 가지 필터의 결과를 비교하여 상황에 맞는 필터를 선택합니다.'],
     'file':'11_noise_removal.json',
     'extra':'Gaussian은 범용, Median은 소금-후추, Bilateral은 에지 보존에 적합합니다.'},

    {'n':12, 'title':'이미지 네거티브 (반전)', 'difficulty':'★',
     'goal':'Bitwise NOT으로 이미지 색상을 반전시킵니다.',
     'concepts':['비트 반전', '네거티브 이미지', 'Bitwise NOT'],
     'pipe':[('Image Read','#4CAF50',''), ('Bitwise NOT','#3F51B5',''), ('Show','#4CAF50','')],
     'steps':[
         'Image Read에 이미지를 업로드합니다.',
         'Bitwise NOT을 연결합니다. 모든 픽셀 값이 255에서 빼진 값으로 변환됩니다.',
         'Image Show를 연결하고 Execute합니다.',
         '원본과 비교: 별도 Image Show에 Image Read를 직접 연결합니다.'],
     'file':'12_image_negative.json',
     'extra':'X-ray 이미지 분석이나 예술적 효과에 활용됩니다.'},

    {'n':13, 'title':'파노라마용 특징 매칭 (Feature Matching)', 'difficulty':'★★★',
     'goal':'두 이미지의 ORB 특징점을 매칭하여 파노라마 합성 가능성을 확인합니다.',
     'concepts':['ORB 특징점', '특징 매칭', '파노라마', '매칭 비율'],
     'pipe':[('Image Read','#4CAF50','Left'), ('Match Features','#CDDC39',''), ('Show','#4CAF50','')],
     'pipe2':[('Image Read','#4CAF50','Right')],
     'steps':[
         'Image Read 두 개를 추가합니다. 겹치는 영역이 있는 좌/우 이미지를 업로드합니다.',
         'Match Features를 추가합니다. nFeatures=500, Match Ratio=0.75.',
         '왼쪽 이미지 -> image, 오른쪽 이미지 -> image2에 연결합니다.',
         'Image Show를 연결하고 Execute합니다.',
         '매칭된 특징점 쌍이 선으로 연결되어 표시됩니다.'],
     'file':'13_panorama_matching.json',
     'extra':'Match Ratio를 낮추면 더 정확한 매칭만 남고, 높이면 더 많은 매칭이 표시됩니다.'},

    {'n':14, 'title':'허프 직선 검출 (Hough Lines)', 'difficulty':'★★★',
     'goal':'Hough 변환으로 이미지에서 직선을 검출합니다.',
     'concepts':['허프 변환', '직선 검출', '캐니 에지', '파라미터 공간'],
     'pipe':[('Image Read','#4CAF50',''), ('CvtColor','#9C27B0','GRAY'), ('Gaussian','#FF9800','k=5'),
             ('Canny','#F44336',''), ('Hough Lines','#CDDC39',''), ('Show','#4CAF50','')],
     'steps':[
         'Image Read에 건물이나 도로 사진을 업로드합니다.',
         'CvtColor(BGR2GRAY) -> Gaussian Blur(ksize=5) -> Canny(50, 150)를 연결합니다.',
         'Hough Lines를 추가합니다. Rho=1, Theta Divisor=180, Threshold=100.',
         'Canny -> Hough Lines -> Image Show를 연결하고 Execute합니다.',
         '검출된 직선이 이미지 위에 빨간 선으로 표시됩니다.'],
     'file':'14_hough_line_detection.json',
     'extra':'Threshold를 낮추면 더 많은 직선이 검출되고, 높이면 확실한 직선만 검출됩니다.'},

    {'n':15, 'title':'허프 원 검출 (Hough Circles)', 'difficulty':'★★★',
     'goal':'Hough 원 변환으로 이미지에서 원형 객체를 검출합니다.',
     'concepts':['허프 원 변환', '원 검출', '가우시안 블러', '파라미터 튜닝'],
     'pipe':[('Image Read','#4CAF50',''), ('CvtColor','#9C27B0','GRAY'), ('Gaussian','#FF9800','k=9'),
             ('Hough Circles','#CDDC39',''), ('Show','#4CAF50','')],
     'steps':[
         'Image Read에 동전이나 공 등 원형 물체 이미지를 업로드합니다.',
         'CvtColor(BGR2GRAY)로 변환합니다.',
         'Gaussian Blur (ksize=9)로 노이즈를 충분히 줄입니다. 원 검출에 중요합니다.',
         'Hough Circles를 추가합니다. dp=1.2, minDist=50, param2=30, minRadius=10, maxRadius=100.',
         'Execute하면 검출된 원이 표시됩니다.'],
     'file':'15_hough_circle_detection.json',
     'extra':'minDist는 원 사이 최소 거리, param2를 낮추면 더 많은 원이 검출됩니다.'},

    {'n':16, 'title':'GrabCut 전경 추출', 'difficulty':'★★',
     'goal':'GrabCut 알고리즘으로 이미지에서 전경 객체를 자동 분리합니다.',
     'concepts':['GrabCut', '전경 분리', 'ROI 영역', '반복 최적화'],
     'pipe':[('Image Read','#4CAF50',''), ('GrabCut','#795548','ROI'), ('Show','#4CAF50','')],
     'steps':[
         'Image Read에 전경 객체가 있는 이미지를 업로드합니다.',
         'GrabCut을 추가합니다. 전경 객체를 포함하는 ROI를 설정합니다: x=50, y=50, width=400, height=300.',
         'Iterations=5로 설정합니다. 반복 횟수가 많을수록 정확합니다.',
         'Image Read -> GrabCut -> Image Show를 연결하고 Execute합니다.',
         '전경 객체만 추출되어 표시됩니다.'],
     'file':'16_grabcut_segmentation.json',
     'extra':'ROI가 전경 객체를 충분히 포함해야 합니다. Iterations를 높이면 결과가 개선됩니다.'},

    {'n':17, 'title':'Watershed 분할', 'difficulty':'★★★',
     'goal':'Watershed 알고리즘으로 이미지를 영역별로 분할합니다.',
     'concepts':['Watershed', '영역 분할', '마커 기반 분할'],
     'pipe':[('Image Read','#4CAF50',''), ('Watershed','#795548',''), ('Show','#4CAF50','')],
     'steps':[
         'Image Read에 여러 객체가 있는 이미지를 업로드합니다.',
         'Watershed를 추가합니다. Marker Size=3.',
         'Image Read -> Watershed -> Image Show를 연결합니다.',
         'Execute하면 영역 경계가 색상으로 표시됩니다.',
         '별도 Image Show에 원본을 연결하여 비교합니다.'],
     'file':'17_watershed_segmentation.json',
     'extra':'겹쳐진 동전, 셀 등의 분리에 효과적입니다.'},

    {'n':18, 'title':'원근 변환 보정 (Perspective Transform)', 'difficulty':'★★★',
     'goal':'비스듬하게 촬영된 이미지를 정면 뷰로 변환합니다.',
     'concepts':['원근 변환', '4점 변환', '호모그래피'],
     'pipe':[('Image Read','#4CAF50',''), ('Warp Perspective','#00BCD4','4점'), ('Show','#4CAF50','')],
     'steps':[
         'Image Read에 비스듬하게 촬영된 문서/간판 이미지를 업로드합니다.',
         'Warp Perspective를 추가합니다.',
         '소스 좌표 4개: 원본 이미지에서 문서의 네 꼭지점을 지정합니다.',
         '대상 좌표 4개: 변환 후 위치를 직사각형으로 지정합니다.',
         'Execute하면 원근이 보정된 정면 뷰 이미지가 생성됩니다.'],
     'file':'18_perspective_transform.json',
     'extra':'문서 스캐너(예제 7)의 Approx Poly로 감지된 꼭지점을 여기에 활용할 수 있습니다.'},

    {'n':19, 'title':'Flood Fill 색칠', 'difficulty':'★★',
     'goal':'Flood Fill로 이미지의 특정 영역을 색칠합니다.',
     'concepts':['Flood Fill', '시드 포인트', '색상 허용 범위', '영역 채우기'],
     'pipe':[('Image Read','#4CAF50',''), ('Flood Fill','#795548','빨강'), ('Flood Fill','#795548','녹색'),
             ('Flood Fill','#795548','파랑'), ('Show','#4CAF50','')],
     'steps':[
         'Image Read에 이미지를 업로드합니다.',
         '첫 번째 Flood Fill: seedX=100, seedY=100, Color=Red(255,0,0), loDiff=30, upDiff=30.',
         '두 번째 Flood Fill: seedX=200, seedY=200, Color=Green(0,255,0).',
         '세 번째 Flood Fill: seedX=300, seedY=150, Color=Blue(0,0,255).',
         '직렬로 연결하여 3개 영역을 순차적으로 색칠합니다.',
         'Execute하면 각 시드 포인트 주변의 유사 색상 영역이 채워집니다.'],
     'file':'19_flood_fill_coloring.json',
     'extra':'loDiff와 upDiff를 높이면 더 넓은 색상 범위가 채워집니다.'},

    {'n':20, 'title':'형태학 연산 비교 (Morphology Comparison)', 'difficulty':'★★',
     'goal':'Open, Close, Gradient, Top Hat 형태학 연산의 차이를 비교합니다.',
     'concepts':['열기(Open)', '닫기(Close)', '그래디언트', 'Top Hat', '이진화'],
     'pipe':[('Image Read','#4CAF50',''), ('CvtColor','#9C27B0','GRAY'), ('Otsu TH','#795548',''),
             ('Morph Open','#607D8B',''), ('Show','#4CAF50','Open')],
     'pipe2':[('Close','#607D8B',''), ('Gradient','#607D8B',''), ('TopHat','#607D8B','')],
     'steps':[
         'Image Read에 이미지를 업로드합니다.',
         'CvtColor(BGR2GRAY) -> Otsu Threshold로 이진화합니다.',
         'Morphology 4개를 추가합니다: MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT.',
         'Otsu 출력에서 4개로 분기하여 각각 Image Show에 연결합니다.',
         'Execute하면 4가지 연산 결과를 비교할 수 있습니다.',
         'Open: 노이즈 제거, Close: 빈 공간 채움, Gradient: 에지, TopHat: 밝은 요소 추출.'],
     'file':'20_morphology_comparison.json',
     'extra':'ksize와 iterations를 조정하면 연산 강도가 달라집니다.'},

    {'n':21, 'title':'샤프닝 강도 비교 (Sharpen Comparison)', 'difficulty':'★',
     'goal':'다양한 샤프닝 강도의 효과를 비교합니다.',
     'concepts':['언샤프 마스크', '샤프닝 강도', '이미지 선명화'],
     'pipe':[('Image Read','#4CAF50',''), ('Sharpen','#FF9800','0.5'), ('Show','#4CAF50','Weak')],
     'pipe2':[('Sharpen','#FF9800','1.5'), ('Sharpen','#FF9800','3.0')],
     'steps':[
         'Image Read에 약간 흐릿한 이미지를 업로드합니다.',
         'Sharpen 3개를 추가합니다: Strength=0.5(약), 1.5(중), 3.0(강).',
         'Image Read에서 3개로 분기하여 각각 Image Show에 연결합니다.',
         'Execute하면 3단계 샤프닝 강도를 비교할 수 있습니다.',
         '너무 강하면 노이즈가 증폭되므로 적절한 값을 선택합니다.'],
     'file':'21_sharpen_compare.json',
     'extra':'Strength=1.0~2.0이 대부분의 경우에 적합합니다.'},

    {'n':22, 'title':'색상 추출 (Color Extraction)', 'difficulty':'★★★',
     'goal':'HSV 마스크로 특정 색상만 추출하여 원본에 적용합니다.',
     'concepts':['HSV 마스크', 'Bitwise AND 마스킹', '색상 분리'],
     'pipe':[('Image Read','#4CAF50',''), ('CvtColor','#9C27B0','HSV'), ('InRange','#E91E63','빨강'),
             ('Bitwise AND','#3F51B5',''), ('Show','#4CAF50','Red Only')],
     'steps':[
         'Image Read에 다양한 색상이 있는 이미지를 업로드합니다.',
         'CvtColor(BGR2HSV) -> InRange로 빨간색 마스크를 생성합니다: H=0~10, S=100~255, V=100~255.',
         'Bitwise AND를 추가합니다.',
         'Image Read(원본) -> image 포트, InRange(마스크) -> image2 포트에 연결합니다.',
         'Execute하면 빨간색 부분만 원래 색상으로, 나머지는 검정으로 표시됩니다.',
         'InRange의 범위를 변경하면 다른 색상도 추출할 수 있습니다.'],
     'file':'22_color_extraction.json',
     'extra':'여러 InRange + Bitwise OR로 복수 색상을 동시에 추출할 수 있습니다.'},

    {'n':23, 'title':'비디오 에지 출력 (Video Edge Export)', 'difficulty':'★★',
     'goal':'비디오의 모든 프레임에 에지 검출을 적용하여 새 비디오로 저장합니다.',
     'concepts':['비디오 처리', '프레임별 에지 검출', '비디오 출력'],
     'pipe':[('Video Read','#FF5722','loop'), ('CvtColor','#9C27B0','GRAY'), ('Canny','#F44336',''),
             ('Image Write','#8BC34A','MP4')],
     'steps':[
         'Video Read를 추가하고 비디오를 업로드합니다. Mode=loop.',
         'CvtColor(BGR2GRAY) -> Canny(100, 200)를 연결합니다.',
         'Image Write를 추가합니다. Video Output=true, filepath=output_edges.mp4, Codec=mp4v, FPS=30.',
         'Canny -> Image Write를 연결합니다.',
         '분기하여 Canny -> Image Show를 연결하면 미리보기도 됩니다.',
         'Execute하면 에지 검출된 비디오가 저장됩니다.'],
     'file':'23_video_edge_output.json',
     'extra':'Canny 대신 다른 에지/필터 노드를 연결하면 다양한 효과의 비디오를 생성합니다.'},

    {'n':24, 'title':'ROI 영역 집중 처리', 'difficulty':'★★',
     'goal':'이미지의 특정 영역(ROI)을 잘라내어 집중적으로 처리합니다.',
     'concepts':['ROI(Region of Interest)', 'Crop', '블러', '샤프닝'],
     'pipe':[('Image Read','#4CAF50',''), ('Crop','#00BCD4','ROI'), ('Gaussian','#FF9800','k=11'), ('Show','#4CAF50','Blur')],
     'pipe2':[('Sharpen','#FF9800','2.0')],
     'steps':[
         'Image Read에 이미지를 업로드합니다.',
         'Crop을 추가합니다. x=100, y=100, width=200, height=200으로 관심 영역을 잘라냅니다.',
         'Crop에서 분기하여 Gaussian Blur(ksize=11)와 Sharpen(strength=2.0)을 각각 연결합니다.',
         '각각 Image Show에 연결하고 Execute합니다.',
         'ROI 영역에 대해 블러와 샤프닝 효과를 비교할 수 있습니다.'],
     'file':'24_roi_focus.json',
     'extra':'Crop 좌표를 변경하여 다양한 영역을 분석할 수 있습니다.'},

    {'n':25, 'title':'이미지 회전 갤러리 (Rotation Gallery)', 'difficulty':'★',
     'goal':'다양한 각도로 이미지를 회전시켜 갤러리를 만듭니다.',
     'concepts':['이미지 회전', '각도 변환', '보간법'],
     'pipe':[('Image Read','#4CAF50',''), ('Rotate','#00BCD4','45'), ('Show','#4CAF50','45도')],
     'pipe2':[('Rotate','#00BCD4','90'), ('Rotate','#00BCD4','180')],
     'steps':[
         'Image Read에 이미지를 업로드합니다.',
         'Rotate 3개를 추가합니다: 45도, 90도, 180도.',
         'Image Read에서 3개로 분기하여 각각 Image Show에 연결합니다.',
         'Execute하면 3가지 회전 결과를 동시에 비교할 수 있습니다.'],
     'file':'25_image_rotation_gallery.json',
     'extra':'Flip 노드를 추가하면 좌우/상하 반전도 함께 비교할 수 있습니다.'},

    {'n':26, 'title':'임계값 방법 비교 (Threshold Comparison)', 'difficulty':'★★',
     'goal':'Binary, Otsu, Adaptive 세 가지 임계값 방법을 비교합니다.',
     'concepts':['고정 임계값', 'Otsu 자동 임계값', '적응형 임계값'],
     'pipe':[('Image Read','#4CAF50',''), ('CvtColor','#9C27B0','GRAY'), ('Threshold','#795548','127'), ('Show','#4CAF50','Binary')],
     'pipe2':[('Otsu TH','#795548','auto'), ('Adaptive TH','#795548','local')],
     'steps':[
         'Image Read에 문서 또는 텍스트 이미지를 업로드합니다.',
         'CvtColor(BGR2GRAY)로 변환합니다.',
         'CvtColor에서 3개로 분기합니다:',
         '  1) Threshold (thresh=127, BINARY): 고정값 이진화.',
         '  2) Otsu Threshold: 자동으로 최적 임계값 결정.',
         '  3) Adaptive Threshold (GAUSSIAN_C, blockSize=11, C=2): 영역별 적응적 이진화.',
         '각각 Image Show에 연결하고 Execute합니다.',
         '조명이 불균일한 이미지에서는 Adaptive가 가장 좋은 결과를 보입니다.'],
     'file':'26_threshold_comparison.json',
     'extra':'blockSize를 변경하면 Adaptive Threshold의 참조 영역이 달라집니다.'},

    {'n':27, 'title':'컨투어 분석 종합', 'difficulty':'★★★',
     'goal':'다양한 컨투어 분석 도구의 결과를 비교합니다.',
     'concepts':['컨투어 그리기', '바운딩 박스', '면적 필터링', '윤곽 분석'],
     'pipe':[('Image Read','#4CAF50',''), ('CvtColor','#9C27B0','GRAY'), ('Otsu TH','#795548',''),
             ('Draw Contours','#009688',''), ('Show','#4CAF50','Contours')],
     'pipe2':[('BBox','#009688',''), ('Area Filter','#009688','')],
     'steps':[
         'Image Read에 여러 객체가 있는 이미지를 업로드합니다.',
         'CvtColor(BGR2GRAY) -> Otsu Threshold로 이진화합니다.',
         'Otsu에서 3개로 분기합니다:',
         '  1) Draw Contours (녹색): 모든 윤곽선을 그립니다.',
         '  2) Bounding Rect (빨간색): 외접 사각형을 그립니다.',
         '  3) Contour Area (파란색, minArea=500): 작은 노이즈를 필터링합니다.',
         '각각 Image Show에 연결하고 Execute합니다.'],
     'file':'27_contour_analysis.json',
     'extra':'Contour Properties를 추가하면 면적, 둘레, 중심점 수치를 확인할 수 있습니다.'},

    {'n':28, 'title':'엠보싱 효과 (Emboss Effect)', 'difficulty':'★',
     'goal':'Filter2D 엠보스 프리셋으로 입체 느낌의 효과를 만듭니다.',
     'concepts':['커스텀 커널', '컨볼루션', '엠보싱', 'Filter2D'],
     'pipe':[('Image Read','#4CAF50',''), ('CvtColor','#9C27B0','GRAY'), ('Filter2D','#FF9800','emboss'), ('Show','#4CAF50','')],
     'steps':[
         'Image Read에 이미지를 업로드합니다.',
         'CvtColor(BGR2GRAY)로 그레이스케일 변환합니다.',
         'Filter2D를 추가합니다. Preset=emboss, Kernel Size=3.',
         'CvtColor -> Filter2D -> Image Show를 연결하고 Execute합니다.',
         '입체적인 양각 효과가 적용됩니다.'],
     'file':'28_emboss_effect.json',
     'extra':'Preset을 sharpen이나 edge_detect로 변경하면 다른 커널 효과를 볼 수 있습니다.'},

    {'n':29, 'title':'특징점 검출 비교 (Feature Detection)', 'difficulty':'★★★',
     'goal':'Harris, Shi-Tomasi, ORB, FAST 네 가지 특징점 검출기를 비교합니다.',
     'concepts':['Harris 코너', 'Shi-Tomasi', 'ORB', 'FAST', '특징점 검출'],
     'pipe':[('Image Read','#4CAF50',''), ('Harris','#CDDC39',''), ('Show','#4CAF50','Harris')],
     'pipe2':[('Good Features','#CDDC39',''), ('ORB','#CDDC39',''), ('FAST','#CDDC39','')],
     'steps':[
         'Image Read에 건물이나 패턴이 많은 이미지를 업로드합니다.',
         'Image Read에서 4개로 분기합니다:',
         '  1) Harris Corner: blockSize=2, ksize=3, k=0.04.',
         '  2) Good Features: maxCorners=100, qualityLevel=0.01, minDistance=10.',
         '  3) ORB Features: nFeatures=500.',
         '  4) FAST Features: threshold=25.',
         '각각 Image Show에 연결하고 Execute합니다.',
         '특징점의 수, 분포, 정확도를 비교할 수 있습니다.'],
     'file':'29_feature_detection_compare.json',
     'extra':'ORB는 회전 불변, FAST는 속도가 빠르며, Harris는 코너에 강합니다.'},

    {'n':30, 'title':'Python 스크립트 연필 스케치', 'difficulty':'★★★★',
     'goal':'Python Script 노드로 연필 드로잉 효과를 직접 구현합니다.',
     'concepts':['커스텀 코드', '연필 스케치', 'cv2.divide', '가우시안 반전'],
     'pipe':[('Image Read','#4CAF50',''), ('Python Script','#FFC107','Pencil'), ('Show','#4CAF50','')],
     'steps':[
         'Image Read에 인물 또는 풍경 이미지를 업로드합니다.',
         'Python Script를 추가하고 아래 코드를 입력합니다:',
         '  gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)',
         '  inv = 255 - gray',
         '  blur = cv2.GaussianBlur(inv, (21,21), 0)',
         '  sketch = cv2.divide(gray, 255-blur, scale=256)',
         '  img_output = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)',
         'Image Read -> Python Script -> Image Show를 연결하고 Execute합니다.',
         '연필로 스케치한 듯한 예술적 효과가 생성됩니다.'],
     'file':'30_custom_filter.json',
     'extra':'GaussianBlur 커널 크기를 변경하면 스케치 선의 굵기가 달라집니다.'},
]


class SamTextbook(FPDF):
    def __init__(self):
        super().__init__('P', 'mm', 'A4')
        self.add_font('mg', '', FONT_REG)
        self.add_font('mg', 'B', FONT_BOLD)
        self.set_auto_page_break(auto=True, margin=25)
        self.toc_entries = []
        self._diagram_counter = 0

    def header(self):
        if self.page_no() > 1:
            self.set_font('mg', '', 8)
            self.set_text_color(150, 150, 150)
            self.cell(0, 6, 'SamOpenCVWeb 교재', align='R')
            self.ln(8)

    def footer(self):
        self.set_y(-15)
        self.set_font('mg', '', 8)
        self.set_text_color(150, 150, 150)
        self.cell(0, 10, str(self.page_no()), align='C')

    def add_toc_entry(self, title, level=0):
        self.toc_entries.append((title, self.page_no(), level))

    def ensure_space(self, needed_mm):
        if self.get_y() + needed_mm > self.h - 25:
            self.add_page()

    def chapter_title(self, text, level=0):
        sizes = {0: 18, 1: 14, 2: 12}
        needed = 28 if level == 0 else 20
        self.ensure_space(needed)
        self.add_toc_entry(text, level)
        if level == 0:
            self.ln(10)
        else:
            self.ln(7)
        self.set_font('mg', 'B', sizes.get(level, 12))
        self.set_text_color(30, 30, 46)
        self.multi_cell(0, sizes.get(level, 12) * 0.75, text)
        self.ln(5)
        if level == 0:
            self.set_draw_color(76, 175, 80)
            self.set_line_width(0.8)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(7)

    def body_text(self, text):
        self.set_font('mg', '', 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 7, text)
        self.ln(4)

    def bold_text(self, text):
        self.set_font('mg', 'B', 10)
        self.set_text_color(50, 50, 50)
        self.multi_cell(0, 7, text)
        self.ln(3)

    def colored_box(self, text, r, g, b):
        self.ensure_space(22)
        self.set_fill_color(r, g, b)
        self.set_text_color(255, 255, 255)
        self.set_font('mg', 'B', 11)
        self.cell(0, 10, '  ' + text, fill=True)
        self.ln(7)
        self.set_text_color(50, 50, 50)

    def add_pipeline_image(self, pipe_nodes):
        buf = render_pipeline(pipe_nodes)
        if buf is None:
            return
        self._diagram_counter += 1
        tmp = os.path.join(BASE_DIR, f'_tmp_diagram_{self._diagram_counter}.png')
        with open(tmp, 'wb') as f:
            f.write(buf.read())
        iw = self.w - self.l_margin - self.r_margin
        self.ensure_space(38)
        self.image(tmp, x=self.l_margin, w=iw, h=28)
        self.ln(30)
        try:
            os.remove(tmp)
        except:
            pass

    def add_graph_image(self, json_path, height_mm=55):
        """Add a 2D node graph image rendered from JSON example file."""
        if not os.path.exists(json_path):
            return
        buf = render_graph_from_json(json_path)
        if buf is None:
            return
        self._diagram_counter += 1
        tmp = os.path.join(BASE_DIR, f'_tmp_diagram_{self._diagram_counter}.png')
        with open(tmp, 'wb') as f:
            f.write(buf.read())
        iw = self.w - self.l_margin - self.r_margin
        self.ensure_space(height_mm + 8)
        self.image(tmp, x=self.l_margin, w=iw, h=height_mm)
        self.ln(height_mm + 3)
        try:
            os.remove(tmp)
        except:
            pass

    def build_title_page(self):
        self.add_page()
        self.ln(40)
        self.set_font('mg', 'B', 32)
        self.set_text_color(30, 30, 46)
        self.cell(0, 15, 'SamOpenCVWeb', align='C')
        self.ln(18)
        self.set_font('mg', 'B', 20)
        self.set_text_color(76, 175, 80)
        self.cell(0, 12, '비주얼 OpenCV 에디터 교재', align='C')
        self.ln(30)
        self.set_font('mg', '', 12)
        self.set_text_color(100, 100, 100)
        self.cell(0, 8, '노드 기반 시각적 프로그래밍으로 배우는 영상처리', align='C')
        self.ln(50)
        self.set_font('mg', '', 10)
        self.cell(0, 7, '본 교재는 SamOpenCVWeb의 모든 기능을 다룹니다.', align='C')
        self.ln(5)
        self.cell(0, 7, '83개 노드 레퍼런스 + 30개 실습 튜토리얼', align='C')

    def build_toc(self):
        self.add_page()
        self.set_font('mg', 'B', 18)
        self.set_text_color(30, 30, 46)
        self.cell(0, 12, '목차', align='C')
        self.ln(10)
        for title, page, level in self.toc_entries:
            indent = level * 8
            self.set_x(self.l_margin + indent)
            fs = 10 if level == 0 else 9
            style = 'B' if level == 0 else ''
            self.set_font('mg', style, fs)
            self.set_text_color(50, 50, 50)
            tw = self.get_string_width(title)
            pw = self.get_string_width(str(page))
            avail = self.w - self.l_margin - self.r_margin - indent - pw - 4
            display = title
            if tw > avail:
                while self.get_string_width(display + '...') > avail and len(display) > 5:
                    display = display[:-1]
                display += '...'
            self.cell(avail, 6, display)
            self.cell(pw + 4, 6, str(page), align='R')
            self.ln(5 if level == 0 else 4)

    def build_part1(self):
        self.add_page()
        self.chapter_title('Part 1: 소개')
        self.chapter_title('SamOpenCVWeb이란?', 1)
        self.body_text('SamOpenCVWeb은 Node-RED 스타일의 비주얼 프로그래밍 환경에서 OpenCV 영상처리를 수행할 수 있는 웹 기반 도구입니다. 코드를 작성하지 않고도 노드를 드래그, 연결하여 복잡한 영상처리 파이프라인을 구성할 수 있습니다.')
        self.body_text('Python Flask 백엔드와 HTML5 Canvas 프론트엔드로 구성되어 있으며, 브라우저에서 바로 사용할 수 있습니다.')
        self.chapter_title('시스템 요구사항', 1)
        self.body_text('• Python 3.8 이상\n• OpenCV (cv2) 라이브러리\n• Flask 웹 프레임워크\n• 최신 웹 브라우저 (Chrome, Firefox, Edge 등)')
        self.chapter_title('설치 및 실행', 1)
        self.body_text('1. 필수 패키지 설치: pip install flask opencv-python numpy\n2. 서버 실행: python app.py\n3. 브라우저에서 http://localhost:5000 접속')
        self.chapter_title('화면 구성', 1)
        self.body_text('• 노드 팔레트 (왼쪽): 17개 카테고리, 83개 노드를 드래그하여 추가\n• 캔버스 (중앙): 노드를 배치하고 연결하는 작업 영역\n• 속성 패널 (오른쪽): 선택한 노드의 속성을 편집\n• 미리보기 패널 (오른쪽 하단): 실행 결과 이미지를 표시\n• 도구 모음 (상단): Execute, Clear, Save, Load, Code 버튼')

    def build_part2(self):
        self.add_page()
        self.chapter_title('Part 2: 기본 사용법')
        self.chapter_title('노드 추가', 1)
        self.body_text('왼쪽 팔레트에서 원하는 노드를 캔버스로 드래그&드롭합니다. 또는 노드를 클릭하면 캔버스 중앙에 추가됩니다.')
        self.chapter_title('노드 연결', 1)
        self.body_text('노드의 출력 포트(오른쪽 원)를 다른 노드의 입력 포트(왼쪽 원)로 드래그하면 연결선이 만들어집니다. 가까이 배치하면 자동 연결됩니다.')
        self.chapter_title('속성 편집', 1)
        self.body_text('노드를 클릭하면 오른쪽 속성 패널에 해당 노드의 속성이 표시됩니다. 값을 변경하면 실행 시 반영됩니다.')
        self.chapter_title('실행 & 미리보기', 1)
        self.body_text('• Execute (Ctrl+Enter): 전체 파이프라인을 실행합니다.\n• 노드를 더블클릭하면 해당 노드까지만 실행합니다.\n• 파일 노드(Image Read 등)를 더블클릭하면 파일 선택 대화상자가 열립니다.')
        self.chapter_title('저장 & 불러오기', 1)
        self.body_text('• Save: 현재 파이프라인을 JSON 파일로 저장합니다.\n• Load: 저장된 파이프라인을 불러옵니다.\n• 브라우저의 로컬 저장소를 사용합니다.')
        self.chapter_title('코드 생성', 1)
        self.body_text('Code 버튼을 클릭하면 현재 파이프라인에 해당하는 Python/OpenCV 코드가 자동 생성됩니다. 이 코드를 복사하여 독립 스크립트로 사용할 수 있습니다.')
        self.chapter_title('단축키', 1)
        shortcuts = [
            ('Ctrl+Enter', '파이프라인 실행'),
            ('Ctrl+Z', '실행 취소 (Undo)'),
            ('Ctrl+C / X / V', '복사 / 잘라내기 / 붙여넣기'),
            ('Ctrl+A', '전체 선택'),
            ('Delete', '선택 노드 삭제'),
            ('Ctrl+S', '파이프라인 저장'),
            ('마우스 휠', '캔버스 확대/축소'),
            ('마우스 드래그 (빈 영역)', '캔버스 이동'),
            ('Shift+드래그', '다중 선택'),
        ]
        for key, desc in shortcuts:
            self.set_font('mg', 'B', 9)
            self.cell(45, 6, key)
            self.set_font('mg', '', 9)
            self.cell(0, 6, desc)
            self.ln(5)

    def build_part3(self):
        self.add_page()
        self.chapter_title('Part 3: 노드 레퍼런스')
        self.body_text('SamOpenCVWeb에서 사용 가능한 83개 노드를 카테고리별로 설명합니다.')
        self.ln(3)
        current_cat = ''
        for nd in NODES:
            cat = nd['cat']
            if cat != current_cat:
                current_cat = cat
                color_hex, cat_kr = CATEGORIES[cat]
                self.ln(4)
                rgb = hex_to_rgb(color_hex)
                self.colored_box(cat_kr, rgb[0], rgb[1], rgb[2])
                self.ln(3)
            # Estimate space needed for this node
            n_props = len(nd['p'])
            needed = 40 + n_props * 6
            self.ensure_space(min(needed, 65))
            # Node header
            self.set_font('mg', 'B', 11)
            rgb = hex_to_rgb(nd['c'])
            self.set_text_color(rgb[0], rgb[1], rgb[2])
            self.cell(0, 8, f"{nd['kr']}  ({nd['en']})")
            self.ln(8)
            # Signature
            self.set_font('mg', '', 9)
            self.set_text_color(100, 100, 100)
            self.cell(0, 5, nd['sig'])
            self.ln(6)
            # Description
            self.set_font('mg', '', 10)
            self.set_text_color(50, 50, 50)
            self.multi_cell(0, 7, nd['desc'])
            self.ln(3)
            # Ports
            inputs = ', '.join([p[1] for p in nd['i']]) if nd['i'] else '없음'
            outputs = ', '.join([p[1] for p in nd['o']]) if nd['o'] else '없음'
            self.set_font('mg', '', 9)
            self.set_text_color(70, 70, 70)
            self.cell(0, 6, f'입력: {inputs}  |  출력: {outputs}')
            self.ln(7)
            # Properties table
            if nd['p']:
                self.ensure_space(15)
                self.set_font('mg', 'B', 9)
                self.set_fill_color(240, 240, 245)
                self.cell(40, 6, '속성', border=1, fill=True)
                self.cell(25, 6, '타입', border=1, fill=True)
                self.cell(25, 6, '기본값', border=1, fill=True)
                self.cell(0, 6, '설명', border=1, fill=True)
                self.ln()
                self.set_font('mg', '', 8)
                for pk, pl, pt, pd, pdesc in nd['p']:
                    self.ensure_space(8)
                    self.cell(40, 5.5, pl, border=1)
                    self.cell(25, 5.5, pt, border=1)
                    self.cell(25, 5.5, str(pd), border=1)
                    self.cell(0, 5.5, pdesc, border=1)
                    self.ln()
            self.ln(8)
            # Separator line
            self.set_draw_color(220, 220, 220)
            self.set_line_width(0.3)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(3)

    def build_part4(self):
        self.add_page()
        self.chapter_title('Part 4: 튜토리얼')
        self.body_text('30개의 실습 튜토리얼을 통해 SamOpenCVWeb의 사용법을 단계별로 배웁니다.')
        self.ln(4)
        for tut in TUTORIALS:
            # Each tutorial starts on a fresh area (new page if not enough space)
            self.ensure_space(90)
            self.chapter_title(f"Tutorial {tut['n']}: {tut['title']}", 1)
            # Goal
            self.set_fill_color(230, 245, 230)
            self.set_font('mg', 'B', 10)
            self.set_text_color(40, 100, 40)
            self.multi_cell(0, 7, '  학습 목표: ' + tut['goal'], fill=True)
            self.ln(5)
            self.set_text_color(50, 50, 50)
            # Pipeline diagram
            self.bold_text('파이프라인 다이어그램:')
            self.add_pipeline_image(tut['pipe'])
            self.ln(2)
            # Steps
            self.bold_text('단계별 설명:')
            self.ln(1)
            for idx, step in enumerate(tut['steps'], 1):
                self.ensure_space(12)
                self.set_font('mg', 'B', 10)
                self.set_text_color(76, 175, 80)
                step_x = self.get_x()
                self.cell(10, 7, str(idx) + '.')
                self.set_font('mg', '', 10)
                self.set_text_color(50, 50, 50)
                self.multi_cell(0, 7, step)
                self.ln(2.5)
            # Extra
            if tut.get('extra'):
                self.ln(3)
                self.ensure_space(14)
                self.set_fill_color(235, 235, 255)
                self.set_font('mg', '', 9)
                self.set_text_color(60, 60, 120)
                self.multi_cell(0, 7, '  [Tip] 더 해보기: ' + tut['extra'], fill=True)
                self.ln(5)
            self.ln(8)
            # Separator
            self.set_draw_color(200, 200, 200)
            self.set_line_width(0.3)
            self.line(self.l_margin + 20, self.get_y(), self.w - self.r_margin - 20, self.get_y())
            self.ln(4)

    def build_part5(self):
        self.add_page()
        self.chapter_title('Part 5: 단축키 & 팁')
        self.chapter_title('키보드 단축키', 1)
        keys = [
            ('Ctrl+Enter', '파이프라인 실행'),
            ('Ctrl+Z', '실행 취소'),
            ('Ctrl+C', '선택 노드 복사'),
            ('Ctrl+X', '선택 노드 잘라내기'),
            ('Ctrl+V', '붙여넣기'),
            ('Ctrl+A', '전체 선택'),
            ('Delete / Backspace', '선택 삭제'),
            ('Ctrl+S', '파이프라인 저장'),
            ('Esc', '선택 해제'),
        ]
        self.set_font('mg', '', 10)
        for k, d in keys:
            self.set_font('mg', 'B', 10)
            self.cell(50, 6, k)
            self.set_font('mg', '', 10)
            self.cell(0, 6, d)
            self.ln(5)
        self.ln(5)
        self.chapter_title('마우스 조작', 1)
        mouse = [
            ('왼쪽 클릭', '노드 선택'),
            ('더블 클릭', '파일 노드: 파일 선택 / 기타: 미리보기 실행'),
            ('드래그 (노드)', '노드 이동'),
            ('드래그 (포트)', '연결선 생성'),
            ('드래그 (빈 영역)', '캔버스 이동'),
            ('Shift+드래그', '다중 선택 영역'),
            ('마우스 휠', '확대/축소'),
        ]
        for k, d in mouse:
            self.set_font('mg', 'B', 10)
            self.cell(50, 6, k)
            self.set_font('mg', '', 10)
            self.cell(0, 6, d)
            self.ln(5)
        self.ln(5)
        self.chapter_title('유용한 팁', 1)
        tips = [
            '노드를 가까이 배치하면 자동으로 연결됩니다.',
            '이미지 크기가 다른 경우 Resize 노드로 맞춰주세요.',
            'Python Script 노드로 커스텀 처리를 구현할 수 있습니다.',
            'Code 버튼으로 생성된 코드를 독립 스크립트로 활용하세요.',
            'Video Read의 loop 모드로 비디오 전체를 한번에 처리합니다.',
            '컨투어 분석 시 이진화를 먼저 수행하면 결과가 좋습니다.',
            'HSV 색상 공간에서 InRange 필터링이 더 효과적입니다.',
        ]
        for tip in tips:
            self.body_text('• ' + tip)


    def build_part6(self):
        self.add_page()
        self.chapter_title('Part 6: 응용 예제 프로젝트')
        self.body_text(f'실전에서 활용할 수 있는 {len(EXAMPLES)}개의 복합 예제 프로젝트입니다. 각 예제는 JSON 저장 파일로 제공되며, Load 기능으로 바로 불러올 수 있습니다.')
        self.ln(2)
        self.set_fill_color(240, 248, 255)
        self.set_font('mg', 'B', 10)
        self.set_text_color(30, 80, 140)
        self.multi_cell(0, 7, '  [안내] examples/ 폴더의 JSON 파일을 Load하면 예제 파이프라인이 자동으로 구성됩니다.', fill=True)
        self.ln(6)
        self.set_text_color(50, 50, 50)
        for ex in EXAMPLES:
            self.ensure_space(95)
            # Example header with number and difficulty
            self.set_draw_color(76, 175, 80)
            self.set_line_width(1.0)
            self.line(self.l_margin, self.get_y(), self.w - self.r_margin, self.get_y())
            self.ln(3)
            self.chapter_title(f"예제 {ex['n']}: {ex['title']}", 1)
            # Difficulty
            self.set_font('mg', '', 9)
            self.set_text_color(150, 100, 30)
            self.cell(0, 6, f"난이도: {ex['difficulty']}    |    저장파일: {ex['file']}")
            self.ln(7)
            # Goal
            self.set_fill_color(230, 245, 230)
            self.set_font('mg', 'B', 10)
            self.set_text_color(40, 100, 40)
            self.multi_cell(0, 7, '  학습 목표: ' + ex['goal'], fill=True)
            self.ln(4)
            self.set_text_color(50, 50, 50)
            # Concepts
            self.set_font('mg', 'B', 9)
            self.set_text_color(80, 80, 120)
            concepts_str = '  |  '.join(ex['concepts'])
            self.multi_cell(0, 6, '핵심 개념: ' + concepts_str)
            self.ln(3)
            self.set_text_color(50, 50, 50)
            # Node graph diagram from JSON (with image previews)
            json_path = os.path.join(BASE_DIR, 'examples', ex['file'])
            self.bold_text('노드 그래프 (각 노드에 처리 결과 미리보기 포함):')
            self.add_graph_image(json_path, height_mm=55)
            self.ln(1)
            # Steps
            self.bold_text('단계별 구성 방법:')
            self.ln(1)
            for idx, step in enumerate(ex['steps'], 1):
                self.ensure_space(12)
                self.set_font('mg', 'B', 10)
                self.set_text_color(76, 175, 80)
                self.cell(10, 7, str(idx) + '.')
                self.set_font('mg', '', 10)
                self.set_text_color(50, 50, 50)
                self.multi_cell(0, 7, step)
                self.ln(2)
            # Extra tips
            if ex.get('extra'):
                self.ln(3)
                self.ensure_space(14)
                self.set_fill_color(235, 235, 255)
                self.set_font('mg', '', 9)
                self.set_text_color(60, 60, 120)
                self.multi_cell(0, 7, '  [Tip] ' + ex['extra'], fill=True)
                self.ln(5)
            self.ln(10)


def main():
    n_examples = len(EXAMPLES)
    print('SamOpenCVWeb 교재 PDF 생성 시작...')
    pdf = SamTextbook()
    print('  [1/7] 표지 생성...')
    pdf.build_title_page()
    print('  [2/7] Part 1: 소개...')
    pdf.build_part1()
    print('  [3/7] Part 2: 기본 사용법...')
    pdf.build_part2()
    print('  [4/7] Part 3: 노드 레퍼런스 (83개)...')
    pdf.build_part3()
    print('  [5/7] Part 4: 튜토리얼 (30개)...')
    pdf.build_part4()
    print(f'  [6/7] Part 5: 단축키 & 팁...')
    pdf.build_part5()
    print(f'  [7/7] Part 6: 응용 예제 ({n_examples}개)...')
    pdf.build_part6()
    pdf.output(OUTPUT_PATH)
    print(f'\nPDF 생성 완료: {OUTPUT_PATH}')
    print(f'총 {pdf.page_no()} 페이지')
    # Cleanup any leftover temp files
    for f in os.listdir(BASE_DIR):
        if f.startswith('_tmp_diagram_') and f.endswith('.png'):
            try:
                os.remove(os.path.join(BASE_DIR, f))
            except:
                pass


if __name__ == '__main__':
    main()
