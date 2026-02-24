"""
NodeOpenCV - Node-RED style visual programming for OpenCV
Flask backend for image processing pipeline execution
Multi-user session support for classroom environments
"""
import os
import uuid
import base64
import traceback
import json
import time
import threading
import shutil
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory, render_template, Response, stream_with_context, g, make_response

app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SESSIONS_DIR = os.path.join(BASE_DIR, 'sessions')
os.makedirs(SESSIONS_DIR, exist_ok=True)


# ---- Session Management for Multi-User Support ----

class SessionData:
    """Per-user session data with isolated storage."""
    def __init__(self, sid):
        self.sid = sid
        self.image_store = {}
        self.script_sessions = {}
        self.video_loop_stop = False
        self.created_at = time.time()
        self.last_access = time.time()
        self.lock = threading.Lock()
        # Session-specific directories
        self.upload_folder = os.path.join(SESSIONS_DIR, sid, 'uploads')
        self.work_folder = os.path.join(SESSIONS_DIR, sid, 'work')
        self.script_folder = os.path.join(SESSIONS_DIR, sid, 'scripts')
        os.makedirs(self.upload_folder, exist_ok=True)
        os.makedirs(self.work_folder, exist_ok=True)
        os.makedirs(self.script_folder, exist_ok=True)

    def touch(self):
        self.last_access = time.time()


_sessions = {}
_sessions_lock = threading.Lock()
SESSION_TTL = 3600 * 4  # 4 hours


def get_session():
    """Get or create session for current request."""
    sid = request.headers.get('X-Session-ID', '')
    if not sid:
        sid = request.cookies.get('session_id', '')
    if not sid:
        sid = uuid.uuid4().hex[:12]
    with _sessions_lock:
        if sid not in _sessions:
            _sessions[sid] = SessionData(sid)
        session = _sessions[sid]
    session.touch()
    return session


def cleanup_expired_sessions():
    """Remove sessions older than SESSION_TTL."""
    now = time.time()
    with _sessions_lock:
        expired = [sid for sid, s in _sessions.items() if now - s.last_access > SESSION_TTL]
        for sid in expired:
            session = _sessions.pop(sid)
            session_dir = os.path.join(SESSIONS_DIR, sid)
            if os.path.exists(session_dir):
                try:
                    shutil.rmtree(session_dir)
                except Exception:
                    pass


def _cleanup_loop():
    """Background thread: cleanup expired sessions every 30 minutes."""
    while True:
        time.sleep(1800)
        try:
            cleanup_expired_sessions()
        except Exception:
            pass


_cleanup_thread = threading.Thread(target=_cleanup_loop, daemon=True)
_cleanup_thread.start()


@app.before_request
def before_request():
    g.session = get_session()


@app.after_request
def after_request(response):
    if hasattr(g, 'session'):
        response.set_cookie('session_id', g.session.sid, max_age=SESSION_TTL, httponly=True, samesite='Lax')
    return response


def encode_image_jpeg(img, quality=80):
    """Encode OpenCV image to base64 JPEG for fast preview."""
    if img is None:
        return None
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    success, buffer = cv2.imencode('.jpg', img, params)
    if not success:
        return None
    return 'data:image/jpeg;base64,' + base64.b64encode(buffer).decode('utf-8')


def encode_image_png(img):
    """Encode OpenCV image to base64 PNG for full quality."""
    if img is None:
        return None
    success, buffer = cv2.imencode('.png', img)
    if not success:
        return None
    return 'data:image/png;base64,' + base64.b64encode(buffer).decode('utf-8')


def decode_image(data_url):
    """Decode base64 data URL to OpenCV image."""
    if not data_url:
        return None
    if ',' in data_url:
        data_url = data_url.split(',')[1]
    img_bytes = base64.b64decode(data_url)
    nparr = np.frombuffer(img_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)


def store_image(img):
    """Store image in current session and return its ID."""
    img_id = str(uuid.uuid4())[:8]
    g.session.image_store[img_id] = img
    return img_id


# ---- Node processing functions ----

def process_image_read(node, inputs):
    """Read image from server store or file path."""
    props = node.get('properties', {})
    # First try server-side stored image
    img_id = props.get('imageId', '')
    if img_id and img_id in g.session.image_store:
        return {'image': g.session.image_store[img_id].copy()}
    # Try file path
    filepath = props.get('filepath', '')
    flags_str = props.get('flags', 'IMREAD_COLOR')
    flags = getattr(cv2, flags_str, cv2.IMREAD_COLOR)
    if filepath and os.path.exists(filepath):
        img = cv2.imread(filepath, flags)
        if img is not None and len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return {'image': img}
    # Try inline base64 (fallback for compat)
    image_data = props.get('imageData', '')
    if image_data:
        img = decode_image(image_data)
        if img is not None:
            # Store it for future use
            new_id = store_image(img)
            return {'image': img}
    return {'image': None, 'error': 'No image loaded. Upload a file first.'}


def process_image_show(node, inputs):
    """Pass through image (display handled on frontend). Also outputs image for chaining."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image connected'}
    return {'image': img.copy()}


def process_cvt_color(node, inputs):
    """Convert color space."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    code_str = node.get('properties', {}).get('code', 'COLOR_BGR2GRAY')
    code = getattr(cv2, code_str, None)
    if code is None:
        return {'image': img, 'error': f'Unknown color code: {code_str}'}
    try:
        result = cv2.cvtColor(img, code)
    except cv2.error as e:
        return {'image': img, 'error': f'cvtColor failed: {e}'}
    if len(result.shape) == 2:
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return {'image': result}


def process_gaussian_blur(node, inputs):
    """Apply Gaussian blur."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    ksize = int(props.get('ksize', 5))
    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1
    sigma_x = float(props.get('sigmaX', 0))
    result = cv2.GaussianBlur(img, (ksize, ksize), sigma_x)
    return {'image': result}


def process_median_blur(node, inputs):
    """Apply median blur."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    ksize = int(node.get('properties', {}).get('ksize', 5))
    if ksize < 3:
        ksize = 3
    if ksize % 2 == 0:
        ksize += 1
    result = cv2.medianBlur(img, ksize)
    return {'image': result}


def process_bilateral_filter(node, inputs):
    """Apply bilateral filter."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    d = int(props.get('d', 9))
    sigma_color = float(props.get('sigmaColor', 75))
    sigma_space = float(props.get('sigmaSpace', 75))
    result = cv2.bilateralFilter(img, d, sigma_color, sigma_space)
    return {'image': result}


def process_canny(node, inputs):
    """Canny edge detection."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    threshold1 = float(props.get('threshold1', 100))
    threshold2 = float(props.get('threshold2', 200))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    result = cv2.Canny(gray, threshold1, threshold2)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return {'image': result}


def process_threshold(node, inputs):
    """Apply threshold."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    thresh_val = float(props.get('thresh', 127))
    max_val = float(props.get('maxval', 255))
    type_str = props.get('type', 'THRESH_BINARY')
    thresh_type = getattr(cv2, type_str, cv2.THRESH_BINARY)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    _, result = cv2.threshold(gray, thresh_val, max_val, thresh_type)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return {'image': result}


def process_resize(node, inputs):
    """Resize image."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    width = int(props.get('width', 0))
    height = int(props.get('height', 0))
    fx = float(props.get('fx', 0.5))
    fy = float(props.get('fy', 0.5))
    interp_str = props.get('interpolation', 'INTER_LINEAR')
    interp = getattr(cv2, interp_str, cv2.INTER_LINEAR)
    if width > 0 and height > 0:
        result = cv2.resize(img, (width, height), interpolation=interp)
    else:
        if fx <= 0:
            fx = 0.5
        if fy <= 0:
            fy = 0.5
        result = cv2.resize(img, None, fx=fx, fy=fy, interpolation=interp)
    return {'image': result}


def process_rotate(node, inputs):
    """Rotate image."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    angle = float(props.get('angle', 90))
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    cos_a = abs(M[0, 0])
    sin_a = abs(M[0, 1])
    new_w = int(h * sin_a + w * cos_a)
    new_h = int(h * cos_a + w * sin_a)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2
    result = cv2.warpAffine(img, M, (new_w, new_h))
    return {'image': result}


def process_morphology(node, inputs):
    """Morphological operations."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    op_str = props.get('operation', 'MORPH_OPEN')
    ksize = int(props.get('ksize', 5))
    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1
    iterations = max(1, int(props.get('iterations', 1)))
    shape_str = props.get('shape', 'MORPH_RECT')
    shape = getattr(cv2, shape_str, cv2.MORPH_RECT)
    kernel = cv2.getStructuringElement(shape, (ksize, ksize))
    op_map = {
        'MORPH_ERODE': cv2.MORPH_ERODE,
        'MORPH_DILATE': cv2.MORPH_DILATE,
        'MORPH_OPEN': cv2.MORPH_OPEN,
        'MORPH_CLOSE': cv2.MORPH_CLOSE,
        'MORPH_GRADIENT': cv2.MORPH_GRADIENT,
        'MORPH_TOPHAT': cv2.MORPH_TOPHAT,
        'MORPH_BLACKHAT': cv2.MORPH_BLACKHAT,
    }
    op = op_map.get(op_str, cv2.MORPH_OPEN)
    result = cv2.morphologyEx(img, op, kernel, iterations=iterations)
    return {'image': result}


def process_dilate(node, inputs):
    """Dilate operation."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    ksize = int(props.get('ksize', 5))
    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1
    iterations = max(1, int(props.get('iterations', 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    result = cv2.dilate(img, kernel, iterations=iterations)
    return {'image': result}


def process_erode(node, inputs):
    """Erode operation."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    ksize = int(props.get('ksize', 5))
    if ksize < 1:
        ksize = 1
    if ksize % 2 == 0:
        ksize += 1
    iterations = max(1, int(props.get('iterations', 1)))
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
    result = cv2.erode(img, kernel, iterations=iterations)
    return {'image': result}


def process_sobel(node, inputs):
    """Sobel edge detection."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    dx = int(props.get('dx', 1))
    dy = int(props.get('dy', 0))
    ksize = int(props.get('ksize', 3))
    if ksize % 2 == 0:
        ksize += 1
    if dx == 0 and dy == 0:
        dx = 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    result = cv2.Sobel(gray, cv2.CV_64F, dx, dy, ksize=ksize)
    result = cv2.convertScaleAbs(result)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return {'image': result}


def process_laplacian(node, inputs):
    """Laplacian edge detection."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    ksize = int(node.get('properties', {}).get('ksize', 3))
    if ksize % 2 == 0:
        ksize += 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    result = cv2.Laplacian(gray, cv2.CV_64F, ksize=ksize)
    result = cv2.convertScaleAbs(result)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return {'image': result}


def process_find_contours(node, inputs):
    """Find contours and output them as data for drawing/analysis nodes."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    mode_str = props.get('mode', 'RETR_EXTERNAL')
    method_str = props.get('method', 'CHAIN_APPROX_SIMPLE')
    mode = getattr(cv2, mode_str, cv2.RETR_EXTERNAL)
    method = getattr(cv2, method_str, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    contours, hierarchy = cv2.findContours(gray, mode, method)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)
    return {'image': result, 'contours': contours, 'info': f'Found {len(contours)} contours'}


def process_hough_lines(node, inputs):
    """Hough line detection."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    rho = float(props.get('rho', 1))
    theta_div = float(props.get('theta_divisor', 180))
    threshold = int(props.get('threshold', 100))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLines(edges, rho, np.pi / theta_div, threshold)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    line_count = 0
    if lines is not None:
        line_count = len(lines)
        for line in lines:
            r, t = line[0]
            a, b = np.cos(t), np.sin(t)
            x0, y0 = a * r, b * r
            pt1 = (int(x0 + 1000 * (-b)), int(y0 + 1000 * a))
            pt2 = (int(x0 - 1000 * (-b)), int(y0 - 1000 * a))
            cv2.line(result, pt1, pt2, (0, 0, 255), 2)
    return {'image': result, 'info': f'Found {line_count} lines'}


def process_adaptive_threshold(node, inputs):
    """Adaptive thresholding."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    max_val = float(props.get('maxval', 255))
    method_str = props.get('adaptiveMethod', 'ADAPTIVE_THRESH_GAUSSIAN_C')
    type_str = props.get('thresholdType', 'THRESH_BINARY')
    block_size = int(props.get('blockSize', 11))
    c = float(props.get('C', 2))
    method = getattr(cv2, method_str, cv2.ADAPTIVE_THRESH_GAUSSIAN_C)
    thresh_type = getattr(cv2, type_str, cv2.THRESH_BINARY)
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    result = cv2.adaptiveThreshold(gray, max_val, method, thresh_type, block_size, c)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return {'image': result}


def process_python_script(node, inputs):
    """Execute custom Python/OpenCV script."""
    img = inputs.get('image')
    props = node.get('properties', {})
    script = props.get('script', '')
    if not script.strip():
        return {'image': img}
    local_vars = {'img_input': img, 'cv2': cv2, 'np': np, 'img_output': None}
    try:
        exec(script, {'__builtins__': __builtins__}, local_vars)
        result = local_vars.get('img_output')
        if result is None:
            return {'image': img, 'error': 'img_output not set in script'}
        return {'image': result}
    except Exception as e:
        return {'image': img, 'error': f'Script error: {e}'}


def _prepare_mask(mask, target_shape):
    """Helper: prepare mask for bitwise operations (ensure 8-bit single channel, matching size)."""
    if mask is None:
        return None
    if len(mask.shape) == 3:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    h, w = target_shape[:2]
    if mask.shape[0] != h or mask.shape[1] != w:
        mask = cv2.resize(mask, (w, h))
    return mask


def process_bitwise_and(node, inputs):
    """Bitwise AND of two images with optional mask."""
    img = inputs.get('image')
    img2 = inputs.get('image2')
    if img is None:
        return {'image': None, 'error': 'No input image on port 1'}
    if img2 is None:
        return {'image': img, 'error': 'No input image on port 2, passing through port 1'}
    if img.shape != img2.shape:
        img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    mask = _prepare_mask(inputs.get('mask'), img.shape)
    result = cv2.bitwise_and(img, img2, mask=mask)
    return {'image': result}


def process_bitwise_or(node, inputs):
    """Bitwise OR of two images with optional mask."""
    img = inputs.get('image')
    img2 = inputs.get('image2')
    if img is None:
        return {'image': None, 'error': 'No input image on port 1'}
    if img2 is None:
        return {'image': img, 'error': 'No input image on port 2, passing through port 1'}
    if img.shape != img2.shape:
        img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    mask = _prepare_mask(inputs.get('mask'), img.shape)
    result = cv2.bitwise_or(img, img2, mask=mask)
    return {'image': result}


def process_bitwise_not(node, inputs):
    """Bitwise NOT with optional mask."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    mask = _prepare_mask(inputs.get('mask'), img.shape)
    result = cv2.bitwise_not(img, mask=mask)
    return {'image': result}


def process_add_weighted(node, inputs):
    """Blend two images."""
    img = inputs.get('image')
    img2 = inputs.get('image2')
    if img is None:
        return {'image': None, 'error': 'No input image on port 1'}
    if img2 is None:
        return {'image': img, 'error': 'No input image on port 2, passing through port 1'}
    props = node.get('properties', {})
    alpha = float(props.get('alpha', 0.5))
    beta = float(props.get('beta', 0.5))
    gamma = float(props.get('gamma', 0))
    if img.shape != img2.shape:
        img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    result = cv2.addWeighted(img, alpha, img2, beta, gamma)
    return {'image': result}


def process_histogram_eq(node, inputs):
    """Histogram equalization."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    use_clahe = props.get('useCLAHE', False)
    if len(img.shape) == 3:
        ycrcb = cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
        if use_clahe:
            clip_limit = float(props.get('clipLimit', 2.0))
            tile_size = int(props.get('tileGridSize', 8))
            if tile_size < 1:
                tile_size = 1
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            ycrcb[:, :, 0] = clahe.apply(ycrcb[:, :, 0])
        else:
            ycrcb[:, :, 0] = cv2.equalizeHist(ycrcb[:, :, 0])
        result = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
    else:
        if use_clahe:
            clip_limit = float(props.get('clipLimit', 2.0))
            tile_size = int(props.get('tileGridSize', 8))
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
            result = clahe.apply(img)
        else:
            result = cv2.equalizeHist(img)
        result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return {'image': result}


def process_in_range(node, inputs):
    """Color range thresholding (inRange)."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    lower = [int(props.get('lowerB', 0)), int(props.get('lowerG', 0)), int(props.get('lowerR', 0))]
    upper = [int(props.get('upperB', 255)), int(props.get('upperG', 255)), int(props.get('upperR', 255))]
    result = cv2.inRange(img, np.array(lower), np.array(upper))
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return {'image': result}


# ---- Control Flow Nodes ----

def process_control_if(node, inputs):
    """Conditional branching: route image to true or false port."""
    img = inputs.get('image')
    if img is None:
        return {'true': None, 'false': None, 'error': 'No input image'}
    props = node.get('properties', {})
    cond_type = props.get('condition', 'not_empty')
    value = float(props.get('value', 100))
    result = False
    try:
        if cond_type == 'not_empty':
            result = img is not None and img.size > 0
        elif cond_type == 'is_color':
            result = len(img.shape) == 3 and img.shape[2] >= 3
        elif cond_type == 'is_grayscale':
            result = len(img.shape) == 2 or (len(img.shape) == 3 and img.shape[2] == 1)
        elif cond_type == 'width_gt':
            result = img.shape[1] > value
        elif cond_type == 'height_gt':
            result = img.shape[0] > value
        elif cond_type == 'mean_gt':
            result = float(np.mean(img)) > value
        elif cond_type == 'custom':
            expr = props.get('customExpr', 'True')
            result = bool(eval(expr, {'__builtins__': {}}, {'img': img, 'cv2': cv2, 'np': np}))
    except Exception as e:
        return {'true': img, 'false': None, 'error': f'Condition error: {e}'}
    if result:
        return {'true': img.copy(), 'false': None, 'info': 'Condition: True'}
    else:
        return {'true': None, 'false': img.copy(), 'info': 'Condition: False'}


def process_control_for(node, inputs):
    """For loop: apply operation N times."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    iterations = max(1, min(100, int(props.get('iterations', 3))))
    operation = props.get('operation', 'gaussian_blur')
    ksize = int(props.get('ksize', 3))
    if ksize < 1: ksize = 1
    if ksize % 2 == 0: ksize += 1
    result = img.copy()
    try:
        for i in range(iterations):
            if operation == 'gaussian_blur':
                result = cv2.GaussianBlur(result, (ksize, ksize), 0)
            elif operation == 'median_blur':
                result = cv2.medianBlur(result, ksize if ksize >= 3 else 3)
            elif operation == 'dilate':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                result = cv2.dilate(result, kernel)
            elif operation == 'erode':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                result = cv2.erode(result, kernel)
            elif operation == 'sharpen':
                blurred = cv2.GaussianBlur(result, (ksize, ksize), 0)
                result = cv2.addWeighted(result, 1.5, blurred, -0.5, 0)
            elif operation == 'custom':
                code = props.get('customCode', '')
                local_vars = {'img': result, 'i': i, 'cv2': cv2, 'np': np}
                exec(code, {'__builtins__': __builtins__}, local_vars)
                result = local_vars.get('img', result)
    except Exception as e:
        return {'image': result, 'error': f'For loop error at iteration {i}: {e}'}
    return {'image': result, 'info': f'Completed {iterations} iterations'}


def process_control_while(node, inputs):
    """While loop: repeat operation while condition holds."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    cond_type = props.get('condition', 'mean_gt')
    value = float(props.get('value', 128))
    operation = props.get('operation', 'gaussian_blur')
    ksize = int(props.get('ksize', 3))
    if ksize < 1: ksize = 1
    if ksize % 2 == 0: ksize += 1
    max_iter = max(1, min(500, int(props.get('maxIter', 50))))
    result = img.copy()
    count = 0
    try:
        while count < max_iter:
            # Evaluate condition
            cond_met = False
            if cond_type == 'mean_gt':
                cond_met = float(np.mean(result)) > value
            elif cond_type == 'mean_lt':
                cond_met = float(np.mean(result)) < value
            elif cond_type == 'std_gt':
                cond_met = float(np.std(result)) > value
            elif cond_type == 'nonzero_gt':
                cond_met = int(np.count_nonzero(result)) > value
            elif cond_type == 'custom':
                expr = props.get('customCond', 'False')
                cond_met = bool(eval(expr, {'__builtins__': {}}, {'img': result, 'cv2': cv2, 'np': np, 'i': count}))
            if not cond_met:
                break
            # Apply operation
            if operation == 'gaussian_blur':
                result = cv2.GaussianBlur(result, (ksize, ksize), 0)
            elif operation == 'median_blur':
                result = cv2.medianBlur(result, ksize if ksize >= 3 else 3)
            elif operation == 'erode':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                result = cv2.erode(result, kernel)
            elif operation == 'dilate':
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (ksize, ksize))
                result = cv2.dilate(result, kernel)
            elif operation == 'threshold_step':
                gray = cv2.cvtColor(result, cv2.COLOR_BGR2GRAY) if len(result.shape) == 3 else result
                _, result = cv2.threshold(gray, int(value), 255, cv2.THRESH_BINARY)
                result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
            elif operation == 'custom':
                code = props.get('customCode', '')
                local_vars = {'img': result, 'i': count, 'cv2': cv2, 'np': np}
                exec(code, {'__builtins__': __builtins__}, local_vars)
                result = local_vars.get('img', result)
            count += 1
    except Exception as e:
        return {'image': result, 'error': f'While loop error at iteration {count}: {e}'}
    return {'image': result, 'info': f'Stopped after {count} iteration(s)'}


def process_control_switch(node, inputs):
    """Switch-case: route image to one of 3 output ports."""
    img = inputs.get('image')
    if img is None:
        return {'case0': None, 'case1': None, 'case2': None, 'error': 'No input image'}
    props = node.get('properties', {})
    switch_on = props.get('switchOn', 'channels')
    case_idx = 0
    try:
        if switch_on == 'channels':
            ch = img.shape[2] if len(img.shape) == 3 else 1
            case_idx = 0 if ch == 1 else (1 if ch == 3 else 2)
        elif switch_on == 'depth':
            if img.dtype == np.uint8:
                case_idx = 0
            elif img.dtype in (np.float32, np.float64):
                case_idx = 1
            else:
                case_idx = 2
        elif switch_on == 'size_class':
            pixels = img.shape[0] * img.shape[1]
            case_idx = 0 if pixels < 100000 else (1 if pixels < 1000000 else 2)
        elif switch_on == 'mean_range':
            m = float(np.mean(img))
            case_idx = 0 if m < 85 else (1 if m < 170 else 2)
        elif switch_on == 'custom':
            expr = props.get('customExpr', '0')
            case_idx = int(eval(expr, {'__builtins__': {}}, {'img': img, 'cv2': cv2, 'np': np}))
            case_idx = max(0, min(2, case_idx))
    except Exception as e:
        return {'case0': img, 'case1': None, 'case2': None, 'error': f'Switch error: {e}'}
    out = {'case0': None, 'case1': None, 'case2': None, 'info': f'Matched case {case_idx}'}
    out[f'case{case_idx}'] = img.copy()
    return out


# ---- IO Nodes ----

def process_image_write(node, inputs):
    """Save image to session work folder and provide download."""
    img = inputs.get('image')
    if img is None:
        return {'error': 'No input image'}
    props = node.get('properties', {})
    filename = props.get('filepath', 'output.png')
    # Sanitize: use only the filename, save into session work folder
    filename = os.path.basename(filename) if filename else 'output.png'
    fmt = props.get('format', 'png').lower()
    quality = int(props.get('quality', 95))
    try:
        filepath = os.path.join(g.session.work_folder, filename)
        if fmt in ('jpg', 'jpeg'):
            params = [cv2.IMWRITE_JPEG_QUALITY, quality]
        elif fmt == 'png':
            params = [cv2.IMWRITE_PNG_COMPRESSION, max(0, min(9, 9 - quality // 11))]
        else:
            params = []
        cv2.imwrite(filepath, img, params)
        return {'image': img, 'info': f'Saved: {filename}', 'downloadFile': filename}
    except Exception as e:
        return {'error': f'Failed to save: {e}'}


def process_video_write(node, inputs):
    """Write video output. Actual writing happens in execute_video_loop; single execution just passes through."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    filepath = props.get('filepath', 'output.mp4')
    return {'image': img, 'info': f'Video Write → {filepath} (use Video Loop to record)'}


def process_video_read(node, inputs):
    """Read a frame from video file."""
    props = node.get('properties', {})
    filepath = props.get('filepath', '')
    frame_index = int(props.get('frameIndex', 0))
    if not filepath:
        return {'image': None, 'error': 'No video filepath specified'}
    try:
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return {'image': None, 'error': f'Cannot open video: {filepath}'}
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {'image': None, 'error': f'Cannot read frame {frame_index}'}
        return {'image': frame, 'info': f'Frame {frame_index}'}
    except Exception as e:
        return {'image': None, 'error': f'Video read error: {e}'}


def process_camera_capture(node, inputs):
    """Capture a frame from camera."""
    props = node.get('properties', {})
    camera_index = int(props.get('cameraIndex', 0))
    try:
        cap = cv2.VideoCapture(camera_index)
        if not cap.isOpened():
            return {'image': None, 'error': f'Cannot open camera {camera_index}'}
        ret, frame = cap.read()
        cap.release()
        if not ret:
            return {'image': None, 'error': 'Cannot capture frame from camera'}
        return {'image': frame}
    except Exception as e:
        return {'image': None, 'error': f'Camera capture error: {e}'}


# ---- Color Nodes ----

def process_split_channels(node, inputs):
    """Split image into channels."""
    img = inputs.get('image')
    if img is None:
        return {'ch0': None, 'ch1': None, 'ch2': None, 'error': 'No input image'}
    if len(img.shape) == 2:
        # Grayscale: all channels same
        gray_bgr = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return {'ch0': gray_bgr, 'ch1': gray_bgr, 'ch2': gray_bgr}
    channels = cv2.split(img)
    # Convert single-channel to BGR for display
    ch0 = cv2.cvtColor(channels[0], cv2.COLOR_GRAY2BGR) if len(channels) > 0 else None
    ch1 = cv2.cvtColor(channels[1], cv2.COLOR_GRAY2BGR) if len(channels) > 1 else None
    ch2 = cv2.cvtColor(channels[2], cv2.COLOR_GRAY2BGR) if len(channels) > 2 else None
    return {'ch0': ch0, 'ch1': ch1, 'ch2': ch2, 'info': f'Split into {len(channels)} channels'}


def process_merge_channels(node, inputs):
    """Merge channels into one image."""
    ch0 = inputs.get('ch0')
    ch1 = inputs.get('ch1')
    ch2 = inputs.get('ch2')
    if ch0 is None and ch1 is None and ch2 is None:
        return {'image': None, 'error': 'No channel inputs connected'}
    # Convert BGR inputs to single channel (take first channel)
    def to_gray(ch):
        if ch is None:
            return None
        if len(ch.shape) == 3:
            return cv2.cvtColor(ch, cv2.COLOR_BGR2GRAY)
        return ch
    g0 = to_gray(ch0)
    g1 = to_gray(ch1)
    g2 = to_gray(ch2)
    # Use first available as reference for shape
    ref = g0 if g0 is not None else (g1 if g1 is not None else g2)
    h, w = ref.shape[:2]
    if g0 is None:
        g0 = np.zeros((h, w), dtype=np.uint8)
    if g1 is None:
        g1 = np.zeros((h, w), dtype=np.uint8)
    if g2 is None:
        g2 = np.zeros((h, w), dtype=np.uint8)
    # Resize to match
    g1 = cv2.resize(g1, (w, h)) if g1.shape[:2] != (h, w) else g1
    g2 = cv2.resize(g2, (w, h)) if g2.shape[:2] != (h, w) else g2
    merged = cv2.merge([g0, g1, g2])
    return {'image': merged}


# ---- Filter Nodes ----

def process_box_filter(node, inputs):
    """Apply box filter."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    kwidth = int(props.get('kwidth', 5))
    kheight = int(props.get('kheight', 5))
    normalize = props.get('normalize', True)
    if isinstance(normalize, str):
        normalize = normalize.lower() in ('true', '1', 'yes')
    if kwidth < 1:
        kwidth = 1
    if kheight < 1:
        kheight = 1
    result = cv2.boxFilter(img, -1, (kwidth, kheight), normalize=normalize)
    return {'image': result}


def process_sharpen(node, inputs):
    """Unsharp mask sharpening."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    strength = float(props.get('strength', 1.0))
    blurred = cv2.GaussianBlur(img, (5, 5), 0)
    result = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
    return {'image': result}


def process_filter2d(node, inputs):
    """Apply custom 2D filter with preset kernels."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    ksize = int(props.get('kernelSize', 3))
    preset = props.get('preset', 'sharpen')
    if preset == 'sharpen':
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    elif preset == 'edge_detect':
        kernel = np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]], dtype=np.float32)
    elif preset == 'emboss':
        kernel = np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]], dtype=np.float32)
    elif preset == 'ridge':
        kernel = np.array([[-1, -1, -1], [-1, 4, -1], [-1, -1, -1]], dtype=np.float32)
    elif preset == 'blur':
        kernel = np.ones((ksize, ksize), dtype=np.float32) / (ksize * ksize)
    else:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    result = cv2.filter2D(img, -1, kernel)
    return {'image': result}


# ---- Edge Nodes ----

def process_scharr(node, inputs):
    """Scharr edge detection."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    dx = int(props.get('dx', 1))
    dy = int(props.get('dy', 0))
    if dx == 0 and dy == 0:
        dx = 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    result = cv2.Scharr(gray, cv2.CV_64F, dx, dy)
    result = cv2.convertScaleAbs(result)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return {'image': result}


# ---- Threshold Nodes ----

def process_otsu_threshold(node, inputs):
    """Otsu's automatic thresholding."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    maxval = float(props.get('maxval', 255))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    thresh_val, result = cv2.threshold(gray, 0, maxval, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    result = cv2.cvtColor(result, cv2.COLOR_GRAY2BGR)
    return {'image': result, 'info': f'Otsu threshold: {thresh_val:.1f}'}


# ---- Morphology Nodes ----

def process_structuring_element(node, inputs):
    """Create a structuring element."""
    props = node.get('properties', {})
    shape_str = props.get('shape', 'MORPH_RECT')
    width = int(props.get('width', 5))
    height = int(props.get('height', 5))
    if width < 1:
        width = 1
    if height < 1:
        height = 1
    shape = getattr(cv2, shape_str, cv2.MORPH_RECT)
    element = cv2.getStructuringElement(shape, (width, height))
    # Scale to visible image for preview
    display = (element * 255).astype(np.uint8)
    display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)
    # Scale up for visibility
    scale = max(1, 200 // max(width, height))
    display = cv2.resize(display, (width * scale, height * scale), interpolation=cv2.INTER_NEAREST)
    return {'image': display, 'info': f'{shape_str} ({width}x{height})'}


# ---- Contour Nodes ----

def _get_contours(img, props):
    """Helper: find contours from image."""
    mode_str = props.get('mode', 'RETR_EXTERNAL')
    method_str = props.get('method', 'CHAIN_APPROX_SIMPLE')
    mode = getattr(cv2, mode_str, cv2.RETR_EXTERNAL)
    method = getattr(cv2, method_str, cv2.CHAIN_APPROX_SIMPLE)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    contours, hierarchy = cv2.findContours(gray, mode, method)
    return contours, hierarchy


def _get_contour_color(props):
    """Helper: get drawing color from properties."""
    r = int(props.get('colorR', 0))
    g = int(props.get('colorG', 255))
    b = int(props.get('colorB', 0))
    return (b, g, r)


def process_draw_contours(node, inputs):
    """Draw contours from Find Contours node."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    contours = inputs.get('contours')
    if contours is None:
        return {'image': img, 'error': 'No contours input. Connect Find Contours node.'}
    props = node.get('properties', {})
    contour_idx = int(props.get('contourIdx', -1))
    thickness = int(props.get('thickness', 2))
    color = _get_contour_color(props)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.drawContours(result, contours, contour_idx, color, thickness)
    return {'image': result, 'info': f'Drew {len(contours)} contours'}


def process_bounding_rect(node, inputs):
    """Draw bounding rectangles around contours."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    contours = inputs.get('contours')
    if contours is None:
        return {'image': img, 'error': 'No contours input. Connect Find Contours node.'}
    props = node.get('properties', {})
    thickness = int(props.get('thickness', 2))
    color = _get_contour_color(props)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return {'image': result, 'info': f'{len(contours)} bounding rects'}


def process_min_enclosing_circle(node, inputs):
    """Draw minimum enclosing circles around contours."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    contours = inputs.get('contours')
    if contours is None:
        return {'image': img, 'error': 'No contours input. Connect Find Contours node.'}
    props = node.get('properties', {})
    thickness = int(props.get('thickness', 2))
    color = _get_contour_color(props)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        if len(cnt) >= 5:
            (cx, cy), radius = cv2.minEnclosingCircle(cnt)
            cv2.circle(result, (int(cx), int(cy)), int(radius), color, thickness)
    return {'image': result, 'info': f'{len(contours)} enclosing circles'}


def process_convex_hull(node, inputs):
    """Draw convex hulls around contours."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    contours = inputs.get('contours')
    if contours is None:
        return {'image': img, 'error': 'No contours input. Connect Find Contours node.'}
    props = node.get('properties', {})
    thickness = int(props.get('thickness', 2))
    color = _get_contour_color(props)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    hulls = [cv2.convexHull(cnt) for cnt in contours]
    cv2.drawContours(result, hulls, -1, color, thickness)
    return {'image': result, 'info': f'{len(hulls)} convex hulls'}


def process_approx_poly(node, inputs):
    """Approximate contour shapes to polygons and draw them."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    contours = inputs.get('contours')
    if contours is None:
        return {'image': img, 'error': 'No contours input. Connect Find Contours node.'}
    props = node.get('properties', {})
    epsilon_pct = float(props.get('epsilon', 0.02))
    closed = props.get('closed', True)
    if isinstance(closed, str):
        closed = closed.lower() in ('true', '1', 'yes')
    thickness = int(props.get('thickness', 2))
    color = _get_contour_color(props)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    approxes = []
    for cnt in contours:
        eps = epsilon_pct * cv2.arcLength(cnt, True)
        approx = cv2.approxPolyDP(cnt, eps, closed)
        approxes.append(approx)
    cv2.drawContours(result, approxes, -1, color, thickness)
    return {'image': result, 'info': f'{len(approxes)} polygons'}


def process_contour_area(node, inputs):
    """Filter contours by area range and draw matching ones."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    contours = inputs.get('contours')
    if contours is None:
        return {'image': img, 'error': 'No contours input. Connect Find Contours node.'}
    props = node.get('properties', {})
    min_area = float(props.get('minArea', 0))
    max_area = float(props.get('maxArea', 1e9))
    thickness = int(props.get('thickness', 2))
    color = _get_contour_color(props)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    filtered = [cnt for cnt in contours if min_area <= cv2.contourArea(cnt) <= max_area]
    cv2.drawContours(result, filtered, -1, color, thickness)
    return {'image': result, 'info': f'{len(filtered)}/{len(contours)} contours in area range'}


def process_contour_properties(node, inputs):
    """Annotate contours with area, perimeter, and center info."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    contours = inputs.get('contours')
    if contours is None:
        return {'image': img, 'error': 'No contours input. Connect Find Contours node.'}
    props = node.get('properties', {})
    show_area = props.get('showArea', True)
    show_perimeter = props.get('showPerimeter', True)
    show_center = props.get('showCenter', True)
    if isinstance(show_area, str):
        show_area = show_area.lower() in ('true', '1', 'yes')
    if isinstance(show_perimeter, str):
        show_perimeter = show_perimeter.lower() in ('true', '1', 'yes')
    if isinstance(show_center, str):
        show_center = show_center.lower() in ('true', '1', 'yes')
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for cnt in contours:
        M = cv2.moments(cnt)
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)
        if M['m00'] > 0:
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
        else:
            x, y, w, h = cv2.boundingRect(cnt)
            cx, cy = x + w // 2, y + h // 2
        y_offset = cy - 10
        if show_area:
            cv2.putText(result, f'A:{area:.0f}', (cx, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            y_offset += 15
        if show_perimeter:
            cv2.putText(result, f'P:{perimeter:.1f}', (cx, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            y_offset += 15
        if show_center:
            cv2.putText(result, f'({cx},{cy})', (cx, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
            cv2.circle(result, (cx, cy), 3, (0, 0, 255), -1)
    return {'image': result, 'info': f'{len(contours)} contours analyzed'}


# ---- Feature Nodes ----

def process_harris_corner(node, inputs):
    """Harris corner detection."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    block_size = int(props.get('blockSize', 2))
    ksize = int(props.get('ksize', 3))
    k = float(props.get('k', 0.04))
    if ksize % 2 == 0:
        ksize += 1
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = np.float32(gray)
    dst = cv2.cornerHarris(gray, block_size, ksize, k)
    dst = cv2.dilate(dst, None)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result[dst > 0.01 * dst.max()] = [0, 0, 255]
    return {'image': result, 'info': f'Harris corners detected'}


def process_good_features(node, inputs):
    """Shi-Tomasi corner detection (goodFeaturesToTrack)."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    max_corners = int(props.get('maxCorners', 100))
    quality_level = float(props.get('qualityLevel', 0.01))
    min_distance = float(props.get('minDistance', 10))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    corners = cv2.goodFeaturesToTrack(gray, max_corners, quality_level, min_distance)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    count = 0
    if corners is not None:
        count = len(corners)
        for corner in corners:
            x, y = corner.ravel()
            cv2.circle(result, (int(x), int(y)), 5, (0, 0, 255), -1)
    return {'image': result, 'info': f'{count} corners detected'}


def process_orb_features(node, inputs):
    """ORB feature detection."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    n_features = int(props.get('nFeatures', 500))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    orb = cv2.ORB_create(nfeatures=n_features)
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = cv2.drawKeypoints(result, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return {'image': result, 'info': f'{len(keypoints)} ORB features'}


def process_fast_features(node, inputs):
    """FAST feature detection."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    threshold = int(props.get('threshold', 10))
    nonmax_suppression = props.get('nonmaxSuppression', True)
    if isinstance(nonmax_suppression, str):
        nonmax_suppression = nonmax_suppression.lower() in ('true', '1', 'yes')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    fast = cv2.FastFeatureDetector_create(threshold=threshold, nonmaxSuppression=nonmax_suppression)
    keypoints = fast.detect(gray, None)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    result = cv2.drawKeypoints(result, keypoints, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return {'image': result, 'info': f'{len(keypoints)} FAST features'}


def process_match_features(node, inputs):
    """ORB feature matching between two images using BFMatcher."""
    img = inputs.get('image')
    img2 = inputs.get('image2')
    if img is None:
        return {'image': None, 'error': 'No input image on port 1'}
    if img2 is None:
        return {'image': img, 'error': 'No input image on port 2'}
    props = node.get('properties', {})
    n_features = int(props.get('nFeatures', 500))
    match_ratio = float(props.get('matchRatio', 0.75))
    gray1 = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY) if len(img2.shape) == 3 else img2
    orb = cv2.ORB_create(nfeatures=n_features)
    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)
    if des1 is None or des2 is None:
        result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        return {'image': result, 'error': 'Not enough features to match'}
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for m_pair in matches:
        if len(m_pair) == 2:
            m, n = m_pair
            if m.distance < match_ratio * n.distance:
                good.append(m)
    img1_c = img if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    img2_c = img2 if len(img2.shape) == 3 else cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    result = cv2.drawMatches(img1_c, kp1, img2_c, kp2, good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    return {'image': result, 'info': f'{len(good)} good matches'}


# ---- Drawing Nodes ----

def process_draw_line(node, inputs):
    """Draw a line on the image."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    x1 = int(props.get('x1', 0))
    y1 = int(props.get('y1', 0))
    x2 = int(props.get('x2', 100))
    y2 = int(props.get('y2', 100))
    r = int(props.get('colorR', 255))
    g = int(props.get('colorG', 0))
    b = int(props.get('colorB', 0))
    thickness = int(props.get('thickness', 2))
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.line(result, (x1, y1), (x2, y2), (b, g, r), thickness)
    return {'image': result}


def process_draw_rectangle(node, inputs):
    """Draw a rectangle on the image."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    x = int(props.get('x', 10))
    y = int(props.get('y', 10))
    w = int(props.get('width', 100))
    h = int(props.get('height', 100))
    r = int(props.get('colorR', 0))
    g = int(props.get('colorG', 255))
    b = int(props.get('colorB', 0))
    thickness = int(props.get('thickness', 2))
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(result, (x, y), (x + w, y + h), (b, g, r), thickness)
    return {'image': result}


def process_draw_circle(node, inputs):
    """Draw a circle on the image."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    cx = int(props.get('centerX', 50))
    cy = int(props.get('centerY', 50))
    radius = int(props.get('radius', 30))
    r = int(props.get('colorR', 0))
    g = int(props.get('colorG', 0))
    b = int(props.get('colorB', 255))
    thickness = int(props.get('thickness', 2))
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.circle(result, (cx, cy), radius, (b, g, r), thickness)
    return {'image': result}


def process_draw_ellipse(node, inputs):
    """Draw an ellipse on the image."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    cx = int(props.get('centerX', 100))
    cy = int(props.get('centerY', 100))
    axes_w = int(props.get('axesW', 50))
    axes_h = int(props.get('axesH', 30))
    angle = float(props.get('angle', 0))
    r = int(props.get('colorR', 255))
    g = int(props.get('colorG', 0))
    b = int(props.get('colorB', 255))
    thickness = int(props.get('thickness', 2))
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.ellipse(result, (cx, cy), (axes_w, axes_h), angle, 0, 360, (b, g, r), thickness)
    return {'image': result}


def process_draw_text(node, inputs):
    """Draw text on the image."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    text = props.get('text', 'Hello')
    x = int(props.get('x', 10))
    y = int(props.get('y', 30))
    font_scale = float(props.get('fontScale', 1.0))
    r = int(props.get('colorR', 255))
    g = int(props.get('colorG', 255))
    b = int(props.get('colorB', 255))
    thickness = int(props.get('thickness', 2))
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    cv2.putText(result, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (b, g, r), thickness)
    return {'image': result}


def process_draw_polylines(node, inputs):
    """Draw polylines on the image."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    points_str = props.get('points', '10,10;100,50;50,100')
    is_closed = props.get('isClosed', True)
    if isinstance(is_closed, str):
        is_closed = is_closed.lower() in ('true', '1', 'yes')
    r = int(props.get('colorR', 0))
    g = int(props.get('colorG', 255))
    b = int(props.get('colorB', 0))
    thickness = int(props.get('thickness', 2))
    try:
        pts = []
        for pair in points_str.split(';'):
            coords = pair.strip().split(',')
            if len(coords) == 2:
                pts.append([int(coords[0].strip()), int(coords[1].strip())])
        if len(pts) < 2:
            return {'image': img, 'error': 'Need at least 2 points'}
        pts_np = np.array([pts], dtype=np.int32)
        result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        cv2.polylines(result, pts_np, is_closed, (b, g, r), thickness)
        return {'image': result}
    except Exception as e:
        return {'image': img, 'error': f'Polylines error: {e}'}


# ---- Transform Nodes (new) ----

def process_flip(node, inputs):
    """Flip image."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    flip_code_str = str(props.get('flipCode', '1'))
    # Parse common string formats
    if 'horizontal' in flip_code_str.lower() or flip_code_str.strip() == '1':
        flip_code = 1
    elif 'vertical' in flip_code_str.lower() or flip_code_str.strip() == '0':
        flip_code = 0
    elif 'both' in flip_code_str.lower() or flip_code_str.strip() == '-1':
        flip_code = -1
    else:
        try:
            flip_code = int(flip_code_str)
        except ValueError:
            flip_code = 1
    result = cv2.flip(img, flip_code)
    return {'image': result}


def process_crop(node, inputs):
    """Crop image using numpy slicing."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    x = int(props.get('x', 0))
    y = int(props.get('y', 0))
    w = int(props.get('width', 100))
    h = int(props.get('height', 100))
    img_h, img_w = img.shape[:2]
    x = max(0, min(x, img_w - 1))
    y = max(0, min(y, img_h - 1))
    x2 = max(x + 1, min(x + w, img_w))
    y2 = max(y + 1, min(y + h, img_h))
    result = img[y:y2, x:x2].copy()
    return {'image': result}


def process_warp_affine(node, inputs):
    """Apply affine warp with 2x3 matrix."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    m00 = float(props.get('m00', 1))
    m01 = float(props.get('m01', 0))
    m02 = float(props.get('m02', 0))
    m10 = float(props.get('m10', 0))
    m11 = float(props.get('m11', 1))
    m12 = float(props.get('m12', 0))
    M = np.float32([[m00, m01, m02], [m10, m11, m12]])
    h, w = img.shape[:2]
    result = cv2.warpAffine(img, M, (w, h))
    return {'image': result}


def process_warp_perspective(node, inputs):
    """Apply perspective warp using 4 source and 4 destination points."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    h, w = img.shape[:2]
    # Source points (default: image corners)
    sx0 = float(props.get('srcX0', 0))
    sy0 = float(props.get('srcY0', 0))
    sx1 = float(props.get('srcX1', w))
    sy1 = float(props.get('srcY1', 0))
    sx2 = float(props.get('srcX2', w))
    sy2 = float(props.get('srcY2', h))
    sx3 = float(props.get('srcX3', 0))
    sy3 = float(props.get('srcY3', h))
    # Destination points (default: image corners)
    dx0 = float(props.get('dstX0', 0))
    dy0 = float(props.get('dstY0', 0))
    dx1 = float(props.get('dstX1', w))
    dy1 = float(props.get('dstY1', 0))
    dx2 = float(props.get('dstX2', w))
    dy2 = float(props.get('dstY2', h))
    dx3 = float(props.get('dstX3', 0))
    dy3 = float(props.get('dstY3', h))
    src_pts = np.float32([[sx0, sy0], [sx1, sy1], [sx2, sy2], [sx3, sy3]])
    dst_pts = np.float32([[dx0, dy0], [dx1, dy1], [dx2, dy2], [dx3, dy3]])
    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    result = cv2.warpPerspective(img, M, (w, h))
    return {'image': result}


def process_remap(node, inputs):
    """Barrel/pincushion distortion using remap."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    distortion_k = float(props.get('distortionK', 0.5))
    interp_str = props.get('interpolation', 'INTER_LINEAR')
    interp = getattr(cv2, interp_str, cv2.INTER_LINEAR)
    h, w = img.shape[:2]
    cx, cy = w / 2.0, h / 2.0
    max_r = np.sqrt(cx * cx + cy * cy)
    # Vectorized remap computation
    ys, xs = np.mgrid[0:h, 0:w].astype(np.float32)
    dx = (xs - cx) / max_r
    dy = (ys - cy) / max_r
    r = np.sqrt(dx * dx + dy * dy)
    r_new = r * (1 + distortion_k * r * r)
    scale = np.where(r > 0, r_new / r, 1.0)
    map_x = (cx + (xs - cx) * scale).astype(np.float32)
    map_y = (cy + (ys - cy) * scale).astype(np.float32)
    result = cv2.remap(img, map_x, map_y, interp)
    return {'image': result}


# ---- Histogram Nodes ----

def process_calc_histogram(node, inputs):
    """Calculate and draw histogram."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    hist_size = int(props.get('histSize', 256))
    do_normalize = props.get('normalize', True)
    if isinstance(do_normalize, str):
        do_normalize = do_normalize.lower() in ('true', '1', 'yes')
    hist_img = np.zeros((300, 512, 3), dtype=np.uint8)
    if len(img.shape) == 3:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        for i, color in enumerate(colors):
            hist = cv2.calcHist([img], [i], None, [hist_size], [0, 256])
            if do_normalize:
                cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
            for j in range(1, hist_size):
                x1 = int((j - 1) * 512 / hist_size)
                x2 = int(j * 512 / hist_size)
                cv2.line(hist_img, (x1, 300 - int(hist[j - 1])), (x2, 300 - int(hist[j])), color, 1)
    else:
        hist = cv2.calcHist([img], [0], None, [hist_size], [0, 256])
        if do_normalize:
            cv2.normalize(hist, hist, 0, 300, cv2.NORM_MINMAX)
        for j in range(1, hist_size):
            x1 = int((j - 1) * 512 / hist_size)
            x2 = int(j * 512 / hist_size)
            cv2.line(hist_img, (x1, 300 - int(hist[j - 1])), (x2, 300 - int(hist[j])), (255, 255, 255), 1)
    return {'image': hist_img}


# ---- Arithmetic Nodes (new) ----

def process_add(node, inputs):
    """Add two images."""
    img = inputs.get('image')
    img2 = inputs.get('image2')
    if img is None:
        return {'image': None, 'error': 'No input image on port 1'}
    if img2 is None:
        return {'image': img, 'error': 'No input image on port 2, passing through port 1'}
    if img.shape != img2.shape:
        img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    result = cv2.add(img, img2)
    return {'image': result}


def process_subtract(node, inputs):
    """Subtract two images."""
    img = inputs.get('image')
    img2 = inputs.get('image2')
    if img is None:
        return {'image': None, 'error': 'No input image on port 1'}
    if img2 is None:
        return {'image': img, 'error': 'No input image on port 2, passing through port 1'}
    if img.shape != img2.shape:
        img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    result = cv2.subtract(img, img2)
    return {'image': result}


def process_multiply(node, inputs):
    """Multiply two images."""
    img = inputs.get('image')
    img2 = inputs.get('image2')
    if img is None:
        return {'image': None, 'error': 'No input image on port 1'}
    if img2 is None:
        return {'image': img, 'error': 'No input image on port 2, passing through port 1'}
    props = node.get('properties', {})
    scale = float(props.get('scale', 1.0))
    if img.shape != img2.shape:
        img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    result = cv2.multiply(img, img2, scale=scale)
    return {'image': result}


def process_absdiff(node, inputs):
    """Absolute difference of two images."""
    img = inputs.get('image')
    img2 = inputs.get('image2')
    if img is None:
        return {'image': None, 'error': 'No input image on port 1'}
    if img2 is None:
        return {'image': img, 'error': 'No input image on port 2, passing through port 1'}
    if img.shape != img2.shape:
        img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    result = cv2.absdiff(img, img2)
    return {'image': result}


def process_bitwise_xor(node, inputs):
    """Bitwise XOR of two images with optional mask."""
    img = inputs.get('image')
    img2 = inputs.get('image2')
    if img is None:
        return {'image': None, 'error': 'No input image on port 1'}
    if img2 is None:
        return {'image': img, 'error': 'No input image on port 2, passing through port 1'}
    if img.shape != img2.shape:
        img2 = cv2.resize(img2, (img.shape[1], img.shape[0]))
    mask = _prepare_mask(inputs.get('mask'), img.shape)
    result = cv2.bitwise_xor(img, img2, mask=mask)
    return {'image': result}


# ---- Detection Nodes ----

def process_haar_cascade(node, inputs):
    """Haar cascade object detection."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    cascade_type = props.get('cascadeType', 'haarcascade_frontalface_default')
    scale_factor = float(props.get('scaleFactor', 1.1))
    min_neighbors = int(props.get('minNeighbors', 5))
    min_w = int(props.get('minWidth', 30))
    min_h = int(props.get('minHeight', 30))
    # Map cascade type to file path
    cascade_map = {
        'face': 'haarcascade_frontalface_default',
        'face_alt': 'haarcascade_frontalface_alt',
        'face_alt2': 'haarcascade_frontalface_alt2',
        'profile': 'haarcascade_profileface',
        'eye': 'haarcascade_eye',
        'eye_glasses': 'haarcascade_eye_tree_eyeglasses',
        'smile': 'haarcascade_smile',
        'upper_body': 'haarcascade_upperbody',
        'lower_body': 'haarcascade_lowerbody',
        'full_body': 'haarcascade_fullbody',
        'cat_face': 'haarcascade_frontalcatface',
    }
    cascade_name = cascade_map.get(cascade_type, cascade_type)
    cascade_path = os.path.join(cv2.data.haarcascades, cascade_name + '.xml')
    if not os.path.exists(cascade_path):
        return {'image': img, 'error': f'Cascade file not found: {cascade_name}'}
    classifier = cv2.CascadeClassifier(cascade_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    detections = classifier.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(min_w, min_h))
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    count = 0
    if len(detections) > 0:
        count = len(detections)
        for (x, y, w, h) in detections:
            cv2.rectangle(result, (x, y), (x + w, y + h), (0, 255, 0), 2)
    return {'image': result, 'info': f'Detected {count} objects'}


def process_hough_circles(node, inputs):
    """Hough circle detection."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    dp = float(props.get('dp', 1.2))
    min_dist = float(props.get('minDist', 50))
    param1 = float(props.get('param1', 100))
    param2 = float(props.get('param2', 30))
    min_radius = int(props.get('minRadius', 0))
    max_radius = int(props.get('maxRadius', 0))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
    gray = cv2.medianBlur(gray, 5)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp, min_dist, param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    count = 0
    if circles is not None:
        circles = np.uint16(np.around(circles))
        count = len(circles[0])
        for c in circles[0, :]:
            cv2.circle(result, (c[0], c[1]), c[2], (0, 255, 0), 2)
            cv2.circle(result, (c[0], c[1]), 2, (0, 0, 255), 3)
    return {'image': result, 'info': f'Found {count} circles'}


def process_template_match(node, inputs):
    """Template matching between two images."""
    img = inputs.get('image')
    template = inputs.get('image2')
    if img is None:
        return {'image': None, 'matches': [], 'error': 'No input image'}
    if template is None:
        return {'image': img, 'matches': [], 'error': 'No template image on port 2'}
    props = node.get('properties', {})
    method_str = props.get('method', 'TM_CCOEFF_NORMED')
    threshold = float(props.get('threshold', 0.8))
    method = getattr(cv2, method_str, cv2.TM_CCOEFF_NORMED)
    # Ensure same type
    if len(img.shape) == 3 and len(template.shape) == 2:
        template = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)
    elif len(img.shape) == 2 and len(template.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    if template.shape[0] > img.shape[0] or template.shape[1] > img.shape[1]:
        return {'image': img, 'matches': [], 'error': 'Template larger than image'}
    match_result = cv2.matchTemplate(img, template, method)
    result = img.copy() if len(img.shape) == 3 else cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    th, tw = template.shape[:2]
    # Find locations above threshold
    if method in (cv2.TM_SQDIFF, cv2.TM_SQDIFF_NORMED):
        locations = np.where(match_result <= (1 - threshold))
    else:
        locations = np.where(match_result >= threshold)
    # Build match coordinate list: [[x1, y1, x2, y2], ...]
    matches = []
    for pt in zip(*locations[::-1]):
        x1, y1 = int(pt[0]), int(pt[1])
        x2, y2 = x1 + tw, y1 + th
        matches.append([x1, y1, x2, y2])
        cv2.rectangle(result, (x1, y1), (x2, y2), (0, 0, 255), 2)
    count = len(matches)
    return {'image': result, 'matches': matches, 'info': f'{count} matches found: {matches[:5]}{"..." if count > 5 else ""}'}


# ---- Segmentation Nodes ----

def process_flood_fill(node, inputs):
    """Flood fill from a seed point."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    props = node.get('properties', {})
    seed_x = int(props.get('seedX', 0))
    seed_y = int(props.get('seedY', 0))
    r = int(props.get('colorR', 255))
    g = int(props.get('colorG', 0))
    b = int(props.get('colorB', 0))
    lo_diff = int(props.get('loDiff', 20))
    up_diff = int(props.get('upDiff', 20))
    result = img.copy()
    h, w = result.shape[:2]
    seed_x = max(0, min(seed_x, w - 1))
    seed_y = max(0, min(seed_y, h - 1))
    mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
    if len(result.shape) == 3:
        new_val = (b, g, r)
        lo = (lo_diff, lo_diff, lo_diff)
        up = (up_diff, up_diff, up_diff)
    else:
        new_val = 255
        lo = lo_diff
        up = up_diff
    cv2.floodFill(result, mask, (seed_x, seed_y), new_val, lo, up)
    return {'image': result}


def process_grabcut(node, inputs):
    """GrabCut segmentation."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    if len(img.shape) != 3:
        return {'image': img, 'error': 'GrabCut requires a color image'}
    props = node.get('properties', {})
    x = int(props.get('x', 10))
    y = int(props.get('y', 10))
    w = int(props.get('width', 100))
    h_val = int(props.get('height', 100))
    iterations = int(props.get('iterations', 5))
    rect = (x, y, w, h_val)
    mask = np.zeros(img.shape[:2], np.uint8)
    bgd_model = np.zeros((1, 65), np.float64)
    fgd_model = np.zeros((1, 65), np.float64)
    try:
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model, iterations, cv2.GC_INIT_WITH_RECT)
        mask2 = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0).astype(np.uint8)
        result = cv2.bitwise_and(img, img, mask=mask2)
        return {'image': result}
    except Exception as e:
        return {'image': img, 'error': f'GrabCut error: {e}'}


def process_watershed(node, inputs):
    """Watershed segmentation."""
    img = inputs.get('image')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    if len(img.shape) != 3:
        return {'image': img, 'error': 'Watershed requires a color image'}
    props = node.get('properties', {})
    marker_size = int(props.get('markerSize', 10))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Noise removal
    kernel = np.ones((3, 3), np.uint8)
    opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=2)
    # Sure background
    sure_bg = cv2.dilate(opening, kernel, iterations=3)
    # Sure foreground via distance transform
    dist_transform = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
    _, sure_fg = cv2.threshold(dist_transform, 0.5 * dist_transform.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg, sure_fg)
    _, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0
    result = img.copy()
    markers = cv2.watershed(result, markers)
    result[markers == -1] = [0, 0, 255]
    return {'image': result, 'info': f'Watershed segmentation complete'}


# ---- Value Nodes ----

def process_val_integer(node, inputs):
    """Output an integer value."""
    props = node.get('properties', {})
    try:
        value = int(props.get('value', 0))
    except (ValueError, TypeError):
        value = 0
    return {'value': value, 'info': f'Integer: {value}'}


def process_val_float(node, inputs):
    """Output a float value."""
    props = node.get('properties', {})
    try:
        value = float(props.get('value', 0.0))
    except (ValueError, TypeError):
        value = 0.0
    return {'value': value, 'info': f'Float: {value}'}


def process_val_boolean(node, inputs):
    """Output a boolean value."""
    props = node.get('properties', {})
    value = props.get('value', False)
    if isinstance(value, str):
        value = value.lower() in ('true', '1', 'yes')
    else:
        value = bool(value)
    return {'value': value, 'info': f'Boolean: {value}'}


def process_val_point(node, inputs):
    """Output a point (x, y)."""
    props = node.get('properties', {})
    try:
        x = int(props.get('x', 0))
        y = int(props.get('y', 0))
    except (ValueError, TypeError):
        x, y = 0, 0
    return {'value': (x, y), 'info': f'Point: ({x}, {y})'}


def process_val_scalar(node, inputs):
    """Output a scalar (v0, v1, v2, v3)."""
    props = node.get('properties', {})
    try:
        v0 = float(props.get('v0', 0))
        v1 = float(props.get('v1', 0))
        v2 = float(props.get('v2', 0))
        v3 = float(props.get('v3', 0))
    except (ValueError, TypeError):
        v0, v1, v2, v3 = 0, 0, 0, 0
    return {'value': (v0, v1, v2, v3), 'info': f'Scalar: ({v0}, {v1}, {v2}, {v3})'}


def process_val_math(node, inputs):
    """Perform math on two input values."""
    props = node.get('properties', {})
    operation = props.get('operation', 'add')
    a = inputs.get('a')
    b = inputs.get('b')
    # Also allow reading from properties as fallback
    if a is None:
        try:
            a = float(props.get('a', 0))
        except (ValueError, TypeError):
            a = 0
    if b is None:
        try:
            b = float(props.get('b', 0))
        except (ValueError, TypeError):
            b = 0
    # If inputs are tuples/values from value nodes, extract
    if isinstance(a, dict) and 'value' in a:
        a = a['value']
    if isinstance(b, dict) and 'value' in b:
        b = b['value']
    try:
        a = float(a)
        b = float(b)
    except (ValueError, TypeError):
        return {'value': 0, 'error': f'Cannot convert inputs to numbers: a={a}, b={b}'}
    try:
        if operation == 'add':
            result = a + b
        elif operation == 'subtract':
            result = a - b
        elif operation == 'multiply':
            result = a * b
        elif operation == 'divide':
            result = a / b if b != 0 else 0
        elif operation == 'power':
            result = a ** b
        elif operation == 'mod':
            result = a % b if b != 0 else 0
        elif operation == 'min':
            result = min(a, b)
        elif operation == 'max':
            result = max(a, b)
        elif operation == 'abs':
            result = abs(a)
        else:
            result = a + b
        return {'value': result, 'info': f'{a} {operation} {b} = {result}'}
    except Exception as e:
        return {'value': 0, 'error': f'Math error: {e}'}


def process_val_list(node, inputs):
    """Index or slice a list."""
    props = node.get('properties', {})
    mode = props.get('mode', 'index')
    input_list = inputs.get('list')
    if input_list is None:
        return {'value': None, 'error': 'No list input'}
    # If input is wrapped in dict with 'value' key, unwrap
    if isinstance(input_list, dict) and 'value' in input_list:
        input_list = input_list['value']
    if not isinstance(input_list, (list, tuple)):
        return {'value': input_list, 'info': f'Input is not a list (type: {type(input_list).__name__}), passed through'}
    if mode == 'index':
        idx = int(props.get('index', 0))
        if abs(idx) >= len(input_list):
            return {'value': None, 'error': f'Index {idx} out of range (list length: {len(input_list)})'}
        result = input_list[idx]
        return {'value': result, 'info': f'list[{idx}] = {result}'}
    else:  # slice
        start = int(props.get('start', 0))
        stop = int(props.get('stop', -1))
        step = int(props.get('step', 1))
        if step < 1:
            step = 1
        if stop == -1:
            stop = None
        result = list(input_list[start:stop:step])
        return {'value': result, 'info': f'list[{start}:{stop}:{step}] = {result[:5]}{"..." if len(result) > 5 else ""}'}


def process_image_extract(node, inputs):
    """Extract a region from an image using coordinates [x1,y1,x2,y2]."""
    img = inputs.get('image')
    coords = inputs.get('coords')
    if img is None:
        return {'image': None, 'error': 'No input image'}
    if coords is None:
        return {'image': img, 'error': 'No coordinates input'}
    # If coords is wrapped in dict with 'value' key, unwrap
    if isinstance(coords, dict) and 'value' in coords:
        coords = coords['value']
    props = node.get('properties', {})
    padding = int(props.get('padding', 0))
    h, w = img.shape[:2]
    # Determine the coordinate to use
    # If coords is a list of lists [[x1,y1,x2,y2], ...], use the first one
    # If coords is a single [x1,y1,x2,y2], use it directly
    if isinstance(coords, (list, tuple)) and len(coords) > 0:
        if isinstance(coords[0], (list, tuple)):
            coord = coords[0]  # list of coords: use first
        else:
            coord = coords  # single coord
    else:
        return {'image': img, 'error': f'Invalid coords format: {coords}'}
    if len(coord) < 4:
        return {'image': img, 'error': f'Coordinate must have 4 values [x1,y1,x2,y2], got {len(coord)}'}
    x1 = max(0, int(coord[0]) - padding)
    y1 = max(0, int(coord[1]) - padding)
    x2 = min(w, int(coord[2]) + padding)
    y2 = min(h, int(coord[3]) + padding)
    if x2 <= x1 or y2 <= y1:
        return {'image': img, 'error': f'Invalid region: ({x1},{y1})-({x2},{y2})'}
    result = img[y1:y2, x1:x2].copy()
    return {'image': result, 'info': f'Extracted ({x1},{y1})-({x2},{y2}) size={x2-x1}x{y2-y1}'}


# Node processor mapping
NODE_PROCESSORS = {
    'image_read': process_image_read,
    'image_show': process_image_show,
    'cvt_color': process_cvt_color,
    'gaussian_blur': process_gaussian_blur,
    'median_blur': process_median_blur,
    'bilateral_filter': process_bilateral_filter,
    'canny': process_canny,
    'threshold': process_threshold,
    'adaptive_threshold': process_adaptive_threshold,
    'resize': process_resize,
    'rotate': process_rotate,
    'morphology': process_morphology,
    'dilate': process_dilate,
    'erode': process_erode,
    'sobel': process_sobel,
    'laplacian': process_laplacian,
    'find_contours': process_find_contours,
    'hough_lines': process_hough_lines,
    'python_script': process_python_script,
    'bitwise_and': process_bitwise_and,
    'bitwise_or': process_bitwise_or,
    'bitwise_not': process_bitwise_not,
    'add_weighted': process_add_weighted,
    'histogram_eq': process_histogram_eq,
    'in_range': process_in_range,
    'control_if': process_control_if,
    'control_for': process_control_for,
    'control_while': process_control_while,
    'control_switch': process_control_switch,
    # IO
    'image_write': process_image_write,
    'video_write': process_video_write,
    'video_read': process_video_read,
    'camera_capture': process_camera_capture,
    # Color
    'split_channels': process_split_channels,
    'merge_channels': process_merge_channels,
    # Filter
    'box_filter': process_box_filter,
    'sharpen': process_sharpen,
    'filter2d': process_filter2d,
    # Edge
    'scharr': process_scharr,
    # Threshold
    'otsu_threshold': process_otsu_threshold,
    # Morphology
    'structuring_element': process_structuring_element,
    # Contour
    'draw_contours': process_draw_contours,
    'bounding_rect': process_bounding_rect,
    'min_enclosing_circle': process_min_enclosing_circle,
    'convex_hull': process_convex_hull,
    'approx_poly': process_approx_poly,
    'contour_area': process_contour_area,
    'contour_properties': process_contour_properties,
    # Feature
    'harris_corner': process_harris_corner,
    'good_features': process_good_features,
    'orb_features': process_orb_features,
    'fast_features': process_fast_features,
    'match_features': process_match_features,
    # Drawing
    'draw_line': process_draw_line,
    'draw_rectangle': process_draw_rectangle,
    'draw_circle': process_draw_circle,
    'draw_ellipse': process_draw_ellipse,
    'draw_text': process_draw_text,
    'draw_polylines': process_draw_polylines,
    # Transform
    'flip': process_flip,
    'crop': process_crop,
    'warp_affine': process_warp_affine,
    'warp_perspective': process_warp_perspective,
    'remap': process_remap,
    # Histogram
    'calc_histogram': process_calc_histogram,
    # Arithmetic
    'add': process_add,
    'subtract': process_subtract,
    'multiply': process_multiply,
    'absdiff': process_absdiff,
    'bitwise_xor': process_bitwise_xor,
    # Detection
    'haar_cascade': process_haar_cascade,
    'hough_circles': process_hough_circles,
    'template_match': process_template_match,
    # Segmentation
    'flood_fill': process_flood_fill,
    'grabcut': process_grabcut,
    'watershed': process_watershed,
    # Value
    'val_integer': process_val_integer,
    'val_float': process_val_float,
    'val_boolean': process_val_boolean,
    'val_point': process_val_point,
    'val_scalar': process_val_scalar,
    'val_math': process_val_math,
    'val_list': process_val_list,
    # Detection (extra)
    'image_extract': process_image_extract,
}


def topological_sort(nodes, connections):
    """Sort nodes in dependency order using topological sort."""
    node_map = {n['id']: n for n in nodes}
    in_edges = {n['id']: [] for n in nodes}
    out_edges = {n['id']: [] for n in nodes}

    for conn in connections:
        src = conn['sourceNode']
        tgt = conn['targetNode']
        if src in node_map and tgt in node_map:
            in_edges[tgt].append(conn)
            out_edges[src].append(conn)

    # Kahn's algorithm
    in_degree = {nid: len(edges) for nid, edges in in_edges.items()}
    queue = [nid for nid, deg in in_degree.items() if deg == 0]
    sorted_ids = []

    while queue:
        nid = queue.pop(0)
        sorted_ids.append(nid)
        for conn in out_edges.get(nid, []):
            tgt = conn['targetNode']
            in_degree[tgt] -= 1
            if in_degree[tgt] == 0:
                queue.append(tgt)

    return sorted_ids, node_map, in_edges


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/upload', methods=['POST'])
def upload_image():
    """Handle image file upload, store server-side, return ID + preview."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    # Save to session-specific disk
    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or '.png'
    filepath = os.path.join(g.session.upload_folder, file_id + ext)
    file.save(filepath)

    img = cv2.imread(filepath)
    if img is None:
        os.remove(filepath)
        return jsonify({'error': 'Could not read image file'}), 400

    # Store in session memory
    g.session.image_store[file_id] = img

    return jsonify({
        'imageId': file_id,
        'filename': file.filename,
        'preview': encode_image_jpeg(img),
        'shape': list(img.shape),
    })


@app.route('/api/upload_video', methods=['POST'])
def upload_video():
    """Handle video file upload, save server-side, return first frame preview."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400

    frame_index = int(request.form.get('frameIndex', 0))

    # Save to session-specific disk
    file_id = str(uuid.uuid4())[:8]
    ext = os.path.splitext(file.filename)[1] or '.mp4'
    filepath = os.path.join(g.session.upload_folder, file_id + ext)
    file.save(filepath)

    # Read frame from video
    cap = cv2.VideoCapture(filepath)
    if not cap.isOpened():
        os.remove(filepath)
        return jsonify({'error': 'Could not open video file'}), 400

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if frame_index > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        os.remove(filepath)
        return jsonify({'error': 'Could not read frame from video'}), 400

    # Store first frame in session memory
    g.session.image_store[file_id] = frame

    return jsonify({
        'imageId': file_id,
        'filename': file.filename,
        'filepath': filepath,
        'totalFrames': total_frames,
        'preview': encode_image_jpeg(frame),
        'shape': list(frame.shape),
    })


@app.route('/api/download/<image_id>')
def download_image(image_id):
    """Download a stored image as a file."""
    img = g.session.image_store.get(image_id)
    if img is None:
        return jsonify({'error': 'Image not found'}), 404
    fmt = request.args.get('format', 'png')
    filename = request.args.get('filename', f'output.{fmt}')
    if fmt in ('jpg', 'jpeg'):
        quality = int(request.args.get('quality', 95))
        success, buffer = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, quality])
        mimetype = 'image/jpeg'
    else:
        success, buffer = cv2.imencode('.png', img)
        mimetype = 'image/png'
    if not success:
        return jsonify({'error': 'Encoding failed'}), 500
    resp = make_response(buffer.tobytes())
    resp.headers['Content-Type'] = mimetype
    resp.headers['Content-Disposition'] = f'attachment; filename="{filename}"'
    return resp


@app.route('/api/download_file/<path:filename>')
def download_file(filename):
    """Download a file from the session work folder."""
    filepath = os.path.join(g.session.work_folder, filename)
    if not os.path.exists(filepath):
        return jsonify({'error': 'File not found'}), 404
    return send_from_directory(g.session.work_folder, filename, as_attachment=True)


@app.route('/api/stop_video_loop', methods=['POST'])
def stop_video_loop():
    """Signal the video loop to stop."""
    g.session.video_loop_stop = True
    return jsonify({'status': 'stopping'})


@app.route('/api/execute_video_loop', methods=['POST'])
def execute_video_loop():
    """Execute pipeline in video loop mode with SSE progress streaming."""
    g.session.video_loop_stop = False
    session = g.session  # capture reference for generator

    data = request.json
    nodes = data.get('nodes', [])
    connections = data.get('connections', [])

    # Find video_read node with mode='loop'
    video_node = None
    for n in nodes:
        if n.get('type') == 'video_read' and n.get('properties', {}).get('mode') == 'loop':
            video_node = n
            break
    if not video_node:
        return jsonify({'error': 'No video_read node with loop mode found'}), 400

    vprops = video_node.get('properties', {})
    filepath = vprops.get('filepath', '')
    if not filepath:
        return jsonify({'error': 'No video filepath specified'}), 400

    start_frame = max(0, int(vprops.get('startFrame', 0)))
    end_frame = int(vprops.get('endFrame', -1))
    step = max(1, int(vprops.get('step', 1)))

    # Find image_write nodes with videoOutput enabled
    video_writers = {}  # node_id -> VideoWriter

    sorted_ids, node_map, in_edges = topological_sort(nodes, connections)

    def generate():
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            yield f"data: {json.dumps({'error': f'Cannot open video: {filepath}'})}\n\n"
            return

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        src_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        actual_end = total_frames if end_frame < 0 else min(end_frame + 1, total_frames)

        yield f"data: {json.dumps({'type': 'start', 'totalFrames': total_frames, 'startFrame': start_frame, 'endFrame': actual_end - 1, 'step': step})}\n\n"

        # Setup VideoWriters for image_write nodes with videoOutput
        writers = {}
        first_frame_shape = None

        frame_count = 0
        preview_interval = max(1, (actual_end - start_frame) // step // 20)  # ~20 previews max

        try:
            for fidx in range(start_frame, actual_end, step):
                if session.video_loop_stop:
                    yield f"data: {json.dumps({'type': 'stopped', 'frame': fidx, 'processedFrames': frame_count})}\n\n"
                    break

                cap.set(cv2.CAP_PROP_POS_FRAMES, fidx)
                ret, frame = cap.read()
                if not ret:
                    break

                if first_frame_shape is None:
                    first_frame_shape = frame.shape

                # Execute pipeline with this frame injected as video_read result
                results = {}
                results[video_node['id']] = {'image': frame, 'info': f'Frame {fidx}'}

                for nid in sorted_ids:
                    if nid == video_node['id']:
                        continue  # Already set

                    node = node_map[nid]
                    node_type = node.get('type', '')
                    processor = NODE_PROCESSORS.get(node_type)
                    if not processor:
                        results[nid] = {'error': f'Unknown node type: {node_type}'}
                        continue

                    inputs = {}
                    for conn in in_edges.get(nid, []):
                        src_id = conn['sourceNode']
                        src_port = conn.get('sourcePort', 'image')
                        tgt_port = conn.get('targetPort', 'image')
                        if src_id in results:
                            src_result = results[src_id]
                            if src_port in src_result and src_result[src_port] is not None:
                                inputs[tgt_port] = src_result[src_port]
                            elif src_result.get('image') is not None:
                                inputs[tgt_port] = src_result['image']

                    try:
                        result = processor(node, inputs)
                        results[nid] = result
                    except Exception as e:
                        results[nid] = {'error': str(e)}

                    # VideoWriter: write frame for video_write nodes
                    if node_type == 'video_write':
                        nprops = node.get('properties', {})
                        if nid not in writers:
                            out_name = os.path.basename(nprops.get('filepath', 'output.mp4'))
                            if not out_name.lower().endswith(('.mp4', '.avi', '.mkv')):
                                out_name = os.path.splitext(out_name)[0] + '.mp4'
                            out_path = os.path.join(session.work_folder, out_name)
                            codec_str = nprops.get('codec', 'mp4v')
                            fps = float(nprops.get('fps', 0)) or src_fps
                            fourcc = cv2.VideoWriter_fourcc(*codec_str)
                            h, w = first_frame_shape[:2]
                            writers[nid] = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

                        out_img = result.get('image')
                        if out_img is not None and nid in writers:
                            if len(out_img.shape) == 2:
                                out_img = cv2.cvtColor(out_img, cv2.COLOR_GRAY2BGR)
                            if out_img.shape[:2] != first_frame_shape[:2]:
                                out_img = cv2.resize(out_img, (first_frame_shape[1], first_frame_shape[0]))
                            writers[nid].write(out_img)

                frame_count += 1

                # Send progress (with preview every N frames)
                send_preview = (frame_count % preview_interval == 0) or (fidx + step >= actual_end)
                progress_data = {
                    'type': 'progress',
                    'frame': fidx,
                    'processedFrames': frame_count,
                    'totalFrames': total_frames,
                    'percent': round(frame_count / max(1, (actual_end - start_frame) / step) * 100, 1)
                }
                if send_preview:
                    # Find best preview image from results
                    preview_img = None
                    for nid_check in reversed(sorted_ids):
                        if nid_check == video_node['id']:
                            continue
                        r = results.get(nid_check, {})
                        for key in ['image', 'true', 'false', 'ch0']:
                            if r.get(key) is not None and isinstance(r.get(key), np.ndarray):
                                preview_img = r[key]
                                break
                        if preview_img is not None:
                            break
                    if preview_img is None:
                        preview_img = frame
                    progress_data['preview'] = encode_image_jpeg(preview_img)
                    progress_data['shape'] = list(preview_img.shape)

                yield f"data: {json.dumps(progress_data)}\n\n"

        except Exception as e:
            yield f"data: {json.dumps({'type': 'error', 'error': str(e), 'frame': fidx, 'processedFrames': frame_count})}\n\n"
        finally:
            cap.release()
            for w in writers.values():
                w.release()

            # Build final results for node previews
            final_response = {}
            if results:
                for nid, result in results.items():
                    node_resp = {}
                    img_found = None
                    for key in ['image', 'true', 'false', 'case0', 'case1', 'case2', 'ch0', 'ch1', 'ch2']:
                        if result.get(key) is not None and isinstance(result.get(key), np.ndarray):
                            img_found = result[key]
                            break
                    if img_found is not None:
                        node_resp['preview'] = encode_image_jpeg(img_found)
                        node_resp['shape'] = list(img_found.shape)
                        rid = store_image(img_found)
                        node_resp['resultImageId'] = rid
                    if result.get('error'):
                        node_resp['error'] = result['error']
                    if result.get('info'):
                        node_resp['info'] = result['info']
                    final_response[nid] = node_resp

            writer_paths = {}
            for nid, w in writers.items():
                nprops = node_map[nid].get('properties', {})
                writer_paths[nid] = nprops.get('filepath', 'output.mp4')

            yield f"data: {json.dumps({'type': 'done', 'processedFrames': frame_count, 'results': final_response, 'videoPaths': writer_paths})}\n\n"

    return Response(stream_with_context(generate()), mimetype='text/event-stream',
                    headers={'Cache-Control': 'no-cache', 'X-Accel-Buffering': 'no'})


@app.route('/api/execute', methods=['POST'])
def execute_pipeline():
    """Execute the node pipeline and return results for all nodes."""
    data = request.json
    nodes = data.get('nodes', [])
    connections = data.get('connections', [])

    if not nodes:
        return jsonify({})

    sorted_ids, node_map, in_edges = topological_sort(nodes, connections)
    results = {}  # nid -> { image: np.array, error: str, info: str }

    for nid in sorted_ids:
        node = node_map[nid]
        node_type = node.get('type', '')
        processor = NODE_PROCESSORS.get(node_type)

        if not processor:
            results[nid] = {'error': f'Unknown node type: {node_type}'}
            continue

        # Gather inputs from connected source nodes
        inputs = {}
        for conn in in_edges.get(nid, []):
            src_id = conn['sourceNode']
            src_port = conn.get('sourcePort', 'image')
            tgt_port = conn.get('targetPort', 'image')
            if src_id in results:
                # Try source port name first (for multi-output nodes like If/Switch)
                src_result = results[src_id]
                if src_port in src_result and src_result[src_port] is not None:
                    inputs[tgt_port] = src_result[src_port]
                elif src_result.get('image') is not None:
                    inputs[tgt_port] = src_result['image']

        try:
            result = processor(node, inputs)
            results[nid] = result
        except Exception as e:
            results[nid] = {'error': str(e), 'traceback': traceback.format_exc()}

    # Build response: encode images as JPEG previews
    response = {}
    for nid, result in results.items():
        node_resp = {}
        # Find the first image-like value in results (handles multi-output nodes)
        img_found = None
        for key in ['image', 'true', 'false', 'case0', 'case1', 'case2', 'ch0', 'ch1', 'ch2']:
            if result.get(key) is not None and isinstance(result.get(key), np.ndarray):
                img_found = result[key]
                break
        if img_found is not None:
            node_resp['preview'] = encode_image_jpeg(img_found)
            node_resp['shape'] = list(img_found.shape)
            result_id = store_image(img_found)
            node_resp['resultImageId'] = result_id
        if result.get('error'):
            node_resp['error'] = result['error']
        if result.get('info'):
            node_resp['info'] = result['info']
        if result.get('downloadFile'):
            node_resp['downloadFile'] = result['downloadFile']
        response[nid] = node_resp

    return jsonify(response)


@app.route('/api/execute_single', methods=['POST'])
def execute_single_node():
    """Execute pipeline up to a specific node for preview."""
    data = request.json
    target_node_id = data.get('targetNodeId')
    nodes = data.get('nodes', [])
    connections = data.get('connections', [])

    if not nodes or not target_node_id:
        return jsonify({'error': 'Missing data'}), 400

    sorted_ids, node_map, in_edges = topological_sort(nodes, connections)

    # Find all ancestors of target node
    ancestors = set()

    def find_ancestors(nid):
        for conn in in_edges.get(nid, []):
            src = conn['sourceNode']
            if src not in ancestors:
                ancestors.add(src)
                find_ancestors(src)

    ancestors.add(target_node_id)
    find_ancestors(target_node_id)

    results = {}
    for nid in sorted_ids:
        if nid not in ancestors:
            continue

        node = node_map[nid]
        node_type = node.get('type', '')
        processor = NODE_PROCESSORS.get(node_type)

        if not processor:
            results[nid] = {'error': f'Unknown node type: {node_type}'}
            continue

        inputs = {}
        for conn in in_edges.get(nid, []):
            src_id = conn['sourceNode']
            src_port = conn.get('sourcePort', 'image')
            tgt_port = conn.get('targetPort', 'image')
            if src_id in results:
                src_result = results[src_id]
                if src_port in src_result and src_result[src_port] is not None:
                    inputs[tgt_port] = src_result[src_port]
                elif src_result.get('image') is not None:
                    inputs[tgt_port] = src_result['image']

        try:
            result = processor(node, inputs)
            results[nid] = result
        except Exception as e:
            results[nid] = {'error': str(e)}

    # Return only the target node result
    target_result = results.get(target_node_id, {})
    resp = {}
    img_found = None
    for key in ['image', 'true', 'false', 'case0', 'case1', 'case2', 'ch0', 'ch1', 'ch2']:
        if target_result.get(key) is not None and isinstance(target_result.get(key), np.ndarray):
            img_found = target_result[key]
            break
    if img_found is not None:
        resp['preview'] = encode_image_jpeg(img_found)
        resp['shape'] = list(img_found.shape)
        result_id = store_image(img_found)
        resp['resultImageId'] = result_id
    if target_result.get('error'):
        resp['error'] = target_result['error']
    if target_result.get('info'):
        resp['info'] = target_result['info']
    if target_result.get('downloadFile'):
        resp['downloadFile'] = target_result['downloadFile']

    return jsonify(resp)


@app.route('/api/image/<image_id>')
def get_image(image_id):
    """Get a stored image as PNG (for download/full-quality view)."""
    img = g.session.image_store.get(image_id)
    if img is None:
        return jsonify({'error': 'Image not found'}), 404
    success, buffer = cv2.imencode('.png', img)
    if not success:
        return jsonify({'error': 'Encoding failed'}), 500
    return Response(buffer.tobytes(), mimetype='image/png')


# ---- Code Generation ----

def generate_python_code(nodes, connections):
    """Convert a node graph into standalone Python OpenCV code."""
    sorted_ids, node_map, in_edges = topological_sort(nodes, connections)

    lines = [
        '"""',
        'Auto-generated by SamOpenCVWeb',
        'Visual OpenCV Pipeline -> Python Code',
        '"""',
        'import cv2',
        'import numpy as np',
        '',
    ]

    # Variable name mapping: node_id -> variable name
    var_map = {}
    var_counter = [0]

    def var_name(nid, suffix=''):
        if nid + suffix not in var_map:
            node = node_map.get(nid)
            ntype = node.get('type', 'unknown') if node else 'unknown'
            label = ntype.replace('control_', '').replace('image_', '')
            var_map[nid + suffix] = f'img_{label}_{var_counter[0]}'
            var_counter[0] += 1
        return var_map[nid + suffix]

    def get_src_var(nid, port_id):
        """Get source variable name for a connection."""
        for conn in in_edges.get(nid, []):
            if conn.get('targetPort', 'image') == port_id:
                src_id = conn['sourceNode']
                src_port = conn.get('sourcePort', 'image')
                src_node = node_map.get(src_id)
                src_type = src_node.get('type', '') if src_node else ''
                # Multi-output nodes
                if src_type in ('control_if', 'control_switch', 'split_channels', 'template_match', 'find_contours'):
                    return var_name(src_id, '_' + src_port)
                else:
                    return var_name(src_id)
        return 'None'

    for nid in sorted_ids:
        node = node_map[nid]
        ntype = node.get('type', '')
        props = node.get('properties', {})
        out = var_name(nid)
        src = get_src_var(nid, 'image')
        src2 = get_src_var(nid, 'image2')

        if ntype == 'image_read':
            fp = props.get('filepath', '') or props.get('filename', 'input.jpg')
            flags_str = props.get('flags', 'IMREAD_COLOR')
            lines.append(f'# Image Read')
            lines.append(f'{out} = cv2.imread(r"{fp}", cv2.{flags_str})')
            if flags_str == 'IMREAD_GRAYSCALE':
                lines.append(f'{out} = cv2.cvtColor({out}, cv2.COLOR_GRAY2BGR)')
            lines.append('')

        elif ntype == 'image_show':
            wname = props.get('windowName', 'Output')
            lines.append(f'# Image Show')
            lines.append(f'{out} = {src}.copy()')
            lines.append(f'cv2.imshow("{wname}", {out})')
            lines.append('')

        elif ntype == 'cvt_color':
            code = props.get('code', 'COLOR_BGR2GRAY')
            lines.append(f'# CvtColor')
            lines.append(f'{out} = cv2.cvtColor({src}, cv2.{code})')
            lines.append(f'if len({out}.shape) == 2: {out} = cv2.cvtColor({out}, cv2.COLOR_GRAY2BGR)')
            lines.append('')

        elif ntype == 'gaussian_blur':
            k = int(props.get('ksize', 5))
            if k % 2 == 0: k += 1
            s = float(props.get('sigmaX', 0))
            lines.append(f'# Gaussian Blur')
            lines.append(f'{out} = cv2.GaussianBlur({src}, ({k}, {k}), {s})')
            lines.append('')

        elif ntype == 'median_blur':
            k = int(props.get('ksize', 5))
            if k % 2 == 0: k += 1
            lines.append(f'# Median Blur')
            lines.append(f'{out} = cv2.medianBlur({src}, {k})')
            lines.append('')

        elif ntype == 'bilateral_filter':
            d = int(props.get('d', 9))
            sc = float(props.get('sigmaColor', 75))
            ss = float(props.get('sigmaSpace', 75))
            lines.append(f'# Bilateral Filter')
            lines.append(f'{out} = cv2.bilateralFilter({src}, {d}, {sc}, {ss})')
            lines.append('')

        elif ntype == 'canny':
            t1 = float(props.get('threshold1', 100))
            t2 = float(props.get('threshold2', 200))
            lines.append(f'# Canny Edge Detection')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'{out} = cv2.Canny(_gray, {t1}, {t2})')
            lines.append(f'{out} = cv2.cvtColor({out}, cv2.COLOR_GRAY2BGR)')
            lines.append('')

        elif ntype == 'threshold':
            tv = float(props.get('thresh', 127))
            mv = float(props.get('maxval', 255))
            tt = props.get('type', 'THRESH_BINARY')
            lines.append(f'# Threshold')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_, {out} = cv2.threshold(_gray, {tv}, {mv}, cv2.{tt})')
            lines.append(f'{out} = cv2.cvtColor({out}, cv2.COLOR_GRAY2BGR)')
            lines.append('')

        elif ntype == 'adaptive_threshold':
            mv = float(props.get('maxval', 255))
            am = props.get('adaptiveMethod', 'ADAPTIVE_THRESH_GAUSSIAN_C')
            tt = props.get('thresholdType', 'THRESH_BINARY')
            bs = int(props.get('blockSize', 11))
            if bs % 2 == 0: bs += 1
            c = float(props.get('C', 2))
            lines.append(f'# Adaptive Threshold')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'{out} = cv2.adaptiveThreshold(_gray, {mv}, cv2.{am}, cv2.{tt}, {bs}, {c})')
            lines.append(f'{out} = cv2.cvtColor({out}, cv2.COLOR_GRAY2BGR)')
            lines.append('')

        elif ntype == 'resize':
            w = int(props.get('width', 0))
            h = int(props.get('height', 0))
            fx = float(props.get('fx', 0.5))
            fy = float(props.get('fy', 0.5))
            interp = props.get('interpolation', 'INTER_LINEAR')
            lines.append(f'# Resize')
            if w > 0 and h > 0:
                lines.append(f'{out} = cv2.resize({src}, ({w}, {h}), interpolation=cv2.{interp})')
            else:
                lines.append(f'{out} = cv2.resize({src}, None, fx={fx}, fy={fy}, interpolation=cv2.{interp})')
            lines.append('')

        elif ntype == 'rotate':
            angle = float(props.get('angle', 90))
            lines.append(f'# Rotate')
            lines.append(f'_h, _w = {src}.shape[:2]')
            lines.append(f'_M = cv2.getRotationMatrix2D((_w//2, _h//2), {angle}, 1.0)')
            lines.append(f'_cos, _sin = abs(_M[0,0]), abs(_M[0,1])')
            lines.append(f'_nw, _nh = int(_h*_sin + _w*_cos), int(_h*_cos + _w*_sin)')
            lines.append(f'_M[0,2] += (_nw - _w)/2; _M[1,2] += (_nh - _h)/2')
            lines.append(f'{out} = cv2.warpAffine({src}, _M, (_nw, _nh))')
            lines.append('')

        elif ntype == 'morphology':
            op = props.get('operation', 'MORPH_OPEN')
            k = int(props.get('ksize', 5))
            if k % 2 == 0: k += 1
            sh = props.get('shape', 'MORPH_RECT')
            it = max(1, int(props.get('iterations', 1)))
            lines.append(f'# Morphology Ex')
            lines.append(f'_kernel = cv2.getStructuringElement(cv2.{sh}, ({k}, {k}))')
            lines.append(f'{out} = cv2.morphologyEx({src}, cv2.{op}, _kernel, iterations={it})')
            lines.append('')

        elif ntype == 'dilate':
            k = int(props.get('ksize', 5))
            if k % 2 == 0: k += 1
            it = max(1, int(props.get('iterations', 1)))
            lines.append(f'# Dilate')
            lines.append(f'_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({k}, {k}))')
            lines.append(f'{out} = cv2.dilate({src}, _kernel, iterations={it})')
            lines.append('')

        elif ntype == 'erode':
            k = int(props.get('ksize', 5))
            if k % 2 == 0: k += 1
            it = max(1, int(props.get('iterations', 1)))
            lines.append(f'# Erode')
            lines.append(f'_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, ({k}, {k}))')
            lines.append(f'{out} = cv2.erode({src}, _kernel, iterations={it})')
            lines.append('')

        elif ntype == 'sobel':
            dx = int(props.get('dx', 1))
            dy = int(props.get('dy', 0))
            k = int(props.get('ksize', 3))
            if k % 2 == 0: k += 1
            lines.append(f'# Sobel')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'{out} = cv2.Sobel(_gray, cv2.CV_64F, {dx}, {dy}, ksize={k})')
            lines.append(f'{out} = cv2.convertScaleAbs({out})')
            lines.append(f'{out} = cv2.cvtColor({out}, cv2.COLOR_GRAY2BGR)')
            lines.append('')

        elif ntype == 'laplacian':
            k = int(props.get('ksize', 3))
            if k % 2 == 0: k += 1
            lines.append(f'# Laplacian')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'{out} = cv2.Laplacian(_gray, cv2.CV_64F, ksize={k})')
            lines.append(f'{out} = cv2.convertScaleAbs({out})')
            lines.append(f'{out} = cv2.cvtColor({out}, cv2.COLOR_GRAY2BGR)')
            lines.append('')

        elif ntype == 'find_contours':
            mode = props.get('mode', 'RETR_EXTERNAL')
            method = props.get('method', 'CHAIN_APPROX_SIMPLE')
            out_img = var_name(nid, '_image')
            out_contours = var_name(nid, '_contours')
            lines.append(f'# Find Contours')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'{out_contours}, _ = cv2.findContours(_gray, cv2.{mode}, cv2.{method})')
            lines.append(f'{out_img} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'cv2.drawContours({out_img}, {out_contours}, -1, (0, 255, 0), 2)')
            lines.append('')

        elif ntype == 'hough_lines':
            rho = float(props.get('rho', 1))
            td = float(props.get('theta_divisor', 180))
            th = int(props.get('threshold', 100))
            lines.append(f'# Hough Lines')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_edges = cv2.Canny(_gray, 50, 150)')
            lines.append(f'_lines = cv2.HoughLines(_edges, {rho}, np.pi/{td}, {th})')
            lines.append(f'{out} = {src}.copy()')
            lines.append(f'if _lines is not None:')
            lines.append(f'    for _line in _lines:')
            lines.append(f'        _r, _t = _line[0]')
            lines.append(f'        _a, _b = np.cos(_t), np.sin(_t)')
            lines.append(f'        _x0, _y0 = _a*_r, _b*_r')
            lines.append(f'        _pt1 = (int(_x0+1000*(-_b)), int(_y0+1000*_a))')
            lines.append(f'        _pt2 = (int(_x0-1000*(-_b)), int(_y0-1000*_a))')
            lines.append(f'        cv2.line({out}, _pt1, _pt2, (0,0,255), 2)')
            lines.append('')

        elif ntype == 'add_weighted':
            a = float(props.get('alpha', 0.5))
            b = float(props.get('beta', 0.5))
            g = float(props.get('gamma', 0))
            lines.append(f'# Add Weighted')
            lines.append(f'{out} = cv2.addWeighted({src}, {a}, {src2}, {b}, {g})')
            lines.append('')

        elif ntype == 'bitwise_and':
            src_mask = get_src_var(nid, 'mask')
            lines.append(f'# Bitwise AND')
            if src_mask != 'None':
                lines.append(f'_mask = cv2.cvtColor({src_mask}, cv2.COLOR_BGR2GRAY) if len({src_mask}.shape) == 3 else {src_mask}')
                lines.append(f'{out} = cv2.bitwise_and({src}, {src2}, mask=_mask)')
            else:
                lines.append(f'{out} = cv2.bitwise_and({src}, {src2})')
            lines.append('')

        elif ntype == 'bitwise_or':
            src_mask = get_src_var(nid, 'mask')
            lines.append(f'# Bitwise OR')
            if src_mask != 'None':
                lines.append(f'_mask = cv2.cvtColor({src_mask}, cv2.COLOR_BGR2GRAY) if len({src_mask}.shape) == 3 else {src_mask}')
                lines.append(f'{out} = cv2.bitwise_or({src}, {src2}, mask=_mask)')
            else:
                lines.append(f'{out} = cv2.bitwise_or({src}, {src2})')
            lines.append('')

        elif ntype == 'bitwise_not':
            src_mask = get_src_var(nid, 'mask')
            lines.append(f'# Bitwise NOT')
            if src_mask != 'None':
                lines.append(f'_mask = cv2.cvtColor({src_mask}, cv2.COLOR_BGR2GRAY) if len({src_mask}.shape) == 3 else {src_mask}')
                lines.append(f'{out} = cv2.bitwise_not({src}, mask=_mask)')
            else:
                lines.append(f'{out} = cv2.bitwise_not({src})')
            lines.append('')

        elif ntype == 'histogram_eq':
            use_clahe = props.get('useCLAHE', False)
            lines.append(f'# Histogram Equalization')
            if use_clahe:
                cl = float(props.get('clipLimit', 2.0))
                ts = int(props.get('tileGridSize', 8))
                lines.append(f'_clahe = cv2.createCLAHE(clipLimit={cl}, tileGridSize=({ts},{ts}))')
                lines.append(f'_ycrcb = cv2.cvtColor({src}, cv2.COLOR_BGR2YCrCb)')
                lines.append(f'_ycrcb[:,:,0] = _clahe.apply(_ycrcb[:,:,0])')
                lines.append(f'{out} = cv2.cvtColor(_ycrcb, cv2.COLOR_YCrCb2BGR)')
            else:
                lines.append(f'_ycrcb = cv2.cvtColor({src}, cv2.COLOR_BGR2YCrCb)')
                lines.append(f'_ycrcb[:,:,0] = cv2.equalizeHist(_ycrcb[:,:,0])')
                lines.append(f'{out} = cv2.cvtColor(_ycrcb, cv2.COLOR_YCrCb2BGR)')
            lines.append('')

        elif ntype == 'in_range':
            lb = int(props.get('lowerB', 0))
            lg = int(props.get('lowerG', 0))
            lr = int(props.get('lowerR', 0))
            ub = int(props.get('upperB', 255))
            ug = int(props.get('upperG', 255))
            ur = int(props.get('upperR', 255))
            lines.append(f'# InRange')
            lines.append(f'{out} = cv2.inRange({src}, np.array([{lb},{lg},{lr}]), np.array([{ub},{ug},{ur}]))')
            lines.append(f'{out} = cv2.cvtColor({out}, cv2.COLOR_GRAY2BGR)')
            lines.append('')

        elif ntype == 'python_script':
            script = props.get('script', '').strip()
            lines.append(f'# Python Script')
            lines.append(f'img_input = {src}')
            for sline in script.split('\n'):
                lines.append(sline)
            lines.append(f'{out} = img_output')
            lines.append('')

        elif ntype == 'control_if':
            cond_type = props.get('condition', 'not_empty')
            val = float(props.get('value', 100))
            true_var = var_name(nid, '_true')
            false_var = var_name(nid, '_false')
            lines.append(f'# If (condition: {cond_type})')
            if cond_type == 'not_empty':
                lines.append(f'_cond = {src} is not None and {src}.size > 0')
            elif cond_type == 'is_color':
                lines.append(f'_cond = len({src}.shape) == 3 and {src}.shape[2] >= 3')
            elif cond_type == 'is_grayscale':
                lines.append(f'_cond = len({src}.shape) == 2')
            elif cond_type == 'width_gt':
                lines.append(f'_cond = {src}.shape[1] > {val}')
            elif cond_type == 'height_gt':
                lines.append(f'_cond = {src}.shape[0] > {val}')
            elif cond_type == 'mean_gt':
                lines.append(f'_cond = float(np.mean({src})) > {val}')
            elif cond_type == 'custom':
                expr = props.get('customExpr', 'True')
                lines.append(f'_cond = bool({expr.replace("img", src)})')
            lines.append(f'if _cond:')
            lines.append(f'    {true_var} = {src}.copy()')
            lines.append(f'    {false_var} = None')
            lines.append(f'else:')
            lines.append(f'    {true_var} = None')
            lines.append(f'    {false_var} = {src}.copy()')
            lines.append('')

        elif ntype == 'control_for':
            it = max(1, int(props.get('iterations', 3)))
            op = props.get('operation', 'gaussian_blur')
            k = int(props.get('ksize', 3))
            if k % 2 == 0: k += 1
            lines.append(f'# For Loop ({it} iterations, op={op})')
            lines.append(f'{out} = {src}.copy()')
            lines.append(f'for _i in range({it}):')
            if op == 'gaussian_blur':
                lines.append(f'    {out} = cv2.GaussianBlur({out}, ({k},{k}), 0)')
            elif op == 'median_blur':
                lines.append(f'    {out} = cv2.medianBlur({out}, {max(3,k)})')
            elif op == 'dilate':
                lines.append(f'    _k = cv2.getStructuringElement(cv2.MORPH_RECT, ({k},{k}))')
                lines.append(f'    {out} = cv2.dilate({out}, _k)')
            elif op == 'erode':
                lines.append(f'    _k = cv2.getStructuringElement(cv2.MORPH_RECT, ({k},{k}))')
                lines.append(f'    {out} = cv2.erode({out}, _k)')
            elif op == 'sharpen':
                lines.append(f'    _b = cv2.GaussianBlur({out}, ({k},{k}), 0)')
                lines.append(f'    {out} = cv2.addWeighted({out}, 1.5, _b, -0.5, 0)')
            elif op == 'custom':
                code = props.get('customCode', '')
                lines.append(f'    img = {out}')
                lines.append(f'    i = _i')
                for cl in code.split('\n'):
                    lines.append(f'    {cl}')
                lines.append(f'    {out} = img')
            lines.append('')

        elif ntype == 'control_while':
            cond_type = props.get('condition', 'mean_gt')
            val = float(props.get('value', 128))
            op = props.get('operation', 'gaussian_blur')
            k = int(props.get('ksize', 3))
            if k % 2 == 0: k += 1
            mi = max(1, int(props.get('maxIter', 50)))
            lines.append(f'# While Loop (condition={cond_type}, max={mi})')
            lines.append(f'{out} = {src}.copy()')
            lines.append(f'_iter = 0')
            if cond_type == 'custom':
                cexpr = props.get('customCond', 'False')
                lines.append(f'while _iter < {mi} and ({cexpr.replace("img", out)}):')
            elif cond_type == 'mean_gt':
                lines.append(f'while _iter < {mi} and float(np.mean({out})) > {val}:')
            elif cond_type == 'mean_lt':
                lines.append(f'while _iter < {mi} and float(np.mean({out})) < {val}:')
            elif cond_type == 'std_gt':
                lines.append(f'while _iter < {mi} and float(np.std({out})) > {val}:')
            elif cond_type == 'nonzero_gt':
                lines.append(f'while _iter < {mi} and np.count_nonzero({out}) > {val}:')
            if op == 'gaussian_blur':
                lines.append(f'    {out} = cv2.GaussianBlur({out}, ({k},{k}), 0)')
            elif op == 'median_blur':
                lines.append(f'    {out} = cv2.medianBlur({out}, {max(3,k)})')
            elif op == 'erode':
                lines.append(f'    _k = cv2.getStructuringElement(cv2.MORPH_RECT, ({k},{k}))')
                lines.append(f'    {out} = cv2.erode({out}, _k)')
            elif op == 'dilate':
                lines.append(f'    _k = cv2.getStructuringElement(cv2.MORPH_RECT, ({k},{k}))')
                lines.append(f'    {out} = cv2.dilate({out}, _k)')
            elif op == 'custom':
                code = props.get('customCode', '')
                lines.append(f'    img = {out}')
                for cl in code.split('\n'):
                    lines.append(f'    {cl}')
                lines.append(f'    {out} = img')
            lines.append(f'    _iter += 1')
            lines.append('')

        elif ntype == 'control_switch':
            sw = props.get('switchOn', 'channels')
            c0 = var_name(nid, '_case0')
            c1 = var_name(nid, '_case1')
            c2 = var_name(nid, '_case2')
            lines.append(f'# Switch-Case (on={sw})')
            if sw == 'channels':
                lines.append(f'_ch = {src}.shape[2] if len({src}.shape)==3 else 1')
                lines.append(f'_case = 0 if _ch==1 else (1 if _ch==3 else 2)')
            elif sw == 'depth':
                lines.append(f'_case = 0 if {src}.dtype==np.uint8 else (1 if {src}.dtype in (np.float32,np.float64) else 2)')
            elif sw == 'size_class':
                lines.append(f'_px = {src}.shape[0]*{src}.shape[1]')
                lines.append(f'_case = 0 if _px<100000 else (1 if _px<1000000 else 2)')
            elif sw == 'mean_range':
                lines.append(f'_m = float(np.mean({src}))')
                lines.append(f'_case = 0 if _m<85 else (1 if _m<170 else 2)')
            elif sw == 'custom':
                expr = props.get('customExpr', '0')
                lines.append(f'_case = int({expr.replace("img", src)})')
            lines.append(f'{c0} = {src}.copy() if _case==0 else None')
            lines.append(f'{c1} = {src}.copy() if _case==1 else None')
            lines.append(f'{c2} = {src}.copy() if _case==2 else None')
            lines.append('')

        # ---- IO (new) ----
        elif ntype == 'image_write':
            fp = props.get('filepath', 'output.jpg')
            lines.append(f'# Image Write')
            lines.append(f'cv2.imwrite(r"{fp}", {src})')
            lines.append(f'{out} = {src}')
            lines.append('')

        elif ntype == 'video_write':
            fp = props.get('filepath', 'output.mp4')
            codec = props.get('codec', 'mp4v')
            fps = int(props.get('fps', 30))
            lines.append(f'# Video Write')
            lines.append(f'if "_vw_{nid}" not in dir():')
            lines.append(f'    _h, _w = {src}.shape[:2]')
            lines.append(f'    _vw_{nid} = cv2.VideoWriter(r"{fp}", cv2.VideoWriter_fourcc(*"{codec}"), {fps}, (_w, _h))')
            lines.append(f'_vw_frame = {src}')
            lines.append(f'if len(_vw_frame.shape) == 2: _vw_frame = cv2.cvtColor(_vw_frame, cv2.COLOR_GRAY2BGR)')
            lines.append(f'_vw_{nid}.write(_vw_frame)')
            lines.append(f'{out} = {src}')
            lines.append('')

        elif ntype == 'video_read':
            fp = props.get('filepath', '')
            mode = props.get('mode', 'single')
            if mode == 'loop':
                start_f = int(props.get('startFrame', 0))
                end_f = int(props.get('endFrame', -1))
                step_f = max(1, int(props.get('step', 1)))
                lines.append(f'# Video Read (loop mode)')
                lines.append(f'_cap = cv2.VideoCapture(r"{fp}")')
                lines.append(f'_total_frames = int(_cap.get(cv2.CAP_PROP_FRAME_COUNT))')
                lines.append(f'_end = _total_frames if {end_f} < 0 else min({end_f} + 1, _total_frames)')
                lines.append(f'for _fidx in range({start_f}, _end, {step_f}):')
                lines.append(f'    _cap.set(cv2.CAP_PROP_POS_FRAMES, _fidx)')
                lines.append(f'    _ret, {out} = _cap.read()')
                lines.append(f'    if not _ret: break')
                # Mark that subsequent code should be indented (inside loop)
                lines.append(f'    # --- Process frame _fidx below ---')
                lines.append('')
            else:
                frame_num = int(props.get('frameIndex', 0))
                lines.append(f'# Video Read')
                lines.append(f'_cap = cv2.VideoCapture(r"{fp}")')
                if frame_num > 0:
                    lines.append(f'_cap.set(cv2.CAP_PROP_POS_FRAMES, {frame_num})')
                lines.append(f'_ret, {out} = _cap.read()')
                lines.append(f'_cap.release()')
                lines.append(f'if not _ret: {out} = np.zeros((480, 640, 3), dtype=np.uint8)')
                lines.append('')

        elif ntype == 'camera_capture':
            cam_idx = int(props.get('cameraIndex', 0))
            lines.append(f'# Camera Capture')
            lines.append(f'_cap = cv2.VideoCapture({cam_idx})')
            lines.append(f'_ret, {out} = _cap.read()')
            lines.append(f'_cap.release()')
            lines.append(f'if not _ret: {out} = np.zeros((480, 640, 3), dtype=np.uint8)')
            lines.append('')

        # ---- Color (new) ----
        elif ntype == 'split_channels':
            ch0 = var_name(nid, '_ch0')
            ch1 = var_name(nid, '_ch1')
            ch2 = var_name(nid, '_ch2')
            lines.append(f'# Split Channels')
            lines.append(f'if len({src}.shape) == 3 and {src}.shape[2] >= 3:')
            lines.append(f'    {ch0}, {ch1}, {ch2} = cv2.split({src})')
            lines.append(f'else:')
            lines.append(f'    {ch0} = {ch1} = {ch2} = {src}')
            lines.append(f'{out} = {src}')
            lines.append('')

        elif ntype == 'merge_channels':
            src_ch0 = get_src_var(nid, 'ch0')
            src_ch1 = get_src_var(nid, 'ch1')
            src_ch2 = get_src_var(nid, 'ch2')
            lines.append(f'# Merge Channels')
            lines.append(f'{out} = cv2.merge([{src_ch0}, {src_ch1}, {src_ch2}])')
            lines.append('')

        # ---- Filter (new) ----
        elif ntype == 'box_filter':
            k = int(props.get('ksize', 5))
            lines.append(f'# Box Filter')
            lines.append(f'{out} = cv2.boxFilter({src}, -1, ({k}, {k}))')
            lines.append('')

        elif ntype == 'sharpen':
            strength = float(props.get('strength', 1.0))
            lines.append(f'# Sharpen')
            lines.append(f'_blurred = cv2.GaussianBlur({src}, (0, 0), 3)')
            lines.append(f'{out} = cv2.addWeighted({src}, 1.0 + {strength}, _blurred, -{strength}, 0)')
            lines.append('')

        elif ntype == 'filter2d':
            preset = props.get('preset', 'sharpen')
            lines.append(f'# Filter2D (preset: {preset})')
            if preset == 'sharpen':
                lines.append(f'_kernel = np.array([[0,-1,0],[-1,5,-1],[0,-1,0]], dtype=np.float32)')
            elif preset == 'edge_detect':
                lines.append(f'_kernel = np.array([[-1,-1,-1],[-1,8,-1],[-1,-1,-1]], dtype=np.float32)')
            elif preset == 'emboss':
                lines.append(f'_kernel = np.array([[-2,-1,0],[-1,1,1],[0,1,2]], dtype=np.float32)')
            elif preset == 'ridge':
                lines.append(f'_kernel = np.array([[-1,-1,-1],[-1,4,-1],[-1,-1,-1]], dtype=np.float32)')
            else:
                lines.append(f'_kernel = np.array([[0,0,0],[0,1,0],[0,0,0]], dtype=np.float32)')
            lines.append(f'{out} = cv2.filter2D({src}, -1, _kernel)')
            lines.append('')

        # ---- Edge (new) ----
        elif ntype == 'scharr':
            dx = int(props.get('dx', 1))
            dy = int(props.get('dy', 0))
            lines.append(f'# Scharr')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'{out} = cv2.Scharr(_gray, cv2.CV_64F, {dx}, {dy})')
            lines.append(f'{out} = cv2.convertScaleAbs({out})')
            lines.append(f'{out} = cv2.cvtColor({out}, cv2.COLOR_GRAY2BGR)')
            lines.append('')

        # ---- Threshold (new) ----
        elif ntype == 'otsu_threshold':
            mv = float(props.get('maxval', 255))
            lines.append(f'# Otsu Threshold')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_, {out} = cv2.threshold(_gray, 0, {mv}, cv2.THRESH_BINARY + cv2.THRESH_OTSU)')
            lines.append(f'{out} = cv2.cvtColor({out}, cv2.COLOR_GRAY2BGR)')
            lines.append('')

        # ---- Morphology (new) ----
        elif ntype == 'structuring_element':
            shape = props.get('shape', 'MORPH_RECT')
            k = int(props.get('ksize', 5))
            lines.append(f'# Structuring Element')
            lines.append(f'_kernel = cv2.getStructuringElement(cv2.{shape}, ({k}, {k}))')
            lines.append(f'{out} = (_kernel * 255).astype(np.uint8)')
            lines.append(f'{out} = cv2.cvtColor({out}, cv2.COLOR_GRAY2BGR)')
            lines.append('')

        # ---- Contour (new) ----
        elif ntype == 'draw_contours':
            src_contours = get_src_var(nid, 'contours')
            contour_idx = int(props.get('contourIdx', -1))
            thickness = int(props.get('thickness', 2))
            r = int(props.get('colorR', 0))
            g = int(props.get('colorG', 255))
            b = int(props.get('colorB', 0))
            lines.append(f'# Draw Contours')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'cv2.drawContours({out}, {src_contours}, {contour_idx}, ({b},{g},{r}), {thickness})')
            lines.append('')

        elif ntype == 'bounding_rect':
            src_contours = get_src_var(nid, 'contours')
            thickness = int(props.get('thickness', 2))
            r = int(props.get('colorR', 0))
            g = int(props.get('colorG', 255))
            b = int(props.get('colorB', 0))
            lines.append(f'# Bounding Rect')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'for _c in {src_contours}:')
            lines.append(f'    _x, _y, _w, _h = cv2.boundingRect(_c)')
            lines.append(f'    cv2.rectangle({out}, (_x, _y), (_x+_w, _y+_h), ({b},{g},{r}), {thickness})')
            lines.append('')

        elif ntype == 'min_enclosing_circle':
            src_contours = get_src_var(nid, 'contours')
            thickness = int(props.get('thickness', 2))
            r = int(props.get('colorR', 0))
            g = int(props.get('colorG', 255))
            b = int(props.get('colorB', 0))
            lines.append(f'# Min Enclosing Circle')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'for _c in {src_contours}:')
            lines.append(f'    if len(_c) >= 5:')
            lines.append(f'        (_cx, _cy), _r = cv2.minEnclosingCircle(_c)')
            lines.append(f'        cv2.circle({out}, (int(_cx), int(_cy)), int(_r), ({b},{g},{r}), {thickness})')
            lines.append('')

        elif ntype == 'convex_hull':
            src_contours = get_src_var(nid, 'contours')
            thickness = int(props.get('thickness', 2))
            r = int(props.get('colorR', 0))
            g = int(props.get('colorG', 255))
            b = int(props.get('colorB', 0))
            lines.append(f'# Convex Hull')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'_hulls = [cv2.convexHull(_c) for _c in {src_contours}]')
            lines.append(f'cv2.drawContours({out}, _hulls, -1, ({b},{g},{r}), {thickness})')
            lines.append('')

        elif ntype == 'approx_poly':
            src_contours = get_src_var(nid, 'contours')
            epsilon_pct = float(props.get('epsilon', 0.02))
            thickness = int(props.get('thickness', 2))
            r = int(props.get('colorR', 0))
            g = int(props.get('colorG', 255))
            b = int(props.get('colorB', 0))
            lines.append(f'# Approx Poly')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'for _c in {src_contours}:')
            lines.append(f'    _eps = {epsilon_pct} * cv2.arcLength(_c, True)')
            lines.append(f'    _approx = cv2.approxPolyDP(_c, _eps, True)')
            lines.append(f'    cv2.drawContours({out}, [_approx], -1, ({b},{g},{r}), {thickness})')
            lines.append('')

        elif ntype == 'contour_area':
            src_contours = get_src_var(nid, 'contours')
            min_area = float(props.get('minArea', 100))
            max_area = float(props.get('maxArea', 100000))
            thickness = int(props.get('thickness', 2))
            r = int(props.get('colorR', 0))
            g = int(props.get('colorG', 255))
            b = int(props.get('colorB', 0))
            lines.append(f'# Contour Area Filter')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'_filtered = [_c for _c in {src_contours} if {min_area} <= cv2.contourArea(_c) <= {max_area}]')
            lines.append(f'cv2.drawContours({out}, _filtered, -1, ({b},{g},{r}), {thickness})')
            lines.append('')

        elif ntype == 'contour_properties':
            src_contours = get_src_var(nid, 'contours')
            lines.append(f'# Contour Properties')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'for _i, _c in enumerate({src_contours}):')
            lines.append(f'    _area = cv2.contourArea(_c)')
            lines.append(f'    _peri = cv2.arcLength(_c, True)')
            lines.append(f'    _x, _y, _w, _h = cv2.boundingRect(_c)')
            lines.append(f'    cv2.putText({out}, f"A:{{int(_area)}}", (_x, _y-5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,0), 1)')
            lines.append(f'    cv2.rectangle({out}, (_x, _y), (_x+_w, _y+_h), (0, 255, 0), 1)')
            lines.append('')

        # ---- Feature (new) ----
        elif ntype == 'harris_corner':
            block_size = int(props.get('blockSize', 2))
            k_size = int(props.get('ksize', 3))
            k_param = float(props.get('k', 0.04))
            thresh = float(props.get('threshold', 0.01))
            lines.append(f'# Harris Corner')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_gray32 = np.float32(_gray)')
            lines.append(f'_dst = cv2.cornerHarris(_gray32, {block_size}, {k_size}, {k_param})')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'{out}[_dst > {thresh} * _dst.max()] = [0, 0, 255]')
            lines.append('')

        elif ntype == 'good_features':
            max_corners = int(props.get('maxCorners', 100))
            quality = float(props.get('qualityLevel', 0.01))
            min_dist = float(props.get('minDistance', 10))
            lines.append(f'# Good Features to Track')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_corners = cv2.goodFeaturesToTrack(_gray, {max_corners}, {quality}, {min_dist})')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'if _corners is not None:')
            lines.append(f'    for _pt in _corners:')
            lines.append(f'        _x, _y = _pt.ravel()')
            lines.append(f'        cv2.circle({out}, (int(_x), int(_y)), 5, (0, 255, 0), -1)')
            lines.append('')

        elif ntype == 'orb_features':
            n_features = int(props.get('nFeatures', 500))
            lines.append(f'# ORB Features')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_orb = cv2.ORB_create(nfeatures={n_features})')
            lines.append(f'_kp = _orb.detect(_gray, None)')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'{out} = cv2.drawKeypoints({out}, _kp, None, color=(0, 255, 0))')
            lines.append('')

        elif ntype == 'fast_features':
            thresh_val = int(props.get('threshold', 25))
            nms = props.get('nonmaxSuppression', True)
            nms_bool = 'True' if nms else 'False'
            lines.append(f'# FAST Features')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_fast = cv2.FastFeatureDetector_create(threshold={thresh_val}, nonmaxSuppression={nms_bool})')
            lines.append(f'_kp = _fast.detect(_gray, None)')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'{out} = cv2.drawKeypoints({out}, _kp, None, color=(0, 255, 0))')
            lines.append('')

        elif ntype == 'match_features':
            n_features = int(props.get('nFeatures', 500))
            ratio = float(props.get('ratioThreshold', 0.75))
            lines.append(f'# Match Features (ORB + BFMatcher)')
            lines.append(f'_gray1 = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_gray2 = cv2.cvtColor({src2}, cv2.COLOR_BGR2GRAY) if len({src2}.shape) == 3 else {src2}')
            lines.append(f'_orb = cv2.ORB_create(nfeatures={n_features})')
            lines.append(f'_kp1, _des1 = _orb.detectAndCompute(_gray1, None)')
            lines.append(f'_kp2, _des2 = _orb.detectAndCompute(_gray2, None)')
            lines.append(f'_bf = cv2.BFMatcher(cv2.NORM_HAMMING)')
            lines.append(f'_matches = _bf.knnMatch(_des1, _des2, k=2) if _des1 is not None and _des2 is not None else []')
            lines.append(f'_good = [m for m, n in _matches if m.distance < {ratio} * n.distance] if _matches else []')
            lines.append(f'{out} = cv2.drawMatches({src}, _kp1, {src2}, _kp2, _good, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)')
            lines.append('')

        # ---- Drawing (new) ----
        elif ntype == 'draw_line':
            x1 = int(props.get('x1', 0))
            y1 = int(props.get('y1', 0))
            x2 = int(props.get('x2', 100))
            y2 = int(props.get('y2', 100))
            color = props.get('color', '0,255,0')
            thickness = int(props.get('thickness', 2))
            cp = [int(x.strip()) for x in color.split(',')]
            lines.append(f'# Draw Line')
            lines.append(f'{out} = {src}.copy()')
            lines.append(f'cv2.line({out}, ({x1}, {y1}), ({x2}, {y2}), ({cp[0]}, {cp[1]}, {cp[2]}), {thickness})')
            lines.append('')

        elif ntype == 'draw_rectangle':
            x1 = int(props.get('x1', 10))
            y1 = int(props.get('y1', 10))
            x2 = int(props.get('x2', 200))
            y2 = int(props.get('y2', 200))
            color = props.get('color', '0,255,0')
            thickness = int(props.get('thickness', 2))
            cp = [int(x.strip()) for x in color.split(',')]
            lines.append(f'# Draw Rectangle')
            lines.append(f'{out} = {src}.copy()')
            lines.append(f'cv2.rectangle({out}, ({x1}, {y1}), ({x2}, {y2}), ({cp[0]}, {cp[1]}, {cp[2]}), {thickness})')
            lines.append('')

        elif ntype == 'draw_circle':
            cx = int(props.get('centerX', 100))
            cy = int(props.get('centerY', 100))
            r = int(props.get('radius', 50))
            color = props.get('color', '0,255,0')
            thickness = int(props.get('thickness', 2))
            cp = [int(x.strip()) for x in color.split(',')]
            lines.append(f'# Draw Circle')
            lines.append(f'{out} = {src}.copy()')
            lines.append(f'cv2.circle({out}, ({cx}, {cy}), {r}, ({cp[0]}, {cp[1]}, {cp[2]}), {thickness})')
            lines.append('')

        elif ntype == 'draw_ellipse':
            cx = int(props.get('centerX', 100))
            cy = int(props.get('centerY', 100))
            ax1 = int(props.get('axes1', 80))
            ax2 = int(props.get('axes2', 50))
            angle = float(props.get('angle', 0))
            color = props.get('color', '0,255,0')
            thickness = int(props.get('thickness', 2))
            cp = [int(x.strip()) for x in color.split(',')]
            lines.append(f'# Draw Ellipse')
            lines.append(f'{out} = {src}.copy()')
            lines.append(f'cv2.ellipse({out}, ({cx}, {cy}), ({ax1}, {ax2}), {angle}, 0, 360, ({cp[0]}, {cp[1]}, {cp[2]}), {thickness})')
            lines.append('')

        elif ntype == 'draw_text':
            text = props.get('text', 'Hello')
            x = int(props.get('x', 10))
            y = int(props.get('y', 30))
            font_scale = float(props.get('fontScale', 1.0))
            color = props.get('color', '255,255,255')
            thickness = int(props.get('thickness', 2))
            cp = [int(c.strip()) for c in color.split(',')]
            lines.append(f'# Draw Text')
            lines.append(f'{out} = {src}.copy()')
            lines.append(f'cv2.putText({out}, "{text}", ({x}, {y}), cv2.FONT_HERSHEY_SIMPLEX, {font_scale}, ({cp[0]}, {cp[1]}, {cp[2]}), {thickness})')
            lines.append('')

        elif ntype == 'draw_polylines':
            pts_str = props.get('points', '10,10;100,10;100,100;10,100')
            closed = props.get('closed', True)
            color = props.get('color', '0,255,0')
            thickness = int(props.get('thickness', 2))
            cp = [int(c.strip()) for c in color.split(',')]
            closed_str = 'True' if closed else 'False'
            lines.append(f'# Draw Polylines')
            lines.append(f'{out} = {src}.copy()')
            lines.append(f'_pts = np.array([{", ".join("[" + p.strip() + "]" for p in pts_str.split(";"))}], np.int32).reshape((-1, 1, 2))')
            lines.append(f'cv2.polylines({out}, [_pts], {closed_str}, ({cp[0]}, {cp[1]}, {cp[2]}), {thickness})')
            lines.append('')

        # ---- Transform (new) ----
        elif ntype == 'flip':
            flip_code = int(props.get('flipCode', 1))
            lines.append(f'# Flip')
            lines.append(f'{out} = cv2.flip({src}, {flip_code})')
            lines.append('')

        elif ntype == 'crop':
            x = int(props.get('x', 0))
            y = int(props.get('y', 0))
            w = int(props.get('width', 100))
            h = int(props.get('height', 100))
            lines.append(f'# Crop')
            lines.append(f'{out} = {src}[{y}:{y}+{h}, {x}:{x}+{w}].copy()')
            lines.append('')

        elif ntype == 'warp_affine':
            lines.append(f'# Warp Affine')
            lines.append(f'_h, _w = {src}.shape[:2]')
            lines.append(f'_pts1 = np.float32([[0,0],[_w,0],[0,_h]])')
            lines.append(f'_pts2 = np.float32([[0,0],[_w,0],[int(_w*0.15),_h]])')
            lines.append(f'_M = cv2.getAffineTransform(_pts1, _pts2)')
            lines.append(f'{out} = cv2.warpAffine({src}, _M, (_w, _h))')
            lines.append('')

        elif ntype == 'warp_perspective':
            lines.append(f'# Warp Perspective')
            lines.append(f'_h, _w = {src}.shape[:2]')
            lines.append(f'_pts1 = np.float32([[0,0],[_w,0],[0,_h],[_w,_h]])')
            lines.append(f'_pts2 = np.float32([[int(_w*0.1),int(_h*0.1)],[int(_w*0.9),0],[int(_w*0.05),_h],[int(_w*0.95),int(_h*0.9)]])')
            lines.append(f'_M = cv2.getPerspectiveTransform(_pts1, _pts2)')
            lines.append(f'{out} = cv2.warpPerspective({src}, _M, (_w, _h))')
            lines.append('')

        elif ntype == 'remap':
            mode = props.get('mode', 'barrel')
            lines.append(f'# Remap ({mode})')
            lines.append(f'_h, _w = {src}.shape[:2]')
            lines.append(f'_map_x = np.zeros((_h, _w), np.float32)')
            lines.append(f'_map_y = np.zeros((_h, _w), np.float32)')
            lines.append(f'for _j in range(_h):')
            lines.append(f'    for _i in range(_w):')
            if mode == 'barrel':
                lines.append(f'        _nx, _ny = 2.0*_i/_w - 1.0, 2.0*_j/_h - 1.0')
                lines.append(f'        _r = np.sqrt(_nx*_nx + _ny*_ny)')
                lines.append(f'        _theta = 1.0 + 0.5*_r*_r')
                lines.append(f'        _map_x[_j, _i] = (_nx*_theta + 1.0)*_w/2.0')
                lines.append(f'        _map_y[_j, _i] = (_ny*_theta + 1.0)*_h/2.0')
            elif mode == 'flip_h':
                lines.append(f'        _map_x[_j, _i] = _w - _i')
                lines.append(f'        _map_y[_j, _i] = _j')
            else:
                lines.append(f'        _map_x[_j, _i] = _i')
                lines.append(f'        _map_y[_j, _i] = _h - _j')
            lines.append(f'{out} = cv2.remap({src}, _map_x, _map_y, cv2.INTER_LINEAR)')
            lines.append('')

        # ---- Histogram (new) ----
        elif ntype == 'calc_histogram':
            channel = int(props.get('channel', 0))
            bins = int(props.get('bins', 256))
            lines.append(f'# Calc Histogram')
            lines.append(f'_hist = cv2.calcHist([{src}], [{channel}], None, [{bins}], [0, 256])')
            lines.append(f'{out} = np.zeros((300, {bins}, 3), dtype=np.uint8)')
            lines.append(f'cv2.normalize(_hist, _hist, 0, 300, cv2.NORM_MINMAX)')
            lines.append(f'for _i in range({bins}):')
            lines.append(f'    cv2.line({out}, (_i, 300), (_i, 300 - int(_hist[_i])), (255, 255, 255), 1)')
            lines.append('')

        # ---- Arithmetic (new) ----
        elif ntype == 'add':
            lines.append(f'# Add')
            lines.append(f'{out} = cv2.add({src}, {src2})')
            lines.append('')

        elif ntype == 'subtract':
            lines.append(f'# Subtract')
            lines.append(f'{out} = cv2.subtract({src}, {src2})')
            lines.append('')

        elif ntype == 'multiply':
            scale = float(props.get('scale', 1.0))
            lines.append(f'# Multiply')
            lines.append(f'{out} = cv2.multiply({src}, {src2}, scale={scale})')
            lines.append('')

        elif ntype == 'absdiff':
            lines.append(f'# Absolute Difference')
            lines.append(f'{out} = cv2.absdiff({src}, {src2})')
            lines.append('')

        elif ntype == 'bitwise_xor':
            src_mask = get_src_var(nid, 'mask')
            lines.append(f'# Bitwise XOR')
            if src_mask != 'None':
                lines.append(f'_mask = cv2.cvtColor({src_mask}, cv2.COLOR_BGR2GRAY) if len({src_mask}.shape) == 3 else {src_mask}')
                lines.append(f'{out} = cv2.bitwise_xor({src}, {src2}, mask=_mask)')
            else:
                lines.append(f'{out} = cv2.bitwise_xor({src}, {src2})')
            lines.append('')

        # ---- Detection (new) ----
        elif ntype == 'haar_cascade':
            cascade = props.get('cascade', 'haarcascade_frontalface_default')
            scale = float(props.get('scaleFactor', 1.1))
            min_n = int(props.get('minNeighbors', 5))
            lines.append(f'# Haar Cascade')
            lines.append(f'_cascade_path = cv2.data.haarcascades + "{cascade}.xml"')
            lines.append(f'_cascade = cv2.CascadeClassifier(_cascade_path)')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_rects = _cascade.detectMultiScale(_gray, scaleFactor={scale}, minNeighbors={min_n})')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'for (_x, _y, _w, _h) in _rects:')
            lines.append(f'    cv2.rectangle({out}, (_x, _y), (_x+_w, _y+_h), (0, 255, 0), 2)')
            lines.append('')

        elif ntype == 'hough_circles':
            dp = float(props.get('dp', 1.2))
            min_dist = float(props.get('minDist', 50))
            p1 = float(props.get('param1', 100))
            p2 = float(props.get('param2', 30))
            lines.append(f'# Hough Circles')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_gray = cv2.medianBlur(_gray, 5)')
            lines.append(f'_circles = cv2.HoughCircles(_gray, cv2.HOUGH_GRADIENT, {dp}, {min_dist}, param1={p1}, param2={p2})')
            lines.append(f'{out} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'if _circles is not None:')
            lines.append(f'    _circles = np.uint16(np.around(_circles))')
            lines.append(f'    for _c in _circles[0]:')
            lines.append(f'        cv2.circle({out}, (_c[0], _c[1]), _c[2], (0, 255, 0), 2)')
            lines.append(f'        cv2.circle({out}, (_c[0], _c[1]), 2, (0, 0, 255), 3)')
            lines.append('')

        elif ntype == 'template_match':
            method = props.get('method', 'TM_CCOEFF_NORMED')
            thresh = float(props.get('threshold', 0.8))
            out_img = var_name(nid, '_image')
            out_matches = var_name(nid, '_matches')
            lines.append(f'# Template Match')
            lines.append(f'_gray_src = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_gray_tpl = cv2.cvtColor({src2}, cv2.COLOR_BGR2GRAY) if len({src2}.shape) == 3 else {src2}')
            lines.append(f'_result = cv2.matchTemplate(_gray_src, _gray_tpl, cv2.{method})')
            if method.startswith('TM_SQDIFF'):
                lines.append(f'_loc = np.where(_result <= {1 - thresh})')
            else:
                lines.append(f'_loc = np.where(_result >= {thresh})')
            lines.append(f'_th, _tw = _gray_tpl.shape[:2]')
            lines.append(f'{out_img} = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'{out_matches} = []')
            lines.append(f'for _pt in zip(*_loc[::-1]):')
            lines.append(f'    _x1, _y1 = int(_pt[0]), int(_pt[1])')
            lines.append(f'    {out_matches}.append([_x1, _y1, _x1 + _tw, _y1 + _th])')
            lines.append(f'    cv2.rectangle({out_img}, (_x1, _y1), (_x1 + _tw, _y1 + _th), (0, 255, 0), 2)')
            lines.append('')

        # ---- Segmentation (new) ----
        elif ntype == 'flood_fill':
            x = int(props.get('seedX', 0))
            y = int(props.get('seedY', 0))
            lo = int(props.get('loDiff', 20))
            hi = int(props.get('upDiff', 20))
            lines.append(f'# Flood Fill')
            lines.append(f'{out} = {src}.copy()')
            lines.append(f'_mask = np.zeros(({out}.shape[0]+2, {out}.shape[1]+2), np.uint8)')
            lines.append(f'cv2.floodFill({out}, _mask, ({x}, {y}), (0, 255, 0), ({lo},{lo},{lo}), ({hi},{hi},{hi}))')
            lines.append('')

        elif ntype == 'grabcut':
            iterations = int(props.get('iterations', 5))
            margin = int(props.get('margin', 10))
            lines.append(f'# GrabCut')
            lines.append(f'_mask = np.zeros({src}.shape[:2], np.uint8)')
            lines.append(f'_bgdModel = np.zeros((1, 65), np.float64)')
            lines.append(f'_fgdModel = np.zeros((1, 65), np.float64)')
            lines.append(f'_h, _w = {src}.shape[:2]')
            lines.append(f'_rect = ({margin}, {margin}, _w - 2*{margin}, _h - 2*{margin})')
            lines.append(f'cv2.grabCut({src}, _mask, _rect, _bgdModel, _fgdModel, {iterations}, cv2.GC_INIT_WITH_RECT)')
            lines.append(f'_mask2 = np.where((_mask == 2) | (_mask == 0), 0, 1).astype("uint8")')
            lines.append(f'{out} = {src} * _mask2[:, :, np.newaxis]')
            lines.append('')

        elif ntype == 'watershed':
            lines.append(f'# Watershed')
            lines.append(f'_gray = cv2.cvtColor({src}, cv2.COLOR_BGR2GRAY) if len({src}.shape) == 3 else {src}')
            lines.append(f'_, _thresh = cv2.threshold(_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)')
            lines.append(f'_kernel = np.ones((3, 3), np.uint8)')
            lines.append(f'_opening = cv2.morphologyEx(_thresh, cv2.MORPH_OPEN, _kernel, iterations=2)')
            lines.append(f'_sure_bg = cv2.dilate(_opening, _kernel, iterations=3)')
            lines.append(f'_dist = cv2.distanceTransform(_opening, cv2.DIST_L2, 5)')
            lines.append(f'_, _sure_fg = cv2.threshold(_dist, 0.5 * _dist.max(), 255, 0)')
            lines.append(f'_sure_fg = np.uint8(_sure_fg)')
            lines.append(f'_unknown = cv2.subtract(_sure_bg, _sure_fg)')
            lines.append(f'_, _markers = cv2.connectedComponents(_sure_fg)')
            lines.append(f'_markers = _markers + 1')
            lines.append(f'_markers[_unknown == 255] = 0')
            lines.append(f'_img3 = {src}.copy() if len({src}.shape) == 3 else cv2.cvtColor({src}, cv2.COLOR_GRAY2BGR)')
            lines.append(f'cv2.watershed(_img3, _markers)')
            lines.append(f'{out} = _img3.copy()')
            lines.append(f'{out}[_markers == -1] = [0, 0, 255]')
            lines.append('')

        # ---- Value (new) ----
        elif ntype == 'val_integer':
            val = int(props.get('value', 0))
            lines.append(f'# Integer Value')
            lines.append(f'{out} = {val}')
            lines.append('')

        elif ntype == 'val_float':
            val = float(props.get('value', 0.0))
            lines.append(f'# Float Value')
            lines.append(f'{out} = {val}')
            lines.append('')

        elif ntype == 'val_boolean':
            val = props.get('value', True)
            lines.append(f'# Boolean Value')
            lines.append(f'{out} = {val}')
            lines.append('')

        elif ntype == 'val_point':
            x = int(props.get('x', 0))
            y = int(props.get('y', 0))
            lines.append(f'# Point Value')
            lines.append(f'{out} = ({x}, {y})')
            lines.append('')

        elif ntype == 'val_scalar':
            v0 = float(props.get('v0', 0))
            v1 = float(props.get('v1', 0))
            v2 = float(props.get('v2', 0))
            v3 = float(props.get('v3', 0))
            lines.append(f'# Scalar Value')
            lines.append(f'{out} = ({v0}, {v1}, {v2}, {v3})')
            lines.append('')

        elif ntype == 'val_math':
            op = props.get('operation', 'add')
            src_a = get_src_var(nid, 'a')
            src_b = get_src_var(nid, 'b')
            lines.append(f'# Math Operation ({op})')
            op_map = {'add': '+', 'subtract': '-', 'multiply': '*', 'divide': '/', 'modulo': '%', 'power': '**'}
            if op in op_map:
                lines.append(f'{out} = {src_a} {op_map[op]} {src_b}')
            elif op == 'min':
                lines.append(f'{out} = min({src_a}, {src_b})')
            elif op == 'max':
                lines.append(f'{out} = max({src_a}, {src_b})')
            else:
                lines.append(f'{out} = {src_a} + {src_b}')
            lines.append('')

        elif ntype == 'val_list':
            mode = props.get('mode', 'index')
            src_list = get_src_var(nid, 'list')
            if mode == 'index':
                idx = int(props.get('index', 0))
                lines.append(f'# List Index')
                lines.append(f'{out} = {src_list}[{idx}]')
            else:
                start = int(props.get('start', 0))
                stop = int(props.get('stop', -1))
                step = int(props.get('step', 1))
                stop_str = '' if stop == -1 else str(stop)
                step_str = '' if step == 1 else f':{step}'
                lines.append(f'# List Slice')
                lines.append(f'{out} = {src_list}[{start}:{stop_str}{step_str}]')
            lines.append('')

        elif ntype == 'image_extract':
            src_coords = get_src_var(nid, 'coords')
            padding = int(props.get('padding', 0))
            lines.append(f'# Image Extract')
            lines.append(f'_coords = {src_coords}')
            lines.append(f'if isinstance(_coords, list) and len(_coords) > 0 and isinstance(_coords[0], list):')
            lines.append(f'    _coords = _coords[0]')
            lines.append(f'_x1 = max(0, int(_coords[0]) - {padding})')
            lines.append(f'_y1 = max(0, int(_coords[1]) - {padding})')
            lines.append(f'_x2 = min({src}.shape[1], int(_coords[2]) + {padding})')
            lines.append(f'_y2 = min({src}.shape[0], int(_coords[3]) + {padding})')
            lines.append(f'{out} = {src}[_y1:_y2, _x1:_x2].copy()')
            lines.append('')

        else:
            lines.append(f'# {ntype} (unsupported for code generation)')
            lines.append(f'{out} = {src}')
            lines.append('')

    lines.append('')
    lines.append('# Display result')
    lines.append('cv2.waitKey(0)')
    lines.append('cv2.destroyAllWindows()')

    return '\n'.join(lines)


@app.route('/api/generate_code', methods=['POST'])
def api_generate_code():
    """Generate Python code from the node graph."""
    data = request.json
    nodes = data.get('nodes', [])
    connections = data.get('connections', [])
    if not nodes:
        return jsonify({'code': '# Empty pipeline - add some nodes first\n'})
    try:
        code = generate_python_code(nodes, connections)
        return jsonify({'code': code})
    except Exception as e:
        return jsonify({'code': f'# Code generation error: {e}\n', 'error': str(e)})


# ---- Web-based Script Editor Support ----

@app.route('/api/script/save', methods=['POST'])
def save_script():
    """Save script content from the web-based editor."""
    data = request.json
    node_id = data.get('nodeId', 'unknown')
    script = data.get('script', '')

    filename = f'node_{node_id}.py'
    filepath = os.path.join(g.session.script_folder, filename)

    with open(filepath, 'w', encoding='utf-8') as f:
        f.write(script)

    g.session.script_sessions[node_id] = {
        'path': filepath,
        'node_id': node_id,
    }

    return jsonify({'status': 'saved', 'filepath': filepath})


@app.route('/api/session/info')
def session_info():
    """Return current session info."""
    return jsonify({
        'sessionId': g.session.sid,
        'created': g.session.created_at,
        'imageCount': len(g.session.image_store),
    })


@app.route('/api/session/list')
def session_list():
    """Return list of active sessions (for teacher monitoring)."""
    with _sessions_lock:
        sessions = []
        for sid, s in _sessions.items():
            sessions.append({
                'sessionId': sid,
                'imageCount': len(s.image_store),
                'lastAccess': s.last_access,
                'age': round(time.time() - s.created_at),
            })
    return jsonify(sessions)


if __name__ == '__main__':
    import socket
    hostname = socket.gethostname()
    local_ip = socket.gethostbyname(hostname)
    print("=" * 50)
    print("  SamOpenCVWeb - Visual OpenCV Editor")
    print("  Multi-User Classroom Mode")
    print("=" * 50)
    print(f"  Local:   http://localhost:5000")
    print(f"  Network: http://{local_ip}:5000")
    print(f"  Students connect via the Network URL above")
    print("=" * 50)
    try:
        from waitress import serve
        serve(app, host='0.0.0.0', port=5000, threads=8)
    except ImportError:
        print("  [!] waitress not installed, using Flask dev server")
        print("  [!] Run: pip install waitress")
        app.run(debug=True, host='0.0.0.0', port=5000)
