# SamOpenCVWeb

Node-RED style visual programming tool for OpenCV image processing.
Drag nodes, connect wires, execute pipelines.

## Getting Started

### Installation

```bash
pip install -r requirements.txt
python app.py
```

### Web Browser

```
http://localhost:5000
```

---

## Quick Start

### 1. Basic Workflow

1. **Add Nodes** - Drag nodes from the left Node Palette onto the canvas.
2. **Load Image** - Click an *Image Read* node, then use the file upload in Properties (right panel).
3. **Connect Nodes** - Drag from an output port (right side) to an input port (left side) of another node.
4. **Set Properties** - Click a node to see/edit its parameters in the Properties panel.
5. **Execute** - Click *Execute* (or `Ctrl+Enter`) to run the full pipeline.
6. **View Results** - Preview panel shows each node's result. *Image Show* nodes open a popup automatically.

### 2. Ports & Connections

- **Input ports** (left side of node) receive data from other nodes.
- **Output ports** (right side of node) send data to other nodes.
- A filled port circle means it is connected.
- Click a connection wire to delete it.
- Each input port accepts only one connection.

### 3. Preview & Image Show

- Click a node, then click *Preview* to see its output without running the full pipeline.
- Double-click the preview image to enlarge.
- *Image Show* nodes automatically open a full-screen popup when executed.
- *Image Show* also has an output port, so you can chain after it.

### 4. Python Script Node

- Write custom OpenCV code using `img_input` (input image) and `img_output` (result image).
- `cv2` and `np` (NumPy) are available.
- Use *VS Code* or *Notepad* buttons to open the script in an external editor.
- After editing externally, click *Reload* to apply changes.

```python
# Example: Convert to grayscale and apply edge detection
gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
img_output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
```

### 5. Control Flow Nodes

- **If**: Evaluates a condition and routes the image to the *true* or *false* output port.
- **For Loop**: Applies an operation N times (e.g., repeated blur or erosion).
- **While Loop**: Repeats an operation while a condition is true (with safety max iterations).
- **Switch-Case**: Routes the image to one of 3 output ports based on a condition (channels, depth, size, etc.).

### 6. Code Generation

- Click *Code* in the toolbar to convert the current node graph into standalone Python code.
- The generated code uses only `cv2` and `numpy`.
- Use *Copy* to copy to clipboard or *Download* to save as .py file.

### 7. Save & Load

- **Save**: ZIP project file (flow.json + images/ folder) when images are present. JSON when no images.
- **Load**: ZIP project file or JSON file. ZIP restores images and previews automatically.

### 8. Panel Toggle

- Use the arrow buttons on the palette/properties headers to collapse/expand panels.
- Collapse panels to get a wider canvas workspace.

### Tips & Tricks

| Feature | Description |
|---------|-------------|
| Double-click node | File nodes (Image Read, Video Read): opens file dialog. Other nodes: refreshes preview immediately. |
| Double-click port | Unconnected port auto-connects to the nearest compatible port on another node. |
| Drag near another node | When a port comes within 1 grid distance of another port, they auto-connect. |
| Property change | Preview auto-updates 0.5s after you change any property value. |
| Ctrl + C / V | Copy nodes with all their parameter settings. Paste creates new instances. |
| Ctrl + D | Duplicate selected nodes instantly. |
| Rubber-band select | Drag on empty canvas to select multiple nodes at once. |
| Middle-click drag | Pan the canvas. Or hold Alt + left-click drag. |
| Scroll wheel | Zoom in/out on the canvas. |
| Code Generation | Click *Code* to convert your entire node graph to a standalone Python script. |
| Save / Load | Save as ZIP project (images included) or JSON. Load ZIP to restore images and previews automatically. |

---

## Keyboard & Mouse Shortcuts

### Keyboard

| Key | Action |
|-----|--------|
| `Ctrl + Enter` | Execute pipeline |
| `Delete` | Delete selected node(s) |
| `Ctrl + C` | Copy selected node(s) with parameters |
| `Ctrl + X` | Cut selected node(s) |
| `Ctrl + V` | Paste copied node(s) |
| `Ctrl + Z` | Undo last action |
| `Ctrl + A` | Select all nodes |
| `Ctrl + D` | Duplicate selected node(s) |
| `ESC` | Close image popup / help dialog |

### Mouse

| Action | Description |
|--------|-------------|
| Left Click (node) | Select node, show properties |
| Ctrl + Click (node) | Add/remove node from multi-selection |
| Left Drag (empty) | Rubber-band select multiple nodes |
| Left Drag (node) | Move node (or all selected nodes) |
| Left Drag (port) | Create connection wire |
| Left Click (wire) | Delete connection |
| Left Click (empty) | Deselect all nodes |
| Right Click (node) | Delete node |
| Right Click (wire) | Delete connection |
| Middle Drag / Alt+Drag | Pan canvas |
| Scroll Wheel | Zoom in / out |
| Double Click (preview) | Enlarge preview image |

### Drag & Drop

| Action | Description |
|--------|-------------|
| Palette -> Canvas | Create a new node at drop position |

---

## Node Reference (83 types, 17 categories)

### Input / Output (5)

| Node | Description |
|------|-------------|
| Image Read | Load an image file (BMP, JPEG, PNG, TIFF). Use file upload or file path. |
| Image Show | Display image in popup (imshow). Has output port for chaining. |
| Image Write | Save image to file (PNG, JPEG, BMP, TIFF). |
| Video Read | Read a frame from a video file at a given frame index. |
| Camera Capture | Capture a frame from a webcam/camera device. |

### Color (5)

| Node | Description |
|------|-------------|
| CvtColor | Convert color space (BGR<->GRAY, BGR<->HSV, BGR<->RGB, etc.) |
| InRange | Color range thresholding. Set lower/upper BGR bounds to create a binary mask. |
| Histogram EQ | Histogram equalization or CLAHE. |
| Split Channels | Split image into 3 individual channels (B, G, R). |
| Merge Channels | Merge 3 individual channels into one color image. |

### Filter / Blur (6)

| Node | Description |
|------|-------------|
| Gaussian Blur | Smooth image with Gaussian kernel. |
| Median Blur | Reduce salt-and-pepper noise. |
| Bilateral Filter | Edge-preserving smoothing. |
| Box Filter | Average blur with configurable kernel. |
| Sharpen | Image sharpening via unsharp mask. |
| Filter2D | Custom kernel convolution (sharpen, edge detect, emboss, etc.). |

### Edge Detection (4)

| Node | Description |
|------|-------------|
| Canny | Canny edge detector with dual thresholds. |
| Sobel | Sobel derivative operator (X/Y direction). |
| Laplacian | Laplacian of Gaussian for edge detection. |
| Scharr | Scharr derivative filter (more accurate than Sobel). |

### Threshold (3)

| Node | Description |
|------|-------------|
| Threshold | Global thresholding (Binary, Otsu, Triangle, etc.). |
| Adaptive Threshold | Local adaptive thresholding. |
| Otsu Threshold | Automatic optimal threshold determination. |

### Morphology (4)

| Node | Description |
|------|-------------|
| Morphology Ex | Open, Close, Gradient, TopHat, BlackHat. Supports custom kernel via grid editor. |
| Dilate | Dilate operation (expand bright areas). Supports custom kernel shape. |
| Erode | Erode operation (shrink bright areas). Supports custom kernel shape. |
| Structuring Element | Create structuring element (Rect, Cross, Ellipse, Custom grid). |

### Contour (7)

| Node | Description |
|------|-------------|
| Draw Contours | Find and draw contours on image. |
| Bounding Rect | Draw bounding rectangles around contours. Outputs: image + coords [[x1,y1,x2,y2],...]. |
| Min Enclosing Circle | Draw minimum enclosing circles for contours. |
| Convex Hull | Draw convex hulls around contours. |
| Approx Poly | Approximate contours to polygons. |
| Contour Area | Filter and draw contours by area range. |
| Contour Properties | Show contour properties (area, perimeter, center). |

### Transform (8)

| Node | Description |
|------|-------------|
| Resize | Resize by pixel or scale factor. |
| Rotate | Rotate image by angle. |
| Flip | Flip horizontal, vertical, or both. |
| Crop | Crop region of interest (ROI). |
| Paste Image | Overlay a smaller image onto a base image at (x, y). Modes: overwrite, blend (opacity), alpha channel. |
| Warp Affine | Affine transformation (2x3 matrix). |
| Warp Perspective | Perspective transformation (4-point mapping). |
| Remap | Barrel/pincushion distortion correction. |

### Histogram (1)

| Node | Description |
|------|-------------|
| Calc Histogram | Calculate and visualize image histogram. |

### Feature Detection (7)

| Node | Description |
|------|-------------|
| Find Contours | Detect contours and draw them. |
| Hough Lines | Detect straight lines via Hough transform. |
| Harris Corner | Harris corner detection. |
| Good Features | Shi-Tomasi corner detection. |
| ORB | ORB feature detection and description. |
| FAST | FAST corner detection (high speed). |
| Match Features | Match keypoints between two images. |

### Drawing (6)

| Node | Description |
|------|-------------|
| Draw Line | Draw a line on image. |
| Draw Rectangle | Draw a rectangle on image. |
| Draw Circle | Draw a circle on image. |
| Draw Ellipse | Draw an ellipse on image. |
| Draw Text | Draw text on image (cv2.putText). |
| Draw Polylines | Draw polygon/polyline on image. |

### Arithmetic (9)

| Node | Description |
|------|-------------|
| Add | Add two images (cv2.add). Size Mismatch option: error / resize. |
| Subtract | Subtract two images (cv2.subtract). Size Mismatch option. |
| Multiply | Multiply two images with scale factor. Size Mismatch option. |
| AbsDiff | Absolute difference of two images. Size Mismatch option. |
| Add Weighted | Blend: alpha*img1 + beta*img2 + gamma. Size Mismatch option. |
| Bitwise AND | Pixel-wise AND of two images. Size Mismatch option. |
| Bitwise OR | Pixel-wise OR of two images. Size Mismatch option. |
| Bitwise XOR | Pixel-wise XOR of two images. Size Mismatch option. |
| Bitwise NOT | Invert all pixel values. |

### Detection (4)

| Node | Description |
|------|-------------|
| Haar Cascade | Object detection (face, eye, smile, etc.) using Haar cascades. Outputs: image + coords [[x1,y1,x2,y2],...]. |
| Hough Circles | Detect circles using Hough transform. |
| Template Match | Find similar regions. Outputs: image (with rectangles) + matches (coordinate list [[x1,y1,x2,y2],...]). |
| Image Extract | Extract a region from image using [x1,y1,x2,y2] coordinates. Connect from Template Match, Haar Cascade, Bounding Rect, or Val Coords. |

### Segmentation (3)

| Node | Description |
|------|-------------|
| Flood Fill | Region fill from seed point. |
| GrabCut | Foreground/background separation. |
| Watershed | Watershed segmentation algorithm. |

### Value (8)

| Node | Description |
|------|-------------|
| Integer | Output integer value. |
| Float | Output floating-point value. |
| Boolean | Output boolean value. |
| Point | Output Point (x, y) value. |
| Scalar | Output Scalar (v0, v1, v2, v3) value. |
| Math Operation | Basic math on two inputs (add, subtract, multiply, etc.). |
| Val Coords | Manual coordinate input for Image Extract. Single (x1,y1,x2,y2) or multi-line mode. |
| List Index/Slice | Select an element (index) or sub-list (slice) from a list. Connect from Template Match matches port, etc. |

### Control Flow (4)

| Node | Description |
|------|-------------|
| If | Conditional branching: routes image to true/false port. |
| For Loop | Applies an operation N times iteratively. |
| While Loop | Repeats operation while condition holds. |
| Switch-Case | Routes image to case 0/1/2 based on condition. |

### Script (1)

| Node | Description |
|------|-------------|
| Python Script | Custom Python/OpenCV code. Use img_input, img_output, cv2, np. |

---

## Program Structure

### Overview

```
 Browser (Frontend)              Server (Backend)
+---------------------+        +---------------------+
| index.html          |  HTTP  | app.py (Flask)      |
| style.css           | <----> | - /api/upload       |
| node-definitions.js |        | - /api/execute      |
| editor.js (Canvas)  |        | - /api/execute_single|
+---------------------+        | - /api/save_project |
                                | - /api/load_project |
                                | - /api/script/*     |
                                | OpenCV + NumPy      |
                                +---------------------+
```

### File Structure

```
NodeOpenCV/
  app.py                 # Flask backend, OpenCV processors
  requirements.txt       # Python dependencies
  templates/
    index.html           # Main page layout
  static/
    style.css            # Catppuccin dark theme styles
    editor.js            # Canvas-based node editor (IIFE)
    node-definitions.js  # NODE_DEFS - 83 node type configs
  sessions/              # Per-user session data (uploads, work, scripts)
  doc/                   # Generated textbook PDF
```

### Frontend (editor.js)

- **Canvas Rendering**: Pure HTML5 Canvas API, no external libraries. Draws nodes, ports, wires, grid.
- **State Management**: Single `state` object tracks nodes, connections, selection, pan/zoom.
- **Port System**: Dynamic port positions based on node height. Input ports on left, output ports on right.
- **Connection Wires**: Bezier curves with source node's color.
- **Hit Testing**: Node/port/connection detection for mouse interaction.
- **Drag & Drop**: HTML5 drag API from palette to canvas.

### Backend (app.py)

- **Image Store**: Server-side `image_store` dict (imageId -> numpy array). Avoids sending base64 repeatedly.
- **JPEG Preview**: Images encoded as JPEG (quality 80) for fast transfer. Full PNG available via `/api/image/<id>`.
- **Pipeline Execution**: Topological sort (Kahn's algorithm) ensures correct dependency order.
- **Single Node Preview**: `/api/execute_single` traces ancestors and only executes needed nodes.
- **Node Processors**: `NODE_PROCESSORS` dict maps node types to processing functions.
- **External Editor**: Writes script to `scripts/` folder, opens VS Code or Notepad via subprocess.

### Node Definitions (node-definitions.js)

- `NODE_DEFS` object contains all 83 node type configurations.
- Each node type defines: label, category, color, inputs[], outputs[], properties[], doc{}.
- The `doc` object provides OpenCV function signature, description, and parameter details.

### Data Flow

1. **Upload**: Image file -> server stores as numpy array, returns imageId + JPEG preview.
2. **Execute**: Frontend sends node graph (without base64). Backend sorts topologically, processes each node, returns previews.
3. **Preview**: Single-node execution traces ancestor subgraph only.

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Main page (index.html) |
| `POST /api/upload` | Upload image file, returns imageId + preview |
| `POST /api/execute` | Execute full pipeline, returns all node results |
| `POST /api/execute_single` | Execute up to target node only |
| `POST /api/save_project` | Save flow + images as ZIP (flow.json + images/ folder) |
| `POST /api/load_project` | Load ZIP project, restore images to session, return previews |
| `GET /api/examples` | List available example projects |
| `GET /api/examples/<file>` | Download example project ZIP |
| `GET /api/image/<id>` | Download full-quality PNG image |
| `POST /api/script/open` | Open script in VS Code or Notepad |
| `POST /api/script/read` | Read edited script back from file |

### Dependencies

| Package | Description |
|---------|-------------|
| Flask >= 2.3 | Web framework for backend API |
| Flask-CORS >= 4.0 | Cross-origin resource sharing |
| OpenCV >= 4.8 | Image processing library (cv2) |
| NumPy >= 1.24 | Array computation (numpy) |
