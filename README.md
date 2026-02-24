# NodeOpenCV

## Terminal
pip install -r requirements.txt

python app.py

## Web Browser
localhost:5000

SamOpenCVWeb
Node-RED style visual programming tool for OpenCV image processing.
Drag nodes, connect wires, execute pipelines.

1. Basic Workflow
1
Add Nodes
Drag nodes from the left Node Palette onto the canvas.
2
Load Image
Click an Image Read node, then use the file upload in Properties (right panel).
3
Connect Nodes
Drag from an output port (right side) to an input port (left side) of another node.
4
Set Properties
Click a node to see/edit its parameters in the Properties panel.
5
Execute
Click Execute (or Ctrl+Enter) to run the full pipeline.
6
View Results
Preview panel shows each node's result. Image Show nodes open a popup automatically.
2. Ports & Connections
Input ports (left side of node) receive data from other nodes.
Output ports (right side of node) send data to other nodes.
A filled port circle means it is connected.
Click a connection wire to delete it.
Each input port accepts only one connection.
3. Preview & Image Show
Click a node, then click Preview to see its output without running the full pipeline.
Double-click the preview image to enlarge.
Image Show nodes automatically open a full-screen popup when executed.
Image Show also has an output port, so you can chain after it.
4. Python Script Node
Write custom OpenCV code using img_input (input image) and img_output (result image).
cv2 and np (NumPy) are available.
Use VS Code or Notepad buttons to open the script in an external editor.
After editing externally, click Reload to apply changes.

# Example: Convert to grayscale and apply edge detection
gray = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 50, 150)
img_output = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
5. Control Flow Nodes
If: Evaluates a condition and routes the image to the true or false output port.
For Loop: Applies an operation N times (e.g., repeated blur or erosion).
While Loop: Repeats an operation while a condition is true (with safety max iterations).
Switch-Case: Routes the image to one of 3 output ports based on a condition (channels, depth, size, etc.).
6. Code Generation
Click Code in the toolbar to convert the current node graph into standalone Python code.
The generated code uses only cv2 and numpy.
Use Copy to copy to clipboard or Download to save as .py file.
7. Save & Load
Save: 이미지가 포함된 경우 ZIP 프로젝트 파일로 저장 (flow.json + images/ 폴더). 이미지가 없으면 JSON으로 저장.
Load: ZIP 프로젝트 파일 또는 JSON 파일을 불러옵니다. ZIP의 경우 이미지도 함께 복원되어 프리뷰가 표시됩니다.
8. Panel Toggle
Use the arrow buttons on the palette/properties headers to collapse/expand panels.
Collapse panels to get a wider canvas workspace.
