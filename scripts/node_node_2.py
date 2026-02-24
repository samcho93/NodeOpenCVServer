# ──────────────────────────────────────────────
# NodeOpenCV - Python Script Editor
# Node: node_2
# ──────────────────────────────────────────────
# Available variables:
#   img_input  : Input image (numpy.ndarray, BGR) or None
#   img_output : Set this to your result image
#   cv2        : OpenCV library
#   np         : NumPy library
# ──────────────────────────────────────────────

# Input image: img_input (numpy array, BGR)
# Output image: img_output (numpy array)
# Available: cv2, np (numpy)

#img_output = img_input.copy()
# Example: draw text
# cv2.putText(img_output, 'Hello', (50,50),
#     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

img_output = cv2.cvtColor(img_input, cv2.COLOR_BGR2GRAY)