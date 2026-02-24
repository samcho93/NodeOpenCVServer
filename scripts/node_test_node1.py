# ──────────────────────────────────────────────
# NodeOpenCV - Python Script Editor
# Node: test_node1
# ──────────────────────────────────────────────
# Available variables:
#   img_input  : Input image (numpy.ndarray, BGR) or None
#   img_output : Set this to your result image
#   cv2        : OpenCV library
#   np         : NumPy library
# ──────────────────────────────────────────────

img_output = cv2.GaussianBlur(img_input, (5,5), 0)
