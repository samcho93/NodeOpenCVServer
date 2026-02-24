/**
 * Node type definitions: properties, ports, colors, and OpenCV documentation.
 */
const NODE_DEFS = {
    image_read: {
        label: 'Image Read',
        category: 'io',
        color: '#4CAF50',
        inputs: [],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'filepath', label: 'File Path', type: 'text', default: '' },
            { key: '_upload', label: 'Upload Image', type: 'file' },
            { key: 'flags', label: 'Flags', type: 'select', default: 'IMREAD_COLOR',
              options: ['IMREAD_COLOR', 'IMREAD_GRAYSCALE', 'IMREAD_UNCHANGED'] }
        ],
        doc: {
            signature: 'cv2.imread(filename, flags=cv2.IMREAD_COLOR)',
            description: 'Loads an image from a file. Supports BMP, JPEG, PNG, TIFF, etc.',
            params: [
                { name: 'filename', desc: 'Name of the file to be loaded (string)' },
                { name: 'flags', desc: 'imread flags: IMREAD_COLOR (1), IMREAD_GRAYSCALE (0), IMREAD_UNCHANGED (-1)' }
            ],
            returns: 'numpy.ndarray - The loaded image (BGR format), or None if failed'
        }
    },

    image_show: {
        label: 'Image Show',
        category: 'io',
        color: '#2196F3',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [],
        properties: [
            { key: 'windowName', label: 'Window Name', type: 'text', default: 'Output' }
        ],
        doc: {
            signature: 'cv2.imshow(winname, mat)',
            description: 'Displays an image in the specified window. In this tool, the result is shown in the Preview panel.',
            params: [
                { name: 'winname', desc: 'Name of the window (string)' },
                { name: 'mat', desc: 'Image to be shown (numpy.ndarray)' }
            ],
            returns: 'None'
        }
    },

    cvt_color: {
        label: 'CvtColor',
        category: 'color',
        color: '#9C27B0',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'code', label: 'Color Code', type: 'select', default: 'COLOR_BGR2GRAY',
              options: [
                'COLOR_BGR2GRAY', 'COLOR_BGR2RGB', 'COLOR_BGR2HSV', 'COLOR_BGR2HLS',
                'COLOR_BGR2LAB', 'COLOR_BGR2YCrCb', 'COLOR_BGR2LUV',
                'COLOR_HSV2BGR', 'COLOR_HLS2BGR', 'COLOR_LAB2BGR',
                'COLOR_GRAY2BGR', 'COLOR_RGB2BGR'
              ]
            }
        ],
        doc: {
            signature: 'cv2.cvtColor(src, code[, dst[, dstCn]])',
            description: 'Converts an image from one color space to another.',
            params: [
                { name: 'src', desc: 'Input image: 8-bit unsigned, 16-bit unsigned, or single-precision floating-point' },
                { name: 'code', desc: 'Color space conversion code (e.g., cv2.COLOR_BGR2GRAY)' },
                { name: 'dstCn', desc: 'Number of channels in the destination image; if 0, derived automatically' }
            ],
            returns: 'numpy.ndarray - Converted image'
        }
    },

    in_range: {
        label: 'InRange',
        category: 'color',
        color: '#E91E63',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'lowerB', label: 'Lower B/H', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'lowerG', label: 'Lower G/S', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'lowerR', label: 'Lower R/V', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'upperB', label: 'Upper B/H', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'upperG', label: 'Upper G/S', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'upperR', label: 'Upper R/V', type: 'number', default: 255, min: 0, max: 255 }
        ],
        doc: {
            signature: 'cv2.inRange(src, lowerb, upperb[, dst])',
            description: 'Checks if array elements lie between the elements of two other arrays. Used for color-based segmentation.',
            params: [
                { name: 'src', desc: 'Input array (image)' },
                { name: 'lowerb', desc: 'Inclusive lower boundary array or scalar' },
                { name: 'upperb', desc: 'Inclusive upper boundary array or scalar' }
            ],
            returns: 'numpy.ndarray - Output array of same size as src and CV_8U type (binary mask)'
        }
    },

    histogram_eq: {
        label: 'Histogram EQ',
        category: 'color',
        color: '#673AB7',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'useCLAHE', label: 'Use CLAHE', type: 'checkbox', default: false },
            { key: 'clipLimit', label: 'CLAHE Clip Limit', type: 'number', default: 2.0, step: 0.1 },
            { key: 'tileGridSize', label: 'CLAHE Tile Size', type: 'number', default: 8, min: 1 }
        ],
        doc: {
            signature: 'cv2.equalizeHist(src[, dst])\ncv2.createCLAHE(clipLimit, tileGridSize)',
            description: 'Equalizes the histogram of a grayscale image. CLAHE (Contrast Limited Adaptive Histogram Equalization) applies equalization locally.',
            params: [
                { name: 'src', desc: 'Source 8-bit single channel image' },
                { name: 'clipLimit', desc: 'CLAHE: Threshold for contrast limiting (default 2.0)' },
                { name: 'tileGridSize', desc: 'CLAHE: Size of grid for histogram equalization (default 8x8)' }
            ],
            returns: 'numpy.ndarray - Equalized image'
        }
    },

    gaussian_blur: {
        label: 'Gaussian Blur',
        category: 'filter',
        color: '#FF9800',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'ksize', label: 'Kernel Size', type: 'number', default: 5, min: 1, step: 2 },
            { key: 'sigmaX', label: 'Sigma X', type: 'number', default: 0, step: 0.1 }
        ],
        doc: {
            signature: 'cv2.GaussianBlur(src, ksize, sigmaX[, dst[, sigmaY[, borderType]]])',
            description: 'Blurs an image using a Gaussian filter. Useful for noise reduction and smoothing.',
            params: [
                { name: 'src', desc: 'Input image; can have any number of channels' },
                { name: 'ksize', desc: 'Gaussian kernel size (width, height). Must be odd and positive.' },
                { name: 'sigmaX', desc: 'Gaussian kernel standard deviation in X direction. 0 = auto-compute from ksize.' },
                { name: 'sigmaY', desc: 'Gaussian kernel standard deviation in Y direction. 0 = same as sigmaX.' },
                { name: 'borderType', desc: 'Pixel extrapolation method (default: BORDER_DEFAULT)' }
            ],
            returns: 'numpy.ndarray - Blurred image with same size and type as src'
        }
    },

    median_blur: {
        label: 'Median Blur',
        category: 'filter',
        color: '#FF9800',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'ksize', label: 'Kernel Size', type: 'number', default: 5, min: 1, step: 2 }
        ],
        doc: {
            signature: 'cv2.medianBlur(src, ksize)',
            description: 'Blurs an image using the median filter. Effective for salt-and-pepper noise removal.',
            params: [
                { name: 'src', desc: 'Input 1-, 3-, or 4-channel image. With ksize=3 or 5, depth can be CV_8U/CV_16U/CV_32F. Larger ksize requires CV_8U.' },
                { name: 'ksize', desc: 'Aperture linear size; must be odd and greater than 1 (3, 5, 7, ...)' }
            ],
            returns: 'numpy.ndarray - Filtered image with same size and type as src'
        }
    },

    bilateral_filter: {
        label: 'Bilateral Filter',
        category: 'filter',
        color: '#FF9800',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'd', label: 'Diameter', type: 'number', default: 9, min: 1 },
            { key: 'sigmaColor', label: 'Sigma Color', type: 'number', default: 75, step: 1 },
            { key: 'sigmaSpace', label: 'Sigma Space', type: 'number', default: 75, step: 1 }
        ],
        doc: {
            signature: 'cv2.bilateralFilter(src, d, sigmaColor, sigmaSpace[, dst[, borderType]])',
            description: 'Applies the bilateral filter to an image. Smooths while keeping edges sharp.',
            params: [
                { name: 'src', desc: 'Source 8-bit or floating-point, 1-channel or 3-channel image' },
                { name: 'd', desc: 'Diameter of each pixel neighborhood. -1 = auto from sigmaSpace.' },
                { name: 'sigmaColor', desc: 'Filter sigma in the color space. Larger value means farther colors will be mixed.' },
                { name: 'sigmaSpace', desc: 'Filter sigma in the coordinate space. Larger value means farther pixels will influence each other.' }
            ],
            returns: 'numpy.ndarray - Filtered image'
        }
    },

    canny: {
        label: 'Canny',
        category: 'edge',
        color: '#F44336',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'threshold1', label: 'Threshold 1', type: 'number', default: 100, min: 0, max: 500 },
            { key: 'threshold2', label: 'Threshold 2', type: 'number', default: 200, min: 0, max: 500 }
        ],
        doc: {
            signature: 'cv2.Canny(image, threshold1, threshold2[, edges[, apertureSize[, L2gradient]]])',
            description: 'Finds edges in an image using the Canny algorithm. Multi-stage: noise reduction, gradient calc, non-maximum suppression, hysteresis thresholding.',
            params: [
                { name: 'image', desc: '8-bit input image (single channel recommended)' },
                { name: 'threshold1', desc: 'First threshold for the hysteresis procedure (lower)' },
                { name: 'threshold2', desc: 'Second threshold for the hysteresis procedure (upper)' },
                { name: 'apertureSize', desc: 'Aperture size for the Sobel operator (default 3)' },
                { name: 'L2gradient', desc: 'Use L2 norm for gradient magnitude (default False = L1)' }
            ],
            returns: 'numpy.ndarray - Output edge map (same size, single channel)'
        }
    },

    sobel: {
        label: 'Sobel',
        category: 'edge',
        color: '#F44336',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'dx', label: 'dx (X order)', type: 'number', default: 1, min: 0, max: 3 },
            { key: 'dy', label: 'dy (Y order)', type: 'number', default: 0, min: 0, max: 3 },
            { key: 'ksize', label: 'Kernel Size', type: 'number', default: 3, min: 1, max: 31, step: 2 }
        ],
        doc: {
            signature: 'cv2.Sobel(src, ddepth, dx, dy[, dst[, ksize[, scale[, delta[, borderType]]]]])',
            description: 'Calculates the first, second, third, or mixed image derivatives using an extended Sobel operator.',
            params: [
                { name: 'src', desc: 'Input image' },
                { name: 'ddepth', desc: 'Output image depth (-1 for same as src, or CV_16S, CV_32F, CV_64F)' },
                { name: 'dx', desc: 'Order of the derivative x (0, 1, 2, 3)' },
                { name: 'dy', desc: 'Order of the derivative y (0, 1, 2, 3)' },
                { name: 'ksize', desc: 'Size of the extended Sobel kernel; must be 1, 3, 5, or 7' }
            ],
            returns: 'numpy.ndarray - Output image of the same size, with the specified depth'
        }
    },

    laplacian: {
        label: 'Laplacian',
        category: 'edge',
        color: '#F44336',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'ksize', label: 'Kernel Size', type: 'number', default: 3, min: 1, max: 31, step: 2 }
        ],
        doc: {
            signature: 'cv2.Laplacian(src, ddepth[, dst[, ksize[, scale[, delta[, borderType]]]]])',
            description: 'Calculates the Laplacian of an image. Used for edge detection by computing second derivatives.',
            params: [
                { name: 'src', desc: 'Source image' },
                { name: 'ddepth', desc: 'Desired depth of the destination image' },
                { name: 'ksize', desc: 'Aperture size used to compute the second-derivative filters. Must be positive and odd.' }
            ],
            returns: 'numpy.ndarray - Laplacian image'
        }
    },

    threshold: {
        label: 'Threshold',
        category: 'threshold',
        color: '#795548',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'thresh', label: 'Threshold', type: 'number', default: 127, min: 0, max: 255 },
            { key: 'maxval', label: 'Max Value', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'type', label: 'Type', type: 'select', default: 'THRESH_BINARY',
              options: ['THRESH_BINARY', 'THRESH_BINARY_INV', 'THRESH_TRUNC', 'THRESH_TOZERO', 'THRESH_TOZERO_INV', 'THRESH_OTSU']
            }
        ],
        doc: {
            signature: 'cv2.threshold(src, thresh, maxval, type[, dst])',
            description: 'Applies a fixed-level threshold to each array element. Used to create binary images.',
            params: [
                { name: 'src', desc: 'Input array (multiple-channel, 8-bit or 32-bit floating point)' },
                { name: 'thresh', desc: 'Threshold value' },
                { name: 'maxval', desc: 'Maximum value to use with THRESH_BINARY and THRESH_BINARY_INV' },
                { name: 'type', desc: 'Thresholding type: THRESH_BINARY, THRESH_BINARY_INV, THRESH_TRUNC, THRESH_TOZERO, THRESH_TOZERO_INV, THRESH_OTSU' }
            ],
            returns: '(retval, dst) - retval is the threshold used, dst is the output image'
        }
    },

    adaptive_threshold: {
        label: 'Adaptive Threshold',
        category: 'threshold',
        color: '#795548',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'maxval', label: 'Max Value', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'adaptiveMethod', label: 'Adaptive Method', type: 'select', default: 'ADAPTIVE_THRESH_GAUSSIAN_C',
              options: ['ADAPTIVE_THRESH_MEAN_C', 'ADAPTIVE_THRESH_GAUSSIAN_C']
            },
            { key: 'thresholdType', label: 'Threshold Type', type: 'select', default: 'THRESH_BINARY',
              options: ['THRESH_BINARY', 'THRESH_BINARY_INV']
            },
            { key: 'blockSize', label: 'Block Size', type: 'number', default: 11, min: 3, step: 2 },
            { key: 'C', label: 'C (Constant)', type: 'number', default: 2, step: 1 }
        ],
        doc: {
            signature: 'cv2.adaptiveThreshold(src, maxValue, adaptiveMethod, thresholdType, blockSize, C[, dst])',
            description: 'Applies an adaptive threshold to an array. The threshold value is calculated for each pixel based on a local neighborhood.',
            params: [
                { name: 'src', desc: '8-bit single-channel image' },
                { name: 'maxValue', desc: 'Non-zero value assigned to pixels for which the condition is satisfied' },
                { name: 'adaptiveMethod', desc: 'ADAPTIVE_THRESH_MEAN_C or ADAPTIVE_THRESH_GAUSSIAN_C' },
                { name: 'thresholdType', desc: 'THRESH_BINARY or THRESH_BINARY_INV' },
                { name: 'blockSize', desc: 'Size of pixel neighborhood (must be odd, e.g., 3, 5, 7, ...)' },
                { name: 'C', desc: 'Constant subtracted from the mean or weighted mean' }
            ],
            returns: 'numpy.ndarray - Destination image of the same size and same type as src'
        }
    },

    resize: {
        label: 'Resize',
        category: 'transform',
        color: '#00BCD4',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'width', label: 'Width (0=use scale)', type: 'number', default: 0, min: 0 },
            { key: 'height', label: 'Height (0=use scale)', type: 'number', default: 0, min: 0 },
            { key: 'fx', label: 'Scale X', type: 'number', default: 0.5, step: 0.1 },
            { key: 'fy', label: 'Scale Y', type: 'number', default: 0.5, step: 0.1 },
            { key: 'interpolation', label: 'Interpolation', type: 'select', default: 'INTER_LINEAR',
              options: ['INTER_NEAREST', 'INTER_LINEAR', 'INTER_AREA', 'INTER_CUBIC', 'INTER_LANCZOS4']
            }
        ],
        doc: {
            signature: 'cv2.resize(src, dsize[, dst[, fx[, fy[, interpolation]]]])',
            description: 'Resizes an image. Specify either dsize or both fx and fy.',
            params: [
                { name: 'src', desc: 'Input image' },
                { name: 'dsize', desc: 'Output image size (width, height). If (0,0), computed from fx, fy.' },
                { name: 'fx', desc: 'Scale factor along the horizontal axis' },
                { name: 'fy', desc: 'Scale factor along the vertical axis' },
                { name: 'interpolation', desc: 'INTER_NEAREST, INTER_LINEAR, INTER_AREA, INTER_CUBIC, INTER_LANCZOS4' }
            ],
            returns: 'numpy.ndarray - Resized image'
        }
    },

    rotate: {
        label: 'Rotate',
        category: 'transform',
        color: '#00BCD4',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'angle', label: 'Angle (degrees)', type: 'number', default: 90, step: 1 }
        ],
        doc: {
            signature: 'cv2.getRotationMatrix2D(center, angle, scale)\ncv2.warpAffine(src, M, dsize)',
            description: 'Rotates an image around its center by the specified angle using affine transformation.',
            params: [
                { name: 'center', desc: 'Center of the rotation in the source image' },
                { name: 'angle', desc: 'Rotation angle in degrees. Positive = counter-clockwise.' },
                { name: 'scale', desc: 'Isotropic scale factor' }
            ],
            returns: 'numpy.ndarray - Rotated image'
        }
    },

    morphology: {
        label: 'Morphology Ex',
        category: 'morph',
        color: '#607D8B',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'operation', label: 'Operation', type: 'select', default: 'MORPH_OPEN',
              options: ['MORPH_ERODE', 'MORPH_DILATE', 'MORPH_OPEN', 'MORPH_CLOSE', 'MORPH_GRADIENT', 'MORPH_TOPHAT', 'MORPH_BLACKHAT']
            },
            { key: 'ksize', label: 'Kernel Size', type: 'number', default: 5, min: 3, max: 31, step: 2 },
            { key: 'shape', label: 'Kernel Shape', type: 'select', default: 'MORPH_RECT',
              options: ['MORPH_RECT', 'MORPH_CROSS', 'MORPH_ELLIPSE', 'custom']
            },
            { key: 'kernelData', label: 'Custom Kernel', type: 'kernel', default: '1,1,1,1,1,1,1,1,1' },
            { key: 'iterations', label: 'Iterations', type: 'number', default: 1, min: 1 }
        ],
        doc: {
            signature: 'cv2.morphologyEx(src, op, kernel)',
            description: 'Performs advanced morphological transformations. Choose a preset kernel shape or define a custom binary kernel (0/1 values).',
            params: [
                { name: 'op', desc: 'MORPH_OPEN, MORPH_CLOSE, MORPH_GRADIENT, MORPH_TOPHAT, MORPH_BLACKHAT' },
                { name: 'kernel', desc: 'Structuring element: preset shape or custom 0/1 grid' },
                { name: 'iterations', desc: 'Number of times the operation is applied' }
            ],
            returns: 'numpy.ndarray - Morphologically transformed image'
        }
    },

    dilate: {
        label: 'Dilate',
        category: 'morph',
        color: '#607D8B',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'ksize', label: 'Kernel Size', type: 'number', default: 5, min: 3, max: 31, step: 2 },
            { key: 'shape', label: 'Kernel Shape', type: 'select', default: 'MORPH_RECT',
              options: ['MORPH_RECT', 'MORPH_CROSS', 'MORPH_ELLIPSE', 'custom']
            },
            { key: 'kernelData', label: 'Custom Kernel', type: 'kernel', default: '1,1,1,1,1,1,1,1,1' },
            { key: 'iterations', label: 'Iterations', type: 'number', default: 1, min: 1 }
        ],
        doc: {
            signature: 'cv2.dilate(src, kernel[, iterations])',
            description: 'Dilates an image using a structuring element. Choose a preset shape or define a custom binary kernel.',
            params: [
                { name: 'kernel', desc: 'Structuring element: preset shape or custom 0/1 grid' },
                { name: 'iterations', desc: 'Number of times dilation is applied' }
            ],
            returns: 'numpy.ndarray - Dilated image'
        }
    },

    erode: {
        label: 'Erode',
        category: 'morph',
        color: '#607D8B',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'ksize', label: 'Kernel Size', type: 'number', default: 5, min: 3, max: 31, step: 2 },
            { key: 'shape', label: 'Kernel Shape', type: 'select', default: 'MORPH_RECT',
              options: ['MORPH_RECT', 'MORPH_CROSS', 'MORPH_ELLIPSE', 'custom']
            },
            { key: 'kernelData', label: 'Custom Kernel', type: 'kernel', default: '1,1,1,1,1,1,1,1,1' },
            { key: 'iterations', label: 'Iterations', type: 'number', default: 1, min: 1 }
        ],
        doc: {
            signature: 'cv2.erode(src, kernel[, iterations])',
            description: 'Erodes an image using a structuring element. Choose a preset shape or define a custom binary kernel.',
            params: [
                { name: 'kernel', desc: 'Structuring element: preset shape or custom 0/1 grid' },
                { name: 'iterations', desc: 'Number of times erosion is applied' }
            ],
            returns: 'numpy.ndarray - Eroded image'
        }
    },

    find_contours: {
        label: 'Find Contours',
        category: 'contour',
        color: '#009688',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'contours', label: 'contours' }],
        properties: [
            { key: 'mode', label: 'Retrieval Mode', type: 'select', default: 'RETR_EXTERNAL',
              options: ['RETR_EXTERNAL', 'RETR_LIST', 'RETR_CCOMP', 'RETR_TREE']
            },
            { key: 'method', label: 'Approximation', type: 'select', default: 'CHAIN_APPROX_SIMPLE',
              options: ['CHAIN_APPROX_NONE', 'CHAIN_APPROX_SIMPLE', 'CHAIN_APPROX_TC89_L1', 'CHAIN_APPROX_TC89_KCOS']
            }
        ],
        doc: {
            signature: 'cv2.findContours(image, mode, method[, contours[, hierarchy[, offset]]])',
            description: 'Finds contours in a binary image. Outputs the detected contours for use by drawing/analysis nodes.',
            params: [
                { name: 'image', desc: 'Source image (8-bit single-channel). Non-zero pixels are treated as 1s.' },
                { name: 'mode', desc: 'Contour retrieval mode: RETR_EXTERNAL (outermost), RETR_LIST (all), RETR_TREE (full hierarchy)' },
                { name: 'method', desc: 'Contour approximation method: CHAIN_APPROX_NONE (all points), CHAIN_APPROX_SIMPLE (compress)' }
            ],
            returns: '(contours, hierarchy) - contours is list of numpy arrays, hierarchy describes nesting'
        }
    },

    hough_lines: {
        label: 'Hough Lines',
        category: 'feature',
        color: '#CDDC39',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'rho', label: 'Rho', type: 'number', default: 1, min: 0.1, step: 0.1 },
            { key: 'theta_divisor', label: 'Theta Divisor (pi/N)', type: 'number', default: 180, min: 1 },
            { key: 'threshold', label: 'Threshold', type: 'number', default: 100, min: 1 }
        ],
        doc: {
            signature: 'cv2.HoughLines(image, rho, theta, threshold[, lines[, srn[, stn[, min_theta[, max_theta]]]]])',
            description: 'Finds lines in a binary image using the standard Hough transform.',
            params: [
                { name: 'image', desc: '8-bit, single-channel binary source image (e.g., from Canny)' },
                { name: 'rho', desc: 'Distance resolution of the accumulator in pixels' },
                { name: 'theta', desc: 'Angle resolution of the accumulator in radians' },
                { name: 'threshold', desc: 'Accumulator threshold parameter. Only lines with enough votes are returned.' }
            ],
            returns: 'numpy.ndarray - Output vector of lines (rho, theta pairs)'
        }
    },

    add_weighted: {
        label: 'Add Weighted',
        category: 'arithmetic',
        color: '#3F51B5',
        inputs: [{ id: 'image', label: 'image' }, { id: 'image2', label: 'image2' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'alpha', label: 'Alpha', type: 'number', default: 0.5, step: 0.1 },
            { key: 'beta', label: 'Beta', type: 'number', default: 0.5, step: 0.1 },
            { key: 'gamma', label: 'Gamma', type: 'number', default: 0, step: 1 },
            { key: 'sizeMismatch', label: 'Size Mismatch', type: 'select', default: 'error',
              options: ['error', 'resize_img2', 'resize_img1'] }
        ],
        doc: {
            signature: 'cv2.addWeighted(src1, alpha, src2, beta, gamma[, dst[, dtype]])',
            description: 'Calculates the weighted sum of two arrays: dst = src1*alpha + src2*beta + gamma',
            params: [
                { name: 'src1', desc: 'First input array' },
                { name: 'alpha', desc: 'Weight of the first array elements' },
                { name: 'src2', desc: 'Second input array of the same size and channel number as src1' },
                { name: 'beta', desc: 'Weight of the second array elements' },
                { name: 'gamma', desc: 'Scalar added to each sum' }
            ],
            returns: 'numpy.ndarray - Output array that has the same size and number of channels as the input arrays'
        }
    },

    bitwise_and: {
        label: 'Bitwise AND',
        category: 'arithmetic',
        color: '#3F51B5',
        inputs: [{ id: 'image', label: 'image' }, { id: 'image2', label: 'image2' }, { id: 'mask', label: 'mask', optional: true }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'sizeMismatch', label: 'Size Mismatch', type: 'select', default: 'error',
              options: ['error', 'resize_img2', 'resize_img1'] }
        ],
        doc: {
            signature: 'cv2.bitwise_and(src1, src2[, dst[, mask]])',
            description: 'Computes bitwise conjunction of two arrays. If mask is provided, operation is applied only where mask is non-zero.',
            params: [
                { name: 'src1', desc: 'First input array' },
                { name: 'src2', desc: 'Second input array' },
                { name: 'mask', desc: 'Optional 8-bit single channel mask. Operation applied only where mask is non-zero.' }
            ],
            returns: 'numpy.ndarray - Output array that has the same size and type as the input arrays'
        }
    },

    bitwise_or: {
        label: 'Bitwise OR',
        category: 'arithmetic',
        color: '#3F51B5',
        inputs: [{ id: 'image', label: 'image' }, { id: 'image2', label: 'image2' }, { id: 'mask', label: 'mask', optional: true }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'sizeMismatch', label: 'Size Mismatch', type: 'select', default: 'error',
              options: ['error', 'resize_img2', 'resize_img1'] }
        ],
        doc: {
            signature: 'cv2.bitwise_or(src1, src2[, dst[, mask]])',
            description: 'Computes bitwise disjunction of two arrays. If mask is provided, operation is applied only where mask is non-zero.',
            params: [
                { name: 'src1', desc: 'First input array' },
                { name: 'src2', desc: 'Second input array' },
                { name: 'mask', desc: 'Optional 8-bit single channel mask. Operation applied only where mask is non-zero.' }
            ],
            returns: 'numpy.ndarray - Output array'
        }
    },

    bitwise_not: {
        label: 'Bitwise NOT',
        category: 'arithmetic',
        color: '#3F51B5',
        inputs: [{ id: 'image', label: 'image' }, { id: 'mask', label: 'mask', optional: true }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [],
        doc: {
            signature: 'cv2.bitwise_not(src[, dst[, mask]])',
            description: 'Inverts every bit of an array. If mask is provided, operation is applied only where mask is non-zero.',
            params: [
                { name: 'src', desc: 'Input array' },
                { name: 'mask', desc: 'Optional 8-bit single channel mask. Operation applied only where mask is non-zero.' }
            ],
            returns: 'numpy.ndarray - Output array (inverted)'
        }
    },

    // ---- Control Flow Nodes ----

    control_if: {
        label: 'If',
        category: 'control',
        color: '#26A69A',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'true', label: 'true' }, { id: 'false', label: 'false' }],
        properties: [
            { key: 'condition', label: 'Condition', type: 'select', default: 'not_empty',
              options: ['not_empty', 'is_color', 'is_grayscale', 'width_gt', 'height_gt', 'mean_gt', 'custom']
            },
            { key: 'value', label: 'Value (for comparisons)', type: 'number', default: 100, step: 1 },
            { key: 'customExpr', label: 'Custom (Python bool expr)', type: 'text', default: 'img.shape[0] > 100' }
        ],
        doc: {
            signature: 'If (condition) -> True / False branch',
            description: 'Evaluates a condition on the image and routes it to the True or False output port. Use for conditional branching in pipelines.',
            params: [
                { name: 'condition', desc: 'Predefined condition: not_empty, is_color, is_grayscale, width_gt, height_gt, mean_gt, or custom expression' },
                { name: 'value', desc: 'Threshold value for width_gt / height_gt / mean_gt comparisons' },
                { name: 'customExpr', desc: 'Python expression that evaluates to bool. Variable "img" is the input image (numpy array).' }
            ],
            returns: 'True port: image if condition met | False port: image if condition not met'
        }
    },

    control_for: {
        label: 'For Loop',
        category: 'control',
        color: '#26A69A',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'iterations', label: 'Iterations', type: 'number', default: 3, min: 1, max: 100 },
            { key: 'operation', label: 'Operation per iteration', type: 'select', default: 'gaussian_blur',
              options: ['gaussian_blur', 'median_blur', 'dilate', 'erode', 'sharpen', 'custom']
            },
            { key: 'ksize', label: 'Kernel Size', type: 'number', default: 3, min: 1, step: 2 },
            { key: 'customCode', label: 'Custom code (use img, i)', type: 'text', default: 'img = cv2.GaussianBlur(img, (3,3), 0)' }
        ],
        doc: {
            signature: 'for i in range(N): apply operation',
            description: 'Applies a selected operation repeatedly N times on the image. Useful for iterative processing like repeated blur or morphology.',
            params: [
                { name: 'iterations', desc: 'Number of loop iterations (1-100)' },
                { name: 'operation', desc: 'Operation to apply each iteration: gaussian_blur, median_blur, dilate, erode, sharpen, or custom code' },
                { name: 'ksize', desc: 'Kernel size for built-in operations' },
                { name: 'customCode', desc: 'Custom Python code executed per iteration. Variables: img (current image), i (iteration index), cv2, np' }
            ],
            returns: 'Image after N iterations of the operation'
        }
    },

    control_while: {
        label: 'While Loop',
        category: 'control',
        color: '#26A69A',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'condition', label: 'Continue while', type: 'select', default: 'mean_gt',
              options: ['mean_gt', 'mean_lt', 'std_gt', 'nonzero_gt', 'custom']
            },
            { key: 'value', label: 'Threshold value', type: 'number', default: 128, step: 1 },
            { key: 'operation', label: 'Operation per iteration', type: 'select', default: 'gaussian_blur',
              options: ['gaussian_blur', 'median_blur', 'erode', 'dilate', 'threshold_step', 'custom']
            },
            { key: 'ksize', label: 'Kernel Size', type: 'number', default: 3, min: 1, step: 2 },
            { key: 'maxIter', label: 'Max Iterations (safety)', type: 'number', default: 50, min: 1, max: 500 },
            { key: 'customCond', label: 'Custom condition expr', type: 'text', default: 'np.mean(img) > 100' },
            { key: 'customCode', label: 'Custom operation code', type: 'text', default: 'img = cv2.GaussianBlur(img, (3,3), 0)' }
        ],
        doc: {
            signature: 'while (condition): apply operation',
            description: 'Repeats an operation on the image while a condition is true, with a safety max iteration limit.',
            params: [
                { name: 'condition', desc: 'Continue condition: mean_gt, mean_lt, std_gt, nonzero_gt, or custom Python expression' },
                { name: 'value', desc: 'Threshold for predefined conditions' },
                { name: 'operation', desc: 'Operation to apply each iteration' },
                { name: 'maxIter', desc: 'Maximum number of iterations (safety limit, default 50)' }
            ],
            returns: 'Image after condition becomes false or max iterations reached'
        }
    },

    control_switch: {
        label: 'Switch-Case',
        category: 'control',
        color: '#26A69A',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'case0', label: 'case 0' }, { id: 'case1', label: 'case 1' }, { id: 'case2', label: 'case 2' }],
        properties: [
            { key: 'switchOn', label: 'Switch on', type: 'select', default: 'channels',
              options: ['channels', 'depth', 'size_class', 'mean_range', 'custom']
            },
            { key: 'customExpr', label: 'Custom expr (returns 0,1,2)', type: 'text', default: '0 if np.mean(img) < 85 else (1 if np.mean(img) < 170 else 2)' }
        ],
        doc: {
            signature: 'switch (expr) -> case 0 / case 1 / case 2',
            description: 'Routes the image to one of 3 output ports based on a condition. Only the matched case port receives the image.',
            params: [
                { name: 'switchOn', desc: 'channels: 1ch->0, 3ch->1, 4ch->2 | depth: uint8->0, float->1, other->2 | size_class: small->0, medium->1, large->2 | mean_range: dark->0, mid->1, bright->2 | custom: Python expression returning 0/1/2' },
                { name: 'customExpr', desc: 'Python expression using "img" that evaluates to 0, 1, or 2' }
            ],
            returns: 'Image routed to the matching case output port (case 0, case 1, or case 2)'
        }
    },

    python_script: {
        label: 'Python Script',
        category: 'script',
        color: '#FFC107',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'script', label: 'Python Code', type: 'textarea', default:
`# Input image: img_input (numpy array, BGR)
# Output image: img_output (numpy array)
# Available: cv2, np (numpy)

img_output = img_input.copy()
# Example: draw text
# cv2.putText(img_output, 'Hello', (50,50),
#     cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
`
            }
        ],
        doc: {
            signature: 'Custom Python Script Node',
            description: 'Execute arbitrary Python code with OpenCV and NumPy. Input image is available as `img_input`. Set `img_output` for the result.',
            params: [
                { name: 'img_input', desc: 'Input image (numpy.ndarray, BGR format). None if no input connected.' },
                { name: 'img_output', desc: 'Set this variable to the output image (numpy.ndarray).' },
                { name: 'cv2', desc: 'OpenCV library (already imported)' },
                { name: 'np', desc: 'NumPy library (already imported)' }
            ],
            returns: 'img_output variable will be passed to connected nodes'
        }
    },

    // ---- IO (new) ----

    image_write: {
        label: 'Image Write', category: 'io', color: '#8BC34A',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [],
        properties: [
            { key: 'filepath', label: 'File Path', type: 'text', default: 'output.png' },
            { key: 'format', label: 'Format', type: 'select', default: 'PNG', options: ['PNG', 'JPEG', 'BMP', 'TIFF'] },
            { key: 'quality', label: 'Quality', type: 'number', default: 95, min: 1, max: 100 }
        ],
        doc: { signature: 'cv2.imwrite(filename, img[, params])', description: 'Saves an image to file. Supports PNG, JPEG, BMP, TIFF formats with quality control.', params: [{ name: 'filepath', desc: 'Output file path (e.g., output.png)' }, { name: 'format', desc: 'Image format' }, { name: 'quality', desc: 'JPEG quality (1-100) or PNG compression' }], returns: 'bool - True if saved successfully' }
    },

    video_read: {
        label: 'Video Read', category: 'io', color: '#FF5722',
        inputs: [],
        outputs: [{ id: 'image', label: 'frame' }],
        properties: [
            { key: 'filepath', label: 'Video File Path', type: 'text', default: '' },
            { key: '_upload', label: 'Upload Video', type: 'file', accept: 'video/*,.mp4,.avi,.mkv,.mov,.wmv,.flv,.webm' },
            { key: 'mode', label: 'Mode', type: 'select', default: 'single', options: ['single', 'loop'] },
            { key: 'frameIndex', label: 'Frame Index', type: 'number', default: 0, min: 0 },
            { key: 'startFrame', label: 'Start Frame (loop)', type: 'number', default: 0, min: 0 },
            { key: 'endFrame', label: 'End Frame (loop, -1=all)', type: 'number', default: -1, min: -1 },
            { key: 'step', label: 'Step (loop)', type: 'number', default: 1, min: 1 }
        ],
        doc: { signature: 'cv2.VideoCapture(filename)', description: 'Reads frames from a video file. Single mode: reads one frame at frameIndex. Loop mode: iterates from startFrame to endFrame with step, executing the entire downstream pipeline for each frame.', params: [{ name: 'mode', desc: 'single = one frame, loop = iterate frames' }, { name: 'startFrame', desc: 'First frame in loop (0-based)' }, { name: 'endFrame', desc: 'Last frame in loop (-1 = end of video)' }, { name: 'step', desc: 'Frame step increment' }], returns: 'numpy.ndarray - The video frame' }
    },

    camera_capture: {
        label: 'Camera Capture', category: 'io', color: '#FF5722',
        inputs: [],
        outputs: [{ id: 'image', label: 'frame' }],
        properties: [
            { key: 'cameraIndex', label: 'Camera Index', type: 'number', default: 0, min: 0 }
        ],
        doc: { signature: 'cv2.VideoCapture(index)', description: 'Captures a single frame from a webcam/camera.', params: [{ name: 'index', desc: 'Camera device index (0 = default camera)' }], returns: 'numpy.ndarray - Captured frame' }
    },

    video_write: {
        label: 'Video Write', category: 'io', color: '#8BC34A',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [],
        properties: [
            { key: 'filepath', label: 'File Path', type: 'text', default: 'output.mp4' },
            { key: 'codec', label: 'Codec', type: 'select', default: 'mp4v', options: ['mp4v', 'XVID', 'MJPG', 'H264', 'avc1'] },
            { key: 'fps', label: 'FPS', type: 'number', default: 30, min: 1, max: 120 }
        ],
        doc: { signature: 'cv2.VideoWriter(filename, fourcc, fps, frameSize)', description: 'Writes processed frames to a video file. Connect to the end of a pipeline that starts with Video Read (loop mode). Execution collects all loop frames and encodes them into a video.', params: [{ name: 'filepath', desc: 'Output video path (.mp4, .avi)' }, { name: 'codec', desc: 'Video codec (mp4v for .mp4, XVID for .avi)' }, { name: 'fps', desc: 'Frames per second' }], returns: 'Info message with frame count' }
    },

    // ---- Color (new) ----

    split_channels: {
        label: 'Split Channels', category: 'color', color: '#9C27B0',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'ch0', label: 'ch 0' }, { id: 'ch1', label: 'ch 1' }, { id: 'ch2', label: 'ch 2' }],
        properties: [],
        doc: { signature: 'cv2.split(m)', description: 'Splits a multi-channel image into separate single-channel images.', params: [{ name: 'm', desc: 'Input multi-channel array (image)' }], returns: 'tuple of numpy.ndarray - Individual channels (B, G, R for BGR images)' }
    },

    merge_channels: {
        label: 'Merge Channels', category: 'color', color: '#9C27B0',
        inputs: [{ id: 'ch0', label: 'ch 0' }, { id: 'ch1', label: 'ch 1' }, { id: 'ch2', label: 'ch 2' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [],
        doc: { signature: 'cv2.merge(mv)', description: 'Merges several single-channel arrays into a multi-channel image.', params: [{ name: 'mv', desc: 'Input array of single-channel matrices' }], returns: 'numpy.ndarray - Merged multi-channel image' }
    },

    // ---- Filter (new) ----

    box_filter: {
        label: 'Box Filter', category: 'filter', color: '#FF9800',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'kwidth', label: 'Kernel Width', type: 'number', default: 5, min: 1, step: 2 },
            { key: 'kheight', label: 'Kernel Height', type: 'number', default: 5, min: 1, step: 2 },
            { key: 'normalize', label: 'Normalize', type: 'checkbox', default: true }
        ],
        doc: { signature: 'cv2.boxFilter(src, ddepth, ksize[, normalize])', description: 'Blurs an image using the box filter (average filter).', params: [{ name: 'src', desc: 'Input image' }, { name: 'ksize', desc: 'Blurring kernel size (width, height)' }, { name: 'normalize', desc: 'Whether to normalize the kernel (true = average, false = sum)' }], returns: 'numpy.ndarray - Filtered image' }
    },

    sharpen: {
        label: 'Sharpen', category: 'filter', color: '#FF9800',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'strength', label: 'Strength', type: 'number', default: 1.5, step: 0.1 }
        ],
        doc: { signature: 'Unsharp Mask Sharpening', description: 'Sharpens an image using unsharp masking: blurs then subtracts from original.', params: [{ name: 'strength', desc: 'Sharpening strength (1.0 = no change, higher = sharper)' }], returns: 'numpy.ndarray - Sharpened image' }
    },

    filter2d: {
        label: 'Filter2D', category: 'filter', color: '#FF9800',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'kernelSize', label: 'Kernel Size', type: 'number', default: 3, min: 3, max: 9, step: 2 },
            { key: 'preset', label: 'Preset Kernel', type: 'select', default: 'sharpen',
              options: ['identity', 'sharpen', 'edge_detect', 'emboss', 'ridge', 'blur', 'custom'] },
            { key: 'kernelData', label: 'Custom Kernel', type: 'kernel', default: '0,-1,0,-1,5,-1,0,-1,0' }
        ],
        doc: { signature: 'cv2.filter2D(src, ddepth, kernel)', description: 'Convolves an image with a custom kernel. Choose from preset kernels or define a custom NxN kernel.', params: [{ name: 'src', desc: 'Input image' }, { name: 'kernel', desc: 'Convolution kernel (numpy array)' }], returns: 'numpy.ndarray - Filtered image' }
    },

    // ---- Edge (new) ----

    scharr: {
        label: 'Scharr', category: 'edge', color: '#F44336',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'dx', label: 'dx (X order)', type: 'number', default: 1, min: 0, max: 1 },
            { key: 'dy', label: 'dy (Y order)', type: 'number', default: 0, min: 0, max: 1 }
        ],
        doc: { signature: 'cv2.Scharr(src, ddepth, dx, dy)', description: 'Calculates the first x- or y- image derivative using Scharr operator. More accurate than Sobel for 3x3 kernels.', params: [{ name: 'src', desc: 'Input image' }, { name: 'dx', desc: 'Order of X derivative (0 or 1)' }, { name: 'dy', desc: 'Order of Y derivative (0 or 1)' }], returns: 'numpy.ndarray - Derivative image' }
    },

    // ---- Threshold (new) ----

    otsu_threshold: {
        label: 'Otsu Threshold', category: 'threshold', color: '#795548',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'maxval', label: 'Max Value', type: 'number', default: 255, min: 0, max: 255 }
        ],
        doc: { signature: 'cv2.threshold(src, 0, maxval, THRESH_BINARY + THRESH_OTSU)', description: 'Applies Otsu algorithm for automatic optimal threshold determination.', params: [{ name: 'src', desc: '8-bit single-channel image' }, { name: 'maxval', desc: 'Maximum value assigned to thresholded pixels' }], returns: 'numpy.ndarray - Binary image' }
    },

    // ---- Morphology (new) ----

    structuring_element: {
        label: 'Structuring Element', category: 'morph', color: '#607D8B',
        inputs: [],
        outputs: [{ id: 'image', label: 'element' }],
        properties: [
            { key: 'shape', label: 'Shape', type: 'select', default: 'MORPH_RECT',
              options: ['MORPH_RECT', 'MORPH_CROSS', 'MORPH_ELLIPSE', 'custom'] },
            { key: 'width', label: 'Width', type: 'number', default: 5, min: 3, max: 31, step: 2 },
            { key: 'height', label: 'Height', type: 'number', default: 5, min: 3, max: 31, step: 2 },
            { key: 'kernelData', label: 'Custom Kernel', type: 'kernel', default: '1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1' }
        ],
        doc: { signature: 'cv2.getStructuringElement(shape, ksize)', description: 'Creates a structuring element for morphological operations. Choose a preset shape or define a custom binary (0/1) kernel.', params: [{ name: 'shape', desc: 'MORPH_RECT, MORPH_CROSS, MORPH_ELLIPSE, or custom' }, { name: 'ksize', desc: 'Size of the structuring element (width, height)' }], returns: 'numpy.ndarray - Structuring element matrix' }
    },

    // ---- Contour (new category) ----

    draw_contours: {
        label: 'Draw Contours', category: 'contour', color: '#009688',
        inputs: [{ id: 'image', label: 'image' }, { id: 'contours', label: 'contours' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'contourIdx', label: 'Contour Index (-1=all)', type: 'number', default: -1 },
            { key: 'thickness', label: 'Thickness', type: 'number', default: 2, min: 1 },
            { key: 'colorR', label: 'Color R', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorG', label: 'Color G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'Color B', type: 'number', default: 0, min: 0, max: 255 }
        ],
        doc: { signature: 'cv2.drawContours(image, contours, contourIdx, color, thickness)', description: 'Draws contours on the image. Connect Find Contours node to provide contour data.', params: [{ name: 'contourIdx', desc: 'Index of contour to draw (-1 = all)' }, { name: 'thickness', desc: 'Line thickness' }], returns: 'numpy.ndarray - Image with contours drawn' }
    },

    bounding_rect: {
        label: 'Bounding Rect', category: 'contour', color: '#009688',
        inputs: [{ id: 'image', label: 'image' }, { id: 'contours', label: 'contours' }],
        outputs: [{ id: 'image', label: 'image' }, { id: 'coords', label: 'coords' }],
        properties: [
            { key: 'thickness', label: 'Thickness', type: 'number', default: 2, min: 1 },
            { key: 'colorR', label: 'Color R', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorG', label: 'Color G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'Color B', type: 'number', default: 0, min: 0, max: 255 }
        ],
        doc: { signature: 'cv2.boundingRect + cv2.rectangle', description: 'Draws bounding rectangles around contours. Outputs coords as [[x1,y1,x2,y2], ...] for connecting to Image Extract.', params: [{ name: 'thickness', desc: 'Line thickness' }], returns: 'numpy.ndarray - Image with bounding rectangles + coords list' }
    },

    min_enclosing_circle: {
        label: 'Min Enclosing Circle', category: 'contour', color: '#009688',
        inputs: [{ id: 'image', label: 'image' }, { id: 'contours', label: 'contours' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'thickness', label: 'Thickness', type: 'number', default: 2, min: 1 },
            { key: 'colorR', label: 'Color R', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorG', label: 'Color G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'Color B', type: 'number', default: 0, min: 0, max: 255 }
        ],
        doc: { signature: 'cv2.minEnclosingCircle + cv2.circle', description: 'Draws minimum enclosing circles for contours. Connect Find Contours node to provide contour data.', params: [{ name: 'thickness', desc: 'Line thickness' }], returns: 'numpy.ndarray - Image with enclosing circles' }
    },

    convex_hull: {
        label: 'Convex Hull', category: 'contour', color: '#009688',
        inputs: [{ id: 'image', label: 'image' }, { id: 'contours', label: 'contours' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'thickness', label: 'Thickness', type: 'number', default: 2, min: 1 },
            { key: 'colorR', label: 'Color R', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorG', label: 'Color G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'Color B', type: 'number', default: 0, min: 0, max: 255 }
        ],
        doc: { signature: 'cv2.convexHull + cv2.drawContours', description: 'Draws convex hulls around contours. Connect Find Contours node to provide contour data.', params: [{ name: 'thickness', desc: 'Line thickness' }], returns: 'numpy.ndarray - Image with convex hulls' }
    },

    approx_poly: {
        label: 'Approx Poly', category: 'contour', color: '#009688',
        inputs: [{ id: 'image', label: 'image' }, { id: 'contours', label: 'contours' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'epsilon', label: 'Epsilon (fraction)', type: 'number', default: 0.02, step: 0.01 },
            { key: 'closed', label: 'Closed', type: 'checkbox', default: true },
            { key: 'thickness', label: 'Thickness', type: 'number', default: 2, min: 1 },
            { key: 'colorR', label: 'Color R', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorG', label: 'Color G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'Color B', type: 'number', default: 0, min: 0, max: 255 }
        ],
        doc: { signature: 'cv2.approxPolyDP + cv2.drawContours', description: 'Approximates contour shapes to polygons and draws them. Connect Find Contours node to provide contour data.', params: [{ name: 'epsilon', desc: 'Approximation accuracy as fraction of contour perimeter' }, { name: 'closed', desc: 'Whether the approximated curve is closed' }], returns: 'numpy.ndarray - Image with polygon approximations' }
    },

    contour_area: {
        label: 'Contour Area', category: 'contour', color: '#009688',
        inputs: [{ id: 'image', label: 'image' }, { id: 'contours', label: 'contours' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'minArea', label: 'Min Area', type: 'number', default: 100, min: 0 },
            { key: 'maxArea', label: 'Max Area', type: 'number', default: 100000, min: 0 },
            { key: 'thickness', label: 'Thickness', type: 'number', default: 2, min: 1 },
            { key: 'colorR', label: 'Color R', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorG', label: 'Color G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'Color B', type: 'number', default: 0, min: 0, max: 255 }
        ],
        doc: { signature: 'cv2.contourArea + cv2.drawContours', description: 'Filters contours by area range and draws matching ones. Connect Find Contours node to provide contour data.', params: [{ name: 'minArea', desc: 'Minimum contour area to include' }, { name: 'maxArea', desc: 'Maximum contour area to include' }], returns: 'numpy.ndarray - Image with area-filtered contours' }
    },

    contour_properties: {
        label: 'Contour Properties', category: 'contour', color: '#009688',
        inputs: [{ id: 'image', label: 'image' }, { id: 'contours', label: 'contours' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'showArea', label: 'Show Area', type: 'checkbox', default: true },
            { key: 'showPerimeter', label: 'Show Perimeter', type: 'checkbox', default: true },
            { key: 'showCenter', label: 'Show Center', type: 'checkbox', default: true }
        ],
        doc: { signature: 'cv2.contourArea + cv2.arcLength + cv2.moments', description: 'Annotates contours with area, perimeter, and center info. Connect Find Contours node to provide contour data.', params: [{ name: 'showArea', desc: 'Display area text on each contour' }, { name: 'showPerimeter', desc: 'Display perimeter text' }, { name: 'showCenter', desc: 'Display center point marker' }], returns: 'numpy.ndarray - Annotated image' }
    },

    // ---- Feature (new) ----

    harris_corner: {
        label: 'Harris Corner', category: 'feature', color: '#CDDC39',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'blockSize', label: 'Block Size', type: 'number', default: 2, min: 1 },
            { key: 'ksize', label: 'Sobel Kernel Size', type: 'number', default: 3, min: 1, step: 2 },
            { key: 'k', label: 'Harris K', type: 'number', default: 0.04, step: 0.01 }
        ],
        doc: { signature: 'cv2.cornerHarris(src, blockSize, ksize, k)', description: 'Detects corners using the Harris corner detector.', params: [{ name: 'blockSize', desc: 'Neighborhood size' }, { name: 'ksize', desc: 'Aperture parameter for Sobel operator' }, { name: 'k', desc: 'Harris detector free parameter' }], returns: 'numpy.ndarray - Image with corners marked in red' }
    },

    good_features: {
        label: 'Good Features', category: 'feature', color: '#CDDC39',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'maxCorners', label: 'Max Corners', type: 'number', default: 100, min: 1 },
            { key: 'qualityLevel', label: 'Quality Level', type: 'number', default: 0.01, step: 0.01 },
            { key: 'minDistance', label: 'Min Distance', type: 'number', default: 10, min: 1 }
        ],
        doc: { signature: 'cv2.goodFeaturesToTrack(image, maxCorners, qualityLevel, minDistance)', description: 'Shi-Tomasi corner detection. Finds the N strongest corners.', params: [{ name: 'maxCorners', desc: 'Maximum number of corners to return' }, { name: 'qualityLevel', desc: 'Minimum quality level (fraction of best corner quality)' }, { name: 'minDistance', desc: 'Minimum distance between detected corners' }], returns: 'numpy.ndarray - Image with corners marked' }
    },

    orb_features: {
        label: 'ORB', category: 'feature', color: '#CDDC39',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'nFeatures', label: 'Num Features', type: 'number', default: 500, min: 1 }
        ],
        doc: { signature: 'cv2.ORB_create(nfeatures)', description: 'ORB (Oriented FAST and Rotated BRIEF) feature detection and description.', params: [{ name: 'nfeatures', desc: 'Maximum number of features to retain' }], returns: 'numpy.ndarray - Image with keypoints drawn' }
    },

    fast_features: {
        label: 'FAST', category: 'feature', color: '#CDDC39',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'threshold', label: 'Threshold', type: 'number', default: 25, min: 1 },
            { key: 'nonmaxSuppression', label: 'Nonmax Suppression', type: 'checkbox', default: true }
        ],
        doc: { signature: 'cv2.FastFeatureDetector_create(threshold, nonmaxSuppression)', description: 'FAST corner detection algorithm (high speed).', params: [{ name: 'threshold', desc: 'Intensity difference threshold for corner detection' }, { name: 'nonmaxSuppression', desc: 'Whether to apply non-maximum suppression' }], returns: 'numpy.ndarray - Image with keypoints drawn' }
    },

    match_features: {
        label: 'Match Features', category: 'feature', color: '#CDDC39',
        inputs: [{ id: 'image', label: 'image' }, { id: 'image2', label: 'image2' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'nFeatures', label: 'Num Features', type: 'number', default: 500, min: 1 },
            { key: 'matchRatio', label: 'Match Ratio', type: 'number', default: 0.75, step: 0.05, min: 0, max: 1 }
        ],
        doc: { signature: 'cv2.ORB_create + cv2.BFMatcher', description: 'Detects and matches keypoints between two images using ORB and brute-force matcher.', params: [{ name: 'nFeatures', desc: 'Number of ORB features' }, { name: 'matchRatio', desc: 'Lowe ratio test threshold for good matches' }], returns: 'numpy.ndarray - Side-by-side image with match lines drawn' }
    },

    // ---- Drawing (new category) ----

    draw_line: {
        label: 'Draw Line', category: 'drawing', color: '#E91E63',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'x1', label: 'X1', type: 'number', default: 10 }, { key: 'y1', label: 'Y1', type: 'number', default: 10 },
            { key: 'x2', label: 'X2', type: 'number', default: 200 }, { key: 'y2', label: 'Y2', type: 'number', default: 200 },
            { key: 'colorR', label: 'R', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorG', label: 'G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'B', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'thickness', label: 'Thickness', type: 'number', default: 2, min: 1 }
        ],
        doc: { signature: 'cv2.line(img, pt1, pt2, color, thickness)', description: 'Draws a line on the image.', params: [{ name: 'pt1', desc: 'Start point (x1, y1)' }, { name: 'pt2', desc: 'End point (x2, y2)' }], returns: 'numpy.ndarray - Image with line drawn' }
    },

    draw_rectangle: {
        label: 'Draw Rectangle', category: 'drawing', color: '#E91E63',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'x', label: 'X', type: 'number', default: 10 }, { key: 'y', label: 'Y', type: 'number', default: 10 },
            { key: 'width', label: 'Width', type: 'number', default: 100 }, { key: 'height', label: 'Height', type: 'number', default: 100 },
            { key: 'colorR', label: 'R', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorG', label: 'G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'B', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'thickness', label: 'Thickness (-1=filled)', type: 'number', default: 2 }
        ],
        doc: { signature: 'cv2.rectangle(img, pt1, pt2, color, thickness)', description: 'Draws a rectangle on the image.', params: [{ name: 'pt1', desc: 'Top-left corner' }, { name: 'pt2', desc: 'Bottom-right corner' }, { name: 'thickness', desc: 'Line thickness (-1 = filled)' }], returns: 'numpy.ndarray - Image with rectangle drawn' }
    },

    draw_circle: {
        label: 'Draw Circle', category: 'drawing', color: '#E91E63',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'centerX', label: 'Center X', type: 'number', default: 100 },
            { key: 'centerY', label: 'Center Y', type: 'number', default: 100 },
            { key: 'radius', label: 'Radius', type: 'number', default: 50 },
            { key: 'colorR', label: 'R', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorG', label: 'G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'B', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'thickness', label: 'Thickness (-1=filled)', type: 'number', default: 2 }
        ],
        doc: { signature: 'cv2.circle(img, center, radius, color, thickness)', description: 'Draws a circle on the image.', params: [{ name: 'center', desc: 'Center point (x, y)' }, { name: 'radius', desc: 'Radius of the circle' }], returns: 'numpy.ndarray - Image with circle drawn' }
    },

    draw_ellipse: {
        label: 'Draw Ellipse', category: 'drawing', color: '#E91E63',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'centerX', label: 'Center X', type: 'number', default: 100 },
            { key: 'centerY', label: 'Center Y', type: 'number', default: 100 },
            { key: 'axesW', label: 'Axes Width', type: 'number', default: 80 },
            { key: 'axesH', label: 'Axes Height', type: 'number', default: 40 },
            { key: 'angle', label: 'Angle', type: 'number', default: 0 },
            { key: 'colorR', label: 'R', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorG', label: 'G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'B', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'thickness', label: 'Thickness (-1=filled)', type: 'number', default: 2 }
        ],
        doc: { signature: 'cv2.ellipse(img, center, axes, angle, startAngle, endAngle, color, thickness)', description: 'Draws an ellipse on the image.', params: [{ name: 'center', desc: 'Center point' }, { name: 'axes', desc: 'Half of the size of the ellipse main axes (width, height)' }, { name: 'angle', desc: 'Rotation angle in degrees' }], returns: 'numpy.ndarray - Image with ellipse drawn' }
    },

    draw_text: {
        label: 'Draw Text', category: 'drawing', color: '#E91E63',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'text', label: 'Text', type: 'text', default: 'Hello' },
            { key: 'x', label: 'X', type: 'number', default: 50 }, { key: 'y', label: 'Y', type: 'number', default: 50 },
            { key: 'fontScale', label: 'Font Scale', type: 'number', default: 1.0, step: 0.1 },
            { key: 'colorR', label: 'R', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorG', label: 'G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'B', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'thickness', label: 'Thickness', type: 'number', default: 2, min: 1 }
        ],
        doc: { signature: 'cv2.putText(img, text, org, fontFace, fontScale, color, thickness)', description: 'Draws text on the image.', params: [{ name: 'text', desc: 'Text string to draw' }, { name: 'org', desc: 'Bottom-left corner of the text (x, y)' }, { name: 'fontScale', desc: 'Font scale factor' }], returns: 'numpy.ndarray - Image with text drawn' }
    },

    draw_polylines: {
        label: 'Draw Polylines', category: 'drawing', color: '#E91E63',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'points', label: 'Points (x,y;x,y;...)', type: 'text', default: '10,10;100,50;50,100' },
            { key: 'isClosed', label: 'Closed', type: 'checkbox', default: true },
            { key: 'colorR', label: 'R', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorG', label: 'G', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorB', label: 'B', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'thickness', label: 'Thickness', type: 'number', default: 2, min: 1 }
        ],
        doc: { signature: 'cv2.polylines(img, [pts], isClosed, color, thickness)', description: 'Draws polygon/polyline on the image.', params: [{ name: 'pts', desc: 'Points as "x,y;x,y;..." string' }, { name: 'isClosed', desc: 'Whether to close the polygon' }], returns: 'numpy.ndarray - Image with polyline drawn' }
    },

    // ---- Transform (new) ----

    flip: {
        label: 'Flip', category: 'transform', color: '#00BCD4',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'flipCode', label: 'Flip Mode', type: 'select', default: 'Horizontal (1)', options: ['Horizontal (1)', 'Vertical (0)', 'Both (-1)'] }
        ],
        doc: { signature: 'cv2.flip(src, flipCode)', description: 'Flips an image horizontally, vertically, or both.', params: [{ name: 'flipCode', desc: '1 = horizontal, 0 = vertical, -1 = both' }], returns: 'numpy.ndarray - Flipped image' }
    },

    crop: {
        label: 'Crop', category: 'transform', color: '#00BCD4',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'x', label: 'X', type: 'number', default: 0, min: 0 },
            { key: 'y', label: 'Y', type: 'number', default: 0, min: 0 },
            { key: 'width', label: 'Width', type: 'number', default: 100, min: 1 },
            { key: 'height', label: 'Height', type: 'number', default: 100, min: 1 }
        ],
        doc: { signature: 'img[y:y+h, x:x+w]', description: 'Crops a rectangular region of interest (ROI) from the image.', params: [{ name: 'x', desc: 'Left edge of ROI' }, { name: 'y', desc: 'Top edge of ROI' }, { name: 'width', desc: 'Width of ROI' }, { name: 'height', desc: 'Height of ROI' }], returns: 'numpy.ndarray - Cropped image' }
    },

    paste_image: {
        label: 'Paste Image', category: 'transform', color: '#00BCD4',
        inputs: [{ id: 'image', label: 'base image' }, { id: 'overlay', label: 'overlay' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'x', label: 'X (Left)', type: 'number', default: 0, min: 0 },
            { key: 'y', label: 'Y (Top)', type: 'number', default: 0, min: 0 },
            { key: 'mode', label: 'Mode', type: 'select', default: 'overwrite',
              options: ['overwrite', 'blend', 'alpha_channel'] },
            { key: 'opacity', label: 'Opacity', type: 'number', default: 1.0, min: 0, max: 1, step: 0.05 }
        ],
        doc: {
            signature: 'base[y:y+oh, x:x+ow] = overlay',
            description: 'Pastes a smaller overlay image onto a base image at the specified (x, y) position. Overwrite mode replaces pixels directly. Blend mode uses the opacity value for alpha blending. Alpha Channel mode uses the overlay\'s 4th channel (BGRA) as a per-pixel mask.',
            params: [
                { name: 'base', desc: 'Background / target image' },
                { name: 'overlay', desc: 'Foreground image to paste (from Crop, etc.)' },
                { name: 'x, y', desc: 'Top-left position on the base image' },
                { name: 'mode', desc: 'overwrite = replace pixels, blend = alpha blending, alpha_channel = use overlay alpha' },
                { name: 'opacity', desc: 'Blending opacity (0.0~1.0), used in blend and alpha_channel modes' }
            ],
            returns: 'numpy.ndarray - Base image with overlay pasted'
        }
    },

    warp_affine: {
        label: 'Warp Affine', category: 'transform', color: '#00BCD4',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'affinePreset', label: 'Preset', type: 'select', default: 'custom',
              options: ['custom', 'identity', 'translate_50_30', 'rotate_30', 'rotate_45', 'rotate_90', 'scale_half', 'scale_double', 'flip_h', 'flip_v', 'shear_x', 'shear_y'] },
            { key: 'm00', label: 'M[0,0]', type: 'number', default: 1, step: 0.1 },
            { key: 'm01', label: 'M[0,1]', type: 'number', default: 0, step: 0.1 },
            { key: 'm02', label: 'M[0,2] (tx)', type: 'number', default: 0, step: 1 },
            { key: 'm10', label: 'M[1,0]', type: 'number', default: 0, step: 0.1 },
            { key: 'm11', label: 'M[1,1]', type: 'number', default: 1, step: 0.1 },
            { key: 'm12', label: 'M[1,2] (ty)', type: 'number', default: 0, step: 1 }
        ],
        doc: { signature: 'cv2.warpAffine(src, M, dsize)', description: 'Applies an affine transformation (2x3 matrix) to the image. Supports translation, rotation, scaling, and shearing.', params: [{ name: 'M', desc: '2x3 transformation matrix' }, { name: 'dsize', desc: 'Output image size' }], returns: 'numpy.ndarray - Transformed image' }
    },

    warp_perspective: {
        label: 'Warp Perspective', category: 'transform', color: '#00BCD4',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'srcPoints', label: 'Source Points', type: 'perspective_points', role: 'src',
              default: '0,0;300,0;300,300;0,300' },
            { key: 'dstPoints', label: 'Dest Points', type: 'perspective_points', role: 'dst',
              default: '0,0;300,0;300,300;0,300' }
        ],
        doc: { signature: 'cv2.getPerspectiveTransform + cv2.warpPerspective', description: 'Applies perspective transformation using 4 source and 4 destination corner points. Click "Pick" to interactively select points on the preview image.', params: [{ name: 'src points', desc: '4 source corner coordinates (top-left, top-right, bottom-right, bottom-left)' }, { name: 'dst points', desc: '4 destination corner coordinates' }], returns: 'numpy.ndarray - Perspective-transformed image' }
    },

    remap: {
        label: 'Remap', category: 'transform', color: '#00BCD4',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'distortionK', label: 'Distortion K', type: 'number', default: 0.5, step: 0.1, min: -2, max: 2 },
            { key: 'interpolation', label: 'Interpolation', type: 'select', default: 'INTER_LINEAR', options: ['INTER_LINEAR', 'INTER_NEAREST', 'INTER_CUBIC'] }
        ],
        doc: { signature: 'cv2.remap(src, map1, map2, interpolation)', description: 'Applies barrel/pincushion distortion correction via remap.', params: [{ name: 'distortionK', desc: 'Distortion coefficient (positive = barrel, negative = pincushion)' }, { name: 'interpolation', desc: 'Interpolation method' }], returns: 'numpy.ndarray - Remapped image' }
    },

    // ---- Histogram (new) ----

    calc_histogram: {
        label: 'Calc Histogram', category: 'histogram', color: '#673AB7',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'histSize', label: 'Hist Size', type: 'number', default: 256, min: 1 },
            { key: 'normalize', label: 'Normalize', type: 'checkbox', default: true }
        ],
        doc: { signature: 'cv2.calcHist([img], channels, mask, histSize, ranges)', description: 'Calculates and visualizes the image histogram as a graph image.', params: [{ name: 'histSize', desc: 'Number of histogram bins' }, { name: 'normalize', desc: 'Whether to normalize the histogram' }], returns: 'numpy.ndarray - Histogram visualization image (256x200)' }
    },

    // ---- Arithmetic (new) ----

    add: {
        label: 'Add', category: 'arithmetic', color: '#3F51B5',
        inputs: [{ id: 'image', label: 'image' }, { id: 'image2', label: 'image2' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'sizeMismatch', label: 'Size Mismatch', type: 'select', default: 'error',
              options: ['error', 'resize_img2', 'resize_img1'] }
        ],
        doc: { signature: 'cv2.add(src1, src2)', description: 'Per-element sum of two images (saturated). Images must be the same size — use Size Mismatch option to handle different sizes.', params: [{ name: 'src1', desc: 'First input image' }, { name: 'src2', desc: 'Second input image' }], returns: 'numpy.ndarray - Sum image' }
    },

    subtract: {
        label: 'Subtract', category: 'arithmetic', color: '#3F51B5',
        inputs: [{ id: 'image', label: 'image' }, { id: 'image2', label: 'image2' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'sizeMismatch', label: 'Size Mismatch', type: 'select', default: 'error',
              options: ['error', 'resize_img2', 'resize_img1'] }
        ],
        doc: { signature: 'cv2.subtract(src1, src2)', description: 'Per-element difference of two images (saturated). Images must be the same size — use Size Mismatch option to handle different sizes.', params: [{ name: 'src1', desc: 'First input image' }, { name: 'src2', desc: 'Second input image' }], returns: 'numpy.ndarray - Difference image' }
    },

    multiply: {
        label: 'Multiply', category: 'arithmetic', color: '#3F51B5',
        inputs: [{ id: 'image', label: 'image' }, { id: 'image2', label: 'image2' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'scale', label: 'Scale', type: 'number', default: 1.0, step: 0.1 },
            { key: 'sizeMismatch', label: 'Size Mismatch', type: 'select', default: 'error',
              options: ['error', 'resize_img2', 'resize_img1'] }
        ],
        doc: { signature: 'cv2.multiply(src1, src2, scale)', description: 'Per-element product of two images with optional scale. Images must be the same size — use Size Mismatch option to handle different sizes.', params: [{ name: 'src1', desc: 'First input image' }, { name: 'src2', desc: 'Second input image' }, { name: 'scale', desc: 'Scale factor' }], returns: 'numpy.ndarray - Product image' }
    },

    absdiff: {
        label: 'AbsDiff', category: 'arithmetic', color: '#3F51B5',
        inputs: [{ id: 'image', label: 'image' }, { id: 'image2', label: 'image2' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'sizeMismatch', label: 'Size Mismatch', type: 'select', default: 'error',
              options: ['error', 'resize_img2', 'resize_img1'] }
        ],
        doc: { signature: 'cv2.absdiff(src1, src2)', description: 'Per-element absolute difference of two images. Images must be the same size — use Size Mismatch option to handle different sizes.', params: [{ name: 'src1', desc: 'First input image' }, { name: 'src2', desc: 'Second input image' }], returns: 'numpy.ndarray - Absolute difference image' }
    },

    bitwise_xor: {
        label: 'Bitwise XOR', category: 'arithmetic', color: '#3F51B5',
        inputs: [{ id: 'image', label: 'image' }, { id: 'image2', label: 'image2' }, { id: 'mask', label: 'mask', optional: true }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'sizeMismatch', label: 'Size Mismatch', type: 'select', default: 'error',
              options: ['error', 'resize_img2', 'resize_img1'] }
        ],
        doc: { signature: 'cv2.bitwise_xor(src1, src2[, dst[, mask]])', description: 'Per-element bitwise exclusive-or of two images. If mask is provided, operation is applied only where mask is non-zero. Images must be the same size — use Size Mismatch option to handle different sizes.', params: [{ name: 'src1', desc: 'First input image' }, { name: 'src2', desc: 'Second input image' }, { name: 'mask', desc: 'Optional 8-bit single channel mask' }], returns: 'numpy.ndarray - XOR result' }
    },

    // ---- Detection (new category) ----

    haar_cascade: {
        label: 'Haar Cascade', category: 'detection', color: '#FF5722',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }, { id: 'coords', label: 'coords' }],
        properties: [
            { key: 'cascadeType', label: 'Cascade Type', type: 'select', default: 'face', options: ['face', 'eye', 'smile', 'body', 'cat_face'] },
            { key: 'scaleFactor', label: 'Scale Factor', type: 'number', default: 1.1, step: 0.05 },
            { key: 'minNeighbors', label: 'Min Neighbors', type: 'number', default: 5, min: 1 },
            { key: 'minWidth', label: 'Min Width', type: 'number', default: 30, min: 1 },
            { key: 'minHeight', label: 'Min Height', type: 'number', default: 30, min: 1 }
        ],
        doc: { signature: 'cv2.CascadeClassifier.detectMultiScale()', description: 'Object detection using Haar cascade classifiers (face, eyes, smile, etc.). Outputs coords as [[x1,y1,x2,y2], ...] for connecting to Image Extract.', params: [{ name: 'cascadeType', desc: 'Type of cascade: face, eye, smile, body, cat_face' }, { name: 'scaleFactor', desc: 'How much image size is reduced at each scale' }, { name: 'minNeighbors', desc: 'How many neighbors each candidate rectangle should have' }], returns: 'numpy.ndarray - Image with detected objects outlined + coords list' }
    },

    hough_circles: {
        label: 'Hough Circles', category: 'detection', color: '#FF5722',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'dp', label: 'dp', type: 'number', default: 1.2, step: 0.1 },
            { key: 'minDist', label: 'Min Distance', type: 'number', default: 50, min: 1 },
            { key: 'param1', label: 'Param1 (Canny)', type: 'number', default: 100 },
            { key: 'param2', label: 'Param2 (Accum)', type: 'number', default: 30 },
            { key: 'minRadius', label: 'Min Radius', type: 'number', default: 0, min: 0 },
            { key: 'maxRadius', label: 'Max Radius', type: 'number', default: 0, min: 0 }
        ],
        doc: { signature: 'cv2.HoughCircles(image, method, dp, minDist, param1, param2, minRadius, maxRadius)', description: 'Detects circles in an image using the Hough circle transform.', params: [{ name: 'dp', desc: 'Inverse ratio of accumulator resolution' }, { name: 'minDist', desc: 'Minimum distance between circle centers' }, { name: 'param1', desc: 'Upper threshold for Canny edge detector' }, { name: 'param2', desc: 'Accumulator threshold for circle centers' }], returns: 'numpy.ndarray - Image with detected circles drawn' }
    },

    template_match: {
        label: 'Template Match', category: 'detection', color: '#FF5722',
        inputs: [{ id: 'image', label: 'image' }, { id: 'image2', label: 'template' }],
        outputs: [{ id: 'image', label: 'image' }, { id: 'matches', label: 'matches' }],
        properties: [
            { key: 'method', label: 'Method', type: 'select', default: 'TM_CCOEFF_NORMED', options: ['TM_CCOEFF_NORMED', 'TM_CCORR_NORMED', 'TM_SQDIFF_NORMED'] },
            { key: 'threshold', label: 'Threshold', type: 'number', default: 0.8, step: 0.05, min: 0, max: 1 }
        ],
        doc: { signature: 'cv2.matchTemplate(image, templ, method)', description: 'Finds locations in an image that match a template image. Output "matches" port provides a list of [x1,y1,x2,y2] coordinate pairs for each match region.', params: [{ name: 'image', desc: 'Source image to search in' }, { name: 'templ', desc: 'Template image to search for' }, { name: 'method', desc: 'Matching method' }, { name: 'threshold', desc: 'Confidence threshold for matches' }], returns: 'image: matched regions outlined, matches: list of [x1,y1,x2,y2]' }
    },

    // ---- Segmentation (new category) ----

    flood_fill: {
        label: 'Flood Fill', category: 'segmentation', color: '#795548',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'seedX', label: 'Seed X', type: 'number', default: 0, min: 0 },
            { key: 'seedY', label: 'Seed Y', type: 'number', default: 0, min: 0 },
            { key: 'colorR', label: 'Fill R', type: 'number', default: 255, min: 0, max: 255 },
            { key: 'colorG', label: 'Fill G', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'colorB', label: 'Fill B', type: 'number', default: 0, min: 0, max: 255 },
            { key: 'loDiff', label: 'Lower Diff', type: 'number', default: 20, min: 0, max: 255 },
            { key: 'upDiff', label: 'Upper Diff', type: 'number', default: 20, min: 0, max: 255 }
        ],
        doc: { signature: 'cv2.floodFill(image, mask, seedPoint, newVal, loDiff, upDiff)', description: 'Fills a connected component starting from seed point with the specified color.', params: [{ name: 'seedPoint', desc: 'Starting point for the flood fill' }, { name: 'newVal', desc: 'New color to fill with (B, G, R)' }, { name: 'loDiff', desc: 'Max lower brightness/color difference' }, { name: 'upDiff', desc: 'Max upper brightness/color difference' }], returns: 'numpy.ndarray - Flood-filled image' }
    },

    grabcut: {
        label: 'GrabCut', category: 'segmentation', color: '#795548',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'x', label: 'Rect X', type: 'number', default: 10, min: 0 },
            { key: 'y', label: 'Rect Y', type: 'number', default: 10, min: 0 },
            { key: 'width', label: 'Rect Width', type: 'number', default: 200, min: 1 },
            { key: 'height', label: 'Rect Height', type: 'number', default: 200, min: 1 },
            { key: 'iterations', label: 'Iterations', type: 'number', default: 5, min: 1, max: 20 }
        ],
        doc: { signature: 'cv2.grabCut(img, mask, rect, bgdModel, fgdModel, iterCount, mode)', description: 'GrabCut algorithm for foreground/background segmentation using a bounding rectangle.', params: [{ name: 'rect', desc: 'Bounding rectangle (x, y, w, h) containing the foreground object' }, { name: 'iterCount', desc: 'Number of iterations' }], returns: 'numpy.ndarray - Image with background removed (black)' }
    },

    watershed: {
        label: 'Watershed', category: 'segmentation', color: '#795548',
        inputs: [{ id: 'image', label: 'image' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'markerSize', label: 'Marker Kernel Size', type: 'number', default: 10, min: 1 }
        ],
        doc: { signature: 'cv2.watershed(image, markers)', description: 'Watershed algorithm for image segmentation. Automatically generates markers via morphological operations.', params: [{ name: 'markerSize', desc: 'Kernel size for marker generation (larger = fewer segments)' }], returns: 'numpy.ndarray - Image with watershed boundaries drawn in red' }
    },

    // ---- Value (new category) ----

    val_integer: {
        label: 'Integer', category: 'value', color: '#78909C',
        inputs: [],
        outputs: [{ id: 'value', label: 'value' }],
        properties: [
            { key: 'value', label: 'Value', type: 'number', default: 0 },
            { key: 'min', label: 'Min', type: 'number', default: 0 },
            { key: 'max', label: 'Max', type: 'number', default: 255 }
        ],
        doc: { signature: 'Integer Value Node', description: 'Outputs a constant integer value.', params: [{ name: 'value', desc: 'Integer value to output' }], returns: 'int - The configured integer value' }
    },

    val_float: {
        label: 'Float', category: 'value', color: '#78909C',
        inputs: [],
        outputs: [{ id: 'value', label: 'value' }],
        properties: [
            { key: 'value', label: 'Value', type: 'number', default: 0.0, step: 0.1 }
        ],
        doc: { signature: 'Float Value Node', description: 'Outputs a constant floating-point value.', params: [{ name: 'value', desc: 'Float value to output' }], returns: 'float - The configured float value' }
    },

    val_boolean: {
        label: 'Boolean', category: 'value', color: '#78909C',
        inputs: [],
        outputs: [{ id: 'value', label: 'value' }],
        properties: [
            { key: 'value', label: 'Value', type: 'checkbox', default: false }
        ],
        doc: { signature: 'Boolean Value Node', description: 'Outputs a constant boolean value (true/false).', params: [{ name: 'value', desc: 'Boolean value to output' }], returns: 'bool - True or False' }
    },

    val_point: {
        label: 'Point', category: 'value', color: '#78909C',
        inputs: [],
        outputs: [{ id: 'value', label: 'point' }],
        properties: [
            { key: 'x', label: 'X', type: 'number', default: 0 },
            { key: 'y', label: 'Y', type: 'number', default: 0 }
        ],
        doc: { signature: 'Point Value Node', description: 'Outputs a 2D point value (x, y).', params: [{ name: 'x', desc: 'X coordinate' }, { name: 'y', desc: 'Y coordinate' }], returns: 'tuple(int, int) - (x, y) point' }
    },

    val_scalar: {
        label: 'Scalar', category: 'value', color: '#78909C',
        inputs: [],
        outputs: [{ id: 'value', label: 'scalar' }],
        properties: [
            { key: 'v0', label: 'V0', type: 'number', default: 0 },
            { key: 'v1', label: 'V1', type: 'number', default: 0 },
            { key: 'v2', label: 'V2', type: 'number', default: 0 },
            { key: 'v3', label: 'V3', type: 'number', default: 0 }
        ],
        doc: { signature: 'Scalar Value Node', description: 'Outputs a 4-element scalar value (used for colors, etc.).', params: [{ name: 'v0-v3', desc: 'Scalar components' }], returns: 'tuple - (v0, v1, v2, v3) scalar' }
    },

    val_math: {
        label: 'Math Operation', category: 'value', color: '#78909C',
        inputs: [{ id: 'a', label: 'A' }, { id: 'b', label: 'B' }],
        outputs: [{ id: 'value', label: 'result' }],
        properties: [
            { key: 'operation', label: 'Operation', type: 'select', default: 'add', options: ['add', 'subtract', 'multiply', 'divide', 'modulo', 'power', 'min', 'max'] }
        ],
        doc: { signature: 'Math Operation Node', description: 'Performs basic mathematical operations on two numeric inputs.', params: [{ name: 'operation', desc: 'Math operation: add, subtract, multiply, divide, modulo, power, min, max' }], returns: 'float - Result of the operation' }
    },

    val_list: {
        label: 'List Index/Slice', category: 'value', color: '#78909C',
        inputs: [{ id: 'list', label: 'list' }],
        outputs: [{ id: 'value', label: 'result' }],
        properties: [
            { key: 'mode', label: 'Mode', type: 'select', default: 'index', options: ['index', 'slice'] },
            { key: 'index', label: 'Index', type: 'number', default: 0 },
            { key: 'start', label: 'Start (slice)', type: 'number', default: 0 },
            { key: 'stop', label: 'Stop (slice, -1=end)', type: 'number', default: -1 },
            { key: 'step', label: 'Step (slice)', type: 'number', default: 1, min: 1 }
        ],
        doc: { signature: 'List Index/Slice Node', description: 'Selects an element or a sub-list from an input list using indexing or slicing.', params: [{ name: 'mode', desc: 'index: single element, slice: sub-list' }, { name: 'index', desc: 'Index for single element (0-based, negative supported)' }, { name: 'start', desc: 'Slice start index' }, { name: 'stop', desc: 'Slice stop index (-1 means end)' }, { name: 'step', desc: 'Slice step' }], returns: 'value - indexed element or sliced sub-list' }
    },

    val_coords: {
        label: 'Coords', category: 'value', color: '#78909C',
        inputs: [],
        outputs: [{ id: 'coords', label: 'coords' }],
        properties: [
            { key: 'mode', label: 'Mode', type: 'select', default: 'single',
              options: ['single', 'multi'] },
            { key: 'x1', label: 'X1 (Left)', type: 'number', default: 0, min: 0 },
            { key: 'y1', label: 'Y1 (Top)', type: 'number', default: 0, min: 0 },
            { key: 'x2', label: 'X2 (Right)', type: 'number', default: 100 },
            { key: 'y2', label: 'Y2 (Bottom)', type: 'number', default: 100 },
            { key: 'coordsList', label: 'Coords List (multi)', type: 'textarea',
              default: '0,0,100,100\n50,50,200,200' }
        ],
        doc: {
            signature: 'Coordinate Value Node',
            description: 'Outputs coordinates in [[x1,y1,x2,y2], ...] format. Single mode uses the 4 number inputs. Multi mode parses the textarea (one "x1,y1,x2,y2" per line). Connect output to Image Extract coords input.',
            params: [
                { name: 'mode', desc: 'single: one rectangle from X1/Y1/X2/Y2. multi: multiple rectangles from textarea' },
                { name: 'x1,y1', desc: 'Top-left corner' },
                { name: 'x2,y2', desc: 'Bottom-right corner' },
                { name: 'coordsList', desc: 'One "x1,y1,x2,y2" per line for multi mode' }
            ],
            returns: 'list - [[x1,y1,x2,y2], ...] coordinate list'
        }
    },

    // ---- Detection (extra) ----

    image_extract: {
        label: 'Image Extract', category: 'detection', color: '#FF5722',
        inputs: [{ id: 'image', label: 'image' }, { id: 'coords', label: 'coords' }],
        outputs: [{ id: 'image', label: 'image' }],
        properties: [
            { key: 'padding', label: 'Padding', type: 'number', default: 0, min: 0 }
        ],
        doc: { signature: 'Image Extract Node', description: 'Extracts a region from the source image using coordinate input [x1,y1,x2,y2]. Accepts a single coordinate or list of coordinates (uses the first one). Connect "matches" output from Template Match or provide a list manually.', params: [{ name: 'coords', desc: 'Coordinate [x1,y1,x2,y2] or list of coordinates from Template Match' }, { name: 'padding', desc: 'Extra padding pixels around the region' }], returns: 'numpy.ndarray - Extracted image region' }
    }
};
