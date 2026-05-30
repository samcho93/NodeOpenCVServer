/**
 * NodeOpenCV - Canvas-based flow editor
 * Complete rewrite with proper backend integration, multi-port support,
 * auto-preview, Image Show popup, and optimized data flow.
 */
(function () {
    'use strict';

    // ===== Session Management =====
    function getSessionId() {
        // Read from cookie or generate new
        const match = document.cookie.match(/(?:^|;\s*)session_id=([^;]*)/);
        if (match) return match[1];
        const sid = crypto.randomUUID ? crypto.randomUUID().replace(/-/g, '').slice(0, 12)
            : Math.random().toString(36).slice(2, 14);
        document.cookie = `session_id=${sid}; path=/; max-age=${3600 * 4}; SameSite=Lax`;
        return sid;
    }
    const SESSION_ID = getSessionId();

    // Wrapper for fetch that includes session header
    const _origFetch = window.fetch;
    function sessionFetch(url, options = {}) {
        options.headers = options.headers || {};
        if (typeof options.headers.set === 'function') {
            options.headers.set('X-Session-ID', SESSION_ID);
        } else {
            options.headers['X-Session-ID'] = SESSION_ID;
        }
        return _origFetch(url, options);
    }

    // ===== State =====
    const state = {
        nodes: [],
        connections: [],
        selectedNode: null,
        selectedNodes: [],          // multi-selection
        draggingNode: null,
        draggingMulti: false,       // dragging multiple selected nodes
        dragOffset: { x: 0, y: 0 },
        dragOffsets: [],            // per-node offsets for multi-drag
        connecting: null,
        connectingMouse: { x: 0, y: 0 },
        pan: { x: 0, y: 0 },
        zoom: 1,
        isPanning: false,
        panStart: { x: 0, y: 0 },
        mousePos: { x: 0, y: 0 },
        nodeResults: {},
        nextNodeId: 1,
        executing: false,
        // Rubber-band selection
        selecting: false,
        selectionRect: null,        // {x1, y1, x2, y2} in world coords
        // Clipboard
        clipboard: null,            // { nodes: [...], connections: [...] }
        // Undo stack
        undoStack: [],
        undoMaxSize: 50,
    };

    const NODE_WIDTH = 170;
    const PORT_RADIUS = 7;

    // ===== Kernel/Matrix Helpers =====
    const KERNEL_PRESETS = {
        identity:    (n) => { const k = new Array(n*n).fill(0); k[Math.floor(n*n/2)] = 1; return k; },
        sharpen:     () => [0,-1,0,-1,5,-1,0,-1,0],
        edge_detect: () => [-1,-1,-1,-1,8,-1,-1,-1,-1],
        emboss:      () => [-2,-1,0,-1,1,1,0,1,2],
        ridge:       () => [-1,-1,-1,-1,4,-1,-1,-1,-1],
        blur:        (n) => new Array(n*n).fill(+(1/(n*n)).toFixed(4)),
    };
    function _getKernelValues(preset, ksize, kernelDataStr) {
        if (preset === 'custom') {
            try { return kernelDataStr.split(',').map(v => parseFloat(v.trim()) || 0); }
            catch { return new Array(ksize * ksize).fill(0); }
        }
        const fn = KERNEL_PRESETS[preset];
        return fn ? fn(ksize) : KERNEL_PRESETS.sharpen();
    }

    const AFFINE_PRESETS = {
        custom:          null, // keep current
        identity:        [1,0,0, 0,1,0],
        translate_50_30: [1,0,50, 0,1,30],
        rotate_30:       [0.866,-0.5,0, 0.5,0.866,0],
        rotate_45:       [0.707,-0.707,0, 0.707,0.707,0],
        rotate_90:       [0,-1,0, 1,0,0],
        scale_half:      [0.5,0,0, 0,0.5,0],
        scale_double:    [2,0,0, 0,2,0],
        flip_h:          [-1,0,0, 0,1,0],
        flip_v:          [1,0,0, 0,-1,0],
        shear_x:         [1,0.3,0, 0,1,0],
        shear_y:         [1,0,0, 0.3,1,0],
    };
    const GRID_SIZE = 20;

    const canvas = document.getElementById('node-canvas');
    const ctx = canvas.getContext('2d');
    const container = document.getElementById('canvas-container');

    // ===== Node preview image cache =====
    const _previewImageCache = {};  // nodeId -> { dataUrl, img (HTMLImageElement), loaded }

    // When true, the next executePipeline() will skip auto-opening Image Show popups.
    // Used so that auto-execute after Load doesn't spam the screen with windows.
    let _suppressImageShowPopup = false;

    function getPreviewImage(nodeId) {
        const result = state.nodeResults[nodeId];
        if (!result || !result.preview) return null;
        const cached = _previewImageCache[nodeId];
        if (cached && cached.dataUrl === result.preview) {
            // Same data URL — return image if loaded, null if still loading
            return cached.loaded ? cached.img : null;
        }
        // New or changed result — create Image object once
        const img = new window.Image();
        const entry = { dataUrl: result.preview, img, loaded: false };
        img.onload = () => { entry.loaded = true; draw(); };
        img.src = result.preview;
        _previewImageCache[nodeId] = entry;
        return null;
    }

    // ===== Port data-type compatibility =====
    const PORT_TYPE_MAP = {
        // Image types (numpy.ndarray)
        'image': 'image', 'image2': 'image', 'mask': 'image',
        'ch0': 'image', 'ch1': 'image', 'ch2': 'image',
        'true': 'image', 'false': 'image',
        'case0': 'image', 'case1': 'image', 'case2': 'image',
        // Contours type (list of numpy arrays from findContours)
        'contours': 'contours',
        // List / coordinate types
        'list': 'list', 'coords': 'list', 'matches': 'list',
        // Value types (number / bool / point / scalar)
        'value': 'value', 'a': 'value', 'b': 'value',
    };
    const PORT_TYPE_LABELS = { image: '이미지', contours: '컨투어', list: '리스트', value: '값' };

    function getPortDataType(portId) {
        return PORT_TYPE_MAP[portId] || 'any';
    }
    function arePortsCompatible(srcPortId, tgtPortId) {
        const s = getPortDataType(srcPortId);
        const t = getPortDataType(tgtPortId);
        if (s === 'any' || t === 'any') return true;
        return s === t;
    }

    // ===== Smart preview: image-output nodes show preview, value-only nodes show info text =====
    const IMAGE_OUTPUT_IDS = new Set(['image', 'ch0', 'ch1', 'ch2', 'true', 'false', 'case0', 'case1', 'case2']);
    const IMAGE_INPUT_SINK_NODES = new Set(['image_show', 'image_write', 'video_write']);
    function nodeNeedsImagePreview(nodeType) {
        if (IMAGE_INPUT_SINK_NODES.has(nodeType)) return true;
        const def = NODE_DEFS[nodeType];
        if (!def) return false;
        return def.outputs.some(o => IMAGE_OUTPUT_IDS.has(o.id));
    }

    // ===== Dynamic node height based on port count + preview =====
    const PREVIEW_THUMB_H = 70;
    const PREVIEW_THUMB_PAD = 4;
    const INFO_TEXT_H = 18;

    function getNodeHeight(node) {
        if (node && node.type === 'text_label') {
            return _textLabelMetrics(node).height;
        }
        const def = NODE_DEFS[node.type];
        if (!def) return 40;
        const maxPorts = Math.max(def.inputs.length, def.outputs.length, 1);
        let baseH = Math.max(40, maxPorts * 24 + 16);
        const result = state.nodeResults[node.id];
        if (result && result.preview && nodeNeedsImagePreview(node.type)) {
            // Image preview area
            baseH += PREVIEW_THUMB_H + PREVIEW_THUMB_PAD * 2;
        } else if (result && result.info && !nodeNeedsImagePreview(node.type)) {
            // Value/info text area (compact)
            baseH += INFO_TEXT_H;
        }
        return baseH;
    }

    // ===== Undo System =====
    function captureState() {
        // Deep-snapshot nodes (with all properties), connections, and nextNodeId
        return {
            nodes: state.nodes.map(n => ({
                id: n.id,
                type: n.type,
                x: n.x,
                y: n.y,
                label: n.label,
                properties: JSON.parse(JSON.stringify(n.properties)),
            })),
            connections: state.connections.map(c => ({ ...c })),
            nextNodeId: state.nextNodeId,
        };
    }

    function pushUndo() {
        state.undoStack.push(captureState());
        if (state.undoStack.length > state.undoMaxSize) {
            state.undoStack.shift();
        }
    }

    function undo() {
        if (state.undoStack.length === 0) {
            setStatus('Nothing to undo', '');
            return;
        }
        const snapshot = state.undoStack.pop();
        state.nodes = snapshot.nodes;
        state.connections = snapshot.connections;
        state.nextNodeId = snapshot.nextNodeId;
        state.selectedNode = null;
        state.selectedNodes = [];
        selectNode(null);
        updateStatusBar();
        draw();
        setStatus('Undo', '');
    }

    // ===== Multi-selection helpers =====
    function isNodeSelected(node) {
        return state.selectedNodes.some(n => n.id === node.id);
    }

    function addToSelection(node) {
        if (!isNodeSelected(node)) {
            state.selectedNodes.push(node);
        }
    }

    function removeFromSelection(node) {
        state.selectedNodes = state.selectedNodes.filter(n => n.id !== node.id);
    }

    function clearSelection() {
        state.selectedNodes = [];
        state.selectedNode = null;
    }

    function getNodesInRect(x1, y1, x2, y2) {
        const minX = Math.min(x1, x2);
        const maxX = Math.max(x1, x2);
        const minY = Math.min(y1, y2);
        const maxY = Math.max(y1, y2);
        return state.nodes.filter(n => {
            const h = getNodeHeight(n);
            const w = getNodeWidth(n);
            return n.x + w >= minX && n.x <= maxX && n.y + h >= minY && n.y <= maxY;
        });
    }

    // ===== Clipboard (Copy / Cut / Paste) =====
    function copySelected() {
        const nodes = state.selectedNodes.length > 0 ? state.selectedNodes : (state.selectedNode ? [state.selectedNode] : []);
        if (nodes.length === 0) {
            setStatus('No nodes selected to copy', '');
            return;
        }
        const nodeIds = new Set(nodes.map(n => n.id));
        // Copy connections that are entirely within the selection
        const conns = state.connections.filter(c => nodeIds.has(c.sourceNode) && nodeIds.has(c.targetNode));
        state.clipboard = {
            nodes: nodes.map(n => ({
                id: n.id,
                type: n.type,
                x: n.x,
                y: n.y,
                label: n.label,
                properties: JSON.parse(JSON.stringify(n.properties)),
            })),
            connections: conns.map(c => ({ ...c })),
        };
        setStatus(`Copied ${nodes.length} node(s)`, 'success');
    }

    function cutSelected() {
        const nodes = state.selectedNodes.length > 0 ? state.selectedNodes : (state.selectedNode ? [state.selectedNode] : []);
        if (nodes.length === 0) {
            setStatus('No nodes selected to cut', '');
            return;
        }
        pushUndo();
        copySelected();
        // Delete the selected nodes
        const nodeIds = new Set(nodes.map(n => n.id));
        state.nodes = state.nodes.filter(n => !nodeIds.has(n.id));
        state.connections = state.connections.filter(c => !nodeIds.has(c.sourceNode) && !nodeIds.has(c.targetNode));
        clearSelection();
        selectNode(null);
        updateStatusBar();
        draw();
        setStatus(`Cut ${nodeIds.size} node(s)`, 'success');
    }

    function pasteClipboard() {
        if (!state.clipboard || state.clipboard.nodes.length === 0) {
            setStatus('Clipboard is empty', '');
            return;
        }
        pushUndo();
        const oldToNew = {};
        const offset = 40; // pixel offset to avoid exact overlap
        const newNodes = [];
        for (const orig of state.clipboard.nodes) {
            const newId = 'node_' + (state.nextNodeId++);
            oldToNew[orig.id] = newId;
            const n = {
                id: newId,
                type: orig.type,
                x: orig.x + offset,
                y: orig.y + offset,
                label: orig.label,
                properties: JSON.parse(JSON.stringify(orig.properties)),
            };
            // Clear imageId references (server-side images won't carry over reliably)
            // Keep other properties intact
            state.nodes.push(n);
            newNodes.push(n);
        }
        // Recreate internal connections with new IDs
        for (const conn of state.clipboard.connections) {
            if (oldToNew[conn.sourceNode] && oldToNew[conn.targetNode]) {
                state.connections.push({
                    sourceNode: oldToNew[conn.sourceNode],
                    sourcePort: conn.sourcePort,
                    targetNode: oldToNew[conn.targetNode],
                    targetPort: conn.targetPort,
                });
            }
        }
        // Select the pasted nodes
        clearSelection();
        state.selectedNodes = newNodes;
        if (newNodes.length === 1) {
            state.selectedNode = newNodes[0];
            selectNode(newNodes[0]);
        } else {
            selectNode(null);
        }
        updateStatusBar();
        draw();
        setStatus(`Pasted ${newNodes.length} node(s)`, 'success');
    }

    // ===== Canvas sizing =====
    function resizeCanvas() {
        canvas.width = container.clientWidth;
        canvas.height = container.clientHeight;
        draw();
    }
    window.addEventListener('resize', resizeCanvas);
    resizeCanvas();

    // ===== Coordinate helpers =====
    function screenToWorld(sx, sy) {
        return {
            x: (sx - state.pan.x) / state.zoom,
            y: (sy - state.pan.y) / state.zoom,
        };
    }

    // ===== Port positions =====
    // Get the label area height (top portion without preview thumbnail)
    function getNodeLabelHeight(node) {
        const def = NODE_DEFS[node.type];
        if (!def) return 40;
        const maxPorts = Math.max(def.inputs.length, def.outputs.length, 1);
        return Math.max(40, maxPorts * 24 + 16);
    }

    // ===== Text Label helpers (annotation-only node with variable size) =====
    function _textLabelFontSpec(node) {
        const p = node.properties || {};
        const size = Math.max(6, Number(p.fontSize) || 24);
        const fam = p.fontFamily || 'Segoe UI';
        const weight = p.bold ? 'bold ' : '';
        const style = p.italic ? 'italic ' : '';
        // Always include sans-serif fallback so missing fonts don't break layout
        return { spec: `${style}${weight}${size}px "${fam}", sans-serif`, size };
    }

    function _textLabelMetrics(node) {
        // Returns { width, height } in world units (canvas pre-transform pixels)
        const text = String((node.properties && node.properties.text) || 'Text');
        const { spec, size } = _textLabelFontSpec(node);
        const lines = text.split(/\r?\n/);
        ctx.save();
        ctx.font = spec;
        let maxW = 0;
        for (const ln of lines) {
            const w = ctx.measureText(ln || ' ').width;
            if (w > maxW) maxW = w;
        }
        ctx.restore();
        const lineH = size * 1.25;
        return {
            width:  Math.max(40, Math.ceil(maxW) + 12),
            height: Math.max(size + 8, Math.ceil(lines.length * lineH) + 8),
            lineH,
            fontSpec: spec,
            fontSize: size,
            lines,
        };
    }

    function getNodeWidth(node) {
        if (node && node.type === 'text_label') {
            return _textLabelMetrics(node).width;
        }
        return NODE_WIDTH;
    }

    function getInputPorts(node) {
        const def = NODE_DEFS[node.type];
        if (!def) return [];
        const h = getNodeLabelHeight(node);
        return def.inputs.map((port, i) => {
            const spacing = h / (def.inputs.length + 1);
            return {
                id: port.id,
                label: port.label,
                x: node.x,
                y: node.y + spacing * (i + 1),
                type: 'input',
                optional: !!port.optional,
            };
        });
    }

    function getOutputPorts(node) {
        const def = NODE_DEFS[node.type];
        if (!def) return [];
        const h = getNodeLabelHeight(node);
        return def.outputs.map((port, i) => {
            const spacing = h / (def.outputs.length + 1);
            return {
                id: port.id,
                label: port.label,
                x: node.x + NODE_WIDTH,
                y: node.y + spacing * (i + 1),
                type: 'output',
            };
        });
    }

    // ===== Hit testing =====
    function hitTestNode(wx, wy) {
        for (let i = state.nodes.length - 1; i >= 0; i--) {
            const n = state.nodes[i];
            const h = getNodeHeight(n);
            const w = getNodeWidth(n);
            if (wx >= n.x && wx <= n.x + w && wy >= n.y && wy <= n.y + h) {
                return n;
            }
        }
        return null;
    }

    function hitTestPort(wx, wy) {
        for (const node of state.nodes) {
            for (const p of [...getInputPorts(node), ...getOutputPorts(node)]) {
                const dx = wx - p.x;
                const dy = wy - p.y;
                if (dx * dx + dy * dy <= (PORT_RADIUS + 4) * (PORT_RADIUS + 4)) {
                    return { nodeId: node.id, port: p };
                }
            }
        }
        return null;
    }

    function hitTestConnection(wx, wy) {
        const threshold = 8;
        for (let i = 0; i < state.connections.length; i++) {
            const conn = state.connections[i];
            const srcNode = state.nodes.find(n => n.id === conn.sourceNode);
            const tgtNode = state.nodes.find(n => n.id === conn.targetNode);
            if (!srcNode || !tgtNode) continue;
            const srcPort = getOutputPorts(srcNode).find(p => p.id === conn.sourcePort);
            const tgtPort = getInputPorts(tgtNode).find(p => p.id === conn.targetPort);
            if (!srcPort || !tgtPort) continue;
            if (pointNearBezier(wx, wy, srcPort.x, srcPort.y, tgtPort.x, tgtPort.y, threshold)) {
                return i;
            }
        }
        return -1;
    }

    function pointNearBezier(px, py, x1, y1, x2, y2, threshold) {
        const cpDist = Math.max(Math.abs(x2 - x1) * 0.4, 30);
        for (let t = 0; t <= 1; t += 0.02) {
            const u = 1 - t;
            const bx = u * u * u * x1 + 3 * u * u * t * (x1 + cpDist) + 3 * u * t * t * (x2 - cpDist) + t * t * t * x2;
            const by = u * u * u * y1 + 3 * u * u * t * y1 + 3 * u * t * t * y2 + t * t * t * y2;
            if (Math.abs(px - bx) < threshold && Math.abs(py - by) < threshold) return true;
        }
        return false;
    }

    // ===== Drawing =====
    function draw() {
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        ctx.save();
        ctx.translate(state.pan.x, state.pan.y);
        ctx.scale(state.zoom, state.zoom);

        drawGrid();
        drawConnections();
        drawNodes();
        drawConnecting();
        drawSelectionRect();

        ctx.restore();
    }

    function drawGrid() {
        const gs = GRID_SIZE;
        const startX = Math.floor(-state.pan.x / state.zoom / gs) * gs - gs;
        const startY = Math.floor(-state.pan.y / state.zoom / gs) * gs - gs;
        const endX = startX + canvas.width / state.zoom + gs * 2;
        const endY = startY + canvas.height / state.zoom + gs * 2;

        ctx.strokeStyle = '#252538';
        ctx.lineWidth = 0.5;
        for (let x = startX; x <= endX; x += gs) {
            ctx.beginPath();
            ctx.moveTo(x, startY);
            ctx.lineTo(x, endY);
            ctx.stroke();
        }
        for (let y = startY; y <= endY; y += gs) {
            ctx.beginPath();
            ctx.moveTo(startX, y);
            ctx.lineTo(endX, y);
            ctx.stroke();
        }
    }

    // Rounded rect helper (compatible with older browsers)
    function roundedRect(x, y, w, h, r) {
        if (typeof r === 'number') r = [r, r, r, r];
        ctx.beginPath();
        ctx.moveTo(x + r[0], y);
        ctx.lineTo(x + w - r[1], y);
        ctx.quadraticCurveTo(x + w, y, x + w, y + r[1]);
        ctx.lineTo(x + w, y + h - r[2]);
        ctx.quadraticCurveTo(x + w, y + h, x + w - r[2], y + h);
        ctx.lineTo(x + r[3], y + h);
        ctx.quadraticCurveTo(x, y + h, x, y + h - r[3]);
        ctx.lineTo(x, y + r[0]);
        ctx.quadraticCurveTo(x, y, x + r[0], y);
        ctx.closePath();
    }

    function drawNodes() {
        for (const node of state.nodes) {
            const def = NODE_DEFS[node.type];
            if (!def) continue;
            const isSelected = (state.selectedNode && state.selectedNode.id === node.id) || isNodeSelected(node);

            // --- Text Label: render plain text, no box, no ports ---
            if (node.type === 'text_label') {
                const m = _textLabelMetrics(node);
                const color = (node.properties && node.properties.color) || '#cdd6f4';
                ctx.save();
                ctx.font = m.fontSpec;
                ctx.fillStyle = color;
                ctx.textAlign = 'left';
                ctx.textBaseline = 'top';
                for (let i = 0; i < m.lines.length; i++) {
                    ctx.fillText(m.lines[i], node.x + 6, node.y + 4 + i * m.lineH);
                }
                if (isSelected) {
                    ctx.setLineDash([5, 4]);
                    ctx.strokeStyle = '#89b4fa';
                    ctx.lineWidth = 1.5;
                    ctx.strokeRect(node.x - 3, node.y - 3, m.width + 6, m.height + 6);
                    ctx.setLineDash([]);
                }
                ctx.restore();
                continue;
            }

            const h = getNodeHeight(node);
            const result = state.nodeResults[node.id];
            const hasError = result?.error;
            const hasResult = result?.preview || (result?.info && !result?.error);

            // Shadow
            ctx.fillStyle = 'rgba(0,0,0,0.25)';
            roundedRect(node.x + 2, node.y + 2, NODE_WIDTH, h, 6);
            ctx.fill();

            // Body
            ctx.fillStyle = isSelected ? '#3a3a5c' : '#2a2a40';
            ctx.strokeStyle = isSelected ? '#89b4fa' : (hasError ? '#f38ba8' : (hasResult ? '#a6e3a1' : '#45475a'));
            ctx.lineWidth = isSelected ? 2.5 : 1;
            roundedRect(node.x, node.y, NODE_WIDTH, h, 6);
            ctx.fill();
            ctx.stroke();

            // Color bar
            ctx.fillStyle = def.color;
            roundedRect(node.x, node.y, 6, h, [6, 0, 0, 6]);
            ctx.fill();

            // Calculate label area (top portion of node, above preview)
            const labelAreaH = getNodeLabelHeight(node);
            const labelCenterY = node.y + labelAreaH / 2;

            // Label (center aligned in label area)
            const centerX = node.x + NODE_WIDTH / 2;
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            if (node.label) {
                ctx.fillStyle = '#cdd6f4';
                ctx.font = 'bold 12px "Segoe UI", sans-serif';
                ctx.fillText(def.label, centerX, labelCenterY - 7);
                ctx.fillStyle = '#7f849c';
                ctx.font = '10px "Segoe UI", sans-serif';
                ctx.fillText(node.label.length > 18 ? node.label.substring(0, 17) + '...' : node.label, centerX, labelCenterY + 7);
            } else {
                ctx.fillStyle = '#cdd6f4';
                ctx.font = 'bold 12px "Segoe UI", sans-serif';
                ctx.fillText(def.label, centerX, labelCenterY);
            }

            // Preview thumbnail inside node (image nodes only)
            if (result?.preview && nodeNeedsImagePreview(node.type)) {
                const previewImg = getPreviewImage(node.id);
                const thumbPad = PREVIEW_THUMB_PAD;
                const thumbX = node.x + 10;
                const thumbY = node.y + labelAreaH + thumbPad;
                const thumbW = NODE_WIDTH - 20;
                const thumbH = PREVIEW_THUMB_H;
                // Dark background for preview area
                ctx.fillStyle = '#1a1a2e';
                ctx.fillRect(thumbX, thumbY, thumbW, thumbH);
                // Draw the image if loaded
                if (previewImg) {
                    try {
                        // Maintain aspect ratio
                        const imgAspect = previewImg.naturalWidth / previewImg.naturalHeight;
                        const boxAspect = thumbW / thumbH;
                        let dw, dh, dx, dy;
                        if (imgAspect > boxAspect) {
                            dw = thumbW;
                            dh = thumbW / imgAspect;
                            dx = thumbX;
                            dy = thumbY + (thumbH - dh) / 2;
                        } else {
                            dh = thumbH;
                            dw = thumbH * imgAspect;
                            dx = thumbX + (thumbW - dw) / 2;
                            dy = thumbY;
                        }
                        ctx.drawImage(previewImg, dx, dy, dw, dh);
                    } catch(e) { /* ignore draw errors */ }
                }
                // Thin border
                ctx.strokeStyle = '#3a3a5c';
                ctx.lineWidth = 1;
                ctx.strokeRect(thumbX, thumbY, thumbW, thumbH);
            }
            // Value/non-image nodes: show info text compactly
            else if (result?.info && !nodeNeedsImagePreview(node.type)) {
                const infoY = node.y + labelAreaH + 12;
                ctx.fillStyle = '#a6e3a1';
                ctx.font = '10px "Segoe UI", sans-serif';
                ctx.textAlign = 'center';
                ctx.textBaseline = 'middle';
                const infoText = result.info.length > 22
                    ? result.info.substring(0, 21) + '\u2026'
                    : result.info;
                ctx.fillText(infoText, node.x + NODE_WIDTH / 2, infoY);
            }

            // Execution spinner indicator
            if (state.executing) {
                ctx.fillStyle = '#f9e2af';
                ctx.font = '10px "Segoe UI", sans-serif';
                ctx.textAlign = 'right';
                ctx.fillText('...', node.x + NODE_WIDTH - 8, node.y + 12);
            }

            // Ports (positioned based on labelAreaH for consistent port placement)
            for (const p of getInputPorts(node)) {
                drawPort(p, def.color, 'input', isConnected(node.id, p.id, 'input'));
            }
            for (const p of getOutputPorts(node)) {
                drawPort(p, def.color, 'output', isConnected(node.id, p.id, 'output'));
            }
        }
    }

    function isConnected(nodeId, portId, portType) {
        return state.connections.some(c =>
            portType === 'input' ? (c.targetNode === nodeId && c.targetPort === portId) :
                (c.sourceNode === nodeId && c.sourcePort === portId)
        );
    }

    function autoConnectNearbyPorts() {
        // 드래그 중인 노드들의 포트와 나머지 노드 포트 간 거리 체크
        const threshold = GRID_SIZE; // 1 그리드 = 20px
        const movingIds = new Set();
        if (state.draggingMulti && state.selectedNodes.length > 0) {
            state.selectedNodes.forEach(n => movingIds.add(n.id));
        } else if (state.draggingNode) {
            movingIds.add(state.draggingNode.id);
        }
        if (movingIds.size === 0) return;

        let made = false;
        for (const movingNode of state.nodes) {
            if (!movingIds.has(movingNode.id)) continue;
            const myOutputs = getOutputPorts(movingNode);
            const myInputs = getInputPorts(movingNode);

            for (const otherNode of state.nodes) {
                if (movingIds.has(otherNode.id)) continue;
                const otherInputs = getInputPorts(otherNode);
                const otherOutputs = getOutputPorts(otherNode);

                // 내 output → 상대 input
                for (const op of myOutputs) {
                    for (const ip of otherInputs) {
                        if (!arePortsCompatible(op.id, ip.id)) continue; // ★ type check
                        const dx = op.x - ip.x, dy = op.y - ip.y;
                        if (dx * dx + dy * dy <= threshold * threshold) {
                            // 이미 연결되어 있으면 skip
                            const exists = state.connections.some(c =>
                                c.sourceNode === movingNode.id && c.sourcePort === op.id &&
                                c.targetNode === otherNode.id && c.targetPort === ip.id);
                            if (exists) continue;
                            // 해당 input에 기존 연결이 있으면 제거 (input은 1개만 연결)
                            if (!made) pushUndo();
                            state.connections = state.connections.filter(
                                c => !(c.targetNode === otherNode.id && c.targetPort === ip.id));
                            state.connections.push({
                                sourceNode: movingNode.id, sourcePort: op.id,
                                targetNode: otherNode.id, targetPort: ip.id,
                            });
                            autoPreviewOnConnect(otherNode.id);
                            made = true;
                        }
                    }
                }

                // 내 input ← 상대 output
                for (const ip of myInputs) {
                    for (const op of otherOutputs) {
                        if (!arePortsCompatible(op.id, ip.id)) continue; // ★ type check
                        const dx = ip.x - op.x, dy = ip.y - op.y;
                        if (dx * dx + dy * dy <= threshold * threshold) {
                            const exists = state.connections.some(c =>
                                c.sourceNode === otherNode.id && c.sourcePort === op.id &&
                                c.targetNode === movingNode.id && c.targetPort === ip.id);
                            if (exists) continue;
                            if (!made) pushUndo();
                            state.connections = state.connections.filter(
                                c => !(c.targetNode === movingNode.id && c.targetPort === ip.id));
                            state.connections.push({
                                sourceNode: otherNode.id, sourcePort: op.id,
                                targetNode: movingNode.id, targetPort: ip.id,
                            });
                            autoPreviewOnConnect(movingNode.id);
                            made = true;
                        }
                    }
                }
            }
        }
        if (made) {
            updateStatusBar();
            draw();
        }
    }

    function drawPort(port, color, type, connected) {
        const isOptional = !!port.optional;
        ctx.fillStyle = connected ? color : '#2a2a40';
        ctx.strokeStyle = isOptional && !connected ? '#7f849c' : color;
        ctx.lineWidth = isOptional && !connected ? 1.5 : 2;
        if (isOptional && !connected) ctx.setLineDash([3, 3]);
        ctx.beginPath();
        ctx.arc(port.x, port.y, PORT_RADIUS, 0, Math.PI * 2);
        ctx.fill();
        ctx.stroke();
        if (isOptional && !connected) ctx.setLineDash([]);

        // Port label (drawn OUTSIDE the node so it never overlaps the node name)
        ctx.fillStyle = isOptional ? '#585b70' : '#7f849c';
        ctx.font = '9px "Segoe UI", sans-serif';
        ctx.textBaseline = 'middle';
        if (type === 'input') {
            // Left side: label goes to the LEFT of the port (outside node)
            ctx.textAlign = 'right';
            ctx.fillText(port.label, port.x - PORT_RADIUS - 3, port.y);
        } else {
            // Right side: label goes to the RIGHT of the port (outside node)
            ctx.textAlign = 'left';
            ctx.fillText(port.label, port.x + PORT_RADIUS + 3, port.y);
        }
    }

    function drawConnections() {
        for (const conn of state.connections) {
            const srcNode = state.nodes.find(n => n.id === conn.sourceNode);
            const tgtNode = state.nodes.find(n => n.id === conn.targetNode);
            if (!srcNode || !tgtNode) continue;
            const srcPort = getOutputPorts(srcNode).find(p => p.id === conn.sourcePort);
            const tgtPort = getInputPorts(tgtNode).find(p => p.id === conn.targetPort);
            if (!srcPort || !tgtPort) continue;

            const srcDef = NODE_DEFS[srcNode.type];
            const lineColor = srcDef ? srcDef.color + '88' : '#585b70';
            drawBezier(srcPort.x, srcPort.y, tgtPort.x, tgtPort.y, lineColor, 2.5);
        }
    }

    function drawConnecting() {
        if (!state.connecting) return;
        const { x, y } = state.connecting;
        const mx = (state.connectingMouse.x - state.pan.x) / state.zoom;
        const my = (state.connectingMouse.y - state.pan.y) / state.zoom;
        // ★ Check compatibility with hovered port for visual feedback
        let color = '#89b4fa'; // default blue
        const hovered = hitTestPort(mx, my);
        if (hovered && hovered.nodeId !== state.connecting.nodeId) {
            const srcId = state.connecting.portType === 'output' ? state.connecting.portId : hovered.port.id;
            const tgtId = state.connecting.portType === 'output' ? hovered.port.id : state.connecting.portId;
            color = arePortsCompatible(srcId, tgtId) ? '#a6e3a1' : '#f38ba8'; // green OK / red incompatible
        }
        if (state.connecting.portType === 'output') {
            drawBezier(x, y, mx, my, color, 2);
        } else {
            drawBezier(mx, my, x, y, color, 2);
        }
    }

    function drawBezier(x1, y1, x2, y2, color, width) {
        const cpDist = Math.max(Math.abs(x2 - x1) * 0.4, 30);
        ctx.strokeStyle = color;
        ctx.lineWidth = width;
        ctx.beginPath();
        ctx.moveTo(x1, y1);
        ctx.bezierCurveTo(x1 + cpDist, y1, x2 - cpDist, y2, x2, y2);
        ctx.stroke();
    }

    function drawSelectionRect() {
        if (!state.selecting || !state.selectionRect) return;
        const r = state.selectionRect;
        const x = Math.min(r.x1, r.x2);
        const y = Math.min(r.y1, r.y2);
        const w = Math.abs(r.x2 - r.x1);
        const h = Math.abs(r.y2 - r.y1);
        ctx.strokeStyle = '#89b4fa';
        ctx.lineWidth = 1;
        ctx.setLineDash([6, 3]);
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = 'rgba(137, 180, 250, 0.08)';
        ctx.fillRect(x, y, w, h);
        ctx.setLineDash([]);
    }

    // ===== Node creation =====
    function createNode(type, x, y) {
        const def = NODE_DEFS[type];
        if (!def) return null;
        const node = {
            id: 'node_' + (state.nextNodeId++),
            type: type,
            x: x,
            y: y,
            label: '',
            properties: {},
        };
        for (const prop of def.properties) {
            if (prop.key && prop.key[0] !== '_') {
                node.properties[prop.key] = prop.default;
            }
        }
        state.nodes.push(node);
        pushUndo();
        updateStatusBar();
        return node;
    }

    // ===== Mouse events =====
    canvas.addEventListener('mousedown', (e) => {
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const w = screenToWorld(sx, sy);

        if (e.button === 1 || (e.button === 0 && e.altKey)) {
            state.isPanning = true;
            state.panStart = { x: e.clientX - state.pan.x, y: e.clientY - state.pan.y };
            canvas.style.cursor = 'grabbing';
            return;
        }

        if (e.button !== 0) return;

        const portHit = hitTestPort(w.x, w.y);
        if (portHit) {
            state.connecting = {
                nodeId: portHit.nodeId,
                portId: portHit.port.id,
                portType: portHit.port.type,
                x: portHit.port.x,
                y: portHit.port.y,
            };
            state.connectingMouse = { x: sx, y: sy };
            return;
        }

        const nodeHit = hitTestNode(w.x, w.y);
        if (nodeHit) {
            if (e.ctrlKey || e.metaKey) {
                // Ctrl+click: toggle node in multi-selection
                if (isNodeSelected(nodeHit)) {
                    removeFromSelection(nodeHit);
                    if (state.selectedNode?.id === nodeHit.id) {
                        state.selectedNode = state.selectedNodes.length > 0 ? state.selectedNodes[state.selectedNodes.length - 1] : null;
                        selectNode(state.selectedNode);
                    }
                } else {
                    addToSelection(nodeHit);
                    state.selectedNode = nodeHit;
                    renderProperties(nodeHit);
                    renderPreview(nodeHit);
                    renderDocs(nodeHit);
                }
            } else {
                // Regular click: if node is part of multi-selection, drag all; otherwise reset selection
                if (!isNodeSelected(nodeHit)) {
                    clearSelection();
                    state.selectedNode = nodeHit;
                    state.selectedNodes = [nodeHit];
                    selectNode(nodeHit);
                } else {
                    state.selectedNode = nodeHit;
                    renderProperties(nodeHit);
                    renderPreview(nodeHit);
                    renderDocs(nodeHit);
                }
            }

            // Setup dragging (single or multi)
            if (state.selectedNodes.length > 1) {
                state.draggingMulti = true;
                state.draggingNode = null;
                state.dragOffsets = state.selectedNodes.map(n => ({ id: n.id, dx: w.x - n.x, dy: w.y - n.y }));
                pushUndo();
            } else {
                state.draggingNode = nodeHit;
                state.draggingMulti = false;
                state.dragOffset = { x: w.x - nodeHit.x, y: w.y - nodeHit.y };
                // Bring to front
                const idx = state.nodes.indexOf(nodeHit);
                state.nodes.splice(idx, 1);
                state.nodes.push(nodeHit);
                pushUndo();
            }
            draw();
            return;
        }

        const connIdx = hitTestConnection(w.x, w.y);
        if (connIdx >= 0) {
            pushUndo();
            state.connections.splice(connIdx, 1);
            updateStatusBar();
            draw();
            return;
        }

        // Empty canvas click: start rubber-band selection
        if (!e.ctrlKey && !e.metaKey) {
            clearSelection();
            selectNode(null);
        }
        state.selecting = true;
        state.selectionRect = { x1: w.x, y1: w.y, x2: w.x, y2: w.y };
        draw();
    });

    canvas.addEventListener('mousemove', (e) => {
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;

        if (state.isPanning) {
            state.pan.x = e.clientX - state.panStart.x;
            state.pan.y = e.clientY - state.panStart.y;
            draw();
            return;
        }

        if (state.draggingMulti) {
            const w = screenToWorld(sx, sy);
            for (const off of state.dragOffsets) {
                const node = state.nodes.find(n => n.id === off.id);
                if (node) {
                    node.x = Math.round((w.x - off.dx) / GRID_SIZE) * GRID_SIZE;
                    node.y = Math.round((w.y - off.dy) / GRID_SIZE) * GRID_SIZE;
                }
            }
            draw();
            return;
        }

        if (state.draggingNode) {
            const w = screenToWorld(sx, sy);
            state.draggingNode.x = Math.round((w.x - state.dragOffset.x) / GRID_SIZE) * GRID_SIZE;
            state.draggingNode.y = Math.round((w.y - state.dragOffset.y) / GRID_SIZE) * GRID_SIZE;
            draw();
            return;
        }

        if (state.connecting) {
            state.connectingMouse = { x: sx, y: sy };
            draw();
            return;
        }

        if (state.selecting) {
            const w = screenToWorld(sx, sy);
            state.selectionRect.x2 = w.x;
            state.selectionRect.y2 = w.y;
            draw();
            return;
        }

        const w = screenToWorld(sx, sy);
        const portHit = hitTestPort(w.x, w.y);
        canvas.style.cursor = portHit ? 'crosshair' : (hitTestNode(w.x, w.y) ? 'move' : 'default');
    });

    canvas.addEventListener('mouseup', (e) => {
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const w = screenToWorld(sx, sy);

        if (state.isPanning) {
            state.isPanning = false;
            canvas.style.cursor = 'default';
            return;
        }

        if (state.connecting) {
            const portHit = hitTestPort(w.x, w.y);
            if (portHit && portHit.nodeId !== state.connecting.nodeId) {
                pushUndo();
                const from = state.connecting;
                const to = portHit;
                let src, tgt;
                if (from.portType === 'output' && to.port.type === 'input') {
                    src = { nodeId: from.nodeId, portId: from.portId };
                    tgt = { nodeId: to.nodeId, portId: to.port.id };
                } else if (from.portType === 'input' && to.port.type === 'output') {
                    src = { nodeId: to.nodeId, portId: to.port.id };
                    tgt = { nodeId: from.nodeId, portId: from.portId };
                }
                if (src && tgt) {
                    // ★ Port type compatibility check
                    if (!arePortsCompatible(src.portId, tgt.portId)) {
                        const sLabel = PORT_TYPE_LABELS[getPortDataType(src.portId)] || src.portId;
                        const tLabel = PORT_TYPE_LABELS[getPortDataType(tgt.portId)] || tgt.portId;
                        setStatus(`⚠ 연결 불가: ${sLabel}(${src.portId}) → ${tLabel}(${tgt.portId}) 타입이 다릅니다`, 'error');
                        state.connecting = null;
                        draw();
                        return;
                    }
                    state.connections = state.connections.filter(
                        c => !(c.targetNode === tgt.nodeId && c.targetPort === tgt.portId)
                    );
                    state.connections.push({
                        sourceNode: src.nodeId,
                        sourcePort: src.portId,
                        targetNode: tgt.nodeId,
                        targetPort: tgt.portId,
                    });
                    updateStatusBar();
                    autoPreviewOnConnect(tgt.nodeId);
                }
            }
            state.connecting = null;
            draw();
            return;
        }

        // Finish rubber-band selection
        if (state.selecting) {
            state.selecting = false;
            if (state.selectionRect) {
                const r = state.selectionRect;
                const selected = getNodesInRect(r.x1, r.y1, r.x2, r.y2);
                if (e.ctrlKey || e.metaKey) {
                    // Add to existing selection
                    for (const n of selected) addToSelection(n);
                } else {
                    state.selectedNodes = selected;
                }
                if (state.selectedNodes.length === 1) {
                    state.selectedNode = state.selectedNodes[0];
                    selectNode(state.selectedNode);
                } else if (state.selectedNodes.length > 1) {
                    state.selectedNode = state.selectedNodes[state.selectedNodes.length - 1];
                    renderProperties(state.selectedNode);
                    renderPreview(state.selectedNode);
                    renderDocs(state.selectedNode);
                    setStatus(`Selected ${state.selectedNodes.length} nodes`, '');
                }
            }
            state.selectionRect = null;
            draw();
            return;
        }

        // Auto-connect: 드래그 종료 시 포트 간 거리가 1 그리드 이내면 자동 연결
        autoConnectNearbyPorts();

        state.draggingNode = null;
        state.draggingMulti = false;
        state.dragOffsets = [];
    });

    canvas.addEventListener('wheel', (e) => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const oldZoom = state.zoom;
        const zoomDelta = e.deltaY < 0 ? 1.1 : 0.9;
        state.zoom = Math.min(3, Math.max(0.2, state.zoom * zoomDelta));
        state.pan.x = sx - (sx - state.pan.x) * (state.zoom / oldZoom);
        state.pan.y = sy - (sy - state.pan.y) * (state.zoom / oldZoom);
        document.getElementById('zoom-display').textContent = Math.round(state.zoom * 100) + '%';
        draw();
    }, { passive: false });

    // 포트 더블클릭 → 가장 가까운 연결 가능 포트에 자동 연결
    canvas.addEventListener('dblclick', (e) => {
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const w = screenToWorld(sx, sy);
        const portHit = hitTestPort(w.x, w.y);
        if (!portHit) {
            // 포트가 아닌 노드 본체를 더블클릭
            const node = hitTestNode(w.x, w.y);
            if (node && !state.executing) {
                selectNode(node);
                const def = NODE_DEFS[node.type];
                const fileProp = def && def.properties.find(p => p.type === 'file');
                if (fileProp) {
                    // file 속성이 있는 노드: 파일 선택 다이얼로그
                    const input = document.createElement('input');
                    input.type = 'file';
                    input.accept = fileProp.accept || 'image/*';
                    input.onchange = (ev) => {
                        pushUndo();
                        const isVideo = (fileProp.accept || '').includes('video');
                        if (isVideo) handleVideoUpload(ev, node);
                        else handleFileUpload(ev, node);
                    };
                    input.click();
                } else {
                    // 그 외 노드: preview 바로 갱신
                    previewSingleNode(node);
                }
            }
            return;
        }

        const clickedPort = portHit.port;
        const clickedNodeId = portHit.nodeId;

        // 이미 연결된 포트면 무시
        if (isConnected(clickedNodeId, clickedPort.id, clickedPort.type)) return;

        // 가장 가까운 반대 타입 포트 찾기
        let bestDist = Infinity;
        let bestPort = null;
        let bestNodeId = null;

        for (const otherNode of state.nodes) {
            if (otherNode.id === clickedNodeId) continue;
            const candidates = clickedPort.type === 'output'
                ? getInputPorts(otherNode)
                : getOutputPorts(otherNode);
            for (const cp of candidates) {
                // 이미 연결된 input은 건너뛰기
                if (clickedPort.type === 'output' && isConnected(otherNode.id, cp.id, 'input')) continue;
                if (clickedPort.type === 'input' && isConnected(otherNode.id, cp.id, 'output')) continue;
                // ★ 타입 호환성 검사
                if (!arePortsCompatible(clickedPort.id, cp.id)) continue;
                const dx = clickedPort.x - cp.x;
                const dy = clickedPort.y - cp.y;
                const dist = dx * dx + dy * dy;
                if (dist < bestDist) {
                    bestDist = dist;
                    bestPort = cp;
                    bestNodeId = otherNode.id;
                }
            }
        }

        if (!bestPort) return;

        pushUndo();
        let src, tgt;
        if (clickedPort.type === 'output') {
            src = { nodeId: clickedNodeId, portId: clickedPort.id };
            tgt = { nodeId: bestNodeId, portId: bestPort.id };
        } else {
            src = { nodeId: bestNodeId, portId: bestPort.id };
            tgt = { nodeId: clickedNodeId, portId: clickedPort.id };
        }
        state.connections.push({
            sourceNode: src.nodeId, sourcePort: src.portId,
            targetNode: tgt.nodeId, targetPort: tgt.portId,
        });
        updateStatusBar();
        autoPreviewOnConnect(tgt.nodeId);
        draw();
    });

    document.addEventListener('keydown', (e) => {
        const onCanvas = (document.activeElement === document.body || document.activeElement === canvas);

        // Delete selected node(s)
        if (e.key === 'Delete' && onCanvas) {
            if (state.selectedNodes.length > 1) {
                pushUndo();
                const ids = new Set(state.selectedNodes.map(n => n.id));
                state.nodes = state.nodes.filter(n => !ids.has(n.id));
                state.connections = state.connections.filter(c => !ids.has(c.sourceNode) && !ids.has(c.targetNode));
                clearSelection();
                selectNode(null);
                updateStatusBar();
                draw();
            } else if (state.selectedNode) {
                pushUndo();
                deleteNode(state.selectedNode.id);
            }
            return;
        }

        // Ctrl+Enter: Execute
        if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            executePipeline();
            return;
        }

        // Keyboard shortcuts (only when not typing in an input)
        if (!onCanvas) return;

        // Ctrl+Z: Undo
        if (e.key === 'z' && (e.ctrlKey || e.metaKey) && !e.shiftKey) {
            e.preventDefault();
            undo();
            return;
        }

        // Ctrl+C: Copy
        if (e.key === 'c' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            copySelected();
            return;
        }

        // Ctrl+X: Cut
        if (e.key === 'x' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            cutSelected();
            return;
        }

        // Ctrl+V: Paste
        if (e.key === 'v' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            pasteClipboard();
            return;
        }

        // Ctrl+A: Select all
        if (e.key === 'a' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            state.selectedNodes = [...state.nodes];
            if (state.nodes.length > 0) {
                state.selectedNode = state.nodes[state.nodes.length - 1];
                renderProperties(state.selectedNode);
                renderPreview(state.selectedNode);
                renderDocs(state.selectedNode);
            }
            draw();
            setStatus(`Selected all ${state.nodes.length} nodes`, '');
            return;
        }

        // Ctrl+D: Duplicate selected
        if (e.key === 'd' && (e.ctrlKey || e.metaKey)) {
            e.preventDefault();
            copySelected();
            pasteClipboard();
            return;
        }
    });

    // ===== Annotate palette nodes with OpenCV function names =====
    document.querySelectorAll('.palette-node').forEach(el => {
        const def = NODE_DEFS[el.dataset.type];
        if (!def || !def.doc || !def.doc.signature) return;
        const sig = def.doc.signature;
        // Extract function name from "cv2.funcName(...)" pattern
        const m = sig.match(/cv2\.(\w+)\s*\(/);
        if (m) {
            const fnSpan = document.createElement('span');
            fnSpan.style.cssText = 'color:#7f849c;font-size:10px;margin-left:3px';
            fnSpan.textContent = `(${m[1]})`;
            el.appendChild(fnSpan);
        }
    });

    // ===== Drag and drop from palette =====
    document.querySelectorAll('.palette-node').forEach(el => {
        el.addEventListener('dragstart', (e) => {
            e.dataTransfer.setData('node-type', el.dataset.type);
            e.dataTransfer.effectAllowed = 'copy';
        });
    });

    container.addEventListener('dragover', (e) => {
        e.preventDefault();
        e.dataTransfer.dropEffect = 'copy';
    });

    container.addEventListener('drop', (e) => {
        e.preventDefault();
        const type = e.dataTransfer.getData('node-type');
        if (!type || !NODE_DEFS[type]) return;
        const rect = canvas.getBoundingClientRect();
        const sx = e.clientX - rect.left;
        const sy = e.clientY - rect.top;
        const w = screenToWorld(sx, sy);
        const x = Math.round(w.x / GRID_SIZE) * GRID_SIZE;
        const y = Math.round(w.y / GRID_SIZE) * GRID_SIZE;
        const node = createNode(type, x, y);
        selectNode(node);
        draw();
    });

    // Category toggle
    document.querySelectorAll('.category-header').forEach(el => {
        el.addEventListener('click', () => {
            el.nextElementSibling.classList.toggle('collapsed');
        });
    });

    // ===== Node selection =====
    function selectNode(node) {
        state.selectedNode = node;
        renderProperties(node);
        renderPreview(node);
        renderDocs(node);
    }

    function deleteNode(nodeId) {
        state.nodes = state.nodes.filter(n => n.id !== nodeId);
        state.connections = state.connections.filter(
            c => c.sourceNode !== nodeId && c.targetNode !== nodeId
        );
        state.selectedNodes = state.selectedNodes.filter(n => n.id !== nodeId);
        if (state.selectedNode?.id === nodeId) selectNode(null);
        updateStatusBar();
        draw();
    }

    // ===== Properties Panel =====
    function renderProperties(node) {
        const el = document.getElementById('prop-content');
        if (!node) {
            el.innerHTML = '<p class="hint">Click a node to edit its properties</p>';
            return;
        }
        const def = NODE_DEFS[node.type];
        if (!def) return;

        let html = `<div class="prop-row"><label>Type</label><input type="text" value="${def.label}" disabled></div>`;
        html += `<div class="prop-row"><label>Label</label><input type="text" value="${escapeAttr(node.label || '')}" data-prop="__label" placeholder="Optional label"></div>`;

        // ===== Special node-type renderings =====
        const _skipProps = new Set();

        // --- Warp Affine: 2x3 matrix grid with presets ---
        if (node.type === 'warp_affine') {
            _skipProps.add('m00'); _skipProps.add('m01'); _skipProps.add('m02');
            _skipProps.add('m10'); _skipProps.add('m11'); _skipProps.add('m12');
            const m = [
                node.properties.m00 ?? 1, node.properties.m01 ?? 0, node.properties.m02 ?? 0,
                node.properties.m10 ?? 0, node.properties.m11 ?? 1, node.properties.m12 ?? 0
            ];
            html += `<div class="prop-row"><label>Transform Matrix (2x3)</label>
                <table class="matrix-grid"><tbody>
                <tr><td><input type="number" step="0.1" value="${m[0]}" data-prop="m00" data-matrix="0"></td>
                    <td><input type="number" step="0.1" value="${m[1]}" data-prop="m01" data-matrix="1"></td>
                    <td><input type="number" step="1"   value="${m[2]}" data-prop="m02" data-matrix="2"></td></tr>
                <tr><td><input type="number" step="0.1" value="${m[3]}" data-prop="m10" data-matrix="3"></td>
                    <td><input type="number" step="0.1" value="${m[4]}" data-prop="m11" data-matrix="4"></td>
                    <td><input type="number" step="1"   value="${m[5]}" data-prop="m12" data-matrix="5"></td></tr>
                </tbody></table>
                <div class="matrix-labels"><span>scale/rot</span><span>scale/rot</span><span>translate</span></div>
            </div>`;
        }

        // --- Warp Perspective: point tables with picker ---
        if (node.type === 'warp_perspective') {
            _skipProps.add('srcPoints'); _skipProps.add('dstPoints');
            // When points aren't set yet, default to the FULL input-image corners
            // (so an unedited dst maps to the whole frame, not a fixed 300px box).
            const psize = _getPerspectiveInputSize(node);
            const cornerDefault = psize
                ? `0,0;${psize.w},0;${psize.w},${psize.h};0,${psize.h}`
                : '0,0;300,0;300,300;0,300';
            const srcStr = node.properties.srcPoints || cornerDefault;
            const dstStr = node.properties.dstPoints || cornerDefault;
            const srcPts = srcStr.split(';').map(p => p.split(',').map(Number));
            const dstPts = dstStr.split(';').map(p => p.split(',').map(Number));
            const labels = ['TL', 'TR', 'BR', 'BL'];

            html += `<div class="prop-row"><label>Source Points
                <button class="prop-btn pick-btn" data-pick-role="src" style="float:right;padding:1px 8px;font-size:10px;background:#89b4fa;color:#1e1e2e">Pick on Image</button></label>
                <table class="points-table"><thead><tr><th></th><th>X</th><th>Y</th></tr></thead><tbody>`;
            for (let i = 0; i < 4; i++) {
                html += `<tr><td><span class="point-badge point-src">${labels[i]}</span></td>
                    <td><input type="number" value="${srcPts[i]?.[0] ?? 0}" data-perspect="srcX${i}" step="1"></td>
                    <td><input type="number" value="${srcPts[i]?.[1] ?? 0}" data-perspect="srcY${i}" step="1"></td></tr>`;
            }
            html += `</tbody></table></div>`;

            html += `<div class="prop-row"><label>Dest Points
                <button class="prop-btn pick-btn" data-pick-role="dst" style="float:right;padding:1px 8px;font-size:10px;background:#f38ba8;color:#1e1e2e">Pick on Image</button></label>
                <table class="points-table"><thead><tr><th></th><th>X</th><th>Y</th></tr></thead><tbody>`;
            for (let i = 0; i < 4; i++) {
                html += `<tr><td><span class="point-badge point-dst">${labels[i]}</span></td>
                    <td><input type="number" value="${dstPts[i]?.[0] ?? 0}" data-perspect="dstX${i}" step="1"></td>
                    <td><input type="number" value="${dstPts[i]?.[1] ?? 0}" data-perspect="dstY${i}" step="1"></td></tr>`;
            }
            html += `</tbody></table></div>`;
        }

        // --- Image Show: trackbar editor ---
        if (node.type === 'image_show') {
            _skipProps.add('trackbars');
            // Ensure storage exists
            if (!Array.isArray(node.properties.trackbars)) node.properties.trackbars = [];

            const allNodes = state.nodes.filter(n => n.id !== node.id);
            html += `<div class="prop-row" style="flex-direction:column; align-items:stretch;">
                <label style="margin-bottom:6px;">Trackbars
                    <button class="prop-btn" data-tb-add style="float:right; padding:1px 8px; font-size:11px; background:#a6e3a1; color:#1e1e2e">+ Add</button>
                </label>
                <div id="tb-list" style="display:flex; flex-direction:column; gap:6px; margin-top:4px;">`;

            node.properties.trackbars.forEach((tb, i) => {
                const numericProps = _getNumericPropsForNode(tb.nodeId);
                html += `<div class="tb-row" data-tb-idx="${i}" style="border:1px solid #45475a; border-radius:6px; padding:6px 8px; background:#1e1e2e;">
                    <div style="display:flex; gap:4px; margin-bottom:4px;">
                        <input type="text" data-tb-field="name" value="${escapeAttr(tb.name || '')}" placeholder="Slider name"
                               style="flex:1; min-width:0; padding:3px 6px; background:#181825; color:#cdd6f4; border:1px solid #45475a; border-radius:3px; font-size:12px;">
                        <button class="prop-btn" data-tb-del style="padding:1px 8px; font-size:11px; background:#f38ba8; color:#1e1e2e">×</button>
                    </div>
                    <div style="display:flex; gap:4px; margin-bottom:4px;">
                        <select data-tb-field="nodeId" style="flex:1; min-width:0; padding:3px 6px; background:#181825; color:#cdd6f4; border:1px solid #45475a; border-radius:3px; font-size:12px;">
                            <option value="">-- target node --</option>
                            ${allNodes.map(n => `<option value="${n.id}" ${tb.nodeId === n.id ? 'selected' : ''}>${escapeHtml(_nodeDisplayName(n))}</option>`).join('')}
                        </select>
                        <select data-tb-field="propKey" style="flex:1; min-width:0; padding:3px 6px; background:#181825; color:#cdd6f4; border:1px solid #45475a; border-radius:3px; font-size:12px;">
                            <option value="">-- property --</option>
                            ${numericProps.map(p => `<option value="${p.key}" ${tb.propKey === p.key ? 'selected' : ''}>${escapeHtml(p.label)} (${escapeHtml(p.key)})</option>`).join('')}
                        </select>
                    </div>
                    <div style="display:flex; gap:4px;">
                        <input type="number" data-tb-field="min"  value="${tb.min ?? 0}"   placeholder="min"  style="flex:1; min-width:0; padding:3px 6px; background:#181825; color:#cdd6f4; border:1px solid #45475a; border-radius:3px; font-size:12px;">
                        <input type="number" data-tb-field="max"  value="${tb.max ?? 100}" placeholder="max"  style="flex:1; min-width:0; padding:3px 6px; background:#181825; color:#cdd6f4; border:1px solid #45475a; border-radius:3px; font-size:12px;">
                        <input type="number" data-tb-field="step" value="${tb.step ?? 1}"  placeholder="step" style="flex:1; min-width:0; padding:3px 6px; background:#181825; color:#cdd6f4; border:1px solid #45475a; border-radius:3px; font-size:12px;">
                    </div>
                </div>`;
            });

            html += `</div>
                <div style="font-size:11px; color:#6c7086; margin-top:6px; line-height:1.4;">
                    Image Show 팝업창에 슬라이더로 나타납니다. 드래그하면 타겟 노드의 속성이 실시간으로 갱신됩니다.
                </div>
            </div>`;
        }

        for (const prop of def.properties) {
            if (_skipProps.has(prop.key)) continue;

            // --- Kernel grid (Filter2D, Morphology, Dilate, Erode, Structuring Element) ---
            if (prop.type === 'kernel') {
                // Determine grid dimensions based on node type
                let kRows, kCols;
                if (node.type === 'structuring_element') {
                    kCols = parseInt(node.properties.width) || 5;
                    kRows = parseInt(node.properties.height) || 5;
                } else {
                    const ks = parseInt(node.properties.kernelSize || node.properties.ksize) || 3;
                    kRows = ks;
                    kCols = ks;
                }
                // Custom mode: check both preset (filter2d) and shape (morph nodes)
                const isCustom = (node.properties.preset === 'custom') || (node.properties.shape === 'custom');
                // Get kernel values
                let kernelValues;
                if (node.properties.preset && node.properties.preset !== 'custom') {
                    // filter2d with a named preset — compute from KERNEL_PRESETS
                    kernelValues = _getKernelValues(node.properties.preset, kCols, node.properties.kernelData || prop.default);
                } else {
                    // morph nodes or custom mode — parse kernelData directly
                    try {
                        kernelValues = (node.properties.kernelData || prop.default || '').split(',').map(v => parseFloat(v.trim()) || 0);
                    } catch(e) { kernelValues = new Array(kRows * kCols).fill(0); }
                }
                const totalCells = kRows * kCols;
                html += `<div class="prop-row"><label>${prop.label} (${kCols}×${kRows})</label>
                    <table class="matrix-grid kernel-grid" data-kernel-key="${prop.key}" data-k-rows="${kRows}" data-k-cols="${kCols}"><tbody>`;
                for (let r = 0; r < kRows; r++) {
                    html += '<tr>';
                    for (let c = 0; c < kCols; c++) {
                        const idx = r * kCols + c;
                        const val = kernelValues[idx] ?? 0;
                        html += `<td><input type="number" step="0.1" value="${val}" data-kernel-idx="${idx}" ${isCustom ? '' : 'disabled'}></td>`;
                    }
                    html += '</tr>';
                }
                html += `</tbody></table></div>`;
                continue;
            }

            if (prop.type === 'file') {
                const acceptAttr = prop.accept || 'image/*';
                html += `<div class="prop-row"><label>${prop.label}</label>
                    <input type="file" accept="${acceptAttr}" data-prop="${prop.key}" data-file-accept="${acceptAttr}" class="prop-file-input">
                    </div>`;
                // Show current file name if loaded
                if (node.properties.filename) {
                    html += `<div class="prop-row"><label>Loaded</label><input type="text" value="${escapeAttr(node.properties.filename)}" disabled></div>`;
                }
            } else if (prop.type === 'select') {
                html += `<div class="prop-row"><label>${prop.label}</label>
                    <select data-prop="${prop.key}">
                    ${prop.options.map(o => `<option value="${o}" ${node.properties[prop.key] === o ? 'selected' : ''}>${o}</option>`).join('')}
                    </select></div>`;
            } else if (prop.type === 'checkbox') {
                html += `<div class="prop-row"><label>
                    <input type="checkbox" data-prop="${prop.key}" ${node.properties[prop.key] ? 'checked' : ''}> ${prop.label}
                    </label></div>`;
            } else if (prop.type === 'color') {
                const cval = node.properties[prop.key] || prop.default || '#cdd6f4';
                html += `<div class="prop-row"><label>${prop.label}</label>
                    <div style="display:flex; gap:6px; align-items:center; flex:1">
                        <input type="color" value="${escapeAttr(String(cval))}" data-prop="${prop.key}" style="width:40px; height:26px; padding:0; background:transparent; border:1px solid #45475a; border-radius:3px; cursor:pointer;">
                        <input type="text" value="${escapeAttr(String(cval))}" data-color-text="${prop.key}" style="flex:1; min-width:0; padding:3px 6px; background:#181825; color:#cdd6f4; border:1px solid #45475a; border-radius:3px; font-family:Consolas,monospace; font-size:12px;">
                    </div></div>`;
            } else if (prop.type === 'textarea') {
                html += `<div class="prop-row"><label>${prop.label}</label>
                    <textarea data-prop="${prop.key}">${escapeHtml(node.properties[prop.key] || prop.default || '')}</textarea>
                    <div style="display:flex;gap:4px;margin-top:4px;">
                        <button class="prop-btn" onclick="window._openCodeEditor('${node.id}','${prop.key}')" style="background:#89b4fa;color:#1e1e2e" title="Open full-screen code editor">Code Editor</button>
                    </div>
                    </div>`;
            } else {
                const attrs = [];
                if (prop.min !== undefined) attrs.push(`min="${prop.min}"`);
                if (prop.max !== undefined) attrs.push(`max="${prop.max}"`);
                if (prop.step !== undefined) attrs.push(`step="${prop.step}"`);
                const val = node.properties[prop.key] !== undefined ? node.properties[prop.key] : (prop.default || '');
                // filepath for write nodes: filename only (saves to session work folder)
                const isWriteFilePath = prop.key === 'filepath' && (node.type === 'image_write' || node.type === 'video_write');
                html += `<div class="prop-row"><label>${isWriteFilePath ? 'Filename' : prop.label}</label>
                    <div style="display:flex;gap:4px;flex:1">
                    <input type="${prop.type}" value="${escapeAttr(String(val))}" data-prop="${prop.key}" ${attrs.join(' ')} style="flex:1" ${isWriteFilePath ? 'placeholder="output.png"' : ''}>
                    ${isWriteFilePath ? `<button class="prop-btn" data-download="${node.id}" style="padding:2px 8px;white-space:nowrap">Download</button>` : ''}
                    </div></div>`;
            }
        }

        html += `<div class="prop-row" style="margin-top:12px; display:flex; gap:6px;">
            <button class="prop-btn primary" onclick="window._previewNode()">Preview</button>
            <button class="prop-btn" onclick="window._deleteSelectedNode()" style="background:#f38ba8;color:#1e1e2e">Delete</button>
        </div>`;

        el.innerHTML = html;

        // Download button for write nodes
        el.querySelectorAll('[data-download]').forEach(btn => {
            btn.addEventListener('click', () => {
                const filename = node.properties.filepath || 'output.png';
                const basename = filename.split(/[\\/]/).pop();
                const result = state.nodeResults[node.id];
                if (result && result.resultImageId) {
                    // Download via image store
                    const link = document.createElement('a');
                    link.href = `/api/download/${result.resultImageId}?filename=${encodeURIComponent(basename)}`;
                    link.download = basename;
                    link.click();
                    setStatus(`Downloading: ${basename}`, 'success');
                } else if (result && result.downloadFile) {
                    // Download from session work folder
                    const link = document.createElement('a');
                    link.href = `/api/download_file/${encodeURIComponent(result.downloadFile)}`;
                    link.download = result.downloadFile;
                    link.click();
                    setStatus(`Downloading: ${result.downloadFile}`, 'success');
                } else {
                    setStatus('Execute pipeline first to generate the output file', 'error');
                }
            });
        });

        // --- Warp Affine preset handler ---
        if (node.type === 'warp_affine') {
            const presetSelect = el.querySelector('[data-prop="affinePreset"]');
            if (presetSelect) {
                const origHandler = presetSelect._handler;
                presetSelect.addEventListener('change', () => {
                    const preset = presetSelect.value;
                    const vals = AFFINE_PRESETS[preset];
                    if (vals) {
                        pushUndo();
                        const keys = ['m00','m01','m02','m10','m11','m12'];
                        keys.forEach((k, i) => {
                            node.properties[k] = vals[i];
                            const inp = el.querySelector(`[data-prop="${k}"]`);
                            if (inp) inp.value = vals[i];
                        });
                        node.properties.affinePreset = preset;
                        scheduleAutoPreview(node);
                    }
                });
            }
        }

        // --- Kernel grid handler (Filter2D, Morphology, Dilate, Erode, Structuring Element) ---
        if (el.querySelector('.kernel-grid')) {
            el.querySelectorAll('[data-kernel-idx]').forEach(input => {
                input.addEventListener('input', () => {
                    const allInputs = el.querySelectorAll('[data-kernel-idx]');
                    const vals = [];
                    allInputs.forEach(inp => vals.push(parseFloat(inp.value) || 0));
                    pushUndo();
                    node.properties.kernelData = vals.join(',');
                    scheduleAutoPreview(node);
                });
            });
            // Re-render kernel grid when relevant properties change
            const reRenderKernel = () => {
                // Auto-resize kernelData when switching to custom or changing size
                let totalCells;
                if (node.type === 'structuring_element') {
                    totalCells = (parseInt(node.properties.width) || 5) * (parseInt(node.properties.height) || 5);
                } else {
                    const ks = parseInt(node.properties.kernelSize || node.properties.ksize) || 3;
                    totalCells = ks * ks;
                }
                const curVals = (node.properties.kernelData || '').split(',').filter(v => v.trim());
                if (curVals.length !== totalCells) {
                    // Resize: fill with 1 for morph nodes, 0 for filter2d
                    const fillVal = node.properties.shape !== undefined ? 1 : 0;
                    node.properties.kernelData = new Array(totalCells).fill(fillVal).join(',');
                }
                setTimeout(() => renderProperties(node), 10);
            };
            ['preset', 'kernelSize', 'shape', 'ksize', 'width', 'height'].forEach(key => {
                const ctrl = el.querySelector(`[data-prop="${key}"]`);
                if (ctrl) ctrl.addEventListener('change', reRenderKernel);
            });
        }

        // --- Warp Perspective points handler ---
        if (node.type === 'warp_perspective') {
            // Bind numeric inputs for points
            el.querySelectorAll('[data-perspect]').forEach(input => {
                input.addEventListener('input', () => {
                    _syncPerspectivePoints(el, node);
                    scheduleAutoPreview(node);
                });
                input.addEventListener('change', () => {
                    _syncPerspectivePoints(el, node);
                    scheduleAutoPreview(node);
                });
            });
            // Pick buttons
            el.querySelectorAll('.pick-btn').forEach(btn => {
                btn.addEventListener('click', () => {
                    const role = btn.dataset.pickRole; // 'src' or 'dst'
                    _startPointPicker(node, role);
                });
            });
        }

        // --- Image Show: trackbar editor events ---
        if (node.type === 'image_show') {
            const list = el.querySelector('#tb-list');
            const addBtn = el.querySelector('[data-tb-add]');

            if (addBtn) {
                addBtn.addEventListener('click', () => {
                    pushUndo();
                    node.properties.trackbars.push({
                        name: 'Slider ' + (node.properties.trackbars.length + 1),
                        nodeId: '',
                        propKey: '',
                        min: 0,
                        max: 100,
                        step: 1,
                    });
                    renderProperties(node);
                });
            }
            if (list) {
                list.querySelectorAll('.tb-row').forEach(row => {
                    const idx = parseInt(row.dataset.tbIdx);
                    const tb = node.properties.trackbars[idx];
                    if (!tb) return;

                    const delBtn = row.querySelector('[data-tb-del]');
                    if (delBtn) {
                        delBtn.addEventListener('click', () => {
                            pushUndo();
                            node.properties.trackbars.splice(idx, 1);
                            renderProperties(node);
                        });
                    }
                    row.querySelectorAll('[data-tb-field]').forEach(input => {
                        const field = input.dataset.tbField;
                        let undoPushed = false;
                        input.addEventListener('focus', () => { undoPushed = false; });
                        input.addEventListener('input', () => {
                            if (!undoPushed) { pushUndo(); undoPushed = true; }
                            if (input.type === 'number') tb[field] = parseFloat(input.value);
                            else tb[field] = input.value;
                            // Re-render if the target node changed so the property dropdown updates
                            if (field === 'nodeId') {
                                tb.propKey = ''; // reset property when target node changes
                                renderProperties(node);
                            }
                        });
                        input.addEventListener('change', () => {
                            if (!undoPushed) { pushUndo(); undoPushed = true; }
                            if (input.type === 'number') tb[field] = parseFloat(input.value);
                            else tb[field] = input.value;
                        });
                    });
                });
            }
        }

        // Bind events
        el.querySelectorAll('[data-prop]').forEach(input => {
            const propKey = input.dataset.prop;
            let undoPushed = false; // push undo once per focus session
            if (propKey === '__label') {
                input.addEventListener('focus', () => { undoPushed = false; });
                input.addEventListener('input', () => {
                    if (!undoPushed) { pushUndo(); undoPushed = true; }
                    node.label = input.value; draw();
                });
                return;
            }
            if (input.type === 'file') {
                const isVideo = (input.dataset.fileAccept || '').includes('video');
                input.addEventListener('change', (e) => {
                    pushUndo();
                    if (isVideo) handleVideoUpload(e, node);
                    else handleFileUpload(e, node);
                });
                return;
            }
            input.addEventListener('focus', () => { undoPushed = false; });
            const handler = () => {
                if (!undoPushed) { pushUndo(); undoPushed = true; }
                if (input.type === 'checkbox') {
                    node.properties[propKey] = input.checked;
                } else if (input.type === 'number') {
                    node.properties[propKey] = parseFloat(input.value);
                } else {
                    node.properties[propKey] = input.value;
                }
                // For warp_affine matrix inputs, also set preset to 'custom'
                if (node.type === 'warp_affine' && input.dataset.matrix !== undefined) {
                    node.properties.affinePreset = 'custom';
                    const ps = el.querySelector('[data-prop="affinePreset"]');
                    if (ps) ps.value = 'custom';
                }
                scheduleAutoPreview(node);
            };
            input.addEventListener('input', handler);
            input.addEventListener('change', handler);
        });

        // Color picker ↔ hex text input two-way sync
        el.querySelectorAll('input[type="color"][data-prop]').forEach(picker => {
            const key = picker.dataset.prop;
            const txt = el.querySelector(`[data-color-text="${key}"]`);
            if (!txt) return;
            picker.addEventListener('input', () => { txt.value = picker.value; });
            const validHex = /^#[0-9a-fA-F]{6}$/;
            txt.addEventListener('input', () => {
                const v = txt.value.trim();
                if (validHex.test(v)) {
                    picker.value = v;
                    node.properties[key] = v;
                    if (state.selectedNode && state.selectedNode.id === node.id) {
                        scheduleAutoPreview(node);
                    } else {
                        draw();
                    }
                }
            });
        });

    }

    window._deleteSelectedNode = () => {
        if (state.selectedNodes.length > 1) {
            pushUndo();
            const ids = new Set(state.selectedNodes.map(n => n.id));
            state.nodes = state.nodes.filter(n => !ids.has(n.id));
            state.connections = state.connections.filter(c => !ids.has(c.sourceNode) && !ids.has(c.targetNode));
            clearSelection();
            selectNode(null);
            updateStatusBar();
            draw();
        } else if (state.selectedNode) {
            pushUndo();
            deleteNode(state.selectedNode.id);
        }
    };

    window._previewNode = () => {
        if (state.selectedNode) previewSingleNode(state.selectedNode);
    };

    // Auto-preview on property change (debounced)
    let _autoPreviewTimer = null;
    function scheduleAutoPreview(node) {
        if (_autoPreviewTimer) clearTimeout(_autoPreviewTimer);
        _autoPreviewTimer = setTimeout(() => {
            _autoPreviewTimer = null;
            if (state.selectedNode && state.selectedNode.id === node.id && !state.executing) {
                previewSingleNode(node);
            }
        }, 500);
    }

    // Auto-preview when a node's input port gets connected
    function autoPreviewOnConnect(nodeId) {
        const node = state.nodes.find(n => n.id === nodeId);
        if (node && !state.executing) {
            previewSingleNode(node);
        }
    }

    // ===== Inline Code Editor (CodeMirror) =====
    let _codeEditorInstance = null;
    let _codeEditorCallback = null;

    window._openCodeEditor = (nodeId, propKey) => {
        const node = state.nodes.find(n => n.id === nodeId);
        if (!node) return;
        const script = node.properties[propKey] || '';

        const modal = document.getElementById('code-editor-modal');
        const area = document.getElementById('code-editor-area');
        area.innerHTML = '';
        modal.style.display = 'flex';

        _codeEditorInstance = CodeMirror(area, {
            value: script,
            mode: 'python',
            theme: 'monokai',
            lineNumbers: true,
            indentUnit: 4,
            tabSize: 4,
            indentWithTabs: false,
            matchBrackets: true,
            extraKeys: {
                'Tab': (cm) => cm.replaceSelection('    ', 'end'),
                'Ctrl-Enter': () => applyCodeEditor(),
                'Escape': () => closeCodeEditor(),
            },
        });
        _codeEditorInstance.focus();
        setTimeout(() => _codeEditorInstance.refresh(), 50);

        _codeEditorCallback = { nodeId, propKey };

        function applyCodeEditor() {
            if (!_codeEditorInstance || !_codeEditorCallback) return;
            const newScript = _codeEditorInstance.getValue();
            const n = state.nodes.find(nd => nd.id === _codeEditorCallback.nodeId);
            if (n) {
                pushUndo();
                n.properties[_codeEditorCallback.propKey] = newScript;
                if (state.selectedNode?.id === n.id) {
                    renderProperties(n);
                }
                setStatus('Script applied', 'success');
                // Auto-save to server
                sessionFetch('/api/script/save', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ nodeId: n.id, script: newScript }),
                });
            }
            closeCodeEditor();
        }

        function closeCodeEditor() {
            modal.style.display = 'none';
            _codeEditorInstance = null;
            _codeEditorCallback = null;
        }

        document.getElementById('code-editor-apply').onclick = applyCodeEditor;
        document.getElementById('code-editor-cancel').onclick = closeCodeEditor;
        modal.querySelector('.code-editor-overlay').onclick = closeCodeEditor;
    };

    // ===== Perspective Point Helpers =====

    // Find the input image size for a warp_perspective node: prefer its own result,
    // else the shape of an upstream connected node's result. Returns {w,h} or null.
    function _getPerspectiveInputSize(node) {
        const r = state.nodeResults[node.id];
        if (r && r.shape && r.shape.length >= 2) return { w: r.shape[1], h: r.shape[0] };
        for (const conn of state.connections) {
            if (conn.targetNode === node.id) {
                const sr = state.nodeResults[conn.sourceNode];
                if (sr && sr.shape && sr.shape.length >= 2) return { w: sr.shape[1], h: sr.shape[0] };
            }
        }
        return null;
    }

    function _syncPerspectivePoints(el, node) {
        // Rebuild srcPoints/dstPoints strings from grid inputs
        for (const role of ['src', 'dst']) {
            const pts = [];
            for (let i = 0; i < 4; i++) {
                const xInput = el.querySelector(`[data-perspect="${role}X${i}"]`);
                const yInput = el.querySelector(`[data-perspect="${role}Y${i}"]`);
                const x = xInput ? parseFloat(xInput.value) || 0 : 0;
                const y = yInput ? parseFloat(yInput.value) || 0 : 0;
                pts.push(`${x},${y}`);
            }
            node.properties[role + 'Points'] = pts.join(';');
        }
    }

    // Interactive point picker on preview image
    let _pointPickerState = null;

    // Find the INPUT image (preview + size) feeding a warp_perspective node.
    // Perspective points must be picked on the source image, never on the warped output.
    function _getPerspectiveInputPreview(node) {
        for (const conn of state.connections) {
            if (conn.targetNode === node.id) {
                const sr = state.nodeResults[conn.sourceNode];
                if (sr && sr.preview && sr.shape && sr.shape.length >= 2) {
                    return { preview: sr.preview, w: sr.shape[1], h: sr.shape[0] };
                }
            }
        }
        return null;
    }

    function _startPointPicker(node, role) {
        // Always pick on the INPUT image (upstream result), not on this node's warped output.
        const input = _getPerspectiveInputPreview(node);
        if (!input) {
            setStatus('입력 이미지를 먼저 실행(Execute)하세요 — 원본 이미지 위에서 점을 찍습니다.', 'error');
            return;
        }
        const imgW = input.w, imgH = input.h;
        const color = role === 'src' ? '#89b4fa' : '#f38ba8';
        const labels = ['TL', 'TR', 'BR', 'BL'];
        const roleName = role === 'src' ? 'Source(원본 사각형)' : 'Dest(목표 위치)';

        // Full-screen modal overlay
        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.85);z-index:11000;display:flex;align-items:center;justify-content:center;flex-direction:column;';

        const titleEl = document.createElement('div');
        titleEl.style.cssText = 'color:#cdd6f4;font-size:14px;margin-bottom:8px;font-family:"Segoe UI",sans-serif;text-align:center;';
        titleEl.innerHTML = `<b>${roleName}</b> 점 4개를 순서대로 클릭: ${labels.join(' → ')} &nbsp;|&nbsp; 입력 이미지 ${imgW}×${imgH}`;

        const wrap = document.createElement('div');
        wrap.style.cssText = 'position:relative;display:inline-block;line-height:0;border:2px solid #45475a;border-radius:6px;overflow:hidden;';

        const img = document.createElement('img');
        img.src = input.preview;
        img.style.cssText = 'display:block;max-width:88vw;max-height:78vh;';

        const cvs = document.createElement('canvas');
        cvs.style.cssText = 'position:absolute;top:0;left:0;cursor:crosshair;';

        const hint = document.createElement('div');
        hint.style.cssText = 'color:#6c7086;font-size:12px;margin-top:8px;font-family:"Segoe UI",sans-serif;';
        hint.textContent = 'ESC 취소 | 마지막(BL) 점을 찍으면 자동 적용';

        wrap.appendChild(img);
        wrap.appendChild(cvs);
        overlay.appendChild(titleEl);
        overlay.appendChild(wrap);
        overlay.appendChild(hint);
        document.body.appendChild(overlay);

        const pickerCtx = cvs.getContext('2d');
        const points = [];

        // Prefill with existing points for this role (shown as faded guide, click still replaces)
        const existingStr = node.properties[role + 'Points'];

        function layout() {
            // Match canvas to the displayed image size
            const dw = img.clientWidth, dh = img.clientHeight;
            cvs.width = dw; cvs.height = dh;
            draw();
        }

        function toDisp(x, y) {
            return { x: (x / imgW) * cvs.width, y: (y / imgH) * cvs.height };
        }

        function draw() {
            pickerCtx.clearRect(0, 0, cvs.width, cvs.height);
            // Existing points (faded) for reference
            if (existingStr && points.length === 0) {
                const ex = existingStr.split(';').map(p => p.split(',').map(Number));
                if (ex.length === 4) {
                    pickerCtx.strokeStyle = 'rgba(200,200,200,0.5)';
                    pickerCtx.lineWidth = 1.5;
                    pickerCtx.setLineDash([3, 3]);
                    pickerCtx.beginPath();
                    ex.forEach((p, i) => {
                        const d = toDisp(p[0], p[1]);
                        if (i === 0) pickerCtx.moveTo(d.x, d.y); else pickerCtx.lineTo(d.x, d.y);
                    });
                    pickerCtx.closePath();
                    pickerCtx.stroke();
                    pickerCtx.setLineDash([]);
                }
            }
            if (points.length > 1) {
                pickerCtx.strokeStyle = color;
                pickerCtx.lineWidth = 2;
                pickerCtx.setLineDash([4, 4]);
                pickerCtx.beginPath();
                pickerCtx.moveTo(points[0].px, points[0].py);
                for (let i = 1; i < points.length; i++) pickerCtx.lineTo(points[i].px, points[i].py);
                if (points.length === 4) pickerCtx.lineTo(points[0].px, points[0].py);
                pickerCtx.stroke();
                pickerCtx.setLineDash([]);
            }
            for (let i = 0; i < points.length; i++) {
                pickerCtx.fillStyle = color;
                pickerCtx.beginPath();
                pickerCtx.arc(points[i].px, points[i].py, 7, 0, Math.PI * 2);
                pickerCtx.fill();
                pickerCtx.fillStyle = '#fff';
                pickerCtx.font = 'bold 10px sans-serif';
                pickerCtx.textAlign = 'center';
                pickerCtx.textBaseline = 'middle';
                pickerCtx.fillText(labels[i], points[i].px, points[i].py);
            }
        }

        const onResize = () => layout();

        function cleanup() {
            document.removeEventListener('keydown', onEsc);
            window.removeEventListener('resize', onResize);
            if (overlay.parentNode) overlay.remove();
            _pointPickerState = null;
        }

        function onEsc(e) {
            if (e.key === 'Escape') { cleanup(); setStatus('Point picking cancelled', ''); }
        }

        overlay.addEventListener('click', (e) => { if (e.target === overlay) { cleanup(); setStatus('Point picking cancelled', ''); } });
        document.addEventListener('keydown', onEsc);
        window.addEventListener('resize', onResize);

        cvs.addEventListener('click', function onPickClick(e) {
            const rect = cvs.getBoundingClientRect();
            const px = e.clientX - rect.left;
            const py = e.clientY - rect.top;
            const realX = Math.round((px / cvs.width) * imgW);
            const realY = Math.round((py / cvs.height) * imgH);
            points.push({ px, py, x: realX, y: realY });
            draw();

            if (points.length === 4) {
                cvs.removeEventListener('click', onPickClick);
                pushUndo();
                const ptsStr = points.map(p => `${p.x},${p.y}`).join(';');
                node.properties[role + 'Points'] = ptsStr;
                const propEl = document.getElementById('prop-content');
                if (propEl) {
                    for (let i = 0; i < 4; i++) {
                        const xInp = propEl.querySelector(`[data-perspect="${role}X${i}"]`);
                        const yInp = propEl.querySelector(`[data-perspect="${role}Y${i}"]`);
                        if (xInp) xInp.value = points[i].x;
                        if (yInp) yInp.value = points[i].y;
                    }
                }
                setStatus(`${roleName} 점 설정됨: ${ptsStr}`, 'success');
                setTimeout(() => { cleanup(); scheduleAutoPreview(node); }, 350);
            } else {
                setStatus(`${points.length}/4 (${labels[points.length - 1]}: ${realX},${realY}) — 다음: ${labels[points.length]}`, '');
            }
        });

        _pointPickerState = { node, role, overlay, points, imgW, imgH };

        if (img.complete && img.clientWidth) layout();
        else img.addEventListener('load', layout);
    }

    // ===== File Upload =====
    async function handleFileUpload(e, node) {
        const file = e.target.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        setStatus('Uploading image...', '');
        try {
            const resp = await sessionFetch('/api/upload', { method: 'POST', body: formData });
            const data = await resp.json();
            if (data.error) {
                setStatus(data.error, 'error');
                return;
            }
            // Store image ID (server-side reference), not base64
            node.properties.imageId = data.imageId;
            node.properties.filename = data.filename;
            node.label = data.filename;
            state.nodeResults[node.id] = { preview: data.preview, shape: data.shape };
            renderProperties(node);
            renderPreview(node);
            draw();
            setStatus(`Loaded: ${data.filename} (${data.shape[1]}x${data.shape[0]})`, 'success');
        } catch (err) {
            setStatus('Upload failed: ' + err.message, 'error');
        }
    }

    async function handleVideoUpload(e, node) {
        const file = e.target.files[0];
        if (!file) return;
        const formData = new FormData();
        formData.append('file', file);
        const frameIndex = node.properties.frameIndex || 0;
        formData.append('frameIndex', frameIndex);
        setStatus('Uploading video...', '');
        try {
            const resp = await sessionFetch('/api/upload_video', { method: 'POST', body: formData });
            const data = await resp.json();
            if (data.error) {
                setStatus(data.error, 'error');
                return;
            }
            node.properties.filepath = data.filepath;
            node.properties.filename = data.filename;
            node.properties.totalFrames = data.totalFrames;
            node.properties.imageId = data.imageId;
            node.label = data.filename;
            state.nodeResults[node.id] = { preview: data.preview, shape: data.shape };
            renderProperties(node);
            renderPreview(node);
            draw();
            setStatus(`Video loaded: ${data.filename} (${data.totalFrames} frames, ${data.shape[1]}x${data.shape[0]})`, 'success');
        } catch (err) {
            setStatus('Video upload failed: ' + err.message, 'error');
        }
    }

    // ===== Preview Panel =====
    function renderPreview(node) {
        const el = document.getElementById('preview-content');
        if (!node) {
            el.innerHTML = '<p class="hint">Execute pipeline to see results</p>';
            return;
        }
        const result = state.nodeResults[node.id];
        if (!result || !result.preview) {
            el.innerHTML = '<p class="hint">No preview yet. Click "Preview" or "Execute".</p>';
            return;
        }
        let html = `<img src="${result.preview}" alt="Preview" ondblclick="window._openImagePopup('${node.id}')">`;
        if (result.shape) {
            html += `<div class="image-info">${result.shape[1]} x ${result.shape[0]}`;
            html += result.shape[2] ? ` x ${result.shape[2]}ch` : ' (grayscale)';
            html += '</div>';
        }
        if (result.info) {
            html += `<div class="image-info" style="color:#a6e3a1">${escapeHtml(result.info)}</div>`;
        }
        if (result.error) {
            html += `<div class="image-info" style="color:#f38ba8">${escapeHtml(result.error)}</div>`;
        }
        html += '<div class="image-info" style="color:#585b70">Double-click image to enlarge</div>';
        el.innerHTML = html;

        // Draw perspective point overlay if warp_perspective node
        if (node.type === 'warp_perspective' && result.shape) {
            const previewImg = el.querySelector('img');
            if (previewImg) {
                previewImg.addEventListener('load', () => _drawPerspectiveOverlay(el, node, result.shape));
                if (previewImg.complete) _drawPerspectiveOverlay(el, node, result.shape);
            }
        }
    }

    function _drawPerspectiveOverlay(el, node, shape) {
        const previewImg = el.querySelector('img');
        if (!previewImg) return;
        // Remove existing overlay
        const oldCvs = el.querySelector('.persp-overlay');
        if (oldCvs) oldCvs.remove();

        const w = previewImg.clientWidth, h = previewImg.clientHeight;
        if (w === 0 || h === 0) return;
        const imgW = shape[1], imgH = shape[0];

        const cvs = document.createElement('canvas');
        cvs.className = 'persp-overlay';
        cvs.width = w; cvs.height = h;
        cvs.style.cssText = `position:absolute;top:0;left:0;pointer-events:none;`;

        // Wrap image if not already wrapped
        let wrap = previewImg.parentNode;
        if (!wrap.classList.contains('preview-canvas-wrap')) {
            wrap = document.createElement('div');
            wrap.className = 'preview-canvas-wrap';
            wrap.style.cssText = 'position:relative;display:inline-block;';
            previewImg.parentNode.insertBefore(wrap, previewImg);
            wrap.appendChild(previewImg);
        }
        wrap.appendChild(cvs);

        const ctx2 = cvs.getContext('2d');
        const labels = ['TL', 'TR', 'BR', 'BL'];

        function drawPointSet(ptsStr, color) {
            if (!ptsStr) return;
            const pts = ptsStr.split(';').map(p => {
                const [x, y] = p.split(',').map(Number);
                return { x: (x / imgW) * w, y: (y / imgH) * h };
            });
            if (pts.length !== 4) return;
            // Draw polygon
            ctx2.strokeStyle = color;
            ctx2.lineWidth = 2;
            ctx2.setLineDash([4, 4]);
            ctx2.beginPath();
            ctx2.moveTo(pts[0].x, pts[0].y);
            for (let i = 1; i < 4; i++) ctx2.lineTo(pts[i].x, pts[i].y);
            ctx2.closePath();
            ctx2.stroke();
            ctx2.setLineDash([]);
            // Draw points
            for (let i = 0; i < 4; i++) {
                ctx2.fillStyle = color;
                ctx2.beginPath();
                ctx2.arc(pts[i].x, pts[i].y, 5, 0, Math.PI * 2);
                ctx2.fill();
                ctx2.fillStyle = '#fff';
                ctx2.font = 'bold 8px sans-serif';
                ctx2.textAlign = 'center';
                ctx2.textBaseline = 'middle';
                ctx2.fillText(labels[i], pts[i].x, pts[i].y);
            }
        }
        drawPointSet(node.properties.srcPoints, '#89b4fa');
        drawPointSet(node.properties.dstPoints, '#f38ba8');
    }

    // ===== Image Popup (Image Show) =====
    window._openImagePopup = (nodeId) => {
        const result = state.nodeResults[nodeId];
        if (!result?.preview) return;
        const node = state.nodes.find(n => n.id === nodeId);
        const title = node ? (NODE_DEFS[node.type]?.label || node.type) + (node.label ? ': ' + node.label : '') : 'Image';
        if (node && node.type === 'image_show') {
            openImageShowWindow(node, result.preview, title, result.shape);
        } else {
            openImageWindow(result.preview, title, result.shape);
        }
    };

    function openImageWindow(src, title, shape) {
        // Create modal overlay
        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:10000;display:flex;align-items:center;justify-content:center;flex-direction:column;cursor:pointer;';
        overlay.onclick = () => overlay.remove();

        const titleEl = document.createElement('div');
        titleEl.textContent = title + (shape ? ` (${shape[1]}x${shape[0]})` : '');
        titleEl.style.cssText = 'color:#cdd6f4;font-size:14px;margin-bottom:10px;font-family:"Segoe UI",sans-serif;';

        const img = document.createElement('img');
        img.src = src;
        img.style.cssText = 'max-width:90vw;max-height:85vh;border:2px solid #45475a;border-radius:6px;';
        img.onclick = (e) => e.stopPropagation();

        const hint = document.createElement('div');
        hint.textContent = 'Click outside to close | ESC to close';
        hint.style.cssText = 'color:#6c7086;font-size:11px;margin-top:8px;font-family:"Segoe UI",sans-serif;';

        overlay.appendChild(titleEl);
        overlay.appendChild(img);
        overlay.appendChild(hint);
        document.body.appendChild(overlay);

        const closeOnEsc = (e) => { if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', closeOnEsc); } };
        document.addEventListener('keydown', closeOnEsc);
    }

    // ===== Image Show popup with trackbars (B from design) =====
    // Per-ImageShow-node debounce + abort state, so concurrent slider drags don't pile up requests.
    const _imageShowRefreshState = new Map();

    function _scheduleImageShowRefresh(showNode, imgEl, shapeBadge) {
        let s = _imageShowRefreshState.get(showNode.id);
        if (!s) { s = { timer: null, abort: null }; _imageShowRefreshState.set(showNode.id, s); }
        if (s.timer) clearTimeout(s.timer);
        s.timer = setTimeout(async () => {
            s.timer = null;
            if (s.abort) s.abort.abort();
            s.abort = new AbortController();
            const payload = buildPayload();
            payload.targetNodeId = showNode.id;
            try {
                const resp = await sessionFetch('/api/execute_single', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                    signal: s.abort.signal,
                });
                const data = await resp.json();
                state.nodeResults[showNode.id] = data;
                if (data.preview && imgEl) imgEl.src = data.preview;
                if (shapeBadge && data.shape) {
                    shapeBadge.textContent = ` (${data.shape[1]}x${data.shape[0]})`;
                }
                draw();
            } catch (err) {
                if (err && err.name !== 'AbortError') {
                    console.error('Trackbar refresh failed:', err);
                    setStatus('Trackbar 갱신 실패: ' + err.message, 'error');
                }
            }
        }, 80);
    }

    function openImageShowWindow(node, src, title, shape) {
        const overlay = document.createElement('div');
        overlay.style.cssText = 'position:fixed;top:0;left:0;width:100%;height:100%;background:rgba(0,0,0,0.85);z-index:10000;display:flex;align-items:center;justify-content:center;flex-direction:column;cursor:pointer;';
        overlay.onclick = (e) => { if (e.target === overlay) overlay.remove(); };

        const titleWrap = document.createElement('div');
        titleWrap.style.cssText = 'color:#cdd6f4;font-size:14px;margin-bottom:10px;font-family:"Segoe UI",sans-serif;';
        const titleText = document.createElement('span');
        titleText.textContent = title;
        const shapeBadge = document.createElement('span');
        shapeBadge.textContent = shape ? ` (${shape[1]}x${shape[0]})` : '';
        shapeBadge.style.color = '#6c7086';
        titleWrap.appendChild(titleText);
        titleWrap.appendChild(shapeBadge);

        const img = document.createElement('img');
        img.src = src;
        img.style.cssText = 'max-width:90vw;max-height:70vh;border:2px solid #45475a;border-radius:6px;display:block;';
        img.onclick = (e) => e.stopPropagation();

        // Trackbar panel (only if at least one valid trackbar is defined)
        const trackbars = Array.isArray(node?.properties?.trackbars) ? node.properties.trackbars : [];
        const validTbs = trackbars.filter(tb => tb.nodeId && tb.propKey);
        let tbPanel = null;
        if (validTbs.length) {
            tbPanel = document.createElement('div');
            tbPanel.onclick = (e) => e.stopPropagation();
            tbPanel.style.cssText = 'margin-top:12px; padding:10px 14px; background:#1e1e2e; border:1px solid #45475a; border-radius:6px; min-width:min(680px, 90vw); max-width:90vw; max-height:25vh; overflow-y:auto; cursor:default;';

            for (const tb of validTbs) {
                const tgtNode = state.nodes.find(n => n.id === tb.nodeId);
                if (!tgtNode) continue;

                const row = document.createElement('div');
                row.style.cssText = 'display:flex; align-items:center; gap:10px; padding:4px 0;';

                const lblText = tb.name || `${_nodeDisplayName(tgtNode)}.${tb.propKey}`;
                const lbl = document.createElement('div');
                lbl.textContent = lblText;
                lbl.title = lblText;
                lbl.style.cssText = 'color:#cdd6f4; font-size:12px; min-width:160px; max-width:200px; overflow:hidden; text-overflow:ellipsis; white-space:nowrap; font-family:"Segoe UI",sans-serif;';

                const slider = document.createElement('input');
                slider.type = 'range';
                slider.min = String(tb.min != null ? tb.min : 0);
                slider.max = String(tb.max != null ? tb.max : 100);
                slider.step = String(tb.step != null ? tb.step : 1);
                const curRaw = tgtNode.properties[tb.propKey];
                const cur = (typeof curRaw === 'number' && !isNaN(curRaw)) ? curRaw : Number(tb.min != null ? tb.min : 0);
                slider.value = String(cur);
                slider.style.cssText = 'flex:1; min-width:200px;';

                const valEl = document.createElement('div');
                valEl.textContent = slider.value;
                valEl.style.cssText = 'color:#a6e3a1; font-size:12px; min-width:64px; text-align:right; font-family:Consolas, monospace;';

                row.appendChild(lbl);
                row.appendChild(slider);
                row.appendChild(valEl);
                tbPanel.appendChild(row);

                slider.addEventListener('input', () => {
                    const v = parseFloat(slider.value);
                    valEl.textContent = slider.value;
                    tgtNode.properties[tb.propKey] = v;
                    // Mirror to properties panel if the target node is currently selected
                    if (state.selectedNode && state.selectedNode.id === tgtNode.id) {
                        const inp = document.querySelector(`#prop-content [data-prop="${tb.propKey}"]`);
                        if (inp && inp !== document.activeElement) inp.value = slider.value;
                    }
                    _scheduleImageShowRefresh(node, img, shapeBadge);
                });
            }
        }

        const hint = document.createElement('div');
        hint.textContent = (validTbs.length ? '슬라이더를 드래그하면 결과가 실시간 갱신됩니다 | ' : '') + 'Click outside to close | ESC to close';
        hint.style.cssText = 'color:#6c7086;font-size:11px;margin-top:8px;font-family:"Segoe UI",sans-serif;';

        overlay.appendChild(titleWrap);
        overlay.appendChild(img);
        if (tbPanel) overlay.appendChild(tbPanel);
        overlay.appendChild(hint);
        document.body.appendChild(overlay);

        const closeOnEsc = (e) => { if (e.key === 'Escape') { overlay.remove(); document.removeEventListener('keydown', closeOnEsc); } };
        document.addEventListener('keydown', closeOnEsc);
    }

    // ===== Docs Panel =====
    function renderDocs(node) {
        const el = document.getElementById('docs-content');
        if (!node) {
            el.innerHTML = '<p class="hint">Click a node to see OpenCV function documentation</p>';
            return;
        }
        const def = NODE_DEFS[node.type];
        if (!def || !def.doc) {
            el.innerHTML = '<p class="hint">No documentation available</p>';
            return;
        }
        const doc = def.doc;
        let html = `<div class="doc-signature">${escapeHtml(doc.signature)}</div>`;
        html += `<p style="margin-bottom:8px">${escapeHtml(doc.description)}</p>`;
        if (doc.params && doc.params.length) {
            html += `<div class="doc-section-title">Parameters</div><dl class="doc-params">`;
            for (const p of doc.params) {
                html += `<dt>${escapeHtml(p.name)}</dt><dd>${escapeHtml(p.desc)}</dd>`;
            }
            html += `</dl>`;
        }
        if (doc.returns) {
            html += `<div class="doc-section-title">Returns</div><p>${escapeHtml(doc.returns)}</p>`;
        }
        el.innerHTML = html;
    }

    function escapeHtml(str) {
        const d = document.createElement('div');
        d.textContent = str;
        return d.innerHTML;
    }

    function escapeAttr(str) {
        return str.replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
    }

    // ===== Trackbar helpers (Image Show node) =====
    function _nodeDisplayName(n) {
        const def = NODE_DEFS[n.type];
        const lbl = n.label || (def ? def.label : n.type);
        return `${lbl} (${n.id})`;
    }

    function _getNumericPropsForNode(nodeId) {
        if (!nodeId) return [];
        const n = state.nodes.find(x => x.id === nodeId);
        if (!n) return [];
        const def = NODE_DEFS[n.type];
        if (!def || !def.properties) return [];
        return def.properties.filter(p => p.type === 'number' || p.type === 'range');
    }

    // ===== Build payload for backend =====
    function buildPayload() {
        return {
            nodes: state.nodes.map(n => {
                const props = { ...n.properties };
                // Don't send large imageData to backend; use imageId instead
                delete props.imageData;
                return { id: n.id, type: n.type, properties: props };
            }),
            connections: state.connections,
        };
    }

    // ===== Pipeline Execution =====
    async function executePipeline() {
        if (state.nodes.length === 0) {
            setStatus('No nodes to execute', 'error');
            return;
        }
        if (state.executing) return;

        // Check for video_read in loop mode
        const loopVideoNode = state.nodes.find(n => n.type === 'video_read' && n.properties.mode === 'loop');
        if (loopVideoNode) {
            return executeVideoLoop();
        }

        state.executing = true;
        setStatus('Executing pipeline...', '');
        draw();

        const payload = buildPayload();

        try {
            const resp = await sessionFetch('/api/execute', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();

            // Merge results
            for (const [nid, result] of Object.entries(data)) {
                state.nodeResults[nid] = result;
            }

            if (state.selectedNode) {
                renderPreview(state.selectedNode);
            }

            // Auto-open Image Show popup for image_show nodes (suppressed during auto-execute after load)
            if (!_suppressImageShowPopup) {
                for (const node of state.nodes) {
                    if (node.type === 'image_show' && data[node.id]?.preview) {
                        const title = 'Image Show' + (node.label ? ': ' + node.label : '') +
                            (node.properties.windowName ? ' [' + node.properties.windowName + ']' : '');
                        openImageShowWindow(node, data[node.id].preview, title, data[node.id].shape);
                    }
                }
            }

            const errors = Object.values(data).filter(r => r.error);
            if (errors.length) {
                setStatus(`Done with ${errors.length} error(s): ${errors[0].error}`, 'error');
            } else {
                setStatus(`Pipeline executed successfully (${Object.keys(data).length} nodes)`, 'success');
            }
        } catch (err) {
            setStatus('Execution failed: ' + err.message, 'error');
        } finally {
            state.executing = false;
            draw();
        }
    }

    // ===== Video Loop Execution (SSE) =====
    let _videoLoopAbort = null;

    async function executeVideoLoop() {
        if (state.executing) return;
        state.executing = true;
        setStatus('Starting video loop...', '');
        showVideoProgress(0, 0, 0);
        draw();

        const payload = buildPayload();
        _videoLoopAbort = new AbortController();

        try {
            const resp = await sessionFetch('/api/execute_video_loop', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
                signal: _videoLoopAbort.signal,
            });

            const reader = resp.body.getReader();
            const decoder = new TextDecoder();
            let buffer = '';

            while (true) {
                const { done, value } = await reader.read();
                if (done) break;

                buffer += decoder.decode(value, { stream: true });
                const lines = buffer.split('\n');
                buffer = lines.pop() || '';

                for (const line of lines) {
                    if (!line.startsWith('data: ')) continue;
                    try {
                        const msg = JSON.parse(line.slice(6));
                        handleVideoLoopMessage(msg);
                    } catch (e) { /* skip parse errors */ }
                }
            }
        } catch (err) {
            if (err.name !== 'AbortError') {
                setStatus('Video loop failed: ' + err.message, 'error');
            }
        } finally {
            state.executing = false;
            _videoLoopAbort = null;
            hideVideoProgress();
            draw();
        }
    }

    function handleVideoLoopMessage(msg) {
        if (msg.type === 'start') {
            setStatus(`Video loop: frames ${msg.startFrame}-${msg.endFrame} (step ${msg.step})`, '');
            showVideoProgress(0, msg.totalFrames, 0);
        } else if (msg.type === 'progress') {
            const pct = msg.percent || 0;
            setStatus(`Processing frame ${msg.frame} (${msg.processedFrames} done, ${pct.toFixed(1)}%)`, '');
            showVideoProgress(pct, msg.totalFrames, msg.processedFrames);
            if (msg.preview) {
                // Update preview for selected node or show the latest frame
                const previewEl = document.getElementById('preview-content');
                if (previewEl) {
                    previewEl.innerHTML = `<img src="${msg.preview}" style="max-width:100%;border-radius:6px;">
                        <p class="hint">Frame ${msg.frame} | ${msg.shape ? msg.shape[1] + 'x' + msg.shape[0] : ''}</p>`;
                }
            }
        } else if (msg.type === 'stopped') {
            setStatus(`Video loop stopped at frame ${msg.frame} (${msg.processedFrames} frames processed)`, 'error');
        } else if (msg.type === 'error') {
            setStatus(`Video loop error at frame ${msg.frame}: ${msg.error}`, 'error');
        } else if (msg.type === 'done') {
            // Merge final results into node state
            if (msg.results) {
                for (const [nid, result] of Object.entries(msg.results)) {
                    state.nodeResults[nid] = result;
                }
            }
            if (state.selectedNode) {
                renderPreview(state.selectedNode);
            }
            let info = `Video loop done: ${msg.processedFrames} frames processed`;
            if (msg.videoPaths && Object.keys(msg.videoPaths).length > 0) {
                const paths = Object.values(msg.videoPaths).join(', ');
                info += ` | Output: ${paths}`;
            }
            setStatus(info, 'success');
        }
    }

    function stopVideoLoop() {
        if (_videoLoopAbort) {
            sessionFetch('/api/stop_video_loop', { method: 'POST' });
            _videoLoopAbort.abort();
            _videoLoopAbort = null;
        }
    }
    window._stopVideoLoop = stopVideoLoop;

    function showVideoProgress(percent, total, processed) {
        const bar = document.getElementById('video-progress');
        const fill = document.getElementById('video-progress-bar');
        const text = document.getElementById('video-progress-text');
        const stopBtn = document.getElementById('btn-stop-video');
        if (bar) { bar.style.display = 'flex'; }
        if (fill) { fill.style.width = percent + '%'; }
        if (text) { text.textContent = `${processed} / ${total > 0 ? Math.ceil(total) : '?'} frames (${percent.toFixed(1)}%)`; }
        if (stopBtn) { stopBtn.style.display = 'inline-block'; }
    }

    function hideVideoProgress() {
        const bar = document.getElementById('video-progress');
        const stopBtn = document.getElementById('btn-stop-video');
        if (bar) { bar.style.display = 'none'; }
        if (stopBtn) { stopBtn.style.display = 'none'; }
    }

    // ===== Single Node Preview =====
    async function previewSingleNode(node) {
        if (state.executing) return;
        state.executing = true;
        setStatus(`Previewing ${NODE_DEFS[node.type]?.label || node.type}...`, '');
        draw();

        const payload = buildPayload();
        payload.targetNodeId = node.id;

        try {
            const resp = await sessionFetch('/api/execute_single', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            state.nodeResults[node.id] = data;
            renderPreview(node);
            draw();

            if (data.error) {
                setStatus(`Preview error: ${data.error}`, 'error');
            } else {
                setStatus('Preview updated', 'success');
            }
        } catch (err) {
            setStatus('Preview failed: ' + err.message, 'error');
        } finally {
            state.executing = false;
            draw();
        }
    }

    // ===== Status Bar =====
    function setStatus(text, type) {
        const el = document.getElementById('status-text');
        el.textContent = text;
        el.className = type || '';
    }

    function updateStatusBar() {
        let text = `Nodes: ${state.nodes.length} | Connections: ${state.connections.length}`;
        if (state.selectedNodes.length > 1) {
            text += ` | Selected: ${state.selectedNodes.length}`;
        }
        text += ` | Session: ${SESSION_ID}`;
        document.getElementById('node-count').textContent = text;
    }

    // ===== Save / Load =====
    async function saveFlow() {
        // Strip imageData from properties to keep file small
        const saveNodes = state.nodes.map(n => ({
            ...n,
            properties: Object.fromEntries(
                Object.entries(n.properties).filter(([k]) => k !== 'imageData')
            ),
        }));
        const flowData = {
            version: 2,
            nodes: saveNodes,
            connections: state.connections,
            pan: state.pan,
            zoom: state.zoom,
            nextNodeId: state.nextNodeId,
        };

        // Check if any node has an imageId (needs project ZIP save)
        const hasImages = state.nodes.some(n => n.properties.imageId);

        if (hasImages) {
            // Save as ZIP project (flow.json + images/)
            setStatus('Saving project with images...', '');
            try {
                const resp = await sessionFetch('/api/save_project', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(flowData),
                });
                if (!resp.ok) {
                    let errMsg = 'HTTP ' + resp.status;
                    try {
                        const ct = resp.headers.get('content-type') || '';
                        if (ct.includes('application/json')) {
                            const err = await resp.json();
                            errMsg = err.error || errMsg;
                        }
                    } catch (_) { /* ignore parse errors */ }
                    setStatus('Save failed: ' + errMsg, 'error');
                    return;
                }
                const blob = await resp.blob();

                // showSaveFilePicker — 클라이언트 로컬 폴더 선택 (Chrome/Edge, secure context)
                if (window.showSaveFilePicker) {
                    try {
                        const handle = await window.showSaveFilePicker({
                            suggestedName: 'nodeopencv-project.zip',
                            types: [{ description: 'ZIP Files', accept: { 'application/zip': ['.zip'] } }],
                        });
                        const writable = await handle.createWritable();
                        await writable.write(blob);
                        await writable.close();
                        setStatus('Project saved (with images): ' + handle.name, 'success');
                        return;
                    } catch (err) {
                        if (err.name === 'AbortError') return;
                        // fall through
                    }
                }

                // Fallback — 브라우저 다운로드
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = 'nodeopencv-project.zip';
                a.click();
                URL.revokeObjectURL(url);
                setStatus('Project saved with images (downloaded)', 'success');
            } catch (err) {
                setStatus('Save project failed: ' + err.message, 'error');
            }
        } else {
            // No images — save as simple JSON
            const jsonStr = JSON.stringify(flowData, null, 2);
            const blob = new Blob([jsonStr], { type: 'application/json' });

            if (window.showSaveFilePicker) {
                try {
                    const handle = await window.showSaveFilePicker({
                        suggestedName: 'nodeopencv-flow.json',
                        types: [{ description: 'JSON Files', accept: { 'application/json': ['.json'] } }],
                    });
                    const writable = await handle.createWritable();
                    await writable.write(blob);
                    await writable.close();
                    setStatus('Flow saved: ' + handle.name, 'success');
                    return;
                } catch (err) {
                    if (err.name === 'AbortError') return;
                }
            }

            const url = URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = 'nodeopencv-flow.json';
            a.click();
            URL.revokeObjectURL(url);
            setStatus('Flow saved (downloaded to default folder)', 'success');
        }
    }

    function applyFlowData(data, imagePreviews) {
        state.nodes = data.nodes || [];
        state.connections = data.connections || [];
        state.pan = data.pan || { x: 0, y: 0 };
        state.zoom = data.zoom || 1;
        state.nextNodeId = data.nextNodeId || 1;
        state.selectedNode = null;
        state.selectedNodes = [];
        state.nodeResults = {};
        Object.keys(_previewImageCache).forEach(k => delete _previewImageCache[k]);
        state.undoStack = [];

        // Restore image/video previews for nodes that have imageId
        if (imagePreviews) {
            for (const node of state.nodes) {
                const imgId = node.properties && node.properties.imageId;
                if (imgId && imagePreviews[imgId]) {
                    const info = imagePreviews[imgId];
                    state.nodeResults[node.id] = {
                        preview: info.preview,
                        shape: info.shape,
                    };
                    // Restore video properties if present
                    if (info.filepath) {
                        node.properties.filepath = info.filepath;
                    }
                    if (info.totalFrames) {
                        node.properties.totalFrames = info.totalFrames;
                    }
                    // Update label with filename if available
                    if (node.properties.filename) {
                        node.label = node.properties.filename;
                    }
                }
            }
        }

        selectNode(null);
        updateStatusBar();
        document.getElementById('zoom-display').textContent = Math.round(state.zoom * 100) + '%';
        draw();
    }

    async function loadFlow() {
        // showOpenFilePicker — 클라이언트 로컬 파일 선택 (Chrome/Edge, secure context)
        if (window.showOpenFilePicker) {
            try {
                const [handle] = await window.showOpenFilePicker({
                    types: [{
                        description: 'Project / Flow Files',
                        accept: {
                            'application/zip': ['.zip'],
                            'application/json': ['.json'],
                        },
                    }],
                    multiple: false,
                });
                const file = await handle.getFile();

                if (file.name.endsWith('.zip')) {
                    await loadProjectZip(file);
                } else {
                    const text = await file.text();
                    const data = JSON.parse(text);
                    applyFlowData(data, null);
                    setStatus('Flow loaded: ' + file.name, 'success');
                }
                return;
            } catch (err) {
                if (err.name === 'AbortError') return;
                // fall through
            }
        }

        // Fallback — hidden file input
        const input = document.createElement('input');
        input.type = 'file';
        input.accept = '.json,.zip';
        input.onchange = async (e) => {
            const file = e.target.files[0];
            if (!file) return;

            if (file.name.endsWith('.zip')) {
                await loadProjectZip(file);
            } else {
                const reader = new FileReader();
                reader.onload = (ev) => {
                    try {
                        const data = JSON.parse(ev.target.result);
                        applyFlowData(data, null);
                        setStatus('Flow loaded: ' + file.name, 'success');
                    } catch (err) {
                        setStatus('Failed to load flow: ' + err.message, 'error');
                    }
                };
                reader.readAsText(file);
            }
        };
        input.click();
    }

    async function loadProjectZip(file) {
        setStatus('Loading project with images...', '');
        try {
            const formData = new FormData();
            formData.append('file', file);
            const resp = await sessionFetch('/api/load_project', {
                method: 'POST',
                body: formData,
            });
            if (!resp.ok) {
                let errMsg = 'HTTP ' + resp.status;
                try {
                    const ct = resp.headers.get('content-type') || '';
                    if (ct.includes('application/json')) {
                        const err = await resp.json();
                        errMsg = err.error || errMsg;
                    }
                } catch (_) { /* ignore parse errors */ }
                setStatus('Load failed: ' + errMsg, 'error');
                return;
            }
            const data = await resp.json();
            if (data.error) {
                setStatus('Load failed: ' + data.error, 'error');
                return;
            }
            const flow = data.flow;
            const imagePreviews = data.images || {};
            applyFlowData(flow, imagePreviews);

            const imgCount = Object.keys(imagePreviews).length;
            setStatus(`Project loaded: ${file.name} (${imgCount} image${imgCount !== 1 ? 's' : ''} restored)`, 'success');

            // If at least one image was restored, run the full pipeline so every
            // downstream node also shows its preview. Skip if a video_read is in loop mode
            // (auto-starting an infinite loop on load would be surprising).
            if (imgCount > 0 && state.nodes.length > 0) {
                const hasLoopVideo = state.nodes.some(n =>
                    n.type === 'video_read' && n.properties && n.properties.mode === 'loop');
                if (!hasLoopVideo) {
                    _suppressImageShowPopup = true;
                    try {
                        await executePipeline();
                    } finally {
                        _suppressImageShowPopup = false;
                    }
                }
            }
        } catch (err) {
            setStatus('Load project failed: ' + err.message, 'error');
        }
    }

    // ===== Toolbar buttons =====
    document.getElementById('btn-execute').addEventListener('click', executePipeline);
    document.getElementById('btn-clear').addEventListener('click', () => {
        if (state.nodes.length === 0) return;
        if (!confirm('Clear all nodes and connections?')) return;
        pushUndo();
        // Cancel any in-flight trackbar refreshes and pending timers
        _imageShowRefreshState.forEach(s => {
            if (s.timer) clearTimeout(s.timer);
            if (s.abort) { try { s.abort.abort(); } catch (e) {} }
        });
        _imageShowRefreshState.clear();
        // Close any open Image Show / Image popup overlays (those are created with z-index:10000)
        document.querySelectorAll('body > div[style*="z-index:10000"]').forEach(el => el.remove());
        state.nodes = [];
        state.connections = [];
        state.selectedNode = null;
        state.selectedNodes = [];
        state.nodeResults = {};
        Object.keys(_previewImageCache).forEach(k => delete _previewImageCache[k]);
        state.nextNodeId = 1;
        selectNode(null);
        updateStatusBar();
        draw();
        setStatus('Canvas cleared', '');
    });
    document.getElementById('btn-save').addEventListener('click', saveFlow);

    // ===== Code Generation =====
    document.getElementById('btn-generate').addEventListener('click', async () => {
        if (state.nodes.length === 0) {
            setStatus('No nodes to generate code from', 'error');
            return;
        }
        setStatus('Generating Python code...', '');
        const payload = buildPayload();
        try {
            const resp = await sessionFetch('/api/generate_code', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(payload),
            });
            const data = await resp.json();
            document.getElementById('code-output').textContent = data.code || '# No code generated';
            document.getElementById('code-overlay').style.display = 'flex';
            setStatus('Code generated', 'success');
        } catch (err) {
            setStatus('Code generation failed: ' + err.message, 'error');
        }
    });

    document.getElementById('code-close-btn').addEventListener('click', () => {
        document.getElementById('code-overlay').style.display = 'none';
    });
    document.getElementById('code-overlay').addEventListener('click', (e) => {
        if (e.target.id === 'code-overlay') e.target.style.display = 'none';
    });
    document.getElementById('code-copy-btn').addEventListener('click', () => {
        const code = document.getElementById('code-output').textContent;
        navigator.clipboard.writeText(code).then(() => {
            setStatus('Code copied to clipboard', 'success');
        }).catch(() => {
            // Fallback
            const ta = document.createElement('textarea');
            ta.value = code;
            document.body.appendChild(ta);
            ta.select();
            document.execCommand('copy');
            ta.remove();
            setStatus('Code copied to clipboard', 'success');
        });
    });
    document.getElementById('code-download-btn').addEventListener('click', () => {
        const code = document.getElementById('code-output').textContent;
        const blob = new Blob([code], { type: 'text/x-python' });
        const url = URL.createObjectURL(blob);
        const a = document.createElement('a');
        a.href = url;
        a.download = 'opencv_pipeline.py';
        a.click();
        URL.revokeObjectURL(url);
        setStatus('Code downloaded as opencv_pipeline.py', 'success');
    });

    document.getElementById('btn-load').addEventListener('click', loadFlow);

    // ===== Examples Modal =====
    document.getElementById('btn-examples').addEventListener('click', async () => {
        const overlay = document.getElementById('examples-overlay');
        const listEl = document.getElementById('examples-list');
        overlay.style.display = 'flex';
        listEl.innerHTML = '<p style="color:#a6adc8;">Loading examples...</p>';

        try {
            const resp = await sessionFetch('/api/examples');
            const examples = await resp.json();
            if (!examples.length) {
                listEl.innerHTML = '<p style="color:#f38ba8;">No examples found. Run bundle_examples.py first.</p>';
                return;
            }

            let html = '';
            for (const ex of examples) {
                const sizeStr = ex.size > 1024 * 1024
                    ? (ex.size / 1024 / 1024).toFixed(1) + ' MB'
                    : (ex.size / 1024).toFixed(0) + ' KB';
                html += `<div class="example-item" data-file="${ex.filename}" style="display:flex; justify-content:space-between; align-items:center; padding:10px 14px; margin:4px 0; background:#313244; border-radius:8px; cursor:pointer; border:1px solid transparent; transition:all 0.15s;">
                    <div>
                        <div style="color:#cdd6f4; font-weight:600; font-size:14px;">${ex.name}</div>
                        <div style="color:#6c7086; font-size:11px; margin-top:2px;">${ex.filename} (${sizeStr})</div>
                    </div>
                    <button class="example-load-btn" data-file="${ex.filename}" style="background:#89b4fa; color:#1e1e2e; border:none; border-radius:6px; padding:6px 16px; font-weight:600; cursor:pointer; font-size:12px; white-space:nowrap;">Load</button>
                </div>`;
            }
            listEl.innerHTML = html;

            // Hover effect
            listEl.querySelectorAll('.example-item').forEach(item => {
                item.addEventListener('mouseenter', () => { item.style.borderColor = '#89b4fa'; item.style.background = '#3b3d50'; });
                item.addEventListener('mouseleave', () => { item.style.borderColor = 'transparent'; item.style.background = '#313244'; });
            });

            // Load button click
            listEl.querySelectorAll('.example-load-btn').forEach(btn => {
                btn.addEventListener('click', async (e) => {
                    e.stopPropagation();
                    const filename = btn.dataset.file;
                    btn.textContent = 'Loading...';
                    btn.disabled = true;
                    await loadExampleFromServer(filename);
                    overlay.style.display = 'none';
                });
            });

            // Row click also loads
            listEl.querySelectorAll('.example-item').forEach(item => {
                item.addEventListener('click', async () => {
                    const filename = item.dataset.file;
                    const btn = item.querySelector('.example-load-btn');
                    if (btn) { btn.textContent = 'Loading...'; btn.disabled = true; }
                    await loadExampleFromServer(filename);
                    overlay.style.display = 'none';
                });
            });
        } catch (err) {
            listEl.innerHTML = `<p style="color:#f38ba8;">Failed to load examples: ${err.message}</p>`;
        }
    });

    document.getElementById('examples-close').addEventListener('click', () => {
        document.getElementById('examples-overlay').style.display = 'none';
    });
    document.getElementById('examples-overlay').addEventListener('click', (e) => {
        if (e.target === e.currentTarget) e.currentTarget.style.display = 'none';
    });

    async function loadExampleFromServer(filename) {
        setStatus(`Loading example: ${filename}...`, '');
        try {
            // Fetch the ZIP from server
            const resp = await sessionFetch(`/api/examples/${filename}`);
            if (!resp.ok) {
                setStatus('Failed to download example: HTTP ' + resp.status, 'error');
                return;
            }
            const zipBlob = await resp.blob();

            // Upload the ZIP to load_project endpoint
            const formData = new FormData();
            formData.append('file', zipBlob, filename);
            const loadResp = await sessionFetch('/api/load_project', {
                method: 'POST',
                body: formData,
            });
            if (!loadResp.ok) {
                let errMsg = 'HTTP ' + loadResp.status;
                try {
                    const ct = loadResp.headers.get('content-type') || '';
                    if (ct.includes('application/json')) {
                        const err = await loadResp.json();
                        errMsg = err.error || errMsg;
                    }
                } catch (_) {}
                setStatus('Load example failed: ' + errMsg, 'error');
                return;
            }
            const data = await loadResp.json();
            if (data.error) {
                setStatus('Load example failed: ' + data.error, 'error');
                return;
            }
            const flow = data.flow;
            const imagePreviews = data.images || {};
            applyFlowData(flow, imagePreviews);

            const imgCount = Object.keys(imagePreviews).length;
            setStatus(`Example loaded: ${filename} (${imgCount} asset${imgCount !== 1 ? 's' : ''} restored)`, 'success');
        } catch (err) {
            setStatus('Load example failed: ' + err.message, 'error');
        }
    }

    // Context menu (right click delete)
    canvas.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        const rect = canvas.getBoundingClientRect();
        const w = screenToWorld(e.clientX - rect.left, e.clientY - rect.top);
        const nodeHit = hitTestNode(w.x, w.y);
        if (nodeHit) {
            pushUndo();
            deleteNode(nodeHit.id);
        } else {
            const connIdx = hitTestConnection(w.x, w.y);
            if (connIdx >= 0) {
                pushUndo();
                state.connections.splice(connIdx, 1);
                updateStatusBar();
                draw();
            }
        }
    });

    // ===== Help Modal =====
    const helpOverlay = document.getElementById('help-overlay');
    const helpCloseBtn = document.getElementById('help-close-btn');
    const btnHelp = document.getElementById('btn-help');

    btnHelp.addEventListener('click', () => {
        helpOverlay.style.display = 'flex';
    });

    helpCloseBtn.addEventListener('click', () => {
        helpOverlay.style.display = 'none';
    });

    helpOverlay.addEventListener('click', (e) => {
        if (e.target === helpOverlay) helpOverlay.style.display = 'none';
    });

    // Tab switching
    document.querySelectorAll('.help-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            const tabId = tab.dataset.tab;
            document.querySelectorAll('.help-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.help-content').forEach(c => c.classList.remove('active'));
            tab.classList.add('active');
            document.querySelector(`.help-content[data-tab="${tabId}"]`).classList.add('active');
        });
    });

    // ESC closes help / examples / report
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            if (helpOverlay.style.display !== 'none') helpOverlay.style.display = 'none';
            const exOverlay = document.getElementById('examples-overlay');
            if (exOverlay && exOverlay.style.display !== 'none') exOverlay.style.display = 'none';
            const reportOverlay = document.getElementById('report-overlay');
            if (reportOverlay && reportOverlay.style.display !== 'none') reportOverlay.style.display = 'none';
        }
    });

    // ===== Report (PDF) Modal =====
    const reportOverlay = document.getElementById('report-overlay');
    const btnReport = document.getElementById('btn-report');
    const btnReportClose = document.getElementById('report-close-btn');
    const btnReportExport = document.getElementById('report-export-btn');

    function _escHtml(s) {
        return String(s == null ? '' : s)
            .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;').replace(/'/g, '&#039;');
    }

    function _topoSort(nodes, connections) {
        const id2node = new Map(nodes.map(n => [n.id, n]));
        const indeg = new Map(nodes.map(n => [n.id, 0]));
        const out = new Map(nodes.map(n => [n.id, []]));
        for (const c of connections) {
            if (id2node.has(c.sourceNode) && id2node.has(c.targetNode)) {
                out.get(c.sourceNode).push(c.targetNode);
                indeg.set(c.targetNode, (indeg.get(c.targetNode) || 0) + 1);
            }
        }
        const q = nodes.filter(n => (indeg.get(n.id) || 0) === 0).map(n => n.id);
        const seen = new Set(q);
        const order = [];
        while (q.length) {
            const id = q.shift();
            order.push(id);
            for (const nx of out.get(id) || []) {
                indeg.set(nx, indeg.get(nx) - 1);
                if (indeg.get(nx) === 0 && !seen.has(nx)) { seen.add(nx); q.push(nx); }
            }
        }
        // Append any remaining (cycles) so all nodes show up in the report
        for (const n of nodes) if (!seen.has(n.id)) order.push(n.id);
        return order.map(id => id2node.get(id));
    }

    // Capture the canvas graph (all nodes + connections) as a PNG data URL.
    // Temporarily resizes the canvas and adjusts pan/zoom to fit the world bbox of all nodes,
    // then restores the original viewport.
    async function _captureGraphImage(scale = 2) {
        if (!state.nodes || state.nodes.length === 0) return null;
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const n of state.nodes) {
            const h = getNodeHeight(n);
            const w = getNodeWidth(n);
            if (n.x < minX) minX = n.x;
            if (n.y < minY) minY = n.y;
            if (n.x + w > maxX) maxX = n.x + w;
            if (n.y + h > maxY) maxY = n.y + h;
        }
        const pad = 30;
        minX -= pad; minY -= pad; maxX += pad; maxY += pad;
        const worldW = Math.max(1, maxX - minX);
        const worldH = Math.max(1, maxY - minY);

        // Cap to prevent runaway memory on huge graphs (cap at ~3000px on the longer side).
        const cap = 3000;
        let s = scale;
        if (worldW * s > cap || worldH * s > cap) {
            s = Math.min(cap / worldW, cap / worldH);
        }
        const targetW = Math.max(1, Math.round(worldW * s));
        const targetH = Math.max(1, Math.round(worldH * s));

        // Save current viewport
        const oldW = canvas.width, oldH = canvas.height;
        const oldPan = { x: state.pan.x, y: state.pan.y };
        const oldZoom = state.zoom;

        try {
            canvas.width = targetW;
            canvas.height = targetH;
            state.zoom = s;
            state.pan.x = -minX * s;
            state.pan.y = -minY * s;
            draw();
            // Wait one frame so any not-yet-decoded preview thumbnails finish painting.
            await new Promise(r => requestAnimationFrame(r));
            return canvas.toDataURL('image/png');
        } finally {
            canvas.width = oldW;
            canvas.height = oldH;
            state.zoom = oldZoom;
            state.pan.x = oldPan.x;
            state.pan.y = oldPan.y;
            draw();
        }
    }

    async function _buildReportPayload() {
        const ordered = _topoSort(state.nodes, state.connections);
        const graphImage = await _captureGraphImage(2);
        return {
            title:       document.getElementById('report-title').value || '제출용 보고서',
            department:  document.getElementById('report-department').value || '',
            studentId:   document.getElementById('report-student-id').value || '',
            name:        document.getElementById('report-name').value || '',
            company:     document.getElementById('report-company').value || '',
            graphImage:  graphImage,
            nodeCount:   state.nodes.length,
            connectionCount: state.connections.length,
            nodes: ordered.map(n => ({ id: n.id, type: n.type, label: n.label || '' })),
        };
    }

    async function _renderReportPreview() {
        const el = document.getElementById('report-preview');
        if (!el) return;
        // Render skeleton synchronously so UI is responsive while we capture.
        el.innerHTML = '<p style="color:#888;">미리보기 생성 중...</p>';

        const payload = await _buildReportPayload();

        const now = new Date();
        const pad = x => String(x).padStart(2, '0');
        const generatedAt = `${now.getFullYear()}-${pad(now.getMonth()+1)}-${pad(now.getDate())} ${pad(now.getHours())}:${pad(now.getMinutes())}:${pad(now.getSeconds())}`;

        let html = '';
        html += `<div style="text-align:center; padding:24px 0 12px; border-bottom:2px solid #1f4f8b;">`;
        html += `<div style="font-size:24px; font-weight:700; color:#1f4f8b;">${_escHtml(payload.title || '제출용 보고서')}</div>`;
        html += `</div>`;

        const cover = [
            ['학과',     payload.department],
            ['학번',     payload.studentId],
            ['이름',     payload.name],
            ['소속회사', payload.company],
            ['생성일자', generatedAt],
            ['노드 수', String(payload.nodeCount)],
            ['연결 수', String(payload.connectionCount)],
        ];
        html += `<table style="width:100%; border-collapse:collapse; margin:18px 0;">`;
        for (const [k, v] of cover) {
            html += `<tr>`;
            html += `<th style="text-align:left; width:30%; padding:8px 10px; background:#eef0f6; border:1px solid #bbbbcc; color:#3b3b5c; font-weight:600;">${_escHtml(k)}</th>`;
            html += `<td style="padding:8px 10px; border:1px solid #bbbbcc;">${_escHtml(v) || '-'}</td>`;
            html += `</tr>`;
        }
        html += `</table>`;

        html += `<div style="margin-top:18px; font-size:15px; font-weight:700; color:#1f4f8b; border-bottom:1px solid #c7c9d6; padding-bottom:4px;">노드 그래프</div>`;

        if (payload.graphImage) {
            html += `<div style="margin-top:10px; text-align:center; background:#1e1e2e; border:1px solid #c7c9d6; border-radius:4px; padding:8px;">`;
            html += `<img src="${payload.graphImage}" alt="graph" style="max-width:100%; height:auto; display:block; margin:0 auto;">`;
            html += `</div>`;
        } else {
            html += `<p style="color:#888; padding:14px 0;">캔버스에 노드가 없습니다. 노드를 추가하고 다시 시도하세요.</p>`;
        }

        el.innerHTML = html;
    }

    async function _openReportModal() {
        if (!reportOverlay) return;
        const titleEl = document.getElementById('report-title');
        if (titleEl && !titleEl.value) titleEl.value = '제출용 보고서';
        reportOverlay.style.display = 'flex';
        await _renderReportPreview();
    }

    if (btnReport) {
        btnReport.addEventListener('click', _openReportModal);
    }
    if (btnReportClose) {
        btnReportClose.addEventListener('click', () => { reportOverlay.style.display = 'none'; });
    }
    if (reportOverlay) {
        reportOverlay.addEventListener('click', (e) => {
            if (e.target === reportOverlay) reportOverlay.style.display = 'none';
        });
        // Re-render preview on any text input change.
        // For text fields the cover info changes but the graph image stays the same — we re-capture
        // anyway for simplicity, but debounce to avoid flicker on rapid typing.
        let _previewTimer = null;
        const debouncedPreview = () => {
            if (_previewTimer) clearTimeout(_previewTimer);
            _previewTimer = setTimeout(() => { _renderReportPreview(); }, 250);
        };
        ['report-title', 'report-department', 'report-student-id', 'report-name', 'report-company']
            .forEach(id => {
                const el = document.getElementById(id);
                if (el) el.addEventListener('input', debouncedPreview);
            });
    }
    if (btnReportExport) {
        btnReportExport.addEventListener('click', async () => {
            if (state.nodes.length === 0) {
                setStatus('보고서로 출력할 노드가 없습니다', 'error');
                return;
            }
            btnReportExport.disabled = true;
            const originalText = btnReportExport.textContent;
            btnReportExport.textContent = '생성 중...';
            try {
                const payload = await _buildReportPayload();
                const resp = await sessionFetch('/api/report', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify(payload),
                });
                if (!resp.ok) {
                    let msg = `HTTP ${resp.status}`;
                    try { const j = await resp.json(); if (j.error) msg = j.error; } catch (e) {}
                    setStatus('보고서 생성 실패: ' + msg, 'error');
                    return;
                }
                const blob = await resp.blob();
                let filename = 'report.pdf';
                const cd = resp.headers.get('Content-Disposition') || '';
                // Prefer RFC 5987 filename*=UTF-8''<percent-encoded> (preserves Korean / Unicode)
                const mStar = /filename\*\s*=\s*UTF-8''([^;]+)/i.exec(cd);
                if (mStar) {
                    try { filename = decodeURIComponent(mStar[1].trim()); } catch (_) { /* keep default */ }
                } else {
                    const m = /filename="?([^";]+)"?/i.exec(cd);
                    if (m) filename = m[1];
                }

                // 1) Native File System (Chrome/Edge, secure context) — 클라이언트가 저장 위치 선택
                if (window.showSaveFilePicker) {
                    try {
                        const handle = await window.showSaveFilePicker({
                            suggestedName: filename,
                            types: [{ description: 'PDF File', accept: { 'application/pdf': ['.pdf'] } }],
                        });
                        const writable = await handle.createWritable();
                        await writable.write(blob);
                        await writable.close();
                        setStatus('보고서 저장됨 (로컬): ' + handle.name, 'success');
                        return;
                    } catch (err) {
                        if (err.name === 'AbortError') {
                            setStatus('보고서 저장 취소됨', '');
                            return;
                        }
                        // showSaveFilePicker 실패 시 다운로드로 폴백
                    }
                }

                // 2) Fallback — 브라우저 기본 다운로드 폴더
                const url = URL.createObjectURL(blob);
                const a = document.createElement('a');
                a.href = url;
                a.download = filename;
                document.body.appendChild(a);
                a.click();
                a.remove();
                URL.revokeObjectURL(url);
                setStatus('보고서 다운로드(기본 폴더): ' + filename, 'success');
            } catch (err) {
                setStatus('보고서 생성 오류: ' + err.message, 'error');
            } finally {
                btnReportExport.disabled = false;
                btnReportExport.textContent = originalText;
            }
        });
    }

    // ===== Panel Collapse/Expand =====
    const palette = document.getElementById('node-palette');
    const rightPanel = document.getElementById('right-panel');
    const btnTogglePalette = document.getElementById('btn-toggle-palette');
    const btnToggleRight = document.getElementById('btn-toggle-right');

    btnTogglePalette.addEventListener('click', () => {
        palette.classList.toggle('collapsed');
        if (palette.classList.contains('collapsed')) {
            btnTogglePalette.innerHTML = '&#9654;'; // ▶ (right arrow = expand)
            btnTogglePalette.title = 'Show Palette';
        } else {
            btnTogglePalette.innerHTML = '&#9664;'; // ◀ (left arrow = collapse)
            btnTogglePalette.title = 'Hide Palette';
        }
        // Resize canvas after transition
        setTimeout(resizeCanvas, 280);
    });

    btnToggleRight.addEventListener('click', () => {
        rightPanel.classList.toggle('collapsed');
        if (rightPanel.classList.contains('collapsed')) {
            btnToggleRight.innerHTML = '&#9664;'; // ◀ (left arrow = expand)
            btnToggleRight.title = 'Show Panel';
        } else {
            btnToggleRight.innerHTML = '&#9654;'; // ▶ (right arrow = collapse)
            btnToggleRight.title = 'Hide Panel';
        }
        // Resize canvas after transition
        setTimeout(resizeCanvas, 280);
    });

    // Initial draw
    draw();
    updateStatusBar();

})();
