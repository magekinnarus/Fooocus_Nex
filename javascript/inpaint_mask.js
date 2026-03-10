(() => {
    const MODES = {
        context: {
            rootId: 'inpaint_canvas',
            fieldId: 'inpaint_context_mask_data',
            canvasId: 'inpaint-context-mask-overlay',
            statusId: 'inpaint-mask-status',
            emptyStatus: 'Load a source image, then paint the Step 1 context mask.',
            capturedStatus: 'Step 1 context mask captured.',
            clearedStatus: 'Step 1 context mask cleared.',
        },
        bb: {
            rootId: 'inpaint_bb_canvas',
            fieldId: 'inpaint_bb_mask_data',
            canvasId: 'inpaint-bb-mask-overlay',
            statusId: 'inpaint-mask-status',
            emptyStatus: 'Load a BB image, then paint the Step 2 BB mask.',
            capturedStatus: 'Step 2 BB mask captured.',
            clearedStatus: 'Step 2 BB mask cleared.',
        },
        outpaint_bb: {
            rootId: 'outpaint_bb_canvas',
            fieldId: 'outpaint_bb_mask_data',
            canvasId: 'outpaint-bb-mask-overlay',
            statusId: 'outpaint-mask-status',
            emptyStatus: 'Load a BB image, then paint the Step 2 BB mask.',
            capturedStatus: 'Step 2 BB mask captured.',
            clearedStatus: 'Step 2 BB mask cleared.',
        },
    };

    const state = {
        activeMode: 'context',
        tool: 'brush',
        brushSize: 36,
        resizeBound: false,
        surfaces: {
            context: createSurface('context'),
            bb: createSurface('bb'),
            outpaint_bb: createSurface('outpaint_bb'),
        },
    };

    function createSurface(mode) {
        return {
            mode,
            root: null,
            host: null,
            img: null,
            canvas: null,
            ctx: null,
            sourceKey: null,
            observersAttached: false,
            drawing: false,
            lastPoint: null,
        };
    }

    function getSurface(mode) {
        return state.surfaces[mode];
    }

    function getRoot(mode) {
        return document.getElementById(MODES[mode].rootId);
    }

    function getImage(root) {
        return root ? root.querySelector('img') : null;
    }

    function getHost(root) {
        if (!root) return null;
        return root.querySelector('.image-container') || root;
    }

    function getMaskField(mode) {
        const fieldId = MODES[mode].fieldId;
        return document.querySelector(`#${fieldId} textarea, #${fieldId} input`);
    }

    function getStatus(mode = state.activeMode) {
        return document.getElementById(MODES[mode].statusId);
    }

    function setStatus(text, mode = state.activeMode) {
        const status = getStatus(mode);
        if (status) {
            status.textContent = text;
        }
    }

    function setMaskValue(mode, value) {
        const field = getMaskField(mode);
        if (!field || field.value === value) return;
        field.value = value;
        field.dispatchEvent(new Event('input', { bubbles: true }));
        field.dispatchEvent(new Event('change', { bubbles: true }));
    }

    function updateModeButtons() {
        const contextBtn = document.getElementById('inpaint-mask-mode-context');
        const bbBtn = document.getElementById('inpaint-mask-mode-bb');
        const outpaintBtn = document.getElementById('outpaint-mask-mode-bb');

        if (contextBtn) contextBtn.classList.toggle('active', state.activeMode === 'context');
        if (bbBtn) bbBtn.classList.toggle('active', state.activeMode === 'bb');
        if (outpaintBtn) outpaintBtn.classList.toggle('active', state.activeMode === 'outpaint_bb');
    }

    function updateToolButtons() {
        ['inpaint', 'outpaint'].forEach(prefix => {
            const brushBtn = document.getElementById(`${prefix}-mask-brush`);
            const eraseBtn = document.getElementById(`${prefix}-mask-erase`);
            if (brushBtn) brushBtn.classList.toggle('active', state.tool === 'brush');
            if (eraseBtn) eraseBtn.classList.toggle('active', state.tool === 'erase');
        });
    }

    function currentModeName() {
        if (state.activeMode === 'context') return 'Context mask';
        if (state.activeMode === 'bb') return 'Inpaint BB mask';
        if (state.activeMode === 'outpaint_bb') return 'Outpaint BB mask';
        return 'Mask';
    }

    function setTool(tool) {
        state.tool = tool;
        updateToolButtons();
        setStatus(tool === 'brush' ? `${currentModeName()} brush active.` : `${currentModeName()} eraser active.`);
    }

    function setActiveMode(mode) {
        if (!MODES[mode]) return;
        state.activeMode = mode;
        updateModeButtons();
        syncCanvasInteractivity();
        const surface = getSurface(mode);
        if (!surface.img || !surface.img.src || surface.img.style.display === 'none') {
            setStatus(MODES[mode].emptyStatus);
            return;
        }
        setStatus(mode === 'context' ? 'Step 1 context mask ready.' :
            mode === 'bb' ? 'Step 2 Inpaint BB mask ready.' : 'Step 2 Outpaint BB mask ready.');
    }

    function hasPaint(surface) {
        if (!surface.ctx || !surface.canvas) return false;
        const alpha = surface.ctx.getImageData(0, 0, surface.canvas.width, surface.canvas.height).data;
        for (let i = 3; i < alpha.length; i += 4) {
            if (alpha[i] !== 0) return true;
        }
        return false;
    }

    function exportMask(mode) {
        const surface = getSurface(mode);
        if (!surface.canvas || !surface.ctx) return;
        if (!hasPaint(surface)) {
            setMaskValue(mode, '');
            if (mode === state.activeMode) {
                setStatus(MODES[mode].emptyStatus);
            }
            return;
        }
        setMaskValue(mode, surface.canvas.toDataURL('image/png'));
        if (mode === state.activeMode) {
            setStatus(MODES[mode].capturedStatus);
        }
    }

    function clearMask(mode = state.activeMode) {
        const surface = getSurface(mode);
        if (!surface.ctx || !surface.canvas) return;
        surface.ctx.clearRect(0, 0, surface.canvas.width, surface.canvas.height);
        setMaskValue(mode, '');
        if (mode === state.activeMode) {
            setStatus(MODES[mode].clearedStatus);
        }
    }

    function pointerToCanvas(surface, event) {
        if (!surface.canvas) return null;
        const rect = surface.canvas.getBoundingClientRect();
        if (!rect.width || !rect.height) return null;
        return {
            x: (event.clientX - rect.left) * (surface.canvas.width / rect.width),
            y: (event.clientY - rect.top) * (surface.canvas.height / rect.height),
        };
    }

    function drawLine(surface, from, to) {
        if (!surface.ctx) return;
        surface.ctx.save();
        surface.ctx.lineCap = 'round';
        surface.ctx.lineJoin = 'round';
        surface.ctx.lineWidth = state.brushSize;
        if (state.tool === 'erase') {
            surface.ctx.globalCompositeOperation = 'destination-out';
            surface.ctx.strokeStyle = 'rgba(0,0,0,1)';
        } else {
            surface.ctx.globalCompositeOperation = 'source-over';
            surface.ctx.strokeStyle = 'rgba(255,255,255,1)';
        }
        surface.ctx.beginPath();
        surface.ctx.moveTo(from.x, from.y);
        surface.ctx.lineTo(to.x, to.y);
        surface.ctx.stroke();
        surface.ctx.restore();
    }

    function attachCanvasEvents(mode) {
        const surface = getSurface(mode);
        surface.canvas.addEventListener('pointerdown', (event) => {
            if (state.activeMode !== mode || !surface.img || !surface.img.src) return;
            const point = pointerToCanvas(surface, event);
            if (!point) return;
            surface.drawing = true;
            surface.lastPoint = point;
            drawLine(surface, point, point);
            surface.canvas.setPointerCapture(event.pointerId);
            event.preventDefault();
        });

        surface.canvas.addEventListener('pointermove', (event) => {
            if (state.activeMode !== mode || !surface.drawing) return;
            const point = pointerToCanvas(surface, event);
            if (!point || !surface.lastPoint) return;
            drawLine(surface, surface.lastPoint, point);
            surface.lastPoint = point;
            event.preventDefault();
        });

        const stopDrawing = (event) => {
            if (!surface.drawing) return;
            surface.drawing = false;
            surface.lastPoint = null;
            try {
                surface.canvas.releasePointerCapture(event.pointerId);
            } catch (e) { }
            exportMask(mode);
            event.preventDefault();
        };

        surface.canvas.addEventListener('pointerup', stopDrawing);
        surface.canvas.addEventListener('pointercancel', stopDrawing);
    }

    function ensureCanvas(mode) {
        const surface = getSurface(mode);
        if (surface.canvas || !surface.host) return;
        surface.host.style.position = 'relative';
        const canvas = document.createElement('canvas');
        canvas.id = MODES[mode].canvasId;
        Object.assign(canvas.style, {
            position: 'absolute',
            zIndex: '20',
            cursor: 'crosshair',
            touchAction: 'none',
            opacity: '0.35',
            pointerEvents: 'none',
        });
        surface.host.appendChild(canvas);
        surface.canvas = canvas;
        surface.ctx = canvas.getContext('2d');
        attachCanvasEvents(mode);
    }

    function getContainSize(img) {
        const { clientWidth: width, clientHeight: height, naturalWidth, naturalHeight } = img;
        if (!width || !height || !naturalWidth || !naturalHeight) return null;

        const imgRatio = naturalWidth / naturalHeight;
        const containerRatio = width / height;

        let w, h, x, y;
        if (imgRatio > containerRatio) {
            w = width;
            h = width / imgRatio;
            x = 0;
            y = (height - h) / 2;
        } else {
            h = height;
            w = height * imgRatio;
            x = (width - w) / 2;
            y = 0;
        }
        return { w, h, x, y };
    }

    function syncCanvasToImage(mode, clearForNewImage = false) {
        const surface = getSurface(mode);
        if (!surface.canvas || !surface.img || !surface.host) return;

        const size = getContainSize(surface.img);
        if (!size) {
            surface.canvas.style.display = 'none';
            return;
        }

        const hostRect = surface.host.getBoundingClientRect();
        const imgRect = surface.img.getBoundingClientRect();

        surface.canvas.style.display = 'block';
        surface.canvas.style.left = `${(imgRect.left - hostRect.left) + size.x}px`;
        surface.canvas.style.top = `${(imgRect.top - hostRect.top) + size.y}px`;
        surface.canvas.style.width = `${size.w}px`;
        surface.canvas.style.height = `${size.h}px`;

        if (clearForNewImage || surface.canvas.width !== surface.img.naturalWidth || surface.canvas.height !== surface.img.naturalHeight) {
            surface.canvas.width = surface.img.naturalWidth;
            surface.canvas.height = surface.img.naturalHeight;
            surface.ctx.clearRect(0, 0, surface.canvas.width, surface.canvas.height);
            setMaskValue(mode, '');
            if (mode === state.activeMode) {
                setStatus(MODES[mode].emptyStatus);
            }
        }
    }

    function syncCanvasInteractivity() {
        Object.keys(MODES).forEach((mode) => {
            const surface = getSurface(mode);
            if (!surface.canvas) return;
            const active = mode === state.activeMode && surface.img && surface.img.src && surface.img.style.display !== 'none';
            surface.canvas.style.pointerEvents = active ? 'auto' : 'none';
            surface.canvas.style.cursor = active ? 'crosshair' : 'default';
            surface.canvas.style.opacity = active ? '1' : '0.35';
        });
    }

    function ensureObservers(mode) {
        const surface = getSurface(mode);
        if (surface.observersAttached || !surface.root) return;
        const observer = new MutationObserver(() => refreshMode(mode));
        observer.observe(surface.root, { childList: true, subtree: true, attributes: true, attributeFilter: ['src', 'style', 'class'] });
        surface.observersAttached = true;
        if (!state.resizeBound) {
            window.addEventListener('resize', refreshAll);
            state.resizeBound = true;
        }
    }

    function attachControls() {
        const brushBtn = document.getElementById('inpaint-mask-brush');
        const eraseBtn = document.getElementById('inpaint-mask-erase');
        const clearBtn = document.getElementById('inpaint-mask-clear');
        const sizeInput = document.getElementById('inpaint-mask-size');
        const contextBtn = document.getElementById('inpaint-mask-mode-context');
        const bbBtn = document.getElementById('inpaint-mask-mode-bb');

        if (brushBtn && !brushBtn.dataset.bound) {
            brushBtn.dataset.bound = '1';
            brushBtn.addEventListener('click', () => setTool('brush'));
        }
        if (eraseBtn && !eraseBtn.dataset.bound) {
            eraseBtn.dataset.bound = '1';
            eraseBtn.addEventListener('click', () => setTool('erase'));
        }
        if (clearBtn && !clearBtn.dataset.bound) {
            clearBtn.dataset.bound = '1';
            clearBtn.addEventListener('click', () => clearMask());
        }
        if (sizeInput && !sizeInput.dataset.bound) {
            sizeInput.dataset.bound = '1';
            sizeInput.addEventListener('input', () => {
                state.brushSize = parseInt(sizeInput.value, 10) || 36;
            });
            state.brushSize = parseInt(sizeInput.value, 10) || 36;
        }
        if (contextBtn && !contextBtn.dataset.bound) {
            contextBtn.dataset.bound = '1';
            contextBtn.addEventListener('click', () => setActiveMode('context'));
        }
        if (bbBtn && !bbBtn.dataset.bound) {
            bbBtn.dataset.bound = '1';
            bbBtn.addEventListener('click', () => setActiveMode('bb'));
        }

        // --- Outpaint Tab Controls ---
        const outpaintBrushBtn = document.getElementById('outpaint-mask-brush');
        const outpaintEraseBtn = document.getElementById('outpaint-mask-erase');
        const outpaintClearBtn = document.getElementById('outpaint-mask-clear');
        const outpaintSizeInput = document.getElementById('outpaint-mask-size');

        if (outpaintBrushBtn && !outpaintBrushBtn.dataset.bound) {
            outpaintBrushBtn.dataset.bound = '1';
            outpaintBrushBtn.addEventListener('click', () => {
                setActiveMode('outpaint_bb');
                setTool('brush');
            });
        }
        if (outpaintEraseBtn && !outpaintEraseBtn.dataset.bound) {
            outpaintEraseBtn.dataset.bound = '1';
            outpaintEraseBtn.addEventListener('click', () => {
                setActiveMode('outpaint_bb');
                setTool('erase');
            });
        }
        if (outpaintClearBtn && !outpaintClearBtn.dataset.bound) {
            outpaintClearBtn.dataset.bound = '1';
            outpaintClearBtn.addEventListener('click', () => {
                setActiveMode('outpaint_bb');
                clearMask('outpaint_bb');
            });
        }
        if (outpaintSizeInput && !outpaintSizeInput.dataset.bound) {
            outpaintSizeInput.dataset.bound = '1';
            outpaintSizeInput.addEventListener('input', () => {
                state.brushSize = parseInt(outpaintSizeInput.value, 10) || 36;
            });
        }
        const outpaintModeBtn = document.getElementById('outpaint-mask-mode-bb');
        if (outpaintModeBtn && !outpaintModeBtn.dataset.bound) {
            outpaintModeBtn.dataset.bound = '1';
            outpaintModeBtn.addEventListener('click', () => setActiveMode('outpaint_bb'));
        }

        updateModeButtons();
        updateToolButtons();
    }

    function refreshMode(mode) {
        const surface = getSurface(mode);
        surface.root = getRoot(mode);
        if (!surface.root) return;
        surface.host = getHost(surface.root);
        surface.img = getImage(surface.root);
        ensureCanvas(mode);
        ensureObservers(mode);

        if (!surface.img || !surface.img.src || surface.img.style.display === 'none') {
            if (surface.canvas) {
                surface.canvas.style.display = 'none';
            }
            surface.sourceKey = null;
            if (mode === state.activeMode) {
                setStatus(MODES[mode].emptyStatus);
            }
            syncCanvasInteractivity();
            return;
        }

        const nextKey = `${surface.img.currentSrc || surface.img.src}|${surface.img.naturalWidth}x${surface.img.naturalHeight}`;
        const clearForNewImage = nextKey !== surface.sourceKey;
        surface.sourceKey = nextKey;
        syncCanvasToImage(mode, clearForNewImage);
        syncCanvasInteractivity();
    }

    function refreshAll() {
        attachControls();
        refreshMode('context');
        refreshMode('bb');
        refreshMode('outpaint_bb');
    }

    function start() {
        refreshAll();
        window.setInterval(refreshAll, 500);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', start, { once: true });
    } else {
        start();
    }
})();
