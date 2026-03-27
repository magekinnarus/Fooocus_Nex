(() => {
    function appRoot() {
        if (typeof gradioApp === 'function') {
            return gradioApp();
        }
        return document;
    }

    function escapeSelector(value) {
        if (window.CSS && typeof window.CSS.escape === 'function') {
            return window.CSS.escape(value);
        }
        return String(value).replace(/([^a-zA-Z0-9_-])/g, '\\$1');
    }

    function getFieldControl(fieldId) {
        if (!fieldId) {
            return null;
        }
        const selector = `#${escapeSelector(fieldId)}`;
        const root = appRoot();
        return root.querySelector(`${selector} textarea, ${selector} input`) ||
            root.querySelector(selector) ||
            document.querySelector(`${selector} textarea, ${selector} input`) ||
            document.querySelector(selector);
    }

    function readFieldValue(fieldId) {
        const field = getFieldControl(fieldId);
        if (!field) {
            return '';
        }
        if (typeof field.value === 'string') {
            return field.value;
        }
        return field.textContent || '';
    }

    function supportsFaceComposer(methodValue) {
        return ['FaceID V2', 'PuLID'].includes((methodValue || '').trim());
    }

    function normalizeDroppedUrl(raw) {
        if (!raw || typeof raw !== 'string') {
            return '';
        }
        const lines = raw.split('\n').map((line) => line.trim()).filter(Boolean);
        for (const line of lines) {
            if (!line.startsWith('#')) {
                return line;
            }
        }
        return '';
    }

    class NexFaceGridComposer {
        constructor() {
            this.slots = new Map();
            this.observer = null;
            this.pollHandle = null;
            this.panel = null;
            this.activeSlot = null;
            this.activeCellIndex = -1;
            this.cellFiles = Array(4).fill(null);
            this.cellObjectUrls = Array(4).fill('');
            this.isDragging = false;
            this.dragOffset = { x: 0, y: 0 };
            this.onMouseMove = (event) => this.handleDrag(event);
            this.onMouseUp = () => this.stopDragging();
            document.addEventListener('mousemove', this.onMouseMove);
            document.addEventListener('mouseup', this.onMouseUp);
            this.init();
        }

        init() {
            this.scanSlots();
            this.observer = new MutationObserver(() => this.scanSlots());
            this.observer.observe(appRoot(), { childList: true, subtree: true });
            this.pollHandle = window.setInterval(() => this.refreshButtonStates(), 300);
        }

        scanSlots() {
            const root = appRoot();
            const slots = root.querySelectorAll('nex-image-slot[data-method-field-id]');
            slots.forEach((slot) => this.ensureSlotButton(slot));
            this.refreshButtonStates();
        }

        ensureSlotButton(slot) {
            if (this.slots.has(slot)) {
                return;
            }
            const actions = slot.querySelector('.nex-slot__actions');
            if (!actions) {
                return;
            }
            const button = document.createElement('button');
            button.type = 'button';
            button.className = 'nex-slot__tool nex-slot__tool--face-grid';
            button.textContent = 'Face Grid';
            button.hidden = true;
            button.addEventListener('click', (event) => {
                event.preventDefault();
                event.stopPropagation();
                this.openPanel(slot);
            });
            actions.insertBefore(button, actions.querySelector('.nex-slot__clear'));
            this.slots.set(slot, button);
        }

        refreshButtonStates() {
            this.slots.forEach((button, slot) => {
                if (!document.body.contains(slot)) {
                    this.slots.delete(slot);
                    return;
                }
                const methodFieldId = slot.dataset.methodFieldId || '';
                const methodValue = readFieldValue(methodFieldId);
                button.hidden = !supportsFaceComposer(methodValue);
            });
        }

        resetCells() {
            this.cellFiles = Array(4).fill(null);
            this.cellObjectUrls.forEach((url) => {
                if (url) {
                    URL.revokeObjectURL(url);
                }
            });
            this.cellObjectUrls = Array(4).fill('');
            this.activeCellIndex = -1;
        }

        openPanel(slot) {
            this.activeSlot = slot;
            this.resetCells();
            this.renderPanel();
        }

        renderPanel() {
            if (this.panel) {
                this.panel.remove();
                this.panel = null;
                this.stopDragging();
            }
            const panel = document.createElement('div');
            panel.id = 'floating-face-grid-panel';
            panel.className = 'floating-panel nex-face-grid-panel';
            panel.innerHTML = `
                <div class="panel-header nex-face-grid-panel__header">
                    <span class="panel-title">Face Grid 2x2</span>
                    <div class="panel-controls">
                        <button type="button" class="nex-face-grid-panel__close" title="Close">X</button>
                    </div>
                </div>
                <div class="panel-content nex-face-grid-panel__content">
                    <div class="nex-face-grid-panel__subtitle">Add 2 to 4 face images into the grid, then compose it back into the slot.</div>
                    <div class="nex-face-grid-panel__status">Add at least 2 images</div>
                    <div class="nex-face-grid-panel__board"></div>
                    <input class="nex-face-grid-panel__file-input" type="file" accept="image/*" hidden>
                    <div class="nex-face-grid-panel__actions">
                        <button type="button" class="nex-face-grid-panel__secondary nex-face-grid-panel__cancel">Close</button>
                        <button type="button" class="nex-face-grid-panel__secondary nex-face-grid-panel__clear">Clear</button>
                        <button type="button" class="nex-face-grid-panel__primary" disabled>Compose To Slot</button>
                    </div>
                </div>
            `;

            const pos = JSON.parse(localStorage.getItem('face-grid-panel-pos') || '{"top":"120px","right":"480px"}');
            Object.assign(panel.style, pos);
            if (!panel.style.left) {
                panel.style.left = 'auto';
            }

            const board = panel.querySelector('.nex-face-grid-panel__board');
            for (let index = 0; index < 4; index += 1) {
                const cell = document.createElement('div');
                cell.className = 'nex-face-grid-panel__cell';
                cell.dataset.index = String(index);
                cell.addEventListener('click', () => {
                    this.activeCellIndex = index;
                    panel.querySelector('.nex-face-grid-panel__file-input').click();
                });
                ['dragenter', 'dragover'].forEach((type) => {
                    cell.addEventListener(type, (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        cell.classList.add('is-dragover');
                    });
                });
                ['dragleave', 'dragend', 'drop'].forEach((type) => {
                    cell.addEventListener(type, (event) => {
                        event.preventDefault();
                        event.stopPropagation();
                        cell.classList.remove('is-dragover');
                    });
                });
                cell.addEventListener('drop', async (event) => {
                    await this.handleCellDrop(index, event);
                });
                board.appendChild(cell);
            }

            const header = panel.querySelector('.nex-face-grid-panel__header');
            header.addEventListener('mousedown', (event) => this.startDragging(event));
            panel.querySelector('.nex-face-grid-panel__close').addEventListener('click', () => this.closePanel());
            panel.querySelector('.nex-face-grid-panel__cancel').addEventListener('click', () => this.closePanel());
            panel.querySelector('.nex-face-grid-panel__clear').addEventListener('click', () => {
                this.resetCells();
                this.refreshPanelState();
            });
            panel.querySelector('.nex-face-grid-panel__file-input').addEventListener('change', (event) => {
                const file = event.target.files && event.target.files[0];
                if (file && this.activeCellIndex >= 0) {
                    this.assignFileToCell(this.activeCellIndex, file);
                    this.refreshPanelState();
                }
                event.target.value = '';
            });
            panel.querySelector('.nex-face-grid-panel__primary').addEventListener('click', async () => {
                await this.composeToSlot(panel);
            });

            document.body.appendChild(panel);
            this.panel = panel;
            this.refreshPanelState();
        }

        startDragging(event) {
            if (!this.panel) {
                return;
            }
            if (event.target.tagName === 'BUTTON') {
                return;
            }
            const rect = this.panel.getBoundingClientRect();
            this.isDragging = true;
            this.dragOffset = {
                x: event.clientX - rect.left,
                y: event.clientY - rect.top,
            };
            this.panel.style.right = 'auto';
            this.panel.style.left = rect.left + 'px';
            this.panel.style.top = rect.top + 'px';
            this.panel.classList.add('dragging');
            event.preventDefault();
        }

        handleDrag(event) {
            if (!this.isDragging || !this.panel) {
                return;
            }
            this.panel.style.left = `${event.clientX - this.dragOffset.x}px`;
            this.panel.style.top = `${event.clientY - this.dragOffset.y}px`;
        }

        stopDragging() {
            if (!this.isDragging || !this.panel) {
                return;
            }
            this.isDragging = false;
            this.panel.classList.remove('dragging');
            localStorage.setItem('face-grid-panel-pos', JSON.stringify({
                top: this.panel.style.top,
                left: this.panel.style.left,
            }));
        }

        assignFileToCell(index, file) {
            if (this.cellObjectUrls[index]) {
                URL.revokeObjectURL(this.cellObjectUrls[index]);
            }
            this.cellFiles[index] = file;
            this.cellObjectUrls[index] = URL.createObjectURL(file);
        }

        clearCell(index) {
            if (this.cellObjectUrls[index]) {
                URL.revokeObjectURL(this.cellObjectUrls[index]);
            }
            this.cellFiles[index] = null;
            this.cellObjectUrls[index] = '';
            this.refreshPanelState();
        }

        refreshPanelState() {
            if (!this.panel) {
                return;
            }
            const filledCount = this.cellFiles.filter(Boolean).length;
            const status = this.panel.querySelector('.nex-face-grid-panel__status');
            const primary = this.panel.querySelector('.nex-face-grid-panel__primary');
            status.textContent = filledCount >= 2 ? `${filledCount} images ready` : 'Add at least 2 images';
            primary.disabled = filledCount < 2;

            const cells = this.panel.querySelectorAll('.nex-face-grid-panel__cell');
            cells.forEach((cell, index) => {
                const file = this.cellFiles[index];
                const previewUrl = this.cellObjectUrls[index];
                cell.classList.toggle('has-image', Boolean(file));
                if (!file || !previewUrl) {
                    cell.innerHTML = `
                        <span class="nex-face-grid-panel__cell-label">Face ${index + 1}</span>
                        <span class="nex-face-grid-panel__cell-hint">Click or drop image</span>
                    `;
                    return;
                }
                cell.innerHTML = `
                    <img src="${previewUrl}" alt="Face ${index + 1}">
                    <button type="button" class="nex-face-grid-panel__cell-clear" aria-label="Remove image">Remove</button>
                `;
                const clearButton = cell.querySelector('.nex-face-grid-panel__cell-clear');
                clearButton.addEventListener('click', (event) => {
                    event.preventDefault();
                    event.stopPropagation();
                    this.clearCell(index);
                });
            });
        }

        async handleCellDrop(index, event) {
            try {
                const dataTransfer = event.dataTransfer;
                if (!dataTransfer) {
                    return;
                }
                const file = dataTransfer.files && dataTransfer.files[0];
                if (file) {
                    this.assignFileToCell(index, file);
                    this.refreshPanelState();
                    return;
                }

                const droppedUrl = normalizeDroppedUrl(
                    dataTransfer.getData('text/uri-list') || dataTransfer.getData('text/plain') || ''
                );
                if (!droppedUrl) {
                    return;
                }

                const stagedPayload = await this.stageUrlToFile(droppedUrl);
                const stagedFile = await this.fetchStagedFile(stagedPayload);
                this.assignFileToCell(index, stagedFile);
                this.refreshPanelState();
            } catch (error) {
                console.error('[face-grid-composer] Drop failed:', error);
                window.alert('Could not use the dropped image. Try dropping a local image file or a staged image again.');
            }
        }

        async stageUrlToFile(url) {
            const formData = new FormData();
            formData.append('url', url);
            const response = await fetch('/staging_api/upload', {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                throw new Error(`Failed to stage dropped image: ${response.status}`);
            }
            return response.json();
        }

        async fetchStagedFile(payload) {
            const response = await fetch(payload.url);
            if (!response.ok) {
                throw new Error(`Failed to fetch staged image: ${response.status}`);
            }
            const blob = await response.blob();
            return new File([blob], payload.file || 'dropped_face.png', { type: blob.type || 'image/png' });
        }

        loadImageFromFile(file) {
            return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                    const image = new Image();
                    image.onload = () => resolve(image);
                    image.onerror = reject;
                    image.src = reader.result;
                };
                reader.onerror = reject;
                reader.readAsDataURL(file);
            });
        }

        drawCoverSquare(ctx, image, x, y, size) {
            const srcWidth = image.naturalWidth || image.width;
            const srcHeight = image.naturalHeight || image.height;
            const side = Math.min(srcWidth, srcHeight);
            const sx = Math.max(0, (srcWidth - side) / 2);
            const sy = Math.max(0, (srcHeight - side) / 2);
            ctx.drawImage(image, sx, sy, side, side, x, y, size, size);
        }

        async buildGridFile() {
            const canvas = document.createElement('canvas');
            const canvasSize = 1024;
            const cellSize = canvasSize / 2;
            canvas.width = canvasSize;
            canvas.height = canvasSize;
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = '#ffffff';
            ctx.fillRect(0, 0, canvas.width, canvas.height);

            for (let index = 0; index < 4; index += 1) {
                const file = this.cellFiles[index];
                const x = (index % 2) * cellSize;
                const y = Math.floor(index / 2) * cellSize;
                if (!file) {
                    ctx.fillStyle = '#f3f4f6';
                    ctx.fillRect(x, y, cellSize, cellSize);
                    continue;
                }
                const image = await this.loadImageFromFile(file);
                this.drawCoverSquare(ctx, image, x, y, cellSize);
            }

            return new Promise((resolve, reject) => {
                canvas.toBlob((blob) => {
                    if (!blob) {
                        reject(new Error('Failed to create grid image blob.'));
                        return;
                    }
                    resolve(new File([blob], 'face_grid_2x2.png', { type: 'image/png' }));
                }, 'image/png');
            });
        }

        async saveGridToStaging(file) {
            const formData = new FormData();
            formData.append('file', file, file.name || 'face_grid_2x2.png');
            const response = await fetch('/staging_api/upload', {
                method: 'POST',
                body: formData,
            });
            if (!response.ok) {
                throw new Error(`Failed to save composed grid: ${response.status}`);
            }
            return response.json();
        }

        async composeToSlot(panel) {
            const primary = panel.querySelector('.nex-face-grid-panel__primary');
            primary.disabled = true;
            primary.textContent = 'Composing...';
            try {
                const file = await this.buildGridFile();
                const payload = await this.saveGridToStaging(file);
                if (!this.activeSlot || typeof this.activeSlot.setFieldValue !== 'function') {
                    throw new Error('Target slot is not available anymore.');
                }
                this.activeSlot.setFieldValue(this.activeSlot.dataset.workspaceFieldId || '', '');
                this.activeSlot.setFieldValue(this.activeSlot.dataset.pathFieldId || '', payload.filepath || '');
                if (typeof this.activeSlot.syncFromApiFields === 'function') {
                    this.activeSlot.syncFromApiFields(true);
                }
                this.closePanel();
            } catch (error) {
                console.error('[face-grid-composer] Compose failed:', error);
                window.alert('Could not compose the face grid. Check the console log for details.');
                primary.disabled = false;
                primary.textContent = 'Compose To Slot';
            }
        }

        closePanel(resetState = true) {
            if (this.panel) {
                this.panel.remove();
                this.panel = null;
            }
            if (resetState) {
                this.activeSlot = null;
            }
            this.activeCellIndex = -1;
            this.stopDragging();
            if (resetState) {
                this.resetCells();
            }
        }
    }

    function initFaceGridComposer() {
        if (!window.__nexFaceGridComposer) {
            window.__nexFaceGridComposer = new NexFaceGridComposer();
        }
    }

    if (document.readyState === 'loading') {
        window.addEventListener('DOMContentLoaded', initFaceGridComposer, { once: true });
    } else {
        initFaceGridComposer();
    }
})();




