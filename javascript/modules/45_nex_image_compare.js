(() => {
    const SLOT_COUNT = 4;
    const MIN_ZOOM = 1;
    const MAX_ZOOM = 8;
    const ZOOM_STEP = 0.16;

    function appRoot() {
        if (typeof gradioApp === 'function') {
            return gradioApp();
        }
        return document;
    }

    function toAbsoluteUrl(value) {
        if (!value) {
            return '';
        }
        try {
            return new URL(String(value), window.location.origin).toString();
        } catch (error) {
            return String(value);
        }
    }

    function toRelativeUrl(value) {
        if (!value) {
            return '';
        }
        try {
            const absolute = new URL(String(value), window.location.origin);
            if (absolute.origin === window.location.origin) {
                return `${absolute.pathname}${absolute.search}`;
            }
        } catch (error) {
            return String(value);
        }
        return String(value);
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function detectStagingName(value) {
        const relative = toRelativeUrl(value);
        const match = relative.match(/\/staging_api\/image\/([^/?#]+)/);
        return match ? decodeURIComponent(match[1]) : '';
    }

    function getFilenameFromUrl(value) {
        const raw = String(value || '').trim();
        if (!raw) {
            return 'compare_image.png';
        }
        const relative = toRelativeUrl(raw);
        if (relative.startsWith('/file=')) {
            const decoded = decodeURIComponent(relative.slice('/file='.length));
            const parts = decoded.split(/[\\/]+/).filter(Boolean);
            return parts.length ? parts[parts.length - 1] : 'compare_image.png';
        }
        const parts = relative.split('/').filter(Boolean);
        return decodeURIComponent(parts.length ? parts[parts.length - 1].split('?')[0] : 'compare_image.png');
    }

    function loadImageMetrics(url) {
        return new Promise((resolve) => {
            const image = new Image();
            image.onload = () => resolve({
                width: image.naturalWidth || 0,
                height: image.naturalHeight || 0,
            });
            image.onerror = () => resolve({
                width: 0,
                height: 0,
            });
            image.src = toAbsoluteUrl(url);
        });
    }

    function getCompareStore() {
        if (!window.__nexImageCompareStore) {
            window.__nexImageCompareStore = {
                slots: Array.from({ length: SLOT_COUNT }, () => null),
                camera: {
                    zoom: 1,
                    ratioX: 0,
                    ratioY: 0,
                },
            };
        }
        return window.__nexImageCompareStore;
    }

    class NexImageCompare extends HTMLElement {
        constructor() {
            super();
            this.state = getCompareStore();
            this.boundGalleryNodes = new WeakSet();
            this.galleryObserver = null;
            this.galleryRoot = null;
            this.appObserver = null;
            this.galleryBindQueued = false;
            this.panState = null;

            this.handleClick = this.handleClick.bind(this);
            this.handleDragStart = this.handleDragStart.bind(this);
            this.handleDragOver = this.handleDragOver.bind(this);
            this.handleDragLeave = this.handleDragLeave.bind(this);
            this.handleDrop = this.handleDrop.bind(this);
            this.handleWheel = this.handleWheel.bind(this);
            this.handleMouseDown = this.handleMouseDown.bind(this);
            this.handleMouseMove = this.handleMouseMove.bind(this);
            this.handleMouseUp = this.handleMouseUp.bind(this);
        }

        connectedCallback() {
            if (this.dataset.ready === 'true') {
                return;
            }
            this.dataset.ready = 'true';
            this.addEventListener('click', this.handleClick);
            this.addEventListener('dragstart', this.handleDragStart);
            this.addEventListener('dragover', this.handleDragOver);
            this.addEventListener('dragleave', this.handleDragLeave);
            this.addEventListener('drop', this.handleDrop);
            this.addEventListener('wheel', this.handleWheel, { passive: false });
            this.addEventListener('mousedown', this.handleMouseDown);
            document.addEventListener('mousemove', this.handleMouseMove);
            document.addEventListener('mouseup', this.handleMouseUp);
            this.observeApp();
            this.render();
            this.queueGalleryBinding();
            this.announceCompareState();
        }

        disconnectedCallback() {
            this.removeEventListener('click', this.handleClick);
            this.removeEventListener('dragstart', this.handleDragStart);
            this.removeEventListener('dragover', this.handleDragOver);
            this.removeEventListener('dragleave', this.handleDragLeave);
            this.removeEventListener('drop', this.handleDrop);
            this.removeEventListener('wheel', this.handleWheel);
            this.removeEventListener('mousedown', this.handleMouseDown);
            document.removeEventListener('mousemove', this.handleMouseMove);
            document.removeEventListener('mouseup', this.handleMouseUp);
            if (this.appObserver) {
                this.appObserver.disconnect();
                this.appObserver = null;
            }
            if (this.galleryObserver) {
                this.galleryObserver.disconnect();
                this.galleryObserver = null;
            }
            this.galleryRoot = null;
        }

        observeApp() {
            if (this.appObserver) {
                return;
            }
            this.appObserver = new MutationObserver(() => this.queueGalleryBinding());
            this.appObserver.observe(appRoot(), { childList: true, subtree: true });
        }

        queueGalleryBinding() {
            if (this.galleryBindQueued) {
                return;
            }
            this.galleryBindQueued = true;
            window.requestAnimationFrame(() => {
                this.galleryBindQueued = false;
                this.bindGalleryNodes();
            });
        }

        bindGalleryNodes() {
            const gallery = appRoot().querySelector('#final_gallery');
            if (!gallery) {
                return;
            }
            if (gallery !== this.galleryRoot) {
                if (this.galleryObserver) {
                    this.galleryObserver.disconnect();
                }
                this.galleryRoot = gallery;
                this.galleryObserver = new MutationObserver(() => this.queueGalleryBinding());
                this.galleryObserver.observe(gallery, { childList: true, subtree: true });
            }

            const images = Array.from(gallery.querySelectorAll('img'));
            const totalCount = images.length;
            images.forEach((image, index) => {
                image.dataset.compareGenerationIndex = String(index + 1);
                image.dataset.compareGenerationTotal = String(totalCount);
                if (this.boundGalleryNodes.has(image)) {
                    return;
                }
                this.boundGalleryNodes.add(image);
                image.draggable = true;
                image.addEventListener('dblclick', async (event) => {
                    event.preventDefault();
                    await this.loadIntoNextEmptySlot(this.buildGalleryDescriptor(image));
                });
                image.addEventListener('dragstart', (event) => {
                    const descriptor = this.buildGalleryDescriptor(image);
                    const payload = JSON.stringify({
                        kind: 'nex-image-source',
                        ...descriptor,
                    });
                    event.dataTransfer.effectAllowed = 'copy';
                    event.dataTransfer.setData('application/json', payload);
                    event.dataTransfer.setData('text/plain', descriptor.absoluteUrl);
                    event.dataTransfer.setData('text/uri-list', descriptor.absoluteUrl);
                });
            });
        }

        buildGalleryDescriptor(image) {
            const absoluteUrl = toAbsoluteUrl(image.currentSrc || image.src || '');
            const relativeUrl = toRelativeUrl(absoluteUrl);
            return {
                sourceKind: 'gallery',
                absoluteUrl,
                relativeUrl,
                stagingName: '',
                generationIndex: Number(image.dataset.compareGenerationIndex || 0) || 0,
                totalCount: Number(image.dataset.compareGenerationTotal || 0) || 0,
            };
        }

        getLoadedEntries() {
            return this.state.slots
                .map((slot, index) => ({ slot, index }))
                .filter((entry) => entry.slot);
        }

        getLayoutMode() {
            const loaded = this.getLoadedEntries();
            if (loaded.length <= 1) {
                return 'single';
            }
            const allLandscape = loaded.every(({ slot }) => slot.width > slot.height * 1.02);
            if (allLandscape) {
                return 'stack';
            }
            if (loaded.length === 2) {
                return 'pair';
            }
            return 'grid';
        }

        render() {
            const loaded = this.getLoadedEntries();
            const layoutMode = this.getLayoutMode();
            const allEntries = this.state.slots.map((slot, index) => ({ slot, index }));

            if (loaded.length === 0) {
                this.innerHTML = `
                    <section class="nic-shell nic-shell--empty">
                        <div class="nic-empty-drop" data-slot-index="-1">
                            <div class="nic-empty-drop__title">Compare Viewer</div>
                            <div class="nic-empty-drop__copy">Double-click gallery images or drag from gallery/staging here. Ctrl + wheel zooms. Ctrl + left-drag pans all loaded images together.</div>
                        </div>
                    </section>
                `;
                return;
            }

            this.innerHTML = `
                <section class="nic-shell nic-shell--${layoutMode}" data-layout-mode="${layoutMode}" data-loaded-count="${loaded.length}">
                    <div class="nic-grid nic-grid--${layoutMode}">
                        ${allEntries.map((entry, renderIndex) => this.renderSlot(entry, renderIndex, loaded.length, layoutMode)).join('')}
                    </div>
                </section>
            `;

            this.syncAllTransforms();
        }

        renderSlot(entry, renderIndex, totalLoaded, layoutMode) {
            const { slot, index } = entry;
            const toolbarPosition = this.getToolbarPosition(layoutMode, renderIndex, totalLoaded);
            if (!slot) {
                return `
                    <article class="nic-slot nic-slot--${layoutMode} nic-slot--empty" data-slot-index="${index}">
                        <div class="nic-slot__viewport nic-slot__viewport--empty" data-slot-index="${index}">
                            <div class="nic-slot__toolbar nic-slot__toolbar--${toolbarPosition}">
                                <span class="nic-slot__label">C${index + 1}</span>
                            </div>
                            <div class="nic-slot__dropcopy">
                                <div class="nic-slot__dropcopy-title">Empty</div>
                                <div class="nic-slot__dropcopy-text">Drag from gallery or staging</div>
                            </div>
                        </div>
                    </article>
                `;
            }
            return `
                <article class="nic-slot nic-slot--${layoutMode}" data-slot-index="${index}">
                    <div class="nic-slot__viewport" data-slot-index="${index}">
                        <img class="nic-slot__image" src="${slot.absoluteUrl}" alt="${slot.filename}" draggable="false" data-slot-index="${index}">
                        <div class="nic-slot__toolbar nic-slot__toolbar--${toolbarPosition}">
                            <span class="nic-slot__label">C${index + 1}</span>
                            <button type="button" class="nic-slot__tool" data-action="reset-view">Reset</button>
                            <button type="button" class="nic-slot__tool" data-action="clear-slot">Clear</button>
                        </div>
                    </div>
                </article>
            `;
        }

        getToolbarPosition(layoutMode, renderIndex, totalLoaded) {
            if (layoutMode === 'stack') {
                return 'side';
            }
            if (layoutMode === 'grid' || layoutMode === 'pair') {
                const isBottomRow = totalLoaded > 2 && renderIndex >= 2;
                return isBottomRow ? 'bottom' : 'top';
            }
            return 'top';
        }

        syncAllTransforms() {
            this.getLoadedEntries().forEach(({ index }) => {
                const image = this.getSlotImage(index);
                if (!image) {
                    return;
                }
                if (!image.dataset.transformBound) {
                    image.dataset.transformBound = 'true';
                    image.addEventListener('load', () => this.applyTransform(index));
                }
                this.applyTransform(index);
            });
        }

        getSlotElement(index) {
            return this.querySelector(`.nic-slot[data-slot-index="${index}"]`);
        }

        getViewport(index) {
            return this.querySelector(`.nic-slot__viewport[data-slot-index="${index}"]`);
        }

        getSlotImage(index) {
            return this.querySelector(`.nic-slot__image[data-slot-index="${index}"]`);
        }

        computeMaxPan(index, zoomOverride = null) {
            const viewport = this.getViewport(index);
            const image = this.getSlotImage(index);
            if (!viewport || !image) {
                return { x: 0, y: 0 };
            }

            const zoom = zoomOverride ?? this.state.camera.zoom;
            const baseWidth = image.clientWidth || image.naturalWidth || 0;
            const baseHeight = image.clientHeight || image.naturalHeight || 0;
            const viewportWidth = viewport.clientWidth || 0;
            const viewportHeight = viewport.clientHeight || 0;

            return {
                x: Math.max(0, ((baseWidth * zoom) - viewportWidth) / 2),
                y: Math.max(0, ((baseHeight * zoom) - viewportHeight) / 2),
            };
        }

        applyTransform(index) {
            const image = this.getSlotImage(index);
            const slot = this.getSlotElement(index);
            if (!image || !slot) {
                return;
            }

            const { zoom, ratioX, ratioY } = this.state.camera;
            const maxPan = this.computeMaxPan(index);
            const panX = maxPan.x * ratioX;
            const panY = maxPan.y * ratioY;
            image.style.transform = `translate(${panX}px, ${panY}px) scale(${zoom})`;
            slot.classList.toggle('is-zoomed', zoom > 1.01);
        }

        handleClick(event) {
            const button = event.target.closest('[data-action]');
            if (!button || !this.contains(button)) {
                return;
            }

            const slotNode = button.closest('.nic-slot');
            const slotIndex = slotNode ? Number(slotNode.dataset.slotIndex || -1) : -1;
            const action = button.dataset.action;

            if (action === 'reset-view') {
                this.resetCamera();
                return;
            }

            if (action === 'clear-slot' && slotIndex >= 0) {
                this.state.slots[slotIndex] = null;
                this.resetCamera();
                this.render();
                this.announceCompareState();
            }
        }

        handleDragStart(event) {
            if (this.contains(event.target)) {
                event.preventDefault();
            }
        }

        handleDragOver(event) {
            if (!this.contains(event.target)) {
                return;
            }
            event.preventDefault();
            const targetSlot = event.target.closest('.nic-slot');
            if (targetSlot) {
                targetSlot.classList.add('is-dragover');
            } else {
                this.querySelector('.nic-shell')?.classList.add('is-dragover');
            }
        }

        handleDragLeave(event) {
            const slot = event.target.closest('.nic-slot');
            if (slot && !slot.contains(event.relatedTarget)) {
                slot.classList.remove('is-dragover');
            }
            if (!this.contains(event.relatedTarget)) {
                this.querySelector('.nic-shell')?.classList.remove('is-dragover');
            }
        }

        async handleDrop(event) {
            if (!this.contains(event.target)) {
                return;
            }
            event.preventDefault();
            this.querySelector('.nic-shell')?.classList.remove('is-dragover');
            this.querySelectorAll('.nic-slot.is-dragover').forEach((node) => node.classList.remove('is-dragover'));

            const targetSlot = event.target.closest('.nic-slot');
            const slotIndex = targetSlot ? Number(targetSlot.dataset.slotIndex || -1) : this.findNextEmptySlot();
            const source = this.parseDroppedSource(event.dataTransfer);
            if (!source) {
                return;
            }

            const finalIndex = slotIndex >= 0 ? slotIndex : 0;
            await this.populateSlot(finalIndex, source);
        }

        parseDroppedSource(dataTransfer) {
            if (!dataTransfer) {
                return null;
            }

            const jsonPayload = dataTransfer.getData('application/json');
            if (jsonPayload) {
                try {
                    const parsed = JSON.parse(jsonPayload);
                    if (parsed && parsed.kind === 'nex-image-source' && parsed.absoluteUrl) {
                        return parsed;
                    }
                } catch (error) {
                    console.warn('[nex-image-compare] Ignoring non-compare JSON payload:', error);
                }
            }

            const url = (
                dataTransfer.getData('text/uri-list') ||
                dataTransfer.getData('text/plain') ||
                ''
            ).split('\n').map((line) => line.trim()).find((line) => line && !line.startsWith('#'));

            if (!url) {
                return null;
            }

            return {
                sourceKind: detectStagingName(url) ? 'staging' : 'external',
                absoluteUrl: toAbsoluteUrl(url),
                relativeUrl: toRelativeUrl(url),
                stagingName: detectStagingName(url),
                generationIndex: 0,
                totalCount: 0,
            };
        }

        handleWheel(event) {
            const viewport = event.target.closest('.nic-slot__viewport');
            if (!viewport || !this.contains(viewport) || !event.ctrlKey) {
                return;
            }

            event.preventDefault();
            const nextZoom = clamp(
                this.state.camera.zoom + (event.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP),
                MIN_ZOOM,
                MAX_ZOOM
            );
            this.state.camera.zoom = nextZoom;
            if (nextZoom <= 1.0001) {
                this.state.camera.ratioX = 0;
                this.state.camera.ratioY = 0;
            }
            this.syncAllTransforms();
        }

        handleMouseDown(event) {
            const viewport = event.target.closest('.nic-slot__viewport');
            if (!viewport || !this.contains(viewport) || event.button !== 0 || !event.ctrlKey) {
                return;
            }
            if (this.state.camera.zoom <= 1.0001) {
                return;
            }

            const slotNode = viewport.closest('.nic-slot');
            const slotIndex = Number(slotNode?.dataset.slotIndex || -1);
            const maxPan = this.computeMaxPan(slotIndex);
            event.preventDefault();
            this.panState = {
                slotIndex,
                startX: event.clientX,
                startY: event.clientY,
                maxPanX: maxPan.x,
                maxPanY: maxPan.y,
                originRatioX: this.state.camera.ratioX,
                originRatioY: this.state.camera.ratioY,
            };
            viewport.classList.add('is-panning');
        }

        handleMouseMove(event) {
            if (!this.panState) {
                return;
            }

            const deltaX = event.clientX - this.panState.startX;
            const deltaY = event.clientY - this.panState.startY;
            const nextRatioX = this.panState.maxPanX > 0
                ? clamp(this.panState.originRatioX + (deltaX / this.panState.maxPanX), -1, 1)
                : 0;
            const nextRatioY = this.panState.maxPanY > 0
                ? clamp(this.panState.originRatioY + (deltaY / this.panState.maxPanY), -1, 1)
                : 0;

            this.state.camera.ratioX = nextRatioX;
            this.state.camera.ratioY = nextRatioY;
            this.syncAllTransforms();
        }

        handleMouseUp() {
            if (!this.panState) {
                return;
            }
            this.getViewport(this.panState.slotIndex)?.classList.remove('is-panning');
            this.panState = null;
        }

        resetCamera() {
            this.state.camera.zoom = 1;
            this.state.camera.ratioX = 0;
            this.state.camera.ratioY = 0;
            this.syncAllTransforms();
        }

        findNextEmptySlot() {
            return this.state.slots.findIndex((slot) => !slot);
        }

        async loadIntoNextEmptySlot(source) {
            const emptyIndex = this.findNextEmptySlot();
            const targetIndex = emptyIndex >= 0 ? emptyIndex : 0;
            await this.populateSlot(targetIndex, source);
        }

        async populateSlot(slotIndex, source) {
            if (slotIndex < 0 || slotIndex >= SLOT_COUNT) {
                return;
            }

            const absoluteUrl = toAbsoluteUrl(source.absoluteUrl || source.url || source.relativeUrl || '');
            const relativeUrl = source.relativeUrl || toRelativeUrl(absoluteUrl);
            if (!absoluteUrl) {
                return;
            }

            const metrics = await loadImageMetrics(absoluteUrl);
            this.state.slots[slotIndex] = {
                absoluteUrl,
                relativeUrl,
                stagingName: source.stagingName || detectStagingName(relativeUrl || absoluteUrl),
                filename: getFilenameFromUrl(relativeUrl || absoluteUrl),
                width: metrics.width || 0,
                height: metrics.height || 0,
                sourceKind: source.sourceKind || 'manual',
                generationIndex: source.generationIndex || 0,
                totalCount: source.totalCount || 0,
            };

            this.resetCamera();
            this.render();
            this.announceCompareState();
        }

        announceCompareState() {
            const stagingMap = {};
            this.state.slots.forEach((slot, index) => {
                if (!slot || !slot.stagingName) {
                    return;
                }
                stagingMap[slot.stagingName] = `C${index + 1}`;
            });
            window.dispatchEvent(new CustomEvent('nex-compare:state-change', {
                detail: { stagingMap },
            }));
        }
    }

    if (!customElements.get('nex-image-compare')) {
        customElements.define('nex-image-compare', NexImageCompare);
    }
})();
