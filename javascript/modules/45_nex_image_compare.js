(() => {
    const SLOT_COUNT = 4;
    const MIN_ZOOM = 1;
    const MAX_ZOOM = 8;
    const ZOOM_STEP = 0.16;
    const PROMPT_PREVIEW_LENGTH = 72;
    const PNG_SIGNATURE = [137, 80, 78, 71, 13, 10, 26, 10];

    function appRoot() {
        if (typeof gradioApp === 'function') {
            return gradioApp();
        }
        return document;
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function escapeSelector(value) {
        if (window.CSS && typeof window.CSS.escape === 'function') {
            return window.CSS.escape(value);
        }
        return String(value).replace(/([^a-zA-Z0-9_-])/g, '\\$1');
    }

    function clamp(value, min, max) {
        return Math.min(max, Math.max(min, value));
    }

    function truncateText(value, maxLength = PROMPT_PREVIEW_LENGTH) {
        const text = String(value || '').trim();
        if (!text) {
            return 'Unknown';
        }
        if (text.length <= maxLength) {
            return text;
        }
        return `${text.slice(0, Math.max(0, maxLength - 3)).trimEnd()}...`;
    }

    function toAbsoluteUrl(value) {
        if (!value) {
            return '';
        }
        try {
            return new URL(String(value), window.location.origin).toString();
        } catch (error) {
            console.warn('[nex-image-compare] Could not build absolute URL:', error);
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

    function detectStagingName(value) {
        const relative = toRelativeUrl(value);
        const match = relative.match(/\/staging_api\/image\/([^/?#]+)/);
        return match ? decodeURIComponent(match[1]) : '';
    }

    function readUint32(bytes, offset) {
        return (
            (bytes[offset] << 24) |
            (bytes[offset + 1] << 16) |
            (bytes[offset + 2] << 8) |
            bytes[offset + 3]
        ) >>> 0;
    }

    function readPngTextEntries(buffer) {
        const bytes = new Uint8Array(buffer);
        if (bytes.length < PNG_SIGNATURE.length) {
            return {};
        }
        for (let index = 0; index < PNG_SIGNATURE.length; index += 1) {
            if (bytes[index] !== PNG_SIGNATURE[index]) {
                return {};
            }
        }

        const decoder = new TextDecoder('utf-8');
        const entries = {};
        let offset = 8;

        while (offset + 8 <= bytes.length) {
            const length = readUint32(bytes, offset);
            offset += 4;
            const type = decoder.decode(bytes.slice(offset, offset + 4));
            offset += 4;

            if (offset + length + 4 > bytes.length) {
                break;
            }

            const chunk = bytes.slice(offset, offset + length);
            offset += length;
            offset += 4;

            if (type === 'tEXt') {
                const separator = chunk.indexOf(0);
                if (separator > 0) {
                    const key = decoder.decode(chunk.slice(0, separator));
                    const value = decoder.decode(chunk.slice(separator + 1));
                    entries[key] = value;
                }
            } else if (type === 'iTXt') {
                let cursor = 0;
                const keywordEnd = chunk.indexOf(0, cursor);
                if (keywordEnd <= 0) {
                    continue;
                }
                const key = decoder.decode(chunk.slice(cursor, keywordEnd));
                cursor = keywordEnd + 1;
                const compressionFlag = chunk[cursor];
                cursor += 1;
                cursor += 1;
                const languageEnd = chunk.indexOf(0, cursor);
                if (languageEnd < 0) {
                    continue;
                }
                cursor = languageEnd + 1;
                const translatedEnd = chunk.indexOf(0, cursor);
                if (translatedEnd < 0) {
                    continue;
                }
                cursor = translatedEnd + 1;
                if (compressionFlag !== 0) {
                    continue;
                }
                entries[key] = decoder.decode(chunk.slice(cursor));
            }

            if (type === 'IEND') {
                break;
            }
        }

        return entries;
    }

    function parseMetadataPayload(textEntries) {
        const rawParameters = textEntries.parameters || '';
        if (!rawParameters) {
            return {
                prompt: 'Unknown',
                seed: 'Unknown',
            };
        }

        try {
            const payload = JSON.parse(rawParameters);
            return {
                prompt: payload.prompt || payload.raw_prompt || payload.full_prompt || payload.positive_prompt || 'Unknown',
                seed: payload.seed || 'Unknown',
            };
        } catch (error) {
            return {
                prompt: 'Unknown',
                seed: 'Unknown',
            };
        }
    }

    async function fetchJson(url, options = {}) {
        const response = await fetch(url, options);
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload?.detail || response.statusText || 'Request failed');
        }
        return payload;
    }

    function getMetadataCache() {
        if (!window.__nexCompareMetadataCache) {
            window.__nexCompareMetadataCache = new Map();
        }
        return window.__nexCompareMetadataCache;
    }

    async function fetchImageMetadata(url) {
        const cache = getMetadataCache();
        const cacheKey = toAbsoluteUrl(url);
        if (cache.has(cacheKey)) {
            return cache.get(cacheKey);
        }

        const task = (async () => {
            try {
                const response = await fetch(cacheKey, { cache: 'force-cache' });
                if (!response.ok) {
                    return {
                        prompt: 'Unknown',
                        seed: 'Unknown',
                    };
                }
                const blob = await response.blob();
                const buffer = await blob.arrayBuffer();
                const entries = readPngTextEntries(buffer);
                if (!entries.parameters) {
                    return {
                        prompt: 'Unknown',
                        seed: 'Unknown',
                    };
                }
                return parseMetadataPayload(entries);
            } catch (error) {
                console.warn('[nex-image-compare] Metadata read failed:', error);
                return {
                    prompt: 'Unknown',
                    seed: 'Unknown',
                };
            }
        })();

        cache.set(cacheKey, task);
        return task;
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

    function buildGenerationLabel(source) {
        if (source.generationLabel) {
            return source.generationLabel;
        }
        if (source.generationIndex && source.totalCount) {
            return `#${source.generationIndex} of ${source.totalCount}`;
        }
        if (source.sourceKind === 'staging') {
            return 'Staging';
        }
        return 'Manual';
    }

    function getCompareStore() {
        if (!window.__nexImageCompareStore) {
            window.__nexImageCompareStore = {
                slots: Array.from({ length: SLOT_COUNT }, () => null),
                openMenuSlot: -1,
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
            this.galleryBindQueued = false;
            this.appObserver = null;
            this.panState = null;
            this.lastTargetSignature = '';

            this.handleClick = this.handleClick.bind(this);
            this.handleDragStart = this.handleDragStart.bind(this);
            this.handleDragOver = this.handleDragOver.bind(this);
            this.handleDragLeave = this.handleDragLeave.bind(this);
            this.handleDrop = this.handleDrop.bind(this);
            this.handleWheel = this.handleWheel.bind(this);
            this.handleMouseDown = this.handleMouseDown.bind(this);
            this.handleMouseMove = this.handleMouseMove.bind(this);
            this.handleMouseUp = this.handleMouseUp.bind(this);
            this.handleDocumentClick = this.handleDocumentClick.bind(this);
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
            document.addEventListener('click', this.handleDocumentClick);
            this.lastTargetSignature = this.buildTargetSignature();
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
            document.removeEventListener('click', this.handleDocumentClick);
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
            this.appObserver = new MutationObserver((mutations) => {
                const hasExternalMutation = mutations.some((mutation) => !this.contains(mutation.target));
                this.queueGalleryBinding();
                if (!hasExternalMutation) {
                    return;
                }
                const nextSignature = this.buildTargetSignature();
                if (nextSignature !== this.lastTargetSignature) {
                    this.lastTargetSignature = nextSignature;
                    this.render();
                }
            });
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
                if (this.boundGalleryNodes.has(image)) {
                    image.dataset.compareGenerationIndex = String(index + 1);
                    image.dataset.compareGenerationTotal = String(totalCount);
                    return;
                }
                this.boundGalleryNodes.add(image);
                image.draggable = true;
                image.dataset.compareGenerationIndex = String(index + 1);
                image.dataset.compareGenerationTotal = String(totalCount);
                image.addEventListener('dblclick', async (event) => {
                    event.preventDefault();
                    const descriptor = this.buildGalleryDescriptor(image);
                    await this.loadIntoNextEmptySlot(descriptor);
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
                generationIndex: Number(image.dataset.compareGenerationIndex || 0) || 0,
                totalCount: Number(image.dataset.compareGenerationTotal || 0) || 0,
                label: image.alt || getFilenameFromUrl(relativeUrl || absoluteUrl),
            };
        }

        getSlotTargets() {
            return Array.from(appRoot().querySelectorAll('nex-image-slot'))
                .filter((node) => node.id)
                .map((node) => ({
                    id: node.id,
                    label: node.dataset.label || node.getAttribute('data-label') || node.id,
                }));
        }

        buildTargetSignature() {
            return this.getSlotTargets()
                .map((target) => `${target.id}:${target.label}`)
                .join('|');
        }

        getLoadedCount() {
            return this.state.slots.filter(Boolean).length;
        }

        render() {
            const targets = this.getSlotTargets();
            const loadedCount = this.getLoadedCount();
            this.innerHTML = `
                <section class="nic-shell" data-loaded-count="${loadedCount}">
                    <header class="nic-header">
                        <div>
                            <h3 class="nic-title">Compare Viewer</h3>
                            <p class="nic-subtitle">Double-click gallery images or drag from gallery/staging into a slot. Ctrl + wheel zooms and Ctrl + left-drag pans.</p>
                        </div>
                        <div class="nic-summary">${loadedCount} / ${SLOT_COUNT} loaded</div>
                    </header>
                    <div class="nic-grid">
                        ${Array.from({ length: SLOT_COUNT }, (_, index) => this.renderSlot(index, targets)).join('')}
                    </div>
                </section>
            `;
            this.syncAllTransforms();
        }

        renderSlot(index, targets) {
            const slot = this.state.slots[index];
            const compareLabel = `C${index + 1}`;
            if (!slot) {
                return `
                    <article class="nic-slot nic-slot--empty" data-slot-index="${index}">
                        <div class="nic-slot__viewport" data-slot-index="${index}">
                            <div class="nic-slot__placeholder">
                                <div class="nic-slot__placeholder-label">${compareLabel}</div>
                                <div class="nic-slot__placeholder-title">Drop Image</div>
                                <div class="nic-slot__placeholder-copy">Drag from gallery or staging, or double-click a gallery image to fill the next empty slot.</div>
                            </div>
                        </div>
                    </article>
                `;
            }

            const targetsMarkup = targets.length
                ? targets.map((target) => `
                    <button type="button" class="nic-menu__target" data-action="send-to-target" data-slot-index="${index}" data-target-id="${escapeHtml(target.id)}">${escapeHtml(target.label)}</button>
                `).join('')
                : '<div class="nic-menu__empty">No image slots found.</div>';

            const revealDisabled = slot.stagingName ? '' : 'disabled';
            const stageLabel = slot.stagingName ? 'Reveal In Staging' : 'Send To Staging';

            return `
                <article class="nic-slot" data-slot-index="${index}">
                    <div class="nic-slot__topline">
                        <div class="nic-slot__badge">${compareLabel}</div>
                        <div class="nic-slot__source">${escapeHtml(slot.sourceLabel)}</div>
                    </div>
                    <div class="nic-slot__viewport" data-slot-index="${index}">
                        <img class="nic-slot__image" src="${escapeHtml(slot.absoluteUrl)}" alt="${escapeHtml(slot.filename)}" draggable="false" data-slot-index="${index}">
                    </div>
                    <div class="nic-slot__meta">
                        <div class="nic-slot__meta-row">
                            <span class="nic-slot__meta-label">Seed</span>
                            <span class="nic-slot__meta-value">${escapeHtml(String(slot.seed || 'Unknown'))}</span>
                        </div>
                        <div class="nic-slot__meta-row">
                            <span class="nic-slot__meta-label">View</span>
                            <span class="nic-slot__meta-value">${escapeHtml(slot.dimensions)} | ${escapeHtml(slot.generationLabel)}</span>
                        </div>
                        <div class="nic-slot__prompt" title="${escapeHtml(slot.promptFull)}">${escapeHtml(slot.promptPreview)}</div>
                    </div>
                    <div class="nic-slot__actions">
                        <button type="button" class="nic-button nic-button--subtle" data-action="reset-view" data-slot-index="${index}">Reset View</button>
                        <button type="button" class="nic-button nic-button--subtle" data-action="clear-slot" data-slot-index="${index}">Clear</button>
                        <details class="nic-menu" ${this.state.openMenuSlot === index ? 'open' : ''}>
                            <summary class="nic-button nic-button--primary" data-action="toggle-menu" data-slot-index="${index}">Actions</summary>
                            <div class="nic-menu__content">
                                <button type="button" class="nic-menu__action" data-action="send-to-staging" data-slot-index="${index}">${stageLabel}</button>
                                <button type="button" class="nic-menu__action" data-action="queue-gimp" data-slot-index="${index}">Queue For GIMP</button>
                                <button type="button" class="nic-menu__action" data-action="reveal-staging" data-slot-index="${index}" ${revealDisabled}>Reveal Staging Tile</button>
                                <div class="nic-menu__group">
                                    <div class="nic-menu__group-title">Send To Image Slot</div>
                                    <div class="nic-menu__targets">${targetsMarkup}</div>
                                </div>
                            </div>
                        </details>
                    </div>
                </article>
            `;
        }

        syncAllTransforms() {
            this.state.slots.forEach((slot, index) => {
                if (!slot) {
                    return;
                }
                const image = this.querySelector(`.nic-slot__image[data-slot-index="${index}"]`);
                if (!image) {
                    return;
                }
                if (!image.dataset.transformBound) {
                    image.dataset.transformBound = 'true';
                    image.addEventListener('load', () => this.applySlotTransform(index));
                }
                this.applySlotTransform(index);
            });
        }

        getViewport(index) {
            return this.querySelector(`.nic-slot__viewport[data-slot-index="${index}"]`);
        }

        getSlotImage(index) {
            return this.querySelector(`.nic-slot__image[data-slot-index="${index}"]`);
        }

        clampPan(index, panX, panY, zoomOverride = null) {
            const slot = this.state.slots[index];
            const viewport = this.getViewport(index);
            const image = this.getSlotImage(index);
            if (!slot || !viewport || !image) {
                return { panX, panY };
            }
            const zoom = zoomOverride ?? slot.zoom ?? 1;
            const baseWidth = image.clientWidth || image.naturalWidth || 0;
            const baseHeight = image.clientHeight || image.naturalHeight || 0;
            const viewportWidth = viewport.clientWidth || 0;
            const viewportHeight = viewport.clientHeight || 0;
            const maxPanX = Math.max(0, ((baseWidth * zoom) - viewportWidth) / 2);
            const maxPanY = Math.max(0, ((baseHeight * zoom) - viewportHeight) / 2);
            return {
                panX: clamp(panX, -maxPanX, maxPanX),
                panY: clamp(panY, -maxPanY, maxPanY),
            };
        }

        applySlotTransform(index) {
            const slot = this.state.slots[index];
            const image = this.getSlotImage(index);
            if (!slot || !image) {
                return;
            }
            const clamped = this.clampPan(index, slot.panX || 0, slot.panY || 0);
            slot.panX = clamped.panX;
            slot.panY = clamped.panY;
            image.style.transform = `translate(${slot.panX}px, ${slot.panY}px) scale(${slot.zoom || 1})`;
            image.closest('.nic-slot')?.classList.toggle('is-zoomed', (slot.zoom || 1) > 1.01);
        }

        handleDocumentClick(event) {
            if (this.contains(event.target)) {
                return;
            }
            if (this.state.openMenuSlot !== -1) {
                this.state.openMenuSlot = -1;
                this.render();
            }
        }

        async handleClick(event) {
            const button = event.target.closest('[data-action]');
            if (!button || !this.contains(button)) {
                return;
            }

            const action = button.dataset.action;
            const slotIndex = Number(button.dataset.slotIndex || -1);

            try {
                if (action === 'toggle-menu') {
                    event.preventDefault();
                    this.state.openMenuSlot = this.state.openMenuSlot === slotIndex ? -1 : slotIndex;
                    this.render();
                    return;
                }

                if (action === 'clear-slot') {
                    this.state.slots[slotIndex] = null;
                    this.state.openMenuSlot = -1;
                    this.render();
                    this.announceCompareState();
                    return;
                }

                if (action === 'reset-view') {
                    this.resetView(slotIndex);
                    return;
                }

                if (action === 'send-to-staging') {
                    await this.sendToStaging(slotIndex);
                    return;
                }

                if (action === 'reveal-staging') {
                    await this.revealInStaging(slotIndex);
                    return;
                }

                if (action === 'queue-gimp') {
                    await this.queueForGimp(slotIndex);
                    return;
                }

                if (action === 'send-to-target') {
                    const targetId = button.dataset.targetId || '';
                    await this.sendToTargetSlot(slotIndex, targetId);
                }
            } catch (error) {
                console.error('[nex-image-compare] Action failed:', error);
                window.alert(error instanceof Error ? error.message : 'Compare viewer action failed.');
            }
        }

        handleDragStart(event) {
            if (!this.contains(event.target)) {
                return;
            }
            const viewport = event.target.closest('.nic-slot__viewport');
            if (!viewport) {
                return;
            }
            const slotIndex = Number(viewport.dataset.slotIndex || -1);
            const slot = this.state.slots[slotIndex];
            if (!slot) {
                return;
            }
            event.preventDefault();
        }

        handleDragOver(event) {
            const slot = event.target.closest('.nic-slot');
            if (!slot || !this.contains(slot)) {
                return;
            }
            event.preventDefault();
            slot.classList.add('is-dragover');
        }

        handleDragLeave(event) {
            const slot = event.target.closest('.nic-slot');
            if (!slot || !this.contains(slot)) {
                return;
            }
            if (!slot.contains(event.relatedTarget)) {
                slot.classList.remove('is-dragover');
            }
        }

        async handleDrop(event) {
            const slot = event.target.closest('.nic-slot');
            if (!slot || !this.contains(slot)) {
                return;
            }
            event.preventDefault();
            slot.classList.remove('is-dragover');
            const slotIndex = Number(slot.dataset.slotIndex || -1);
            const source = this.parseDroppedSource(event.dataTransfer);
            if (!source) {
                return;
            }
            await this.populateSlot(slotIndex, source);
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
            };
        }

        async handleWheel(event) {
            const viewport = event.target.closest('.nic-slot__viewport');
            if (!viewport || !this.contains(viewport) || !event.ctrlKey) {
                return;
            }
            event.preventDefault();
            const slotIndex = Number(viewport.dataset.slotIndex || -1);
            const slot = this.state.slots[slotIndex];
            if (!slot) {
                return;
            }

            const previousZoom = slot.zoom || 1;
            const nextZoom = clamp(
                previousZoom + (event.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP),
                MIN_ZOOM,
                MAX_ZOOM
            );
            if (Math.abs(nextZoom - previousZoom) < 0.0001) {
                return;
            }

            const rect = viewport.getBoundingClientRect();
            const cursorX = event.clientX - rect.left - (rect.width / 2);
            const cursorY = event.clientY - rect.top - (rect.height / 2);
            const ratio = nextZoom / previousZoom;
            let panX = ((slot.panX || 0) - cursorX) * ratio + cursorX;
            let panY = ((slot.panY || 0) - cursorY) * ratio + cursorY;

            if (nextZoom <= 1.0001) {
                panX = 0;
                panY = 0;
            }

            const clamped = this.clampPan(slotIndex, panX, panY, nextZoom);
            slot.zoom = nextZoom;
            slot.panX = clamped.panX;
            slot.panY = clamped.panY;
            this.applySlotTransform(slotIndex);
        }

        handleMouseDown(event) {
            const viewport = event.target.closest('.nic-slot__viewport');
            if (!viewport || !this.contains(viewport) || event.button !== 0 || !event.ctrlKey) {
                return;
            }
            const slotIndex = Number(viewport.dataset.slotIndex || -1);
            const slot = this.state.slots[slotIndex];
            if (!slot || (slot.zoom || 1) <= 1.0001) {
                return;
            }
            event.preventDefault();
            this.panState = {
                slotIndex,
                startX: event.clientX,
                startY: event.clientY,
                originPanX: slot.panX || 0,
                originPanY: slot.panY || 0,
            };
            viewport.classList.add('is-panning');
        }

        handleMouseMove(event) {
            if (!this.panState) {
                return;
            }
            const { slotIndex, startX, startY, originPanX, originPanY } = this.panState;
            const slot = this.state.slots[slotIndex];
            if (!slot) {
                return;
            }
            const deltaX = event.clientX - startX;
            const deltaY = event.clientY - startY;
            const clamped = this.clampPan(slotIndex, originPanX + deltaX, originPanY + deltaY);
            slot.panX = clamped.panX;
            slot.panY = clamped.panY;
            this.applySlotTransform(slotIndex);
        }

        handleMouseUp() {
            if (!this.panState) {
                return;
            }
            const viewport = this.getViewport(this.panState.slotIndex);
            viewport?.classList.remove('is-panning');
            this.panState = null;
        }

        resetView(slotIndex) {
            const slot = this.state.slots[slotIndex];
            if (!slot) {
                return;
            }
            slot.zoom = 1;
            slot.panX = 0;
            slot.panY = 0;
            this.applySlotTransform(slotIndex);
            this.state.openMenuSlot = -1;
            this.render();
        }

        async loadIntoNextEmptySlot(source) {
            const emptyIndex = this.state.slots.findIndex((slot) => !slot);
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

            const [metrics, metadata] = await Promise.all([
                loadImageMetrics(absoluteUrl),
                fetchImageMetadata(relativeUrl || absoluteUrl),
            ]);

            const generationLabel = buildGenerationLabel(source);
            const dimensions = metrics.width && metrics.height
                ? `${metrics.width}x${metrics.height}`
                : 'Unknown';
            const prompt = metadata.prompt || source.prompt || 'Unknown';
            const seed = metadata.seed || source.seed || 'Unknown';
            const stagingName = source.stagingName || detectStagingName(relativeUrl || absoluteUrl);
            const filename = getFilenameFromUrl(relativeUrl || absoluteUrl);

            this.state.slots[slotIndex] = {
                sourceKind: source.sourceKind || (stagingName ? 'staging' : 'manual'),
                sourceLabel: stagingName ? `Staging: ${stagingName}` : (source.sourceKind === 'gallery' ? 'Gallery' : 'Manual'),
                absoluteUrl,
                relativeUrl,
                stagingName,
                filename,
                promptFull: prompt || 'Unknown',
                promptPreview: truncateText(prompt),
                seed: String(seed || 'Unknown'),
                dimensions,
                generationLabel,
                zoom: 1,
                panX: 0,
                panY: 0,
            };
            this.state.openMenuSlot = -1;
            this.render();
            this.announceCompareState();
        }

        async ensureSlotStaged(slotIndex) {
            const slot = this.state.slots[slotIndex];
            if (!slot) {
                throw new Error('No image loaded in that compare slot.');
            }
            if (slot.stagingName) {
                return slot;
            }

            const sourceUrl = slot.relativeUrl && slot.relativeUrl.startsWith('/file=')
                ? slot.relativeUrl
                : slot.absoluteUrl;
            const formData = new FormData();
            formData.append('url', sourceUrl);
            const payload = await fetchJson('/staging_api/upload', {
                method: 'POST',
                body: formData,
            });

            slot.stagingName = payload.file || '';
            this.state.openMenuSlot = -1;
            this.render();
            this.announceCompareState();
            return slot;
        }

        openStagingPanel() {
            window.dispatchEvent(new CustomEvent('nex-staging:open-request'));
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

        async sendToStaging(slotIndex) {
            await this.ensureSlotStaged(slotIndex);
            this.openStagingPanel();
            await this.revealInStaging(slotIndex);
        }

        async revealInStaging(slotIndex) {
            const slot = await this.ensureSlotStaged(slotIndex);
            this.openStagingPanel();
            window.dispatchEvent(new CustomEvent('nex-staging:reveal-request', {
                detail: { name: slot.stagingName },
            }));
            this.state.openMenuSlot = -1;
            this.render();
        }

        async queueForGimp(slotIndex) {
            const slot = await this.ensureSlotStaged(slotIndex);
            await fetchJson(`/staging_api/gimp_target?name=${encodeURIComponent(slot.stagingName)}`, {
                method: 'POST',
            });
            this.state.openMenuSlot = -1;
            this.render();
            this.openStagingPanel();
        }

        getTargetElement(targetId) {
            if (!targetId) {
                return null;
            }
            const selector = `#${escapeSelector(targetId)}`;
            return appRoot().querySelector(selector) || document.querySelector(selector);
        }

        async fileFromSlot(slotIndex) {
            const slot = this.state.slots[slotIndex];
            if (!slot) {
                throw new Error('No image loaded in that compare slot.');
            }
            const response = await fetch(slot.absoluteUrl);
            if (!response.ok) {
                throw new Error(`Image fetch failed with status ${response.status}`);
            }
            const blob = await response.blob();
            return new File([blob], slot.filename || 'compare_image.png', { type: blob.type || 'image/png' });
        }

        async sendToTargetSlot(slotIndex, targetId) {
            const target = this.getTargetElement(targetId);
            if (!target || typeof target.handleFile !== 'function') {
                throw new Error(`Target slot ${targetId} is not available.`);
            }
            const file = await this.fileFromSlot(slotIndex);
            await target.handleFile(file);
            this.state.openMenuSlot = -1;
            this.render();
        }
    }

    if (!customElements.get('nex-image-compare')) {
        customElements.define('nex-image-compare', NexImageCompare);
    }
})();
