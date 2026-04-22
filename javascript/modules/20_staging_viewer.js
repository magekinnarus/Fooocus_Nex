(function () {
    let panel = null;
    let imagesContainer = null;
    let lastImagesJson = '';
    let isDragging = false;
    let dragOffset = { x: 0, y: 0 };
    let pollInterval = null;
    let currentGimpQueue = [];
    let selectedImage = null;
    let currentCompareMap = {};
    let pendingRevealName = '';
    let latestImages = [];
    let activeImageDragCount = 0;
    let renderPendingAfterDrag = false;

    function escapeSelector(value) {
        if (window.CSS && typeof window.CSS.escape === 'function') {
            return window.CSS.escape(value);
        }
        return String(value).replace(/([^a-zA-Z0-9_-])/g, '\\$1');
    }

    function createPanel() {
        if (panel) return;

        panel = document.createElement('div');
        panel.id = 'floating-staging-panel';
        panel.className = 'floating-panel';
        panel.innerHTML = `
            <div class="panel-header" id="staging-panel-header">
                <span class="panel-title">Staging Area</span>
                <div class="panel-controls">
                    <button id="staging-panel-refresh" title="Refresh Now">Refresh</button>
                    <button id="staging-panel-clear" title="Clear All Staging" style="color: #ff4d4d;">Clear</button>
                    <button id="staging-panel-toggle" title="Minimize">-</button>
                    <button id="staging-panel-close" title="Close Palette">x</button>
                </div>
            </div>
            <div class="panel-content">
                <div id="staging-images-grid" class="staging-grid">
                    <div class="empty-msg">Drop images here to stage</div>
                </div>
            </div>
            <div class="resize-handle" id="staging-panel-resize"></div>
        `;

        document.body.appendChild(panel);
        imagesContainer = panel.querySelector('#staging-images-grid');

        // Snap Logic: default position at top-right, aligned with tab titles
        const snapToDefault = () => {
            panel.style.top = '60px'; // Raised from 100px to align with tabs
            panel.style.right = '20px';
            panel.style.left = 'auto'; // Clear drag left property
        };

        // Drag logic
        const header = panel.querySelector('#staging-panel-header');
        header.addEventListener('mousedown', startDragging);
        document.addEventListener('mousemove', (e) => {
            drag(e);
            resize(e);
        });
        document.addEventListener('mouseup', () => {
            stopDragging();
            stopResizing();
        });

        // Resize logic
        const resizeHandle = panel.querySelector('#staging-panel-resize');
        resizeHandle.addEventListener('mousedown', startResizing);

        // Controls
        panel.querySelector('#staging-panel-refresh').addEventListener('click', fetchImages);
        panel.querySelector('#staging-panel-clear').addEventListener('click', clearStaging);

        panel.querySelector('#staging-panel-toggle').addEventListener('click', () => {
            const isMinimized = panel.classList.toggle('minimized');
            if (isMinimized) {
                // Snap on minimize
                snapToDefault();
            }
        });

        panel.querySelector('#staging-panel-close').addEventListener('click', () => {
            panel.style.display = 'none';
        });

        // Drop zone for uploads
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            panel.addEventListener(eventName, (e) => {
                e.preventDefault();
                e.stopPropagation();
            }, false);
        });

        panel.addEventListener('dragover', () => panel.classList.add('drag-over'));
        panel.addEventListener('dragleave', () => panel.classList.remove('drag-over'));
        panel.addEventListener('drop', handleDrop);

        // Load position and size from localStorage
        const pos = JSON.parse(localStorage.getItem('staging-panel-pos') || '{"top": "60px", "right": "20px"}');
        const size = JSON.parse(localStorage.getItem('staging-panel-size') || '{"width": "440px", "height": "auto"}');
        Object.assign(panel.style, pos);
        Object.assign(panel.style, size);

        fetchImages();
        pollInterval = setInterval(fetchImages, 3000);
    }

    function openPanel() {
        if (!panel) {
            return;
        }
        panel.style.display = 'flex';
        panel.classList.remove('minimized');
    }

    function startDragging(e) {
        if (e.target.tagName === 'BUTTON') return;
        isDragging = true;
        const rect = panel.getBoundingClientRect();
        dragOffset = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
        panel.style.right = 'auto';
        panel.style.left = rect.left + 'px';
        panel.style.top = rect.top + 'px';
        panel.classList.add('dragging');
        e.preventDefault();
    }

    function drag(e) {
        if (!isDragging) return;
        panel.style.left = (e.clientX - dragOffset.x) + 'px';
        panel.style.top = (e.clientY - dragOffset.y) + 'px';
    }

    function stopDragging() {
        if (!isDragging) return;
        isDragging = false;
        panel.classList.remove('dragging');
        // Save position
        localStorage.setItem('staging-panel-pos', JSON.stringify({
            top: panel.style.top,
            left: panel.style.left
        }));
    }

    let isResizing = false;
    let resizeStartSize = { w: 0, h: 0 };
    let resizeStartPos = { x: 0, y: 0 };

    function startResizing(e) {
        isResizing = true;
        resizeStartSize = {
            w: panel.offsetWidth,
            h: panel.offsetHeight
        };
        resizeStartPos = {
            x: e.clientX,
            y: e.clientY
        };
        panel.classList.add('resizing');
        e.preventDefault();
        e.stopPropagation();
    }

    function resize(e) {
        if (!isResizing) return;
        const deltaX = e.clientX - resizeStartPos.x;
        const deltaY = e.clientY - resizeStartPos.y;
        panel.style.width = (resizeStartSize.w + deltaX) + 'px';
        panel.style.height = (resizeStartSize.h + deltaY) + 'px';
    }

    function stopResizing() {
        if (!isResizing) return;
        isResizing = false;
        panel.classList.remove('resizing');
        // Save size
        localStorage.setItem('staging-panel-size', JSON.stringify({
            width: panel.style.width,
            height: panel.style.height
        }));
    }

    async function fetchImages() {
        try {
            const res = await fetch('/staging_api/images');
            const data = await res.json();
            currentGimpQueue = Array.isArray(data.gimp_queue) ? data.gimp_queue : [];
            latestImages = Array.isArray(data.images) ? data.images : [];
            const json = JSON.stringify({ images: data.images, gimp_queue: currentGimpQueue });
            if (json === lastImagesJson) return; // No change
            lastImagesJson = json;

            if (activeImageDragCount > 0) {
                renderPendingAfterDrag = true;
                return;
            }
            renderImages(latestImages);
        } catch (e) {
            console.error('[Staging] Fetch error:', e);
        }
    }

    function refreshAfterDragIfNeeded() {
        if (activeImageDragCount > 0 || !renderPendingAfterDrag) {
            return;
        }
        renderPendingAfterDrag = false;
        renderImages(latestImages);
    }

    function updateCompareBadges() {
        if (!panel) {
            return;
        }
        panel.querySelectorAll('.staging-item').forEach((item) => {
            const imageName = item.dataset.imageName || '';
            const compareBadge = currentCompareMap[imageName] || '';
            item.classList.toggle('is-in-compare', !!compareBadge);
            if (compareBadge) {
                item.dataset.compareSlot = compareBadge;
            } else {
                delete item.dataset.compareSlot;
            }

            let badge = item.querySelector('.staging-compare-badge');
            if (compareBadge) {
                if (!badge) {
                    badge = document.createElement('div');
                    badge.className = 'staging-compare-badge';
                    item.appendChild(badge);
                }
                badge.textContent = `${compareBadge} in Compare`;
            } else if (badge) {
                badge.remove();
            }
        });
    }

    function renderImages(images) {
        if (images.length === 0) {
            imagesContainer.innerHTML = '<div class="empty-msg">Drop images here to stage</div>';
            return;
        }

        imagesContainer.innerHTML = '';
        images.forEach(img => {
            const item = document.createElement('div');
            item.className = 'staging-item';
            item.dataset.imageName = img.name;
            if (currentGimpQueue.includes(img.name)) {
                item.classList.add('gimp-targeted');
            }
            const compareBadge = currentCompareMap[img.name];
            if (compareBadge) {
                item.classList.add('is-in-compare');
                item.dataset.compareSlot = compareBadge;
            }

            // Standard img with draggable=true for Fooocus slots
            const imgEl = document.createElement('img');
            imgEl.src = img.url;
            imgEl.alt = img.name;
            imgEl.draggable = true;
            imgEl.title = 'Drag to slots';

            // Critical for dragging into Gradio slots: set absolute URL in dataTransfer
            imgEl.addEventListener('dragstart', (e) => {
                activeImageDragCount += 1;
                const absoluteUrl = window.location.origin + img.url;
                const payload = JSON.stringify({
                    kind: 'nex-image-source',
                    sourceKind: 'staging',
                    absoluteUrl: absoluteUrl,
                    relativeUrl: img.url,
                    stagingName: img.name,
                });
                e.dataTransfer.setData('text/plain', absoluteUrl);
                e.dataTransfer.setData('text/uri-list', absoluteUrl);
                e.dataTransfer.setData('application/json', payload);
                e.dataTransfer.setData('fooocus/staging-internal', 'true'); // Flag to prevent self-drop
                console.log('[Staging] Drag start:', absoluteUrl);
            });
            imgEl.addEventListener('dragend', () => {
                activeImageDragCount = Math.max(0, activeImageDragCount - 1);
                refreshAfterDragIfNeeded();
            });

            item.appendChild(imgEl);

            if (compareBadge) {
                const badge = document.createElement('div');
                badge.className = 'staging-compare-badge';
                badge.textContent = `${compareBadge} in Compare`;
                item.appendChild(badge);
            }

            // Action buttons container
            const actionsContainer = document.createElement('div');
            actionsContainer.className = 'item-actions';

            // GIMP button
            const gimpBtn = document.createElement('button');
            gimpBtn.className = 'item-action-btn btn-gimp';
            gimpBtn.innerHTML = 'G';
            gimpBtn.title = 'Queue for GIMP import';
            gimpBtn.onclick = async (e) => {
                e.stopPropagation();
                try {
                    const res = await fetch(`/staging_api/gimp_target?name=${encodeURIComponent(img.name)}`, {
                        method: 'POST'
                    });
                    const result = await res.json();
                    if (result.status === 'success') {
                        currentGimpQueue = Array.isArray(result.queue) ? result.queue : [];
                        panel.querySelectorAll('.staging-item').forEach(el => {
                            const isQueued = currentGimpQueue.includes(el.dataset.imageName);
                            el.classList.toggle('gimp-targeted', isQueued);
                        });
                        if (result.queued) {
                            console.log('[Staging] GIMP queued:', img.name);
                        } else {
                            console.log('[Staging] GIMP dequeued:', img.name);
                        }
                    }
                } catch (err) {
                    console.error('[Staging] GIMP Target error:', err);
                }
            };
            actionsContainer.appendChild(gimpBtn);

            // Delete button
            const delBtn = document.createElement('button');
            delBtn.className = 'item-action-btn btn-delete';
            delBtn.innerHTML = 'X';
            delBtn.title = 'Remove from staging';
            delBtn.onclick = (e) => {
                e.stopPropagation();
                deleteImage(img.name);
            };
            actionsContainer.appendChild(delBtn);

            item.appendChild(actionsContainer);

            imagesContainer.appendChild(item);
        });

        if (pendingRevealName) {
            flashRevealTarget(pendingRevealName);
        }
    }

    function flashRevealTarget(name) {
        if (!panel || !name) {
            return;
        }
        const target = panel.querySelector(`.staging-item[data-image-name="${escapeSelector(name)}"]`);
        if (!target) {
            return;
        }
        pendingRevealName = '';
        target.classList.add('is-revealed');
        target.scrollIntoView({ behavior: 'smooth', block: 'center' });
        window.setTimeout(() => target.classList.remove('is-revealed'), 1800);
    }

    async function deleteImage(name) {
        if (!confirm('Remove this image from staging?')) return;
        try {
            const res = await fetch(`/staging_api/delete?name=${encodeURIComponent(name)}`, {
                method: 'DELETE'
            });
            const result = await res.json();
            if (result.status === 'success') {
                fetchImages();
            }
        } catch (e) {
            console.error('[Staging] Delete error:', e);
        }
    }

    async function clearStaging() {
        if (!confirm('Clear ALL images from staging?')) return;
        try {
            const res = await fetch('/staging_api/clear', {
                method: 'POST'
            });
            const result = await res.json();
            if (result.status === 'success') {
                fetchImages();
            }
        } catch (e) {
            console.error('[Staging] Clear error:', e);
        }
    }

    async function handleDrop(e) {
        e.preventDefault();
        e.stopPropagation();
        panel.classList.remove('drag-over');

        const dataTransfer = e.dataTransfer;

        // Prevent duplicate upload if dragged from within the staging panel
        if (dataTransfer.types.includes('fooocus/staging-internal')) {
            console.log('[Staging] Ignored internal drag to prevent duplication.');
            return;
        }

        const files = dataTransfer.files;
        const html = dataTransfer.getData('text/html');
        const plain = dataTransfer.getData('text/plain');

        console.log('[Staging] Drop detected:', {
            fileCount: files.length,
            hasHtml: !!html,
            hasPlain: !!plain,
            plainData: plain.substring(0, 100)
        });

        let url = null;
        if (html) {
            const doc = new DOMParser().parseFromString(html, 'text/html');
            const img = doc.querySelector('img');
            if (img && img.src) url = img.src;
        }

        if (!url && plain) {
            if (plain.startsWith('http') || plain.startsWith('file') || plain.startsWith('data:image')) {
                url = plain;
            }
        }

        if (files.length > 0) {
            console.log('[Staging] Processing files...');
            for (let file of files) {
                const formData = new FormData();
                formData.append('file', file);
                await uploadImage(formData);
            }
        } else if (url) {
            console.log('[Staging] Processing URL:', url.substring(0, 50) + '...');
            const formData = new FormData();
            formData.append('url', url);
            await uploadImage(formData);
        } else {
            console.warn('[Staging] No supported data found in drop.');
        }
    }

    async function uploadImage(formData) {
        try {
            const res = await fetch('/staging_api/upload', {
                method: 'POST',
                body: formData
            });
            const result = await res.json();
            if (result.status === 'success') {
                fetchImages();
            }
        } catch (e) {
            console.error('[Staging] Upload error:', e);
        }
    }

    // Initialize when Gradio is ready
    function init() {
        if (window.gradioApp) {
            createPanel();

            window.addEventListener('nex-compare:state-change', (event) => {
                currentCompareMap = (event && event.detail && event.detail.stagingMap) || {};
                if (latestImages.length > 0) {
                    updateCompareBadges();
                } else {
                    fetchImages();
                }
            });

            window.addEventListener('nex-staging:open-request', () => {
                openPanel();
            });

            window.addEventListener('nex-staging:reveal-request', (event) => {
                const name = event && event.detail && event.detail.name;
                if (!name) {
                    return;
                }
                pendingRevealName = name;
                openPanel();
                fetchImages();
                window.setTimeout(() => flashRevealTarget(name), 180);
            });

            // Global listener for the launcher button (survives Gradio DOM swaps)
            document.addEventListener('click', (e) => {
                const launcher = e.target.closest('#staging-panel-launcher');
                if (launcher && panel) {
                    openPanel();
                    panel.scrollIntoView({ behavior: 'smooth', block: 'center' });
                    // Pop effect
                    panel.style.transform = 'scale(1.05)';
                    setTimeout(() => panel.style.transform = 'scale(1)', 200);
                }
            });
        } else {
            setTimeout(init, 500);
        }
    }

    init();
})();


