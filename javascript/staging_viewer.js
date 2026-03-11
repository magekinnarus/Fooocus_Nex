(function () {
    let panel = null;
    let imagesContainer = null;
    let lastImagesJson = '';
    let isDragging = false;
    let dragOffset = { x: 0, y: 0 };
    let pollInterval = null;
    let selectedImage = null;

    function createPanel() {
        if (panel) return;

        panel = document.createElement('div');
        panel.id = 'floating-staging-panel';
        panel.className = 'floating-panel';
        panel.innerHTML = `
            <div class="panel-header" id="staging-panel-header">
                <span class="panel-title">Staging Area</span>
                <div class="panel-controls">
                    <button id="staging-panel-refresh" title="Refresh Now">🔄</button>
                    <button id="staging-panel-clear" title="Clear All Staging" style="color: #ff4d4d;">🗑️</button>
                    <button id="staging-panel-toggle" title="Minimize">－</button>
                    <button id="staging-panel-close" title="Close Palette">×</button>
                </div>
            </div>
            <div class="panel-content">
                <div id="staging-images-grid" class="staging-grid">
                    <div class="empty-msg">Drop images here to stage</div>
                </div>
            </div>
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
        document.addEventListener('mousemove', drag);
        document.addEventListener('mouseup', stopDragging);

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

        // Load position from localStorage
        const pos = JSON.parse(localStorage.getItem('staging-panel-pos') || '{"top": "60px", "right": "20px"}');
        Object.assign(panel.style, pos);

        fetchImages();
        pollInterval = setInterval(fetchImages, 3000);
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

    async function fetchImages() {
        try {
            const res = await fetch('/staging_api/images');
            const data = await res.json();
            const json = JSON.stringify(data.images);
            if (json === lastImagesJson) return; // No change
            lastImagesJson = json;

            renderImages(data.images);
        } catch (e) {
            console.error('[Staging] Fetch error:', e);
        }
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

            // Standard img with draggable=true for Fooocus slots
            const imgEl = document.createElement('img');
            imgEl.src = img.url;
            imgEl.alt = img.name;
            imgEl.draggable = true;
            imgEl.title = 'Drag to slots';

            // Critical for dragging into Gradio slots: set absolute URL in dataTransfer
            imgEl.addEventListener('dragstart', (e) => {
                const absoluteUrl = window.location.origin + img.url;
                e.dataTransfer.setData('text/plain', absoluteUrl);
                e.dataTransfer.setData('text/uri-list', absoluteUrl);
                e.dataTransfer.setData('fooocus/staging-internal', 'true'); // Flag to prevent self-drop
                console.log('[Staging] Drag start:', absoluteUrl);
            });

            item.appendChild(imgEl);

            // Action buttons container
            const actionsContainer = document.createElement('div');
            actionsContainer.className = 'item-actions';

            // GIMP button
            const gimpBtn = document.createElement('button');
            gimpBtn.className = 'item-action-btn btn-gimp';
            gimpBtn.innerHTML = 'G';
            gimpBtn.title = 'Send to GIMP';
            gimpBtn.onclick = (e) => {
                e.stopPropagation();
                alert('GIMP Bridge: Sending ' + img.name + ' to GIMP via outputs/staging...');
                // Integration code for GIMP will go here
            };
            actionsContainer.appendChild(gimpBtn);

            // Delete button
            const delBtn = document.createElement('button');
            delBtn.className = 'item-action-btn btn-delete';
            delBtn.innerHTML = '×';
            delBtn.title = 'Remove from staging';
            delBtn.onclick = (e) => {
                e.stopPropagation();
                deleteImage(img.name);
            };
            actionsContainer.appendChild(delBtn);

            item.appendChild(actionsContainer);

            imagesContainer.appendChild(item);
        });
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

            // Global listener for the launcher button (survives Gradio DOM swaps)
            document.addEventListener('click', (e) => {
                const launcher = e.target.closest('#staging-panel-launcher');
                if (launcher && panel) {
                    panel.style.display = 'flex';
                    panel.classList.remove('minimized');
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
