(function () {
    function getRoot() {
        return window.gradioApp ? window.gradioApp() : document;
    }

    function createThumbStrip(root, workspace) {
        let thumbStrip = root.querySelector('#nex-preview-thumbnails');
        if (!thumbStrip) {
            thumbStrip = document.createElement('div');
            thumbStrip.id = 'nex-preview-thumbnails';
            workspace.appendChild(thumbStrip);
        }
        return thumbStrip;
    }

    function ensureCloseButton(root, workspace, galleryCol) {
        let closeBtn = root.querySelector('#nex-close-gallery-btn');
        if (!closeBtn) {
            closeBtn = document.createElement('button');
            closeBtn.id = 'nex-close-gallery-btn';
            closeBtn.type = 'button';
            closeBtn.innerHTML = '&times;';
            closeBtn.title = 'Close Gallery View';
            closeBtn.setAttribute('aria-label', 'Close Gallery View');
            closeBtn.addEventListener('click', (event) => {
                event.preventDefault();
                event.stopPropagation();
                workspace.classList.remove('gallery-open');
            });
            galleryCol.insertBefore(closeBtn, galleryCol.firstChild);
        }
        return closeBtn;
    }

    function updateThumbnails(finalGallery, thumbStrip, workspace) {
        const thumbnailButtons = Array.from(finalGallery.querySelectorAll('button')).filter((button) =>
            button.querySelector('img')
        );
        const images = [];

        thumbnailButtons.forEach((button) => {
            const img = button.querySelector('img');
            const src = img ? img.src : '';
            if (!src || images.some((item) => item.src === src)) {
                return;
            }
            images.push({ src, button });
        });

        thumbStrip.innerHTML = '';
        thumbStrip.style.display = images.length > 0 ? 'flex' : 'none';

        images.forEach((imgData) => {
            const thumb = document.createElement('button');
            thumb.type = 'button';
            thumb.className = 'nex-thumb-item';

            const img = document.createElement('img');
            img.src = imgData.src;
            img.alt = 'Generated image thumbnail';
            thumb.appendChild(img);

            const isSelected = imgData.button.classList.contains('selected')
                || imgData.button.getAttribute('aria-selected') === 'true'
                || imgData.button.classList.contains('border-primary');
            if (isSelected) {
                thumb.classList.add('active');
            }

            thumb.addEventListener('click', (event) => {
                event.preventDefault();
                event.stopPropagation();
                imgData.button.click();
                workspace.classList.add('gallery-open');
                thumbStrip.querySelectorAll('.nex-thumb-item').forEach((item) => item.classList.remove('active'));
                thumb.classList.add('active');
            });

            thumbStrip.appendChild(thumb);
        });
    }

    function initPreviewWorkspace() {
        const root = getRoot();
        const workspace = root.querySelector('#preview_workspace');
        const finalGallery = root.querySelector('#final_gallery');

        if (!workspace || !finalGallery) {
            setTimeout(initPreviewWorkspace, 100);
            return;
        }

        if (workspace.dataset.nexPreviewReady === '1') {
            return;
        }
        workspace.dataset.nexPreviewReady = '1';

        const thumbStrip = createThumbStrip(root, workspace);
        const galleryCol = finalGallery.closest('.gradio-column') || finalGallery.parentElement;
        const previewImg = root.querySelector('.preview_panel');
        const previewCol = previewImg ? (previewImg.closest('.gradio-column') || previewImg.parentElement) : null;

        if (galleryCol) {
            galleryCol.classList.add('nex-gallery-column');
            ensureCloseButton(root, workspace, galleryCol);
        }
        if (previewCol) {
            previewCol.classList.add('nex-preview-column');
        }

        const observer = new MutationObserver(() => {
            updateThumbnails(finalGallery, thumbStrip, workspace);
        });
        observer.observe(finalGallery, { childList: true, subtree: true, attributes: true });

        updateThumbnails(finalGallery, thumbStrip, workspace);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', initPreviewWorkspace);
    } else {
        initPreviewWorkspace();
    }
})();
