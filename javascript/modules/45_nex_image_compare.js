(() => {
    const MIN_ZOOM = 1;
    const MAX_ZOOM = 8;
    const ZOOM_STEP = 0.16;
    const MIN_COMPACT_WIDTH = 360;
    const MIN_COMPACT_HEIGHT = 220;
    const VIEWPORT_MARGIN = 12;

    const state = {
        panel: null,
        content: null,
        items: [],
        mode: "full",
        camera: {
            zoom: 1,
            ratioX: 0,
            ratioY: 0,
        },
        panState: null,
        dragState: null,
        resizeState: null,
    };

    function toAbsoluteUrl(value) {
        if (!value) {
            return "";
        }
        try {
            return new URL(String(value), window.location.origin).toString();
        } catch (error) {
            return String(value);
        }
    }

    function toRelativeUrl(value) {
        if (!value) {
            return "";
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

    function getFilenameFromUrl(value) {
        const raw = String(value || "").trim();
        if (!raw) {
            return "compare_image.png";
        }
        const relative = toRelativeUrl(raw);
        if (relative.startsWith("/file=")) {
            const decoded = decodeURIComponent(relative.slice("/file=".length));
            const parts = decoded.split(/[\\/]+/).filter(Boolean);
            return parts.length ? parts[parts.length - 1] : "compare_image.png";
        }
        const parts = relative.split("/").filter(Boolean);
        return decodeURIComponent(parts.length ? parts[parts.length - 1].split("?")[0] : "compare_image.png");
    }

    function clampCompactRect(left, top, width, height) {
        const viewportWidth = window.innerWidth || document.documentElement.clientWidth || 0;
        const viewportHeight = window.innerHeight || document.documentElement.clientHeight || 0;
        const maxLeft = Math.max(VIEWPORT_MARGIN, viewportWidth - width - VIEWPORT_MARGIN);
        const maxTop = Math.max(VIEWPORT_MARGIN, viewportHeight - height - VIEWPORT_MARGIN);
        return {
            left: clamp(left, VIEWPORT_MARGIN, maxLeft),
            top: clamp(top, VIEWPORT_MARGIN, maxTop),
            width: clamp(width, MIN_COMPACT_WIDTH, Math.max(MIN_COMPACT_WIDTH, viewportWidth - (VIEWPORT_MARGIN * 2))),
            height: clamp(height, MIN_COMPACT_HEIGHT, Math.max(MIN_COMPACT_HEIGHT, viewportHeight - (VIEWPORT_MARGIN * 2))),
        };
    }

    function setCompactRect(left, top, width, height) {
        if (!state.panel) {
            return;
        }
        const rect = clampCompactRect(left, top, width, height);
        state.panel.style.left = `${rect.left}px`;
        state.panel.style.top = `${rect.top}px`;
        state.panel.style.width = `${rect.width}px`;
        state.panel.style.height = `${rect.height}px`;
        state.panel.style.right = "auto";
        state.panel.style.bottom = "auto";
    }

    function ensurePanel() {
        if (state.panel) {
            return;
        }

        const panel = document.createElement("div");
        panel.id = "nex-compare-overlay";
        panel.className = "floating-panel nex-compare-overlay is-hidden";
        panel.innerHTML = `
            <div class="panel-header nex-compare-overlay__header">
                <span class="panel-title">Compare Viewer</span>
                <div class="panel-controls">
                    <button id="nex-compare-minimize" class="panel-window-btn" title="Switch to windowed view">[]</button>
                    <button id="nex-compare-close" class="panel-window-btn" title="Close">x</button>
                </div>
            </div>
            <div class="panel-content nex-compare-overlay__content">
                <div class="nco-empty">Select up to 4 staged images, then press Compare.</div>
            </div>
            <div class="resize-handle nex-compare-overlay__resize"></div>
        `;

        document.body.appendChild(panel);
        state.panel = panel;
        state.content = panel.querySelector(".nex-compare-overlay__content");

        panel.querySelector("#nex-compare-close").addEventListener("click", closeCompare);
        panel.querySelector("#nex-compare-minimize").addEventListener("click", toggleCompactMode);
        panel.querySelector(".nex-compare-overlay__header").addEventListener("mousedown", startDragging);
        panel.querySelector(".nex-compare-overlay__resize").addEventListener("mousedown", startResizing);
        panel.addEventListener("wheel", handleWheel, { passive: false });
        panel.addEventListener("mousedown", handleMouseDown);
        document.addEventListener("mousemove", handleMouseMove);
        document.addEventListener("mouseup", handleMouseUp);
        window.addEventListener("resize", handleWindowResize);

        applyWindowMode();
    }

    function openCompare(items) {
        ensurePanel();
        state.items = Array.isArray(items)
            ? items.slice(0, 4).map((item, index) => ({
                absoluteUrl: toAbsoluteUrl(item.absoluteUrl || item.url || item.relativeUrl || ""),
                relativeUrl: item.relativeUrl || toRelativeUrl(item.absoluteUrl || item.url || ""),
                stagingName: item.stagingName || "",
                filename: item.filename || getFilenameFromUrl(item.relativeUrl || item.absoluteUrl || item.url || ""),
                slotLabel: `C${index + 1}`,
            })).filter((item) => item.absoluteUrl)
            : [];

        resetCamera();
        state.panel.classList.remove("is-hidden");
        applyWindowMode();
        render();
        announceCompareState();
    }

    function closeCompare() {
        ensurePanel();
        state.items = [];
        resetCamera();
        state.panel.classList.add("is-hidden");
        state.panel.classList.remove("dragging", "resizing");
        state.dragState = null;
        state.resizeState = null;
        render();
        announceCompareState();
        window.dispatchEvent(new CustomEvent("nex-compare:closed"));
    }

    function toggleCompactMode() {
        if (!state.panel || state.panel.classList.contains("is-hidden")) {
            return;
        }
        state.mode = state.mode === "full" ? "compact" : "full";
        applyWindowMode();
        resetCamera();
        render();
    }

    function applyWindowMode() {
        if (!state.panel) {
            return;
        }

        const toggleButton = state.panel.querySelector("#nex-compare-minimize");
        if (state.mode === "compact") {
            state.panel.classList.add("is-compact");
            if (!state.panel.dataset.compactPositioned) {
                setCompactRect(24, 88, 640, 360);
                state.panel.dataset.compactPositioned = "true";
            } else if (!state.panel.style.width || !state.panel.style.height) {
                setCompactRect(24, 88, 640, 360);
            } else {
                const current = state.panel.getBoundingClientRect();
                setCompactRect(current.left, current.top, current.width, current.height);
            }
            if (toggleButton) {
                toggleButton.textContent = "[]";
                toggleButton.title = "Expand to full window";
            }
            return;
        }

        state.panel.classList.remove("is-compact");
        state.panel.style.left = "";
        state.panel.style.top = "";
        state.panel.style.width = "";
        state.panel.style.height = "";
        state.panel.style.right = "";
        state.panel.style.bottom = "";
        if (toggleButton) {
            toggleButton.textContent = "[]";
            toggleButton.title = "Switch to windowed view";
        }
    }

    function render() {
        if (!state.content) {
            return;
        }

        if (!state.items.length) {
            state.content.innerHTML = `<div class="nco-empty">Select up to 4 staged images, then press Compare.</div>`;
            return;
        }

        state.content.innerHTML = `
            <div class="nco-grid" style="--nco-cols:${state.items.length};">
                ${state.items.map((item, index) => `
                    <article class="nco-cell" data-slot-index="${index}">
                        <span class="nco-label">${item.slotLabel}</span>
                        <div class="nco-viewport" data-slot-index="${index}">
                            <img
                                class="nco-image"
                                data-slot-index="${index}"
                                src="${item.absoluteUrl}"
                                alt="${item.filename}"
                                draggable="false"
                            >
                        </div>
                    </article>
                `).join("")}
            </div>
        `;

        syncAllTransforms();
    }

    function getViewport(index) {
        return state.panel?.querySelector(`.nco-viewport[data-slot-index="${index}"]`) || null;
    }

    function getImage(index) {
        return state.panel?.querySelector(`.nco-image[data-slot-index="${index}"]`) || null;
    }

    function getCell(index) {
        return state.panel?.querySelector(`.nco-cell[data-slot-index="${index}"]`) || null;
    }

    function computeMaxPan(index, zoomOverride = null) {
        const viewport = getViewport(index);
        const image = getImage(index);
        if (!viewport || !image) {
            return { x: 0, y: 0 };
        }

        const zoom = zoomOverride ?? state.camera.zoom;
        const baseWidth = image.clientWidth || image.naturalWidth || 0;
        const baseHeight = image.clientHeight || image.naturalHeight || 0;
        const viewportWidth = viewport.clientWidth || 0;
        const viewportHeight = viewport.clientHeight || 0;

        return {
            x: Math.max(0, ((baseWidth * zoom) - viewportWidth) / 2),
            y: Math.max(0, ((baseHeight * zoom) - viewportHeight) / 2),
        };
    }

    function applyTransform(index) {
        const image = getImage(index);
        const cell = getCell(index);
        if (!image || !cell) {
            return;
        }

        const maxPan = computeMaxPan(index);
        const panX = maxPan.x * state.camera.ratioX;
        const panY = maxPan.y * state.camera.ratioY;
        image.style.transform = `translate(${panX}px, ${panY}px) scale(${state.camera.zoom})`;
        cell.classList.toggle("is-zoomed", state.camera.zoom > 1.01);
    }

    function syncAllTransforms() {
        state.items.forEach((item, index) => {
            const image = getImage(index);
            if (!image) {
                return;
            }
            if (!image.dataset.transformBound) {
                image.dataset.transformBound = "true";
                image.addEventListener("load", () => applyTransform(index));
            }
            applyTransform(index);
        });
    }

    function handleWheel(event) {
        if (!state.items.length || !event.ctrlKey) {
            return;
        }

        const viewport = event.target.closest(".nco-viewport");
        if (!viewport || !state.panel?.contains(viewport)) {
            return;
        }

        event.preventDefault();

        const slotIndex = Number(viewport.dataset.slotIndex || -1);
        if (slotIndex < 0) {
            return;
        }

        const oldZoom = state.camera.zoom;
        const newZoom = clamp(
            oldZoom + (event.deltaY < 0 ? ZOOM_STEP : -ZOOM_STEP),
            MIN_ZOOM,
            MAX_ZOOM,
        );

        if (Math.abs(newZoom - oldZoom) < 0.0001) {
            return;
        }

        const oldMax = computeMaxPan(slotIndex, oldZoom);
        const newMax = computeMaxPan(slotIndex, newZoom);
        const rect = viewport.getBoundingClientRect();
        const dx = event.clientX - (rect.left + rect.width / 2);
        const dy = event.clientY - (rect.top + rect.height / 2);
        const oldPanX = oldMax.x * state.camera.ratioX;
        const oldPanY = oldMax.y * state.camera.ratioY;

        state.camera.zoom = newZoom;
        if (newZoom <= 1.0001) {
            state.camera.ratioX = 0;
            state.camera.ratioY = 0;
        } else {
            const newPanX = clamp(dx - ((dx - oldPanX) * newZoom / oldZoom), -newMax.x, newMax.x);
            const newPanY = clamp(dy - ((dy - oldPanY) * newZoom / oldZoom), -newMax.y, newMax.y);
            state.camera.ratioX = newMax.x > 0 ? clamp(newPanX / newMax.x, -1, 1) : 0;
            state.camera.ratioY = newMax.y > 0 ? clamp(newPanY / newMax.y, -1, 1) : 0;
        }

        syncAllTransforms();
    }

    function handleMouseDown(event) {
        if (!state.items.length || event.button !== 0 || !event.ctrlKey) {
            return;
        }

        const viewport = event.target.closest(".nco-viewport");
        if (!viewport || !state.panel?.contains(viewport) || state.camera.zoom <= 1.0001) {
            return;
        }

        const slotIndex = Number(viewport.dataset.slotIndex || -1);
        if (slotIndex < 0) {
            return;
        }

        const maxPan = computeMaxPan(slotIndex);
        state.panState = {
            slotIndex,
            startX: event.clientX,
            startY: event.clientY,
            originPanX: maxPan.x * state.camera.ratioX,
            originPanY: maxPan.y * state.camera.ratioY,
        };
        viewport.classList.add("is-panning");
        event.preventDefault();
    }

    function startDragging(event) {
        if (!state.panel || state.mode !== "compact") {
            return;
        }
        if (event.target.closest(".panel-controls")) {
            return;
        }
        if (event.button !== 0) {
            return;
        }

        const rect = state.panel.getBoundingClientRect();
        state.dragState = {
            offsetX: event.clientX - rect.left,
            offsetY: event.clientY - rect.top,
            width: rect.width,
            height: rect.height,
        };
        state.panel.classList.add("dragging");
        event.preventDefault();
    }

    function startResizing(event) {
        if (!state.panel || state.mode !== "compact" || event.button !== 0) {
            return;
        }
        const rect = state.panel.getBoundingClientRect();
        state.resizeState = {
            startX: event.clientX,
            startY: event.clientY,
            width: rect.width,
            height: rect.height,
            left: rect.left,
            top: rect.top,
        };
        state.panel.classList.add("resizing");
        event.preventDefault();
        event.stopPropagation();
    }

    function handleMouseMove(event) {
        if (state.dragState && state.panel && state.mode === "compact") {
            const nextLeft = event.clientX - state.dragState.offsetX;
            const nextTop = event.clientY - state.dragState.offsetY;
            setCompactRect(nextLeft, nextTop, state.dragState.width, state.dragState.height);
        }

        if (state.resizeState && state.panel && state.mode === "compact") {
            const nextWidth = state.resizeState.width + (event.clientX - state.resizeState.startX);
            const nextHeight = state.resizeState.height + (event.clientY - state.resizeState.startY);
            setCompactRect(state.resizeState.left, state.resizeState.top, nextWidth, nextHeight);
        }

        if (!state.panState) {
            return;
        }

        const maxPan = computeMaxPan(state.panState.slotIndex);
        const deltaX = event.clientX - state.panState.startX;
        const deltaY = event.clientY - state.panState.startY;
        const nextPanX = clamp(state.panState.originPanX + deltaX, -maxPan.x, maxPan.x);
        const nextPanY = clamp(state.panState.originPanY + deltaY, -maxPan.y, maxPan.y);
        state.camera.ratioX = maxPan.x > 0 ? clamp(nextPanX / maxPan.x, -1, 1) : 0;
        state.camera.ratioY = maxPan.y > 0 ? clamp(nextPanY / maxPan.y, -1, 1) : 0;
        syncAllTransforms();
    }

    function handleMouseUp() {
        if (state.dragState && state.panel) {
            state.panel.classList.remove("dragging");
            state.dragState = null;
        }
        if (state.resizeState && state.panel) {
            state.panel.classList.remove("resizing");
            state.resizeState = null;
        }
        if (!state.panState) {
            return;
        }
        getViewport(state.panState.slotIndex)?.classList.remove("is-panning");
        state.panState = null;
    }

    function handleWindowResize() {
        if (!state.panel || state.mode !== "compact") {
            return;
        }
        const rect = state.panel.getBoundingClientRect();
        setCompactRect(rect.left, rect.top, rect.width, rect.height);
        syncAllTransforms();
    }

    function resetCamera() {
        state.camera.zoom = 1;
        state.camera.ratioX = 0;
        state.camera.ratioY = 0;
        syncAllTransforms();
    }

    function announceCompareState() {
        const stagingMap = {};
        state.items.forEach((item, index) => {
            if (!item.stagingName) {
                return;
            }
            stagingMap[item.stagingName] = `C${index + 1}`;
        });
        window.dispatchEvent(new CustomEvent("nex-compare:state-change", {
            detail: { stagingMap },
        }));
    }

    window.addEventListener("nex-compare:open", (event) => {
        const items = event?.detail?.items || [];
        if (!items.length) {
            closeCompare();
            return;
        }
        openCompare(items);
    });

    window.addEventListener("nex-compare:close-request", closeCompare);
})();
