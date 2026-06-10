(function () {
    let pollHandle = null;
    let fetchInFlight = false;
    let queuedFetch = false;
    let lastStatusSignature = '';
    let lastRunningSignature = '';
    let lastPendingSignature = '';
    let lastCompletedSignature = '';
    let lastPreviewSignature = '';
    let lastRuntimeState = null;
    let previewResizeHandle = null;

    function gradioRoot() {
        return window.gradioApp ? window.gradioApp() : document;
    }

    function escapeHtml(value) {
        return String(value ?? '')
            .replace(/&/g, '&amp;')
            .replace(/</g, '&lt;')
            .replace(/>/g, '&gt;')
            .replace(/"/g, '&quot;')
            .replace(/'/g, '&#39;');
    }

    function statusRoot() {
        return gradioRoot().querySelector('#nex-runtime-status-panel');
    }

    function queueRoot() {
        return gradioRoot().querySelector('#nex-runtime-queue-panel');
    }

    function previewRoot() {
        return gradioRoot().querySelector('#nex-runtime-preview-panel');
    }

    function buildPreviewImageUrl(baseUrl, viewport) {
        const resolvedBaseUrl = String(baseUrl || '');
        if (!resolvedBaseUrl) {
            return '';
        }

        let viewportWidth = 0;
        let viewportHeight = 0;
        if (viewport) {
            viewportWidth = Math.max(0, Math.floor(viewport.clientWidth || 0));
            viewportHeight = Math.max(0, Math.floor(viewport.clientHeight || 0));
        }

        try {
            const url = new URL(resolvedBaseUrl, window.location.origin);
            if (viewportWidth > 0) {
                url.searchParams.set('max_width', String(viewportWidth));
            }
            if (viewportHeight > 0) {
                url.searchParams.set('max_height', String(viewportHeight));
            }
            return `${url.pathname}${url.search}`;
        } catch (error) {
            return resolvedBaseUrl;
        }
    }

    function buildTaskMeta(task) {
        const parts = [];
        if (task && task.workflow_name) {
            parts.push(String(task.workflow_name));
        }
        if (task && task.model_name) {
            parts.push(String(task.model_name));
        }
        if (task && task.seed !== undefined && task.seed !== null && String(task.seed) !== '') {
            parts.push(`Seed: ${String(task.seed)}`);
        }
        return parts.join(' | ');
    }

    function renderTaskWorkflow(task) {
        if (!task || !task.workflow_name) {
            return '';
        }
        return `<p class="task-meta"><strong>Workflow:</strong> ${escapeHtml(task.workflow_name)}</p>`;
    }

    function renderTaskPrompt(task) {
        if (!task || !task.show_prompt || !task.prompt_preview) {
            return '';
        }
        const label = task.prompt_label || 'Prompt';
        return `<p class="task-prompt"><strong>${escapeHtml(label)}:</strong> "${escapeHtml(task.prompt_preview)}"</p>`;
    }

    function renderTaskModel(task) {
        if (!task) {
            return '';
        }
        const label = task.model_label || 'Model';
        const modelName = task.model_name || '';
        const seedText = task.seed !== undefined && task.seed !== null && String(task.seed) !== ''
            ? ` | Seed: ${escapeHtml(task.seed)}`
            : '';
        return `<p class="task-meta"><strong>${escapeHtml(label)}:</strong> ${escapeHtml(modelName)}${seedText}</p>`;
    }

    function renderStatus(state) {
        const root = statusRoot();
        if (!root) {
            return;
        }

        if (root.dataset.ready !== '1') {
            root.dataset.ready = '1';
            root.innerHTML = `
                <div class="nex-runtime-status-card">
                    <div class="nex-runtime-status-card__header">
                        <span class="nex-runtime-status-card__title"></span>
                        <span class="nex-runtime-status-card__meta"></span>
                    </div>
                    <div class="nex-runtime-progress-track" aria-hidden="true">
                        <div class="nex-runtime-progress-fill"></div>
                    </div>
                    <div class="nex-runtime-status-card__body"></div>
                </div>
            `;
        }

        const progress = state.progress || {};
        const running = state.running || null;
        const visible = !!progress.visible;
        const number = Math.max(0, Math.min(Number(progress.number || 0), 100));
        const statusMeta = running ? buildTaskMeta(running) : 'No task running.';

        const signature = JSON.stringify({
            visible,
            number,
            text: String(progress.text || (running ? running.status_text || 'Working ...' : 'Idle.')),
            statusMeta,
            runningId: running ? running.task_id : null,
        });
        if (signature === lastStatusSignature) {
            return;
        }
        lastStatusSignature = signature;

        const card = root.querySelector('.nex-runtime-status-card');
        if (!card) {
            return;
        }
        card.classList.toggle('is-active', visible);
        card.classList.toggle('is-idle', !visible);
        card.querySelector('.nex-runtime-status-card__title').textContent = visible ? 'Generation Status' : 'Runtime Status';
        card.querySelector('.nex-runtime-status-card__meta').textContent = running
            ? statusMeta
            : 'No task running.';
        card.querySelector('.nex-runtime-status-card__body').textContent = progress.text || (running ? running.status_text || 'Working ...' : 'Idle.');
        card.querySelector('.nex-runtime-progress-fill').style.width = `${visible ? number : 0}%`;
    }

    function renderPreview(state) {
        const root = previewRoot();
        if (!root) {
            return;
        }

        if (root.dataset.ready !== '1') {
            root.dataset.ready = '1';
            root.innerHTML = `
                <div class="nex-runtime-preview-frame">
                    <div class="nex-runtime-preview-frame__header">
                        <span class="nex-runtime-preview-frame__title">Live Preview</span>
                    </div>
                    <div class="nex-runtime-preview-frame__viewport">
                        <img class="nex-runtime-preview-image" alt="Live Preview" hidden />
                        <div class="nex-runtime-preview-empty">Preview will appear here.</div>
                    </div>
                </div>
            `;
        }

        const preview = state.preview || {};
        const available = !!preview.available && !!preview.image_url;
        const revision = Math.max(0, Number(preview.revision || 0));
        const viewport = root.querySelector('.nex-runtime-preview-frame__viewport');
        const imageUrl = available ? buildPreviewImageUrl(preview.image_url, viewport) : '';
        const signature = JSON.stringify({ available, revision, imageUrl });
        if (signature === lastPreviewSignature) {
            return;
        }
        lastPreviewSignature = signature;

        const image = root.querySelector('.nex-runtime-preview-image');
        const empty = root.querySelector('.nex-runtime-preview-empty');
        if (!image || !empty) {
            return;
        }

        if (!available) {
            if (revision === 0) {
                image.hidden = true;
                image.removeAttribute('src');
                image.removeAttribute('data-revision');
                empty.hidden = false;
                empty.textContent = 'Preview will appear here.';
            }
            return;
        }

        image.hidden = false;
        image.setAttribute('data-revision', String(revision));
        image.src = imageUrl;
        empty.hidden = true;
    }

    function renderPendingTasks(pending) {
        if (!pending.length) {
            return '<p class="empty-queue-msg">No queued tasks.</p>';
        }

        return `<div class="nex-queue-list">${pending.map((task, index) => `
            <div class="nex-queue-item pending-task">
                <div class="nex-queue-item-header pending-header">
                    <div class="nex-queue-item-summary">
                        <span class="badge pending-badge">Queued #${index + 1}</span>
                        <span class="task-id">ID: ${escapeHtml(task.task_id)}</span>
                    </div>
                    <button class="queue-btn btn-cancel pending-inline-action" data-runtime-action="cancel" data-task-id="${escapeHtml(task.task_id)}">Cancel</button>
                </div>
                <div class="task-details">
                    ${renderTaskWorkflow(task)}
                    ${renderTaskPrompt(task)}
                    ${renderTaskModel(task)}
                </div>
            </div>
        `).join('')}</div>`;
    }

    function renderCompletedTasks(completed) {
        if (!completed.length) {
            return '<p class="empty-queue-msg">No completed tasks.</p>';
        }

        return `<div class="nex-queue-list">${completed.map((task) => {
            const thumbs = (task.image_urls || []).map((url) => `
                <div class="task-thumbnail-wrapper">
                    <a href="${escapeHtml(url)}" target="_blank" title="Click to view full image">
                        <img src="${escapeHtml(url)}" style="width: 100%; height: 100%; object-fit: cover;" />
                    </a>
                </div>
            `).join('');
            const encodedImages = escapeHtml(JSON.stringify(task.images || []));
            return `
                <div class="nex-queue-item completed-task">
                    <div class="nex-queue-item-header completed-header">
                        <div class="nex-queue-item-summary">
                            <span class="badge completed-badge">Completed</span>
                            <span class="task-id">ID: ${escapeHtml(task.task_id)}</span>
                        </div>
                        <button class="queue-btn btn-delete nex-runtime-inline-btn" data-runtime-action="delete_completed" data-task-id="${escapeHtml(task.task_id)}">Delete</button>
                    </div>
                    <div class="task-details">
                        ${renderTaskWorkflow(task)}
                        ${renderTaskPrompt(task)}
                        ${renderTaskModel(task)}
                        <div class="task-thumbnails">${thumbs}</div>
                        <div class="task-actions">
                            <button class="queue-btn btn-stage" data-stage-images='${encodedImages}'>Stage</button>
                        </div>
                    </div>
                </div>
            `;
        }).join('')}</div>`;
    }

    function renderQueue(state) {
        const root = queueRoot();
        if (!root) {
            return;
        }

        if (root.dataset.ready !== '1') {
            root.dataset.ready = '1';
            root.innerHTML = `
                <div class="nex-runtime-queue-toolbar">
                    <button class="queue-btn btn-cancel nex-runtime-inline-btn" data-runtime-action="refresh">Reconnect</button>
                    <button class="queue-btn btn-stop nex-runtime-inline-btn" data-runtime-action="clear_all">Clear All Tasks</button>
                </div>
                <div class="nex-queue-section-title">Running Task</div>
                <div id="nex-runtime-running-section"></div>
                <div class="nex-queue-section-title">Queued Tasks</div>
                <div id="nex-runtime-pending-section"></div>
                <div class="nex-queue-section-title">Completed Tasks</div>
                <div id="nex-runtime-completed-section"></div>
            `;
        }

        const running = state.running || null;
        const pending = Array.isArray(state.pending) ? state.pending : [];
        const completed = Array.isArray(state.completed) ? state.completed : [];
        const runningSignature = JSON.stringify(running);
        const pendingSignature = JSON.stringify(pending);
        const completedSignature = JSON.stringify(completed);

        if (runningSignature !== lastRunningSignature) {
            lastRunningSignature = runningSignature;
            const runningMarkup = running ? `
                <div class="nex-running-card">
                    <div class="nex-running-card__header">
                        <span class="badge active-badge">Running</span>
                        <span class="task-id">ID: ${escapeHtml(running.task_id)}</span>
                    </div>
                    ${renderTaskWorkflow(running)}
                    ${renderTaskPrompt(running)}
                    ${renderTaskModel(running)}
                    <p class="nex-running-status">${escapeHtml(running.status_text || 'Working ...')}</p>
                    <button class="queue-btn btn-skip nex-runtime-inline-btn" data-runtime-action="skip">Skip</button>
                </div>
            ` : '<p class="empty-queue-msg">No task running.</p>';
            const runningSection = root.querySelector('#nex-runtime-running-section');
            if (runningSection) {
                runningSection.innerHTML = runningMarkup;
            }
        }

        if (pendingSignature !== lastPendingSignature) {
            lastPendingSignature = pendingSignature;
            const pendingSection = root.querySelector('#nex-runtime-pending-section');
            if (pendingSection) {
                pendingSection.innerHTML = renderPendingTasks(pending);
            }
        }

        if (completedSignature !== lastCompletedSignature) {
            lastCompletedSignature = completedSignature;
            const completedSection = root.querySelector('#nex-runtime-completed-section');
            if (completedSection) {
                completedSection.innerHTML = renderCompletedTasks(completed);
            }
        }
    }

    async function fetchState() {
        if (fetchInFlight) {
            queuedFetch = true;
            return;
        }

        fetchInFlight = true;
        try {
            const response = await fetch('/runtime_surface_api/state');
            const payload = await response.json();
            if (payload.status !== 'success') {
                return;
            }
            lastRuntimeState = payload.state || {};
            renderPreview(lastRuntimeState);
            renderStatus(lastRuntimeState);
            renderQueue(lastRuntimeState);
        } catch (error) {
            console.error('[nex-runtime-surface] Failed to fetch runtime state:', error);
        } finally {
            fetchInFlight = false;
            if (queuedFetch) {
                queuedFetch = false;
                window.setTimeout(fetchState, 0);
            }
        }
    }

    async function postAction(action, taskId = '') {
        if (action === 'refresh') {
            fetchState();
            return;
        }

        try {
            const response = await fetch('/runtime_surface_api/action', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ action, task_id: taskId }),
            });
            const payload = await response.json();
            if (payload.status === 'success') {
                lastRuntimeState = payload.state || {};
                renderPreview(lastRuntimeState);
                renderStatus(lastRuntimeState);
                renderQueue(lastRuntimeState);
            }
        } catch (error) {
            console.error('[nex-runtime-surface] Failed to post runtime action:', error);
        }
    }

    function handlePreviewResize() {
        if (previewResizeHandle) {
            window.clearTimeout(previewResizeHandle);
        }
        previewResizeHandle = window.setTimeout(() => {
            if (!lastRuntimeState) {
                return;
            }
            lastPreviewSignature = '';
            renderPreview(lastRuntimeState);
        }, 120);
    }

    function bindDelegatedEvents() {
        document.addEventListener('click', (event) => {
            const actionButton = event.target.closest('[data-runtime-action]');
            if (actionButton) {
                event.preventDefault();
                const action = actionButton.getAttribute('data-runtime-action') || '';
                const taskId = actionButton.getAttribute('data-task-id') || '';
                postAction(action, taskId);
                return;
            }

            const stageButton = event.target.closest('[data-stage-images]');
            if (stageButton) {
                event.preventDefault();
                try {
                    const images = JSON.parse(stageButton.getAttribute('data-stage-images') || '[]');
                    if (typeof stageAllImages === 'function') {
                        stageAllImages(images, stageButton);
                    }
                } catch (error) {
                    console.error('[nex-runtime-surface] Failed to parse staged images payload:', error);
                }
            }
        });
    }

    function startPolling() {
        if (pollHandle) {
            return;
        }
        fetchState();
        pollHandle = window.setInterval(fetchState, 500);
    }

    function init() {
        if (!statusRoot() || !queueRoot() || !previewRoot()) {
            window.setTimeout(init, 250);
            return;
        }
        window.addEventListener('resize', handlePreviewResize);
        bindDelegatedEvents();
        startPolling();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
