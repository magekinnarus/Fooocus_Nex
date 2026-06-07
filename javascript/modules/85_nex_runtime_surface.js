(function () {
    let pollHandle = null;
    let fetchInFlight = false;
    let queuedFetch = false;
    let lastStatusSignature = '';
    let lastRunningSignature = '';
    let lastPendingSignature = '';
    let lastCompletedSignature = '';

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
        const title = running
            ? `${escapeHtml(running.prompt_preview || 'Image generation')} | ${escapeHtml(running.model_name || '')}`
            : 'No task running.';

        const signature = JSON.stringify({
            visible,
            number,
            text: String(progress.text || (running ? running.status_text || 'Working ...' : 'Idle.')),
            title,
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
            ? `${running.prompt_preview || 'Image generation'} | ${running.model_name || ''}`
            : 'No task running.';
        card.querySelector('.nex-runtime-status-card__body').textContent = progress.text || (running ? running.status_text || 'Working ...' : 'Idle.');
        card.querySelector('.nex-runtime-progress-fill').style.width = `${visible ? number : 0}%`;
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
                    <p class="task-prompt">"${escapeHtml(task.prompt_preview)}"</p>
                    <p class="task-meta">Model: ${escapeHtml(task.model_name || '')} | Seed: ${escapeHtml(task.seed ?? '')}</p>
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
                    </div>
                    <div class="task-details">
                        <p class="task-prompt">"${escapeHtml(task.prompt_preview)}"</p>
                        <p class="task-meta">Model: ${escapeHtml(task.model_name || '')} | Seed: ${escapeHtml(task.seed ?? '')}</p>
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
                    <p class="task-prompt"><strong>Prompt:</strong> "${escapeHtml(running.prompt_preview || 'Image generation')}"</p>
                    <p class="task-meta">Model: ${escapeHtml(running.model_name || '')} | Seed: ${escapeHtml(running.seed ?? '')}</p>
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
            renderStatus(payload.state || {});
            renderQueue(payload.state || {});
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
                renderStatus(payload.state || {});
                renderQueue(payload.state || {});
            }
        } catch (error) {
            console.error('[nex-runtime-surface] Failed to post runtime action:', error);
        }
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
        if (!statusRoot() || !queueRoot()) {
            window.setTimeout(init, 250);
            return;
        }
        bindDelegatedEvents();
        startPolling();
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }
})();
