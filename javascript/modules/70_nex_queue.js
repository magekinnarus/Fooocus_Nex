function triggerQueueAction(taskId, actionType) {
    const root = window.gradioApp ? window.gradioApp() : document;
    const actionIdContainer = root.querySelector('#queue_action_id');
    const actionTypeContainer = root.querySelector('#queue_action_type');
    const actionBtnContainer = root.querySelector('#queue_action_btn');
    const actionBtnEl = actionBtnContainer
        ? (actionBtnContainer.matches('button') ? actionBtnContainer : actionBtnContainer.querySelector('button'))
        : null;
    
    if (actionIdContainer && actionTypeContainer && actionBtnEl) {
        const actionIdEl = actionIdContainer.querySelector('input') || actionIdContainer.querySelector('textarea');
        const actionTypeEl = actionTypeContainer.querySelector('input') || actionTypeContainer.querySelector('textarea');
        
        if (actionIdEl && actionTypeEl) {
            actionIdEl.value = taskId;
            actionIdEl.dispatchEvent(new Event('input', { bubbles: true }));
            actionIdEl.dispatchEvent(new Event('change', { bubbles: true }));
            actionTypeEl.value = actionType;
            actionTypeEl.dispatchEvent(new Event('input', { bubbles: true }));
            actionTypeEl.dispatchEvent(new Event('change', { bubbles: true }));
            
            setTimeout(() => {
                actionBtnEl.click();
            }, 50);
        }
    }
}

function setStageButtonState(btn, { disabled = false, text = 'Stage', staged = false } = {}) {
    if (!btn) {
        return;
    }
    btn.disabled = disabled;
    btn.textContent = text;
    btn.classList.toggle('staged', staged);
}

function flashStageButtonSuccess(btn, text) {
    if (!btn) {
        return;
    }
    setStageButtonState(btn, { disabled: false, text, staged: true });
    window.setTimeout(() => {
        if (btn.isConnected) {
            setStageButtonState(btn, { disabled: false, text: 'Stage', staged: false });
        }
    }, 1600);
}

async function stageGeneratedImage(filepath, btn) {
    setStageButtonState(btn, { disabled: true, text: 'Staging...', staged: false });
    try {
        const url = `/file=${encodeURIComponent(filepath)}`;
        const res = await fetch('/staging_api/upload', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/x-www-form-urlencoded',
            },
            body: `url=${encodeURIComponent(url)}`
        });
        const data = await res.json();
        if (data.status === 'success') {
            flashStageButtonSuccess(btn, 'Staged!');
            const refreshBtn = (window.gradioApp ? window.gradioApp() : document).querySelector('#staging-panel-refresh');
            if (refreshBtn) {
                refreshBtn.click();
            }
        } else {
            alert('Failed to stage image: ' + (data.message || 'unknown error'));
            setStageButtonState(btn, { disabled: false, text: 'Stage', staged: false });
        }
    } catch (err) {
        console.error('Error staging image:', err);
        alert('Error staging image: ' + err.message);
        setStageButtonState(btn, { disabled: false, text: 'Stage', staged: false });
    }
}

async function stageAllImages(filepaths, btn) {
    setStageButtonState(btn, { disabled: true, text: 'Staging...', staged: false });
    let successCount = 0;
    for (const filepath of filepaths) {
        try {
            const url = `/file=${encodeURIComponent(filepath)}`;
            const res = await fetch('/staging_api/upload', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/x-www-form-urlencoded',
                },
                body: `url=${encodeURIComponent(url)}`
            });
            const data = await res.json();
            if (data.status === 'success') {
                successCount++;
            }
        } catch (err) {
            console.error('Error staging image:', filepath, err);
        }
    }
    if (successCount === filepaths.length) {
        flashStageButtonSuccess(btn, 'Staged!');
    } else if (successCount > 0) {
        setStageButtonState(btn, {
            disabled: false,
            text: `Staged (${successCount}/${filepaths.length})`,
            staged: true,
        });
        window.setTimeout(() => {
            if (btn && btn.isConnected) {
                setStageButtonState(btn, { disabled: false, text: 'Stage', staged: false });
            }
        }, 1600);
    } else {
        setStageButtonState(btn, { disabled: false, text: 'Stage', staged: false });
    }

    if (successCount < filepaths.length) {
        console.warn(`Stage all completed with partial success: ${successCount}/${filepaths.length}`);
    }

    const refreshBtn = (window.gradioApp ? window.gradioApp() : document).querySelector('#staging-panel-refresh');
    if (refreshBtn) {
        refreshBtn.click();
    }
}

window.triggerQueueAction = triggerQueueAction;
window.stageGeneratedImage = stageGeneratedImage;
window.stageAllImages = stageAllImages;
