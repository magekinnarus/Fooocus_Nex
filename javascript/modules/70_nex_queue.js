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
