
class NexMonitor {
    constructor() {
        this.pollInterval = 2000; // 2 seconds
        this.panel = null;
        this.isDragging = false;
        this.dragOffset = { x: 0, y: 0 };
        this.isMinimized = false;
        this.init();
    }

    init() {
        if (this.panel) return;

        // Create the floating panel
        this.panel = document.createElement('div');
        this.panel.id = 'floating-monitor-panel';
        this.panel.className = 'floating-panel';
        this.panel.innerHTML = `
            <div class="panel-header" id="monitor-panel-header">
                <span class="panel-title">Resource Dashboard</span>
                <div class="panel-controls">
                    <button id="monitor-panel-toggle" title="Minimize">－</button>
                    <button id="monitor-panel-close" title="Close Dashboard">×</button>
                </div>
            </div>
            <div class="panel-content">
                <div id="nex-monitor-container" class="nex-monitor-grid">
                    <div class="nex-monitor-item" id="cpu-monitor">
                        <div class="nex-monitor-ring-container">
                            <svg width="52" height="52" viewBox="0 0 64 64">
                                <circle class="nex-monitor-ring-bg" cx="32" cy="32" r="28"></circle>
                                <circle class="nex-monitor-ring-bar" cx="32" cy="32" r="28" stroke-dasharray="175.9" stroke-dashoffset="175.9"></circle>
                            </svg>
                            <div class="nex-monitor-label">0%</div>
                        </div>
                        <div class="nex-monitor-title">CPU</div>
                    </div>
                    <div class="nex-monitor-item" id="gpu-monitor">
                        <div class="nex-monitor-ring-container">
                            <svg width="52" height="52" viewBox="0 0 64 64">
                                <circle class="nex-monitor-ring-bg" cx="32" cy="32" r="28"></circle>
                                <circle class="nex-monitor-ring-bar" cx="32" cy="32" r="28" stroke-dasharray="175.9" stroke-dashoffset="175.9"></circle>
                            </svg>
                            <div class="nex-monitor-label">0%</div>
                        </div>
                        <div class="nex-monitor-title">GPU</div>
                    </div>
                    <div class="nex-monitor-item" id="ram-monitor">
                        <div class="nex-monitor-ring-container">
                            <svg width="52" height="52" viewBox="0 0 64 64">
                                <circle class="nex-monitor-ring-bg" cx="32" cy="32" r="28"></circle>
                                <circle class="nex-monitor-ring-bar" cx="32" cy="32" r="28" stroke-dasharray="175.9" stroke-dashoffset="175.9"></circle>
                            </svg>
                            <div class="nex-monitor-label">0%</div>
                        </div>
                        <div class="nex-monitor-title">RAM</div>
                    </div>
                    <div class="nex-monitor-item" id="vram-monitor">
                        <div class="nex-monitor-ring-container">
                            <svg width="52" height="52" viewBox="0 0 64 64">
                                <circle class="nex-monitor-ring-bg" cx="32" cy="32" r="28"></circle>
                                <circle class="nex-monitor-ring-bar" cx="32" cy="32" r="28" stroke-dasharray="175.9" stroke-dashoffset="175.9"></circle>
                            </svg>
                            <div class="nex-monitor-label">0%</div>
                        </div>
                        <div class="nex-monitor-title">VRAM</div>
                    </div>
                </div>
                <div class="vram-info" id="vram-details" style="font-size: 10px; text-align: center; margin-top: 4px; opacity: 0.8;">Initializing...</div>
            </div>
        `;

        document.body.appendChild(this.panel);

        // Position Logic
        const snapToDefault = () => {
            this.panel.style.top = '140px'; // Below staging panel by default
            this.panel.style.right = '20px';
            this.panel.style.left = 'auto';
        };

        // Load position and visibility
        const pos = JSON.parse(localStorage.getItem('monitor-panel-pos') || '{"top": "140px", "right": "20px"}');
        const isHidden = localStorage.getItem('monitor-panel-hidden') === 'true';
        Object.assign(this.panel.style, pos);
        if (isHidden) this.panel.style.display = 'none';

        // Drag events
        const header = this.panel.querySelector('#monitor-panel-header');
        header.addEventListener('mousedown', (e) => this.startDragging(e));
        document.addEventListener('mousemove', (e) => this.drag(e));
        document.addEventListener('mouseup', () => this.stopDragging());

        // Controls
        this.panel.querySelector('#monitor-panel-toggle').addEventListener('click', () => {
            this.isMinimized = this.panel.classList.toggle('minimized');
            if (this.isMinimized) snapToDefault();
        });

        this.panel.querySelector('#monitor-panel-close').addEventListener('click', () => {
            this.panel.style.display = 'none';
            localStorage.setItem('monitor-panel-hidden', 'true');
        });

        // Global listener for launcher button
        document.addEventListener('click', (e) => {
            const launcher = e.target.closest('#monitor-panel-launcher');
            if (launcher) {
                this.panel.style.display = 'flex';
                localStorage.setItem('monitor-panel-hidden', 'false');
                this.panel.classList.remove('minimized');
                this.panel.style.transform = 'scale(1.05)';
                setTimeout(() => this.panel.style.transform = 'scale(1)', 200);
            }
        });

        this.startPolling();
    }

    startDragging(e) {
        if (e.target.tagName === 'BUTTON') return;
        this.isDragging = true;
        const rect = this.panel.getBoundingClientRect();
        this.dragOffset = {
            x: e.clientX - rect.left,
            y: e.clientY - rect.top
        };
        this.panel.style.right = 'auto';
        this.panel.style.left = rect.left + 'px';
        this.panel.style.top = rect.top + 'px';
        this.panel.classList.add('dragging');
        e.preventDefault();
    }

    drag(e) {
        if (!this.isDragging) return;
        this.panel.style.left = (e.clientX - this.dragOffset.x) + 'px';
        this.panel.style.top = (e.clientY - this.dragOffset.y) + 'px';
    }

    stopDragging() {
        if (!this.isDragging) return;
        this.isDragging = false;
        this.panel.classList.remove('dragging');
        localStorage.setItem('monitor-panel-pos', JSON.stringify({
            top: this.panel.style.top,
            left: this.panel.style.left
        }));
    }

    async startPolling() {
        while (true) {
            try {
                if (this.panel.style.display !== 'none') {
                    const response = await fetch('/nex_api/monitor');
                    const data = await response.json();
                    if (!data.error) this.updateRings(data);
                }
            } catch (e) {
                console.debug("[Nex-Monitor] Polling failed:", e);
            }
            await new Promise(resolve => setTimeout(resolve, this.pollInterval));
        }
    }

    updateRings(data) {
        this.updateRing('cpu-monitor', data.cpu);
        this.updateRing('gpu-monitor', data.gpu);
        this.updateRing('ram-monitor', data.ram);
        this.updateRing('vram-monitor', data.vram);
        
        const vramDetails = document.getElementById('vram-details');
        if (vramDetails && data.vram_details) {
            const used = Math.round(data.vram_details.total - data.vram_details.free);
            vramDetails.innerText = `${used}MB / ${data.vram_details.total}MB (${data.device})`;
        }
    }

    updateRing(id, percent) {
        const item = document.getElementById(id);
        if (!item) return;
        const bar = item.querySelector('.nex-monitor-ring-bar');
        const label = item.querySelector('.nex-monitor-label');
        const circumference = 175.9;
        const offset = circumference - (percent / 100) * circumference;
        bar.style.strokeDashoffset = offset;
        label.innerText = `${Math.round(percent)}%`;
        bar.classList.remove('ring-warning', 'ring-danger');
        if (percent > 85) bar.classList.add('ring-danger');
        else if (percent > 70) bar.classList.add('ring-warning');
    }
}

// Start when Gradio is ready
function initNexMonitor() {
    if (window.gradioApp) {
        new NexMonitor();
    } else {
        setTimeout(initNexMonitor, 500);
    }
}
initNexMonitor();
