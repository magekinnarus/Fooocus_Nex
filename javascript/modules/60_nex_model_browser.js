(() => {
    const TAB_DEFS = {
        checkpoints: {
            label: 'Checkpoints',
            rootKeys: ['checkpoints'],
            subTabs: [
                { key: 'sd15', label: 'SD15', match: (record) => record.architecture === 'sd15' },
                { key: 'sdxl', label: 'SDXL', match: (record) => record.architecture === 'sdxl' && (record.sub_architecture === 'base' || record.sub_architecture === 'general') },
                { key: 'pony', label: 'Pony', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'pony' },
                { key: 'illustrious', label: 'Illustrious', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'illustrious' },
                { key: 'noob', label: 'Noob', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'noob' },
            ],
        },
        loras: {
            label: 'LoRAs',
            rootKeys: ['loras'],
            subTabs: [
                { key: 'sd15', label: 'SD15', match: (record) => record.architecture === 'sd15' },
                { key: 'sdxl', label: 'SDXL', match: (record) => record.architecture === 'sdxl' && (record.sub_architecture === 'base' || record.sub_architecture === 'general') },
                { key: 'pony', label: 'Pony', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'pony' },
                { key: 'illustrious', label: 'Illustrious', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'illustrious' },
            ],
        },
        unet: {
            label: 'UNet',
            rootKeys: ['unet'],
            subTabs: [
                { key: 'sdxl', label: 'SDXL', match: (record) => record.architecture === 'sdxl' && (record.sub_architecture === 'base' || record.sub_architecture === 'general') },
                { key: 'pony', label: 'Pony', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'pony' },
                { key: 'illustrious', label: 'Illustrious', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'illustrious' },
                { key: 'noob', label: 'Noob', match: (record) => record.architecture === 'sdxl' && record.sub_architecture === 'noob' },
            ],
        },
        others: {
            label: 'Others',
            rootKeys: ['clip', 'vae', 'embeddings'],
            subTabs: [
                { key: 'clip', label: 'CLIP', match: (record) => record.root_key === 'clip' },
                { key: 'sd15_vae', label: 'SD15 VAE', match: (record) => record.root_key === 'vae' && record.architecture === 'sd15' },
                { key: 'sdxl_vae', label: 'SDXL VAE', match: (record) => record.root_key === 'vae' && record.architecture === 'sdxl' },
                { key: 'sd15_embeddings', label: 'SD15 Embeddings', match: (record) => record.root_key === 'embeddings' && record.architecture === 'sd15' },
                { key: 'sdxl_embeddings', label: 'SDXL Embeddings', match: (record) => record.root_key === 'embeddings' && record.architecture === 'sdxl' },
            ],
        },
    };

    const SECTION_KEYS = ['installed_registered', 'installed_unregistered', 'available_registered'];
    const SECTION_LABELS = {
        installed_registered: 'Installed and Registered',
        installed_unregistered: 'Installed and Unregistered',
        available_registered: 'Available for Download',
    };
    const MODEL_TYPE_OPTIONS = ['checkpoint', 'lora', 'unet', 'clip', 'vae', 'embedding'];
    const ARCHITECTURE_OPTIONS = ['unknown', 'sd15', 'sdxl'];
    const SUB_ARCHITECTURE_OPTIONS = ['general', 'none', 'base', 'pony', 'illustrious', 'noob'];
    const SOURCE_PROVIDER_OPTIONS = ['local', 'civitai', 'huggingface'];
    const DRAG_ROOTS = new Set(['checkpoints', 'unet', 'vae', 'clip', 'loras', 'embeddings']);
    const DROP_TARGETS = [
        { selector: '#model_base_dropdown', target: 'base_model', roots: ['checkpoints', 'unet'] },
        { selector: '#model_vae_dropdown', target: 'vae_model', roots: ['vae'] },
        { selector: '#model_clip_dropdown', target: 'clip_model', roots: ['clip'] },
        { selector: '[id^="lora_model_dropdown_"]', target: (node) => `lora_model:${String(node.id).split('_').pop()}`, roots: ['loras'] },
    ];
    const PROMPT_TARGETS = [
        { selector: '#positive_prompt', label: 'positive prompt' },
        { selector: '#negative_prompt', label: 'negative prompt' },
    ];

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

    async function fetchJson(url, options = {}) {
        const response = await fetch(url, {
            headers: { 'Content-Type': 'application/json' },
            ...options,
        });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) {
            throw new Error(payload?.detail || response.statusText || 'Request failed');
        }
        return payload;
    }

    function sortRecords(records) {
        return records.slice().sort((a, b) => String(a.display_name || a.name).localeCompare(String(b.display_name || b.name)));
    }

    class NexModelBrowser extends HTMLElement {
        constructor() {
            super();
            this.state = {
                activeTab: 'checkpoints',
                activeSubTabs: {},
                tabData: {},
                loadingTabs: new Set(),
                selectedByRoot: {},
                drawer: null,
                jobs: {},
                status: '',
                statusTone: 'info',
                thumbnailRevision: 0,
            };
            Object.entries(TAB_DEFS).forEach(([tabKey, def]) => {
                this.state.activeSubTabs[tabKey] = def.subTabs[0].key;
                def.rootKeys.forEach((rootKey) => {
                    this.state.selectedByRoot[rootKey] = new Set();
                });
            });
            this.handleClick = this.handleClick.bind(this);
            this.handleInput = this.handleInput.bind(this);
            this.handleChange = this.handleChange.bind(this);
            this.handleDragStart = this.handleDragStart.bind(this);
            this.thumbnailSession = String(Date.now());
            this.pollHandle = null;
        }

        connectedCallback() {
            if (this.dataset.ready === 'true') return;
            this.dataset.ready = 'true';
            this.addEventListener('click', this.handleClick);
            this.addEventListener('input', this.handleInput);
            this.addEventListener('change', this.handleChange);
            this.addEventListener('dragstart', this.handleDragStart);
            this.render();
            this.ensureTabLoaded(this.state.activeTab);
        }

        disconnectedCallback() {
            this.removeEventListener('click', this.handleClick);
            this.removeEventListener('input', this.handleInput);
            this.removeEventListener('change', this.handleChange);
            this.removeEventListener('dragstart', this.handleDragStart);
            if (this.pollHandle) {
                window.clearTimeout(this.pollHandle);
                this.pollHandle = null;
            }
        }

        queryById(id) {
            if (!id) return null;
            return document.getElementById(id) || gradioRoot().querySelector(`#${id}`);
        }

        sleep(ms) {
            return new Promise((resolve) => window.setTimeout(resolve, ms));
        }

        get refreshButton() {
            return this.queryById(this.dataset.refreshButtonId);
        }

        get applyDataField() {
            return this.queryById(this.dataset.applyDataId);
        }

        thumbnailUrl(record) {
            const params = new URLSearchParams({
                selector: record.id,
                rev: `${this.thumbnailSession}.${this.state.thumbnailRevision}.${record.thumbnail_library_relative || ''}`,
            });
            return `/api/models/thumbnail/file?${params.toString()}`;
        }

        embeddingToken(name) {
            const stem = String(name || '').replace(/\.[^.]+$/, '').trim();
            return stem ? `(embedding:${stem}:1.0)` : '';
        }

        normalizeSubArchitectureValue(rootKey, modelType, subArchitecture) {
            if (rootKey === 'vae' || rootKey === 'embeddings' || modelType === 'vae' || modelType === 'embedding') {
                return 'none';
            }
            return subArchitecture || 'general';
        }

        normalizeDrawerForm(form, rootKey = '') {
            return {
                ...form,
                sub_architecture: this.normalizeSubArchitectureValue(rootKey, form.model_type, form.sub_architecture),
            };
        }

        resolveTextField(wrapper) {
            return wrapper?.matches('input, textarea') ? wrapper : wrapper?.querySelector('input, textarea');
        }

        setStatus(message, tone = 'info') {
            this.state.status = message || '';
            this.state.statusTone = tone;
            this.render();
        }

        clearStatus() {
            this.state.status = '';
            this.state.statusTone = 'info';
        }

        switchTab(tabKey) {
            if (!TAB_DEFS[tabKey]) return;
            this.state.activeTab = tabKey;
            this.clearStatus();
            this.render();
            this.ensureTabLoaded(tabKey);
        }

        switchSubTab(subTabKey) {
            this.state.activeSubTabs[this.state.activeTab] = subTabKey;
            this.clearStatus();
            this.render();
        }

        tabSelectedCount(tabKey) {
            return TAB_DEFS[tabKey].rootKeys.reduce((count, rootKey) => count + this.state.selectedByRoot[rootKey].size, 0);
        }

        activeRootKeys() {
            return TAB_DEFS[this.state.activeTab].rootKeys;
        }

        currentSubTabs() {
            const top = TAB_DEFS[this.state.activeTab];
            if (this.state.activeTab !== 'others') return top.subTabs;
            const visible = top.subTabs.filter((subTab) => this.subTabCount(subTab.key) > 0);
            return visible.length ? visible : top.subTabs;
        }

        activeSubTabDef() {
            const subTabs = this.currentSubTabs();
            const activeKey = this.state.activeSubTabs[this.state.activeTab];
            const resolved = subTabs.find((subTab) => subTab.key === activeKey) || subTabs[0];
            if (resolved && resolved.key !== activeKey) {
                this.state.activeSubTabs[this.state.activeTab] = resolved.key;
            }
            return resolved;
        }

        activeDownloadSelectors() {
            const subTab = this.activeSubTabDef();
            const tabData = this.state.tabData[this.state.activeTab] || {};
            return this.activeRootKeys().flatMap((rootKey) => {
                const selected = this.state.selectedByRoot[rootKey] || new Set();
                return (tabData[rootKey]?.available_registered || [])
                    .filter((record) => selected.has(record.id) && subTab.match(record))
                    .map((record) => record.id);
            });
        }

        toggleSelection(rootKey, selector) {
            const selection = this.state.selectedByRoot[rootKey] || new Set();
            if (selection.has(selector)) selection.delete(selector);
            else selection.add(selector);
            this.state.selectedByRoot[rootKey] = selection;
            this.render();
        }

        clearActiveSelection() {
            const activeSelectorSet = new Set(this.activeDownloadSelectors());
            this.activeRootKeys().forEach((rootKey) => {
                const selection = this.state.selectedByRoot[rootKey] || new Set();
                activeSelectorSet.forEach((selector) => selection.delete(selector));
                this.state.selectedByRoot[rootKey] = selection;
            });
            this.render();
        }

        async ensureTabLoaded(tabKey, { force = false } = {}) {
            if (this.state.loadingTabs.has(tabKey)) return;
            if (!force && this.state.tabData[tabKey]) return;
            this.state.loadingTabs.add(tabKey);
            this.render();
            try {
                const rootPayloads = {};
                await Promise.all(TAB_DEFS[tabKey].rootKeys.map(async (rootKey) => {
                    const params = new URLSearchParams({ root_key: rootKey });
                    rootPayloads[rootKey] = await fetchJson(`/api/models/browser?${params.toString()}`);
                }));
                this.state.tabData[tabKey] = rootPayloads;
            } catch (error) {
                this.setStatus(error.message || 'Failed to load models.', 'error');
            } finally {
                this.state.loadingTabs.delete(tabKey);
                this.render();
            }
        }
        async refreshAllData() {
            this.state.thumbnailRevision += 1;
            try {
                await fetchJson('/api/models/refresh', { method: 'POST' });
                if (this.refreshButton) this.refreshButton.click();
                await Promise.all(Object.keys(TAB_DEFS).map((tabKey) => this.ensureTabLoaded(tabKey, { force: true })));
            } catch (error) {
                this.setStatus(error.message || 'Failed to refresh models.', 'error');
            }
        }

        async queueActiveDownloads() {
            const selectors = this.activeDownloadSelectors();
            if (!selectors.length) {
                this.setStatus('Select one or more available models in this subtab first.', 'warning');
                return;
            }
            this.setStatus('Queueing downloads...', 'info');
            try {
                const payload = await fetchJson('/api/models/downloads/batch', {
                    method: 'POST',
                    body: JSON.stringify({ root_keys: this.activeRootKeys(), selectors }),
                });
                if (!payload.jobs?.length) {
                    const details = (payload.skipped || []).slice(0, 3).map((item) => `${item.selector || 'model'}: ${String(item.reason || 'skipped').replaceAll('_', ' ')}`).join(' | ');
                    this.setStatus(details || 'No downloadable models were queued from this subtab.', 'warning');
                    return;
                }
                payload.jobs.forEach((job) => { this.state.jobs[job.job_id] = job; });
                this.setStatus(`Queued ${payload.jobs.length} download${payload.jobs.length === 1 ? '' : 's'} in this subtab.`, 'success');
                this.startPollingJobs();
                this.render();
            } catch (error) {
                this.setStatus(error.message || 'Failed to queue downloads.', 'error');
            }
        }

        startPollingJobs() {
            if (this.pollHandle) return;
            const poll = async () => {
                const pendingIds = Object.keys(this.state.jobs).filter((jobId) => {
                    const status = this.state.jobs[jobId]?.status;
                    return status !== 'succeeded' && status !== 'failed';
                });
                if (!pendingIds.length) {
                    this.pollHandle = null;
                    return;
                }
                if (this.offsetParent === null) {
                    this.pollHandle = window.setTimeout(poll, 1800);
                    return;
                }
                try {
                    const results = await Promise.all(pendingIds.map((jobId) => fetchJson(`/api/models/downloads/${jobId}`)));
                    let completed = false;
                    results.forEach((job) => {
                        this.state.jobs[job.job_id] = job;
                        if (job.status === 'succeeded' || job.status === 'failed') completed = true;
                        if (job.status === 'succeeded') {
                            Object.keys(this.state.selectedByRoot).forEach((rootKey) => this.state.selectedByRoot[rootKey].delete(job.entry_id));
                        }
                    });
                    const failed = results.filter((job) => job.status === 'failed');
                    if (failed.length) {
                        const job = failed[0];
                        this.setStatus(job.message || job.error || `Download failed for ${job.entry_id}.`, 'error');
                    }
                    if (completed) await this.refreshAllData();
                } catch (error) {
                    this.setStatus(error.message || 'Download polling failed.', 'error');
                }
                this.render();
                this.pollHandle = window.setTimeout(poll, 1200);
            };
            this.pollHandle = window.setTimeout(poll, 1200);
        }

        async openDrawer(selector, sourceProvider = '', sourceVersionId = '', matchedSelector = '') {
            this.state.drawer = {
                loading: true,
                selector,
                matchedSelector: matchedSelector || '',
                mode: 'registration',
                form: {
                    display_name: '',
                    name: '',
                    installed_relative_path: '',
                    relative_path: '',
                    model_type: 'checkpoint',
                    architecture: 'unknown',
                    sub_architecture: 'general',
                    thumbnail_library_relative: '',
                    source_provider: sourceProvider || 'local',
                    source_version_id: sourceVersionId || '',
                    source_url: '',
                    companion_clip_selector: '',
                    companion_clip_relative_path: '',
                },
                suggestions: [],
                companionClip: null,
                error: '',
                installedLink: null,
                canEditCatalogFields: true,
            };
            this.render();
            try {
                const params = new URLSearchParams({ selector, suggest_limit: '3' });
                if (sourceProvider) params.set('source_provider', sourceProvider);
                if (sourceVersionId) params.set('source_version_id', sourceVersionId);
                if (matchedSelector) params.set('matched_selector', matchedSelector);
                const payload = await fetchJson(`/api/models/registration?${params.toString()}`);
                const companionClip = payload.companion_clip || null;
                this.state.drawer = {
                    ...this.state.drawer,
                    loading: false,
                    mode: 'registration',
                    matchedSelector: matchedSelector || this.state.drawer.matchedSelector,
                    entry: payload.entry,
                    companionClip,
                    form: this.normalizeDrawerForm({
                        display_name: payload.entry.display_name || '',
                        name: payload.entry.name || '',
                        installed_relative_path: payload.entry.installed_relative_path || payload.entry.relative_path || '',
                        relative_path: payload.entry.relative_path || '',
                        model_type: payload.entry.model_type || 'checkpoint',
                        architecture: payload.entry.architecture || 'unknown',
                        sub_architecture: payload.entry.sub_architecture || 'general',
                        thumbnail_library_relative: payload.entry.thumbnail_library_relative || payload.thumbnail?.relative_path || '',
                        source_provider: sourceProvider || payload.entry.source_provider || 'local',
                        source_version_id: sourceVersionId || payload.entry.source_version_id || '',
                        source_url: payload.entry.source?.url || '',
                        companion_clip_selector: companionClip?.recommended_selector || '',
                        companion_clip_relative_path: '',
                    }, payload.entry.root_key || ''),
                    suggestions: payload.suggestions || [],
                };
            } catch (error) {
                this.state.drawer = { ...this.state.drawer, loading: false, error: error.message || 'Failed to load registration details.' };
            }
            this.render();
        }

        async openInstalledDrawer(selector) {
            this.state.drawer = {
                loading: true,
                selector,
                matchedSelector: '',
                mode: 'installed_link',
                form: {
                    display_name: '',
                    name: '',
                    installed_relative_path: '',
                    relative_path: '',
                    model_type: 'checkpoint',
                    architecture: 'unknown',
                    sub_architecture: 'general',
                    thumbnail_library_relative: '',
                    source_provider: 'local',
                    source_version_id: '',
                    source_url: '',
                },
                suggestions: [],
                error: '',
                installedLink: null,
                canEditCatalogFields: false,
            };
            this.render();
            try {
                const params = new URLSearchParams({ selector, suggest_limit: '3' });
                const payload = await fetchJson(`/api/models/installed-link?${params.toString()}`);
                this.state.drawer = {
                    ...this.state.drawer,
                    loading: false,
                    mode: 'installed_link',
                    entry: payload.entry,
                    installedLink: payload.installed_link || null,
                    suggestions: payload.suggestions || [],
                    canEditCatalogFields: Boolean(payload.can_edit_catalog_fields),
                    form: this.normalizeDrawerForm({
                        display_name: payload.entry.display_name || '',
                        name: payload.entry.name || '',
                        installed_relative_path: payload.installed_link?.installed_relative_path || payload.entry.installed_relative_path || '',
                        relative_path: payload.entry.relative_path || '',
                        model_type: payload.entry.model_type || 'checkpoint',
                        architecture: payload.entry.architecture || 'unknown',
                        sub_architecture: payload.entry.sub_architecture || 'general',
                        thumbnail_library_relative: payload.entry.thumbnail_library_relative || payload.thumbnail?.relative_path || '',
                        source_provider: payload.entry.source_provider || 'local',
                        source_version_id: payload.entry.source_version_id || '',
                        source_url: payload.entry.source?.url || '',
                    }, payload.entry.root_key || ''),
                };
            } catch (error) {
                this.state.drawer = { ...this.state.drawer, loading: false, error: error.message || 'Failed to load installed model details.' };
            }
            this.render();
        }


        closeDrawer() {
            this.state.drawer = null;
            this.render();
        }

        async refreshSuggestions() {
            if (!this.state.drawer) return;
            const { selector, form, matchedSelector, mode } = this.state.drawer;
            if (mode === 'installed_link') {
                await this.openInstalledDrawer(selector);
                return;
            }
            await this.openDrawer(selector, form.source_provider, form.source_version_id, matchedSelector);
        }

        chooseSuggestion(selector) {
            if (!this.state.drawer) return;
            const suggestion = (this.state.drawer.suggestions || []).find((item) => item.entry.id === selector);
            if (!suggestion) return;
            this.state.drawer.matchedSelector = suggestion.entry.id;
            this.state.drawer.form = this.normalizeDrawerForm({
                ...this.state.drawer.form,
                display_name: suggestion.entry.display_name || this.state.drawer.form.display_name,
                name: suggestion.entry.name || this.state.drawer.form.name,
                relative_path: suggestion.entry.relative_path || this.state.drawer.form.relative_path,
                model_type: suggestion.entry.model_type || this.state.drawer.form.model_type,
                architecture: suggestion.entry.architecture || this.state.drawer.form.architecture,
                sub_architecture: suggestion.entry.sub_architecture || this.state.drawer.form.sub_architecture,
                thumbnail_library_relative: suggestion.entry.thumbnail_library_relative || this.state.drawer.form.thumbnail_library_relative,
                source_provider: suggestion.entry.source_provider || this.state.drawer.form.source_provider,
                source_version_id: suggestion.entry.source_version_id || this.state.drawer.form.source_version_id,
                source_url: suggestion.entry.source?.url || this.state.drawer.form.source_url,
            }, this.state.drawer.entry?.root_key || '');
            this.render();
        }

        async uploadDrawerThumbnail(file) {
            if (!this.state.drawer || !file) return;
            const { selector } = this.state.drawer;
            this.setStatus(`Uploading thumbnail ${file.name}...`, 'info');
            const formData = new FormData();
            formData.append('selector', selector);
            formData.append('file', file, file.name);
            try {
                const response = await fetch('/api/models/thumbnail/upload', {
                    method: 'POST',
                    body: formData,
                });
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) {
                    throw new Error(payload?.detail || response.statusText || 'Thumbnail upload failed');
                }
                this.state.thumbnailRevision += 1;
                this.state.drawer.entry = payload.entry || this.state.drawer.entry;
                this.state.drawer.form = this.normalizeDrawerForm({
                    ...this.state.drawer.form,
                    thumbnail_library_relative: payload.entry?.thumbnail_library_relative || payload.thumbnail?.relative_path || this.state.drawer.form.thumbnail_library_relative,
                }, this.state.drawer.entry?.root_key || '');
                this.setStatus(`Updated thumbnail for ${payload.entry?.display_name || payload.entry?.name || 'model'}.`, 'success');
                this.render();
            } catch (error) {
                this.setStatus(error.message || 'Failed to upload thumbnail.', 'error');
            }
        }

        async saveRegistration() {
            if (!this.state.drawer) return;
            const { selector, matchedSelector, form, mode } = this.state.drawer;
            this.state.drawer.loading = true;
            this.state.drawer.error = '';
            this.render();
            try {
                const endpoint = mode === 'installed_link' ? '/api/models/installed-link' : '/api/models/registration';
                await fetchJson(endpoint, {
                    method: 'POST',
                    body: JSON.stringify({
                        selector,
                        matched_selector: matchedSelector || undefined,
                        updates: {
                            display_name: form.display_name,
                            name: form.name,
                            installed_relative_path: form.installed_relative_path,
                            relative_path: form.relative_path,
                            model_type: form.model_type,
                            architecture: form.architecture,
                            sub_architecture: form.sub_architecture,
                            source_provider: form.source_provider,
                            source_version_id: form.source_version_id,
                            source_url: form.source_url,
                            thumbnail_library_relative: form.thumbnail_library_relative,
                        },
                    }),
                });
                await this.refreshAllData();
                this.setStatus(mode === 'installed_link' ? 'Installed model link saved.' : 'Model registration saved.', 'success');
                this.closeDrawer();
            } catch (error) {
                this.state.drawer.loading = false;
                this.state.drawer.error = error.message || (mode === 'installed_link' ? 'Failed to save installed model link.' : 'Failed to save registration.');
                this.render();
            }
        }

        recordsForSection(sectionKey) {
            const subTab = this.activeSubTabDef();
            const tabData = this.state.tabData[this.state.activeTab] || {};
            const records = this.activeRootKeys().flatMap((rootKey) => (tabData[rootKey]?.[sectionKey] || []).filter((record) => subTab.match(record)));
            return sortRecords(records);
        }

        subTabCount(subTabKey) {
            const subTab = TAB_DEFS[this.state.activeTab].subTabs.find((item) => item.key === subTabKey);
            if (!subTab) return 0;
            return SECTION_KEYS.reduce((total, sectionKey) => {
                const tabData = this.state.tabData[this.state.activeTab] || {};
                const sectionRecords = this.activeRootKeys().flatMap((rootKey) => tabData[rootKey]?.[sectionKey] || []);
                return total + sectionRecords.filter((record) => subTab.match(record)).length;
            }, 0);
        }

        setBridgeValue(wrapper, value) {
            if (!wrapper) return false;
            const input = this.resolveTextField(wrapper);
            if (!input) return false;
            input.value = value ?? '';
            input.dispatchEvent(new Event('input', { bubbles: true }));
            input.dispatchEvent(new Event('change', { bubbles: true }));
            return true;
        }

        resolveDragPayload(event) {
            const raw = event.dataTransfer?.getData('application/json') || event.dataTransfer?.getData('text/plain');
            if (!raw) return null;
            try {
                const payload = JSON.parse(raw);
                if (payload?.selector && payload?.rootKey) return payload;
            } catch (_) {
                return null;
            }
            return null;
        }

        async applyDropToTarget(payload, targetKey, acceptedRoots) {
            if (!payload?.selector || !payload?.rootKey) return;
            if (!acceptedRoots.includes(payload.rootKey)) {
                this.setStatus(`This drop target only accepts ${acceptedRoots.join(' / ')} models.`, 'warning');
                return;
            }
            const field = this.applyDataField;
            if (!field) {
                this.setStatus('Model apply bridge is not ready yet. Try reloading the UI.', 'error');
                return;
            }
            const applied = this.setBridgeValue(field, JSON.stringify({
                selector: payload.selector,
                target: targetKey,
                aspect_ratio: this.currentAspectRatioValue(),
                ts: Date.now(),
            }));
            if (!applied) {
                this.setStatus('Model apply bridge could not update the apply field. Try reloading the UI.', 'error');
                return;
            }
            this.setStatus(`Applying ${payload.rootKey} to ${targetKey.replace('_', ' ')}...`, 'info');
        }

        currentAspectRatioValue() {
            const root = gradioRoot();
            const checked = root.querySelector('#aspect_ratios_selection input[type="radio"]:checked') || root.querySelector('.aspect_ratios input[type="radio"]:checked');
            if (checked && checked.value) return String(checked.value);
            const checkedLabel = checked?.closest('label');
            const text = checkedLabel?.querySelector('span')?.textContent || checkedLabel?.textContent;
            return text ? String(text).trim() : '';
        }

        applyPromptDrop(payload, targetNode, label) {
            if (!payload?.token || payload.rootKey !== 'embeddings') return;
            const field = this.resolveTextField(targetNode);
            if (!field) {
                this.setStatus(`Could not find the ${label} field.`, 'error');
                return;
            }
            const currentValue = String(field.value || '').trim();
            const token = payload.token.trim();
            const nextValue = currentValue ? `${currentValue}, ${token}` : token;
            field.value = nextValue;
            field.dispatchEvent(new Event('input', { bubbles: true }));
            field.dispatchEvent(new Event('change', { bubbles: true }));
            field.focus();
            this.setStatus(`Inserted ${token} into the ${label}.`, 'success');
        }

        resolvePreferredLoraTarget() {
            const root = gradioRoot();
            const dropdowns = Array.from(root.querySelectorAll('[id^="lora_model_dropdown_"]'));
            if (!dropdowns.length) return 'lora_model:1';
            for (const node of dropdowns) {
                const field = this.resolveTextField(node);
                const value = String(field?.value || '').trim();
                if (!value || value === 'None') {
                    const suffix = String(node.id).split('_').pop() || '1';
                    return `lora_model:${suffix}`;
                }
            }
            const suffix = String(dropdowns[0].id).split('_').pop() || '1';
            return `lora_model:${suffix}`;
        }

        async applyInstalledSelector(selector, rootKey) {
            if (!selector || !rootKey) return;
            let targetKey = null;
            let acceptedRoots = [rootKey];
            if (rootKey === 'checkpoints' || rootKey === 'unet') {
                targetKey = 'base_model';
                acceptedRoots = ['checkpoints', 'unet'];
            } else if (rootKey === 'vae') {
                targetKey = 'vae_model';
            } else if (rootKey === 'clip') {
                targetKey = 'clip_model';
            } else if (rootKey === 'loras') {
                targetKey = this.resolvePreferredLoraTarget();
            }
            if (!targetKey) {
                this.setStatus(`Apply is not supported for ${rootKey}.`, 'warning');
                return;
            }
            await this.applyDropToTarget({ selector, rootKey, token: '' }, targetKey, acceptedRoots);
        }

        insertEmbeddingIntoTarget(token, targetSelector, label) {
            if (!token) return;
            const root = gradioRoot();
            const targetNode = root.querySelector(targetSelector);
            if (!targetNode) {
                this.setStatus(`Could not find the ${label}.`, 'error');
                return;
            }
            this.applyPromptDrop({ rootKey: 'embeddings', token }, targetNode, label);
        }

        installDropTargets() {
            const root = gradioRoot();
            DROP_TARGETS.forEach((binding) => {
                root.querySelectorAll(binding.selector).forEach((node) => {
                    if (node.dataset.nmbDropReady === 'true') return;
                    node.dataset.nmbDropReady = 'true';
                    node.classList.add('nmb-browser-drop-target');
                    const targetKey = typeof binding.target === 'function' ? binding.target(node) : binding.target;
                    const acceptedRoots = binding.roots;
                    node.addEventListener('dragover', (event) => {
                        const payload = this.resolveDragPayload(event);
                        if (!payload || !acceptedRoots.includes(payload.rootKey)) return;
                        event.preventDefault();
                        node.classList.add('is-drop-ready');
                    });
                    node.addEventListener('dragleave', () => {
                        node.classList.remove('is-drop-ready');
                    });
                    node.addEventListener('drop', (event) => {
                        const payload = this.resolveDragPayload(event);
                        node.classList.remove('is-drop-ready');
                        if (!payload || !acceptedRoots.includes(payload.rootKey)) return;
                        event.preventDefault();
                        this.applyDropToTarget(payload, targetKey, acceptedRoots);
                    });
                });
            });
            PROMPT_TARGETS.forEach((binding) => {
                root.querySelectorAll(binding.selector).forEach((node) => {
                    if (node.dataset.nmbPromptDropReady === 'true') return;
                    node.dataset.nmbPromptDropReady = 'true';
                    node.classList.add('nmb-browser-drop-target');
                    node.addEventListener('dragover', (event) => {
                        const payload = this.resolveDragPayload(event);
                        if (!payload || payload.rootKey !== 'embeddings') return;
                        event.preventDefault();
                        node.classList.add('is-drop-ready');
                    });
                    node.addEventListener('dragleave', () => {
                        node.classList.remove('is-drop-ready');
                    });
                    node.addEventListener('drop', (event) => {
                        const payload = this.resolveDragPayload(event);
                        node.classList.remove('is-drop-ready');
                        if (!payload || payload.rootKey !== 'embeddings') return;
                        event.preventDefault();
                        this.applyPromptDrop(payload, node, binding.label);
                    });
                });
            });
        }

        handleDragStart(event) {
            const card = event.target.closest('.nmb-card[draggable="true"]');
            if (!card) return;
            const payload = {
                selector: card.dataset.selector,
                rootKey: card.dataset.rootKey,
                token: card.dataset.token || '',
            };
            event.dataTransfer.effectAllowed = 'copy';
            event.dataTransfer.setData('application/json', JSON.stringify(payload));
            event.dataTransfer.setData('text/plain', JSON.stringify(payload));
        }
        renderTopTabs() {
            return Object.entries(TAB_DEFS).map(([tabKey, tab]) => {
                const count = this.tabSelectedCount(tabKey);
                return `<button type="button" class="nmb-top-tab ${this.state.activeTab === tabKey ? 'is-active' : ''}" data-action="switch-top-tab" data-tab-key="${escapeHtml(tabKey)}">${escapeHtml(tab.label)}${count ? ` <span class="nmb-pill">${count}</span>` : ''}</button>`;
            }).join('');
        }

        renderSubTabs() {
            const activeKey = this.activeSubTabDef()?.key;
            return this.currentSubTabs().map((subTab) => {
                const count = this.subTabCount(subTab.key);
                return `<button type="button" class="nmb-sub-tab ${activeKey === subTab.key ? 'is-active' : ''}" data-action="switch-sub-tab" data-sub-tab-key="${escapeHtml(subTab.key)}">${escapeHtml(subTab.label)} <span class="nmb-sub-tab__count">${count}</span></button>`;
            }).join('');
        }

        renderCard(record, sectionKey) {
            const selectable = sectionKey === 'available_registered';
            const unregistered = sectionKey === 'installed_unregistered';
            const installedRegistered = sectionKey === 'installed_registered';
            const draggable = installedRegistered && record.root_key === 'embeddings';
            const selected = this.state.selectedByRoot[record.root_key]?.has(record.id);
            const surfaceAction = unregistered
                ? 'open-drawer'
                : (installedRegistered ? 'open-installed-drawer' : (selectable ? 'toggle-selection' : ''));
            const stateLabel = unregistered
                ? 'Registration Required'
                : (installedRegistered
                    ? 'Installed'
                    : (selectable ? (selected ? 'Selected for Download' : 'Click to Select') : 'Installed'));

            let actions = '';
            if (installedRegistered) {
                if (record.root_key === 'embeddings') {
                    actions = `
                        <div class="nmb-card__actions">
                            <button type="button" class="nmb-secondary nmb-card__action" data-action="insert-embedding-positive" data-token="${escapeHtml(this.embeddingToken(record.name))}">Insert Positive</button>
                            <button type="button" class="nmb-secondary nmb-card__action" data-action="insert-embedding-negative" data-token="${escapeHtml(this.embeddingToken(record.name))}">Insert Negative</button>
                            <button type="button" class="nmb-secondary nmb-card__action" data-action="open-installed-drawer" data-selector="${escapeHtml(record.id)}">Review</button>
                        </div>
                    `;
                } else {
                    actions = `
                        <div class="nmb-card__actions">
                            <button type="button" class="nmb-primary nmb-card__action" data-action="apply-installed-default" data-selector="${escapeHtml(record.id)}" data-root-key="${escapeHtml(record.root_key)}">Apply</button>
                            <button type="button" class="nmb-secondary nmb-card__action" data-action="open-installed-drawer" data-selector="${escapeHtml(record.id)}">Review</button>
                        </div>
                    `;
                }
            } else if (unregistered) {
                actions = `
                    <div class="nmb-card__actions">
                        <button type="button" class="nmb-primary nmb-card__action" data-action="open-drawer" data-selector="${escapeHtml(record.id)}">Register</button>
                    </div>
                `;
            }

            return `
                <div class="nmb-card ${selected ? 'is-selected' : ''} ${unregistered ? 'is-unregistered' : ''} ${draggable ? 'is-draggable' : ''}" data-root-key="${escapeHtml(record.root_key)}" data-selector="${escapeHtml(record.id)}">
                    <button type="button" class="nmb-card__surface" ${surfaceAction ? `data-action="${surfaceAction}"` : ''} ${surfaceAction ? `data-selector="${escapeHtml(record.id)}"` : ''} ${surfaceAction === 'toggle-selection' ? `data-root-key="${escapeHtml(record.root_key)}"` : ''} ${draggable ? 'draggable="true"' : ''} data-root-key="${escapeHtml(record.root_key)}" data-selector="${escapeHtml(record.id)}" data-token="${escapeHtml(this.embeddingToken(record.name))}">
                        <div class="nmb-card__media">
                            <img class="nmb-card__thumb" src="${escapeHtml(this.thumbnailUrl(record))}" alt="${escapeHtml(record.display_name || record.name || 'Model thumbnail')}" loading="lazy">
                            <div class="nmb-card__badge">${escapeHtml((record.display_name || record.name || 'N').slice(0, 3).toUpperCase())}</div>
                        </div>
                        <div class="nmb-card__content">
                            <div class="nmb-card__title">${escapeHtml(record.display_name || record.name)}</div>
                            <div class="nmb-card__filename">${escapeHtml(record.name)}</div>
                            <div class="nmb-card__meta">${escapeHtml(record.root_key)}${record.source_version_id ? ` | ${escapeHtml(record.source_version_id)}` : ''}</div>
                            <div class="nmb-card__state">${escapeHtml(stateLabel)}</div>
                        </div>
                    </button>
                    ${actions}
                </div>
            `;
        }

        renderSection(sectionKey) {
            const records = this.recordsForSection(sectionKey);
            const cards = records.length ? records.map((record) => this.renderCard(record, sectionKey)).join('') : '<div class="nmb-empty">No models in this section.</div>';
            return `
                <section class="nmb-section nmb-section--${escapeHtml(sectionKey)}">
                    <div class="nmb-section__header">
                        <div>
                            <h3 class="nmb-section__title">${escapeHtml(SECTION_LABELS[sectionKey])}</h3>
                            <div class="nmb-section__count">${records.length} model${records.length === 1 ? '' : 's'}</div>
                        </div>
                    </div>
                    <div class="nmb-card-grid">${cards}</div>
                </section>
            `;
        }

        renderJobs() {
            const jobs = Object.values(this.state.jobs).sort((a, b) => Number(b.created_at || 0) - Number(a.created_at || 0)).slice(0, 5);
            if (!jobs.length) return '';
            return `<div class="nmb-jobs">${jobs.map((job) => {
                const detail = job.error || job.message || '';
                return `<div class="nmb-job nmb-job--${escapeHtml(job.status || 'queued')}"><div class="nmb-job__main"><div class="nmb-job__title">${escapeHtml(job.entry_id)}</div>${detail ? `<div class="nmb-job__detail">${escapeHtml(detail)}</div>` : ''}</div><div class="nmb-job__status">${escapeHtml(job.status)}</div></div>`;
            }).join('')}</div>`;
        }

        renderSelect(field, options, value, disabled = false) {
            const resolved = value || options[0];
            return `<select data-field="${escapeHtml(field)}" ${disabled ? 'disabled' : ''}>${options.map((option) => `<option value="${escapeHtml(option)}" ${option === resolved ? 'selected' : ''}>${escapeHtml(option)}</option>`).join('')}</select>`;
        }

        renderDrawer() {
            const drawer = this.state.drawer;
            if (!drawer) return '';
            const form = drawer.form || {};
            const entry = drawer.entry || {};
            const installedLink = drawer.installedLink || {};
            const installedLinkMode = drawer.mode === 'installed_link';
            const canEditCatalogFields = !installedLinkMode || drawer.canEditCatalogFields;
            const drawerTitle = installedLinkMode ? 'Edit Installed Model' : 'Register Model';
            const drawerSubtitle = installedLinkMode
                ? 'Review the linked installed path and relink this installed file if the current catalog match is wrong.'
                : 'Review this unregistered model and optionally apply a suggested match.';
            const currentInstalledPath = form.installed_relative_path || installedLink.installed_relative_path || entry.installed_relative_path || entry.relative_path || '';
            const currentCatalogPath = entry.relative_path || form.relative_path || '';
            const saveLabel = installedLinkMode ? 'Save Installed Link' : 'Save Registration';
            const suggestions = (drawer.suggestions || []).length
                ? drawer.suggestions.map((item) => `
                    <button type="button" class="nmb-suggestion ${drawer.matchedSelector === item.entry.id ? 'is-selected' : ''}" data-action="choose-suggestion" data-selector="${escapeHtml(item.entry.id)}">
                        <div class="nmb-suggestion__title">${escapeHtml(item.entry.display_name || item.entry.name)}</div>
                        <div class="nmb-suggestion__meta">${escapeHtml(item.entry.source_provider || 'local')}${item.entry.source_version_id ? ` | ${escapeHtml(item.entry.source_version_id)}` : ''}</div>
                        <div class="nmb-suggestion__score">Score ${escapeHtml(item.score)}</div>
                    </button>
                `).join('')
                : '<div class="nmb-empty">No suggestions yet.</div>';
            return `
                <aside class="nmb-drawer ${drawer.loading ? 'is-loading' : ''}">
                    <div class="nmb-drawer__header">
                        <div>
                            <h3>${escapeHtml(drawerTitle)}</h3>
                            <p>${escapeHtml(drawerSubtitle)}</p>
                        </div>
                        <button type="button" class="nmb-secondary" data-action="close-drawer">Close</button>
                    </div>
                    <div class="nmb-drawer__current">
                        <div class="nmb-drawer__current-label">Selected Model</div>
                        <div class="nmb-drawer__current-name">${escapeHtml(entry.display_name || entry.name || form.name || 'Unnamed model')}</div>
                        <div class="nmb-drawer__current-path">Installed: ${escapeHtml(currentInstalledPath)}</div>
                        <div class="nmb-drawer__current-path">Catalog: ${escapeHtml(currentCatalogPath)}</div>
                        <div class="nmb-drawer__current-meta">${escapeHtml(entry.root_key || '')}${form.architecture ? ` | ${escapeHtml(form.architecture)}` : ''}${form.sub_architecture ? ` | ${escapeHtml(form.sub_architecture)}` : ''}</div>
                    </div>
                    ${drawer.error ? `<div class="nmb-status nmb-status--error">${escapeHtml(drawer.error)}</div>` : ''}
                    <div class="nmb-form-grid">
                        <label class="nmb-field--wide"><span>Installed Relative Path</span><input data-field="installed_relative_path" value="${escapeHtml(form.installed_relative_path || '')}" placeholder="Path under the configured model root"></label>
                        <label><span>Display Name</span><input data-field="display_name" value="${escapeHtml(form.display_name || '')}" ${canEditCatalogFields ? '' : 'readonly'}></label>
                        <label><span>Canonical Name</span><input data-field="name" value="${escapeHtml(form.name || '')}" ${canEditCatalogFields ? '' : 'readonly'}></label>
                        <label class="nmb-field--wide"><span>Catalog Relative Path</span><input data-field="relative_path" value="${escapeHtml(form.relative_path || '')}" ${canEditCatalogFields ? '' : 'readonly'}></label>
                        <label><span>Model Type</span>${this.renderSelect('model_type', MODEL_TYPE_OPTIONS, form.model_type, !canEditCatalogFields)}</label>
                        <label><span>Architecture</span>${this.renderSelect('architecture', ARCHITECTURE_OPTIONS, form.architecture, !canEditCatalogFields)}</label>
                        <label><span>Sub-Architecture</span>${this.renderSelect('sub_architecture', SUB_ARCHITECTURE_OPTIONS, form.sub_architecture, !canEditCatalogFields)}</label>
                        <label><span>Source Provider</span>${this.renderSelect('source_provider', SOURCE_PROVIDER_OPTIONS, form.source_provider, !canEditCatalogFields)}</label>
                        <label><span>Version ID</span><input data-field="source_version_id" value="${escapeHtml(form.source_version_id || '')}" ${canEditCatalogFields ? '' : 'readonly'}></label>
                        <label class="nmb-field--wide nmb-field--picker">
                            <span>Thumbnail Path</span>
                            <div class="nmb-field__picker">
                                <input data-field="thumbnail_library_relative" value="${escapeHtml(form.thumbnail_library_relative || '')}" placeholder="Optional thumbnail path under the thumbnail library" ${canEditCatalogFields ? '' : 'readonly'}>
                                <button type="button" class="nmb-secondary nmb-field__picker-button" data-action="pick-thumbnail-file" ${canEditCatalogFields ? '' : 'disabled'}>Browse...</button>
                            </div>
                        </label>
                        <label class="nmb-field--wide"><span>Source URL</span><input data-field="source_url" value="${escapeHtml(form.source_url || '')}" placeholder="Optional" ${canEditCatalogFields ? '' : 'readonly'}></label>
                    </div>
                    <div class="nmb-drawer__thumbnail-actions">
                        <input type="file" accept="image/*" data-thumbnail-upload hidden ${canEditCatalogFields ? '' : 'disabled'}>
                        <button type="button" class="nmb-secondary" data-action="pick-thumbnail-file" ${canEditCatalogFields ? '' : 'disabled'}>Choose Thumbnail Image</button>
                    </div>
                    <div class="nmb-drawer__actions">
                        <button type="button" class="nmb-secondary" data-action="refresh-suggestions">Refresh Suggestions</button>
                        <button type="button" class="nmb-primary" data-action="save-registration">${escapeHtml(saveLabel)}</button>
                    </div>
                    <div class="nmb-drawer__suggestions">
                        <h4>Possible Matches</h4>
                        <p class="nmb-drawer__hint">Selecting a suggestion pre-fills the canonical metadata. Save to confirm the catalog link for this installed file.</p>
                        <div class="nmb-suggestions">${suggestions}</div>
                    </div>
                </aside>
            `;
        }
        handleClick(event) {
            const actionNode = event.target.closest('[data-action]');
            if (!actionNode) return;
            const { action } = actionNode.dataset;
            if (action === 'switch-top-tab') return this.switchTab(actionNode.dataset.tabKey);
            if (action === 'switch-sub-tab') return this.switchSubTab(actionNode.dataset.subTabKey);
            if (action === 'toggle-selection') return this.toggleSelection(actionNode.dataset.rootKey, actionNode.dataset.selector);
            if (action === 'open-drawer') return this.openDrawer(actionNode.dataset.selector);
            if (action === 'open-installed-drawer') return this.openInstalledDrawer(actionNode.dataset.selector);
            if (action === 'close-drawer') return this.closeDrawer();
            if (action === 'refresh-suggestions') return this.refreshSuggestions();
            if (action === 'choose-suggestion') return this.chooseSuggestion(actionNode.dataset.selector);
            if (action === 'apply-installed-default') return this.applyInstalledSelector(actionNode.dataset.selector, actionNode.dataset.rootKey);
            if (action === 'insert-embedding-positive') return this.insertEmbeddingIntoTarget(actionNode.dataset.token, '#positive_prompt', 'positive prompt');
            if (action === 'insert-embedding-negative') return this.insertEmbeddingIntoTarget(actionNode.dataset.token, '#negative_prompt', 'negative prompt');
            if (action === 'pick-thumbnail-file') {
                const uploadInput = this.querySelector('[data-thumbnail-upload]');
                if (uploadInput && !uploadInput.disabled) uploadInput.click();
                return;
            }
            if (action === 'save-registration') return this.saveRegistration();
            if (action === 'download-active') return this.queueActiveDownloads();
            if (action === 'clear-selection') return this.clearActiveSelection();
            if (action === 'refresh-browser') return this.refreshAllData();
        }

        handleInput(event) {
            if (!this.state.drawer) return;
            const input = event.target.closest('[data-field]');
            if (!input) return;
            this.state.drawer.form[input.dataset.field] = input.value;
            this.state.drawer.form = this.normalizeDrawerForm(this.state.drawer.form, this.state.drawer.entry?.root_key || '');
        }

        handleChange(event) {
            const uploadInput = event.target.closest('[data-thumbnail-upload]');
            if (uploadInput?.files?.length) {
                const [file] = uploadInput.files;
                uploadInput.value = '';
                this.uploadDrawerThumbnail(file);
                return;
            }
            this.handleInput(event);
        }

        render() {
            const loading = this.state.loadingTabs.has(this.state.activeTab) && !this.state.tabData[this.state.activeTab];
            const selectedCount = this.activeDownloadSelectors().length;
            const body = loading
                ? '<div class="nmb-empty">Loading models...</div>'
                : SECTION_KEYS.map((sectionKey) => this.renderSection(sectionKey)).join('');
            this.innerHTML = `
                <div class="nmb-shell ${this.state.drawer ? 'has-drawer' : ''}">
                    <div class="nmb-main">
                        <div class="nmb-top-tabs">${this.renderTopTabs()}</div>
                        <div class="nmb-sub-tabs">${this.renderSubTabs()}</div>
                        <div class="nmb-toolbar">
                            <div>
                                <div class="nmb-toolbar__title">${escapeHtml(TAB_DEFS[this.state.activeTab].label)} / ${escapeHtml(this.activeSubTabDef().label)}</div>
                                <div class="nmb-toolbar__subtitle">Selections can stay staged across subtabs, but download and clear actions only affect the current subtab.</div>
                            </div>
                            <div class="nmb-toolbar__actions">
                                <button type="button" class="nmb-secondary" data-action="clear-selection">Clear This Subtab</button>
                                <button type="button" class="nmb-secondary" data-action="refresh-browser">Reload Browser</button>
                                <button type="button" class="nmb-primary" data-action="download-active" ${selectedCount ? '' : 'disabled'}>Download Selected in This Subtab${selectedCount ? ` (${selectedCount})` : ''}</button>
                            </div>
                        </div>
                        ${this.state.status ? `<div class="nmb-status nmb-status--${escapeHtml(this.state.statusTone)}">${escapeHtml(this.state.status)}</div>` : ''}
                        ${this.renderJobs()}
                        <div class="nmb-sections">${body}</div>
                    </div>
                    ${this.renderDrawer()}
                </div>
            `;
            window.requestAnimationFrame(() => this.installDropTargets());
        }
    }

    if (!customElements.get('nex-model-browser')) {
        customElements.define('nex-model-browser', NexModelBrowser);
    }
})();























