function gradioApp() {
    const elems = document.getElementsByTagName('gradio-app');
    const elem = elems.length == 0 ? document : elems[0];

    if (elem !== document) {
        elem.getElementById = function (id) {
            return document.getElementById(id);
        };
    }
    return elem.shadowRoot ? elem.shadowRoot : elem;
}

var uiUpdateCallbacks = [];
var uiAfterUpdateCallbacks = [];

function onUiUpdate(callback) {
    uiUpdateCallbacks.push(callback);
}

function onAfterUiUpdate(callback) {
    uiAfterUpdateCallbacks.push(callback);
}

function executeCallbacks(queue, arg) {
    for (const callback of queue) {
        try {
            callback(arg);
        } catch (e) {
            console.error("error running callback", callback, ":", e);
        }
    }
}

function htmlDecode(input) {
    var doc = new DOMParser().parseFromString(input, "text/html");
    return doc.documentElement.textContent;
}

// Localization System
var re_num = /^[.\d]+$/;
var original_lines = {};
var translated_lines = {};

function hasLocalization() {
    return window.localization && Object.keys(window.localization).length > 0;
}

function getTranslation(text) {
    if (!text) return undefined;
    if (translated_lines[text] === undefined) {
        original_lines[text] = 1;
    }
    var tl = localization[text];
    if (tl !== undefined) {
        translated_lines[tl] = 1;
    }
    return tl;
}

function processTextNode(node) {
    var text = node.textContent.trim();
    if (!text || !node.parentElement) return;
    var parentType = node.parentElement.nodeName;
    if (parentType == 'SCRIPT' || parentType == 'STYLE' || parentType == 'TEXTAREA') return;
    if (re_num.test(text)) return;

    var tl = getTranslation(text);
    if (tl !== undefined) {
        node.textContent = tl;
        if (text && node.parentElement) {
            node.parentElement.setAttribute("data-original-text", text);
        }
    }
}

function processNode(node) {
    if (node.nodeType == 3) {
        processTextNode(node);
        return;
    }
    if (node.title) {
        let tl = getTranslation(node.title);
        if (tl !== undefined) node.title = tl;
    }
    if (node.placeholder) {
        let tl = getTranslation(node.placeholder);
        if (tl !== undefined) node.placeholder = tl;
    }

    // Find text nodes
    var walk = document.createTreeWalker(node, NodeFilter.SHOW_TEXT, null, false);
    var n;
    while ((n = walk.nextNode())) {
        processTextNode(n);
    }
}

function refresh_style_localization() {
    const stylesNode = gradioApp().querySelector('.style_selections');
    if (!stylesNode) return;
    processNode(stylesNode);
}

function refresh_aspect_ratios_label(value) {
    const root = gradioApp();
    const label = root.querySelector('#aspect_ratios_accordion summary .label-wrap span') ||
        root.querySelector('#aspect_ratios_accordion .label-wrap span') ||
        root.querySelector('#aspect_ratios_accordion summary span') ||
        root.querySelector('#aspect_ratios_accordion span');

    if (!label) {
        return;
    }

    let translation = getTranslation("Aspect Ratios");
    if (typeof translation == "undefined") {
        translation = "Aspect Ratios";
    }
    label.textContent = translation + " " + htmlDecode(value);
}

function localizeWholePage() {
    processNode(gradioApp());
}

document.addEventListener("DOMContentLoaded", function () {
    const mutationObserver = new MutationObserver(function (m) {
        if (gradioApp().querySelector('#generate_button')) {
            executeCallbacks(uiUpdateCallbacks, m);
            // scheduleAfterUiUpdate omitted for simplicity if not heavily used
        }
    });
    mutationObserver.observe(gradioApp(), { childList: true, subtree: true });

    if (hasLocalization()) {
        localizeWholePage();
    }
});
