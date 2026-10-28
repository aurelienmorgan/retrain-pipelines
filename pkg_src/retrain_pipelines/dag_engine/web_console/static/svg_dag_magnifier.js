/**
 * @file svg_dag_magnifier.js
 * @description 2x absolute zoom viewport. Clones canvas-group + toolbar, tracks cursor.
 * Exposes: magnifierState.
 * Dependencies: Core (ctx.svg, ctx.config).
 */
function initDagMagnifier(ctx) {
    const { svg, config } = ctx;

    const magnifierBtn = document.getElementById('magnify-' + config.id_prefix);
    const magnifierGroup = document.getElementById('magnifier-' + config.id_prefix);
    const magnifierClipRect = document.getElementById('magnifier-clip-rect-' + config.id_prefix);
    const magnifierBg = document.getElementById('magnifier-bg-' + config.id_prefix);
    const magnifierBorder = document.getElementById('magnifier-border-' + config.id_prefix);
    const magnifierInner = svg.querySelector('#magnifier-inner-' + config.id_prefix);

    let isMagnifierActive = false;

    // Set magnifier background color to seamlessly match the canvas.
    // Traverses up the DOM from the SVG's parent to find the first element
    // with an explicit opaque background color. This handles cases where the
    // direct parent is transparent (e.g. deeply nested divs in the webconsole)
    // by falling back to the page background (e.g. <body> or <html>).
    let currentEl = svg.parentElement;
    let bgFound = null;
    while (currentEl) {
        const bg = window.getComputedStyle(currentEl).backgroundColor;
        if (bg && bg !== 'rgba(0, 0, 0, 0)' && bg !== 'transparent') {
            bgFound = bg;
            break;
        }
        currentEl = currentEl.parentElement;
    }
    magnifierBg.setAttribute('fill', bgFound || '#ffffff');

    function updateMagnifierSize() {
        const canvasWidth = parseFloat(window.getComputedStyle(svg).width) || svg.getBoundingClientRect().width || 800;
        const magW = canvasWidth * 0.50;
        const magH = magW * 2; // width/height ratio of 1/2 -> W/H = 1/2 -> H = 2*W

        magnifierClipRect.setAttribute('width', magW);
        magnifierClipRect.setAttribute('height', magH);
        magnifierBg.setAttribute('width', magW);
        magnifierBg.setAttribute('height', magH);
        magnifierBorder.setAttribute('width', magW);
        magnifierBorder.setAttribute('height', magH);
        magnifierInner.setAttribute('width', magW);
        magnifierInner.setAttribute('height', magH);
    }

    let magnifierRAF = null;
    // Clone the canvas-group (which already nests the tooltips-group) and the
    // top-level toolbar foreignObject into the magnifier inner <svg>. Tooltips
    // are NOT cloned separately: they live inside canvas-group and ride along
    // with its pan/zoom transform. Cloning them standalone would drop that
    // transform, producing a phantom tooltip with x/y offset and constant zoom.
    function updateMagnifierContent() {
        if (magnifierRAF) return;
        magnifierRAF = requestAnimationFrame(() => {
            magnifierRAF = null;
            const inner = magnifierInner;
            if (!inner) return;

            // Clear previous content
            while (inner.firstChild) {
                inner.removeChild(inner.firstChild);
            }

            const svgWidth = parseFloat(window.getComputedStyle(svg).width) || svg.getBoundingClientRect().width;

            // Helper to clone and clean elements
            function cloneAndClean(el, cloneId, fixWidth) {
                if (!el) return;
                const clone = el.cloneNode(true);
                clone.id = cloneId;
                clone.querySelectorAll('[id]').forEach(e => e.removeAttribute('id'));
                if (fixWidth) {
                    clone.setAttribute('width', svgWidth);
                    // inline style overrides CSS class
                    // (actual fix to width percentages issues in cloning sub-regions)
                    clone.style.width = svgWidth + 'px';
                }
                inner.appendChild(clone);
            }

            const canvasGroupCloneSrc = document.getElementById('canvas-group-' + config.id_prefix);
            const toolbarFo = svg.querySelector(':scope > foreignObject.dag-toolbar-fo');

            cloneAndClean(canvasGroupCloneSrc, 'magnifier-clone-canvas');
            cloneAndClean(toolbarFo, 'magnifier-clone-toolbar', true);
        });
    }

    // Expose magnifier state to the tooltip IIFE via the SVG DOM node.
    // This bridges the two isolated IIFE scopes without relying on global
    // variables, ensuring strict per-instance encapsulation in multi-DAG environments.
    ctx.magnifierState = {
        get isActive() {
            // console.log("isMagnifierActive", isMagnifierActive, magnifierInner);
            return isMagnifierActive;
        },
        updateContent: updateMagnifierContent,
        lastEscTime: 0,
        hide: function() {
            isMagnifierActive = false;
            magnifierGroup.style.display = 'none';
        }
    };

    magnifierBtn.addEventListener('mousedown', (e) => { if (e.button === 0) e.stopPropagation(); });
    magnifierBtn.addEventListener('click', (e) => {
        e.stopPropagation();
        isMagnifierActive = !isMagnifierActive;
        magnifierGroup.style.display = isMagnifierActive ? '' : 'none';
        if (isMagnifierActive) {
            updateMagnifierSize();
            updateMagnifierContent();
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (isMagnifierActive) {
            const pt = svg.createSVGPoint();
            pt.x = e.clientX;
            pt.y = e.clientY;
            const p = pt.matrixTransform(svg.getScreenCTM().inverse());
            const magW = parseFloat(magnifierClipRect.getAttribute('width'));
            const magH = parseFloat(magnifierClipRect.getAttribute('height'));

            const tx = p.x - magW / 2;
            const ty = p.y - (2 / 3) * magH;
            magnifierGroup.setAttribute('transform', `translate(${tx}, ${ty})`);

            const magScale = 2.0;
            const vw = magW / magScale;
            const vh = magH / magScale;
            const vx = p.x - (p.x - tx) / magScale;
            const vy = p.y - (p.y - ty) / magScale;

            magnifierInner.setAttribute('viewBox', `${vx} ${vy} ${vw} ${vh}`);
            updateMagnifierContent();
        }
    });
}
