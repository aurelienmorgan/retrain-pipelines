/**
 * @file svg_dag_core.js
 * @description Core utilities and shared state for the DAG SVG renderer.
 * Exposes: state.currentTransform, parallelParentMap, updateTransform, 
 * getMouseSVGCoords, parseTranslate, getOffset, splitIds, getAbsY, getBottom.
 * Dependencies: None.
 */
function initDagCore(ctx) {
    const { svg, canvasGroup, config } = ctx;

    ctx.state = {
        currentTransform: {
            x: 0,
            y: 0,
            scale: 1.2 // 120% magnifying at initial load time
        }
    };

    // Build parallel parent map for ancestor traversal
    ctx.parallelParentMap = {};
    canvasGroup.querySelectorAll('.parallel-dag').forEach(pEl => {
        const pId = pEl.id.replace(/^parallel-/, '');
        const containedPs = pEl.dataset.containedParallels
            ? pEl.dataset.containedParallels.split(',').filter(Boolean)
            : [];
        containedPs.forEach(npId => {
            ctx.parallelParentMap[npId] = pId;
        });
    });

    function updateTransform() {
        canvasGroup.setAttribute(
            'transform',
            `translate(${ctx.state.currentTransform.x},
                       ${ctx.state.currentTransform.y}) scale(${ctx.state.currentTransform.scale})`
        );
    }
    ctx.updateTransform = updateTransform;

    function getMouseSVGCoords(evt) {
        const pt = svg.createSVGPoint();
        pt.x = evt.clientX;
        pt.y = evt.clientY;
        const transformed = pt.matrixTransform(svg.getScreenCTM().inverse());

        // Convert to canvas coordinate system (account for pan and zoom)
        return {
            x: (transformed.x - ctx.state.currentTransform.x) / ctx.state.currentTransform.scale,
            y: (transformed.y - ctx.state.currentTransform.y) / ctx.state.currentTransform.scale
        };
    }
    ctx.getMouseSVGCoords = getMouseSVGCoords;

    // Parse the translate(x, y) transform of an element (defaults to 0, 0)
    function parseTranslate(el) {
        const transform = el.getAttribute('transform');
        if (!transform) return { x: 0, y: 0 };
        // Robust regex to handle signs, decimals, and scientific notation (e.g. 1e-7)
        const match = transform.match(/translate\(\s*([^,]+)\s*,\s*([^\)]+)\s*\)/);
        if (!match) return { x: 0, y: 0 };
        const x = parseFloat(match[1]);
        const y = parseFloat(match[2]);
        return { x: isNaN(x) ? 0 : x, y: isNaN(y) ? 0 : y };
    }
    ctx.parseTranslate = parseTranslate;

    // Sum of local translate transforms from el up to (but not including) stopEl.
    // This natively resolves an element's position in the coordinate space of stopEl.
    function getOffset(el, stopEl) {
        let ox = 0, oy = 0;
        let cur = el;
        while (cur && cur !== stopEl) {
            const t = parseTranslate(cur);
            ox += t.x;
            oy += t.y;
            cur = cur.parentElement;
        }
        return { x: ox, y: oy };
    }
    ctx.getOffset = getOffset;

    // Split a comma-separated data attribute value into a filtered id list
    function splitIds(value) {
        return value ? value.split(',').filter(Boolean) : [];
    }
    ctx.splitIds = splitIds;

    // Absolute canvas Y coordinate of an element (node, taskgroup, or parallel-dag)
    function getAbsY(el) {
        if (el.classList.contains('node')) {
            return getOffset(el, canvasGroup).y;
        } else if (el.classList.contains('taskgroup')) {
            const r = el.querySelector(':scope > rect.taskgroup-container');
            if (r) {
                return parseFloat(r.getAttribute('y')) + getOffset(el, canvasGroup).y;
            }
        } else if (el.classList.contains('parallel-dag')) {
            const r = el.querySelector(':scope > rect.parallel-container');
            if (r) {
                return parseFloat(r.getAttribute('y')) + getOffset(el, canvasGroup).y;
            }
        }
        return Infinity;
    }
    ctx.getAbsY = getAbsY;

    // Absolute canvas bottom Y coordinate of an element
    function getBottom(el) {
        if (el.classList.contains('node')) {
            return getOffset(el, canvasGroup).y + config.nodeHeight;
        } else if (el.classList.contains('taskgroup')) {
            const r = el.querySelector(':scope > rect.taskgroup-container');
            if (r) {
                return parseFloat(r.getAttribute('y')) + getOffset(el, canvasGroup).y + parseFloat(r.getAttribute('height'));
            }
        } else if (el.classList.contains('parallel-dag')) {
            const r = el.querySelector(':scope > rect.parallel-container');
            if (r) {
                return parseFloat(r.getAttribute('y')) + getOffset(el, canvasGroup).y + parseFloat(r.getAttribute('height'));
            }
        }
        return Infinity;
    }
    ctx.getBottom = getBottom;
}
