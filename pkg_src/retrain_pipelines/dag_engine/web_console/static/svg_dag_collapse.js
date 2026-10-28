/**
 * @file svg_dag_collapse.js
 * @description Animated collapse/expand of containers, batch toolbar operations,
 * and global serialization lock.
 * Exposes: collapseElement, expandElement.
 * Dependencies: Core (ctx.svg, ctx.config, ctx.getOffset, ctx.getAbsY, ctx.getBottom, 
 * ctx.parseTranslate, ctx.splitIds), Bounds (ctx.updateTaskgroupBounds, 
 * ctx.updateParallelBounds at runtime), Arrows (ctx.updateLines at runtime), 
 * Canvas (ctx.adjustSVGHeight at runtime), Magnifier (ctx.magnifierState at runtime).
 */
function initDagCollapse(ctx) {
    const { svg, canvasGroup, config } = ctx;

    // Collapse generic dimensions initialized on "tg" defaults
    const collapsedMarginBottom = config.tgMarginBottom;
    const collapsedLabelMarginTop = config.tgLabelMarginTop;
    const collapsedLabelHeight = config.tgLabelHeight;
    const collapsedWidth = 200;
    const collapsedHeight = collapsedLabelHeight + collapsedLabelMarginTop + collapsedMarginBottom;
    const borderStroke = 0.75; // Prevents border cropping during collapse clipping

    // Global serialization lock: no two collapse/expand operations (individual
    // or batch) may run concurrently. Acquired before any collapseElement /
    // expandElement call, released in its onComplete callback.
    let opInProgress = false;

    // Helper to hide/show the first-level content of a collapsed container
    function setContainerContentVisibility(container, isVisible) {
        Array.from(container.children).forEach(child => {
            if (child.tagName.toLowerCase() === 'clippath') return;
            if (child.tagName.toLowerCase() === 'defs') return;
            if (child.classList.contains('taskgroup-container') ||
                child.classList.contains('parallel-container') ||
                child.classList.contains('taskgroup-underlay-group')) return;
            if (child.classList.contains('collapse-expand-fo')) return;
            if (child.tagName.toLowerCase() === 'text') return;

            child.style.display = isVisible ? '' : 'none';
        });
    }

    // Animate the collapse of a taskgroup or parallel container.
    // Progressively clips content, resizes the container rect, and
    // recursively shifts all elements below upwards.
    // duration (ms, default 300) controls animation speed;
    // batch operations pass 30 for faster feedback.
    ctx.collapseElement = function(btn, onComplete, duration = 300) {
        if (btn.dataset.collapsing === 'true' || btn.dataset.collapsed === 'true') {
            if (onComplete) onComplete();
            return;
        }
        btn.dataset.collapsing = 'true';

        const container = btn.closest('.taskgroup') || btn.closest('.parallel-dag');
        if (!container) return;

        const isTaskgroup = container.classList.contains('taskgroup');
        const isParallel = container.classList.contains('parallel-dag');

        const maxPd = isTaskgroup ? parseFloat(container.dataset.maxParallelDepth || 0) : 0;
        const maxUnderlayOffset = maxPd > 0 && !container.dataset.parentTaskgroup ? 3.5 * maxPd : 0;
        const rects = container.querySelectorAll(':scope > rect, :scope > g.taskgroup-underlay-group > rect'); if (!rects.length) return; const rect = rects[0];
        const originalX = parseFloat(rect.getAttribute('x'));
        const originalY = parseFloat(rect.getAttribute('y'));
        const originalWidth = parseFloat(rect.getAttribute('width'));
        const originalHeight = parseFloat(rect.getAttribute('height'));

        const clipId = 'clip-' + container.id;
        let clipPath = document.getElementById(clipId);
        if (!clipPath) {
            clipPath = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
            clipPath.id = clipId;
            container.insertBefore(clipPath, container.firstChild);
            const newClipRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            clipPath.appendChild(newClipRect);
        }
        const clipRect = clipPath.querySelector('rect');

        clipRect.setAttribute('x', (originalX - borderStroke).toFixed(3));
        clipRect.setAttribute('y', (originalY - borderStroke).toFixed(3));
        clipRect.setAttribute('width', (originalWidth + 2 * borderStroke + maxUnderlayOffset).toFixed(3));
        clipRect.setAttribute('height', (originalHeight + 2 * borderStroke + maxUnderlayOffset).toFixed(3));
        clipRect.setAttribute('rx', 6);
        clipRect.setAttribute('ry', 6);
        container.setAttribute('clip-path', `url(#${clipId})`);

        const startTime = performance.now();
        let prevD = 0;

        function animateCollapse(time) {
            const elapsed = time - startTime;
            let p = Math.min(1, elapsed / duration);
            const ease = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;

            const currD = ease * (originalHeight - collapsedHeight);
            const currdW = ease * (originalWidth - collapsedWidth);
            const currdX = ease * ((originalWidth - collapsedWidth) / 2);
            const delta = currD - prevD;

            const newHeight = originalHeight - currD;
            const newWidth = originalWidth - currdW;
            const newX = originalX + currdX;

            rects.forEach(r => {
                r.setAttribute('height', newHeight.toFixed(3));
                r.setAttribute('width', newWidth.toFixed(3));
                r.setAttribute('x', newX.toFixed(3));
            });

            clipRect.setAttribute('x', (newX - borderStroke).toFixed(3));
            clipRect.setAttribute('y', (originalY - borderStroke).toFixed(3));
            clipRect.setAttribute('width', (newWidth + 2 * borderStroke + maxUnderlayOffset).toFixed(3));
            clipRect.setAttribute('height', (newHeight + 2 * borderStroke + maxUnderlayOffset).toFixed(3));
            clipRect.setAttribute('rx', 6);
            clipRect.setAttribute('ry', 6);

            const fo = container.querySelector(':scope > foreignObject.collapse-expand-fo');
            if (fo) {
                fo.setAttribute('x', newX.toFixed(3));
                fo.setAttribute('y', originalY.toFixed(3));
                fo.setAttribute('width', newWidth.toFixed(3));
                fo.setAttribute('height', newHeight.toFixed(3));
            }

            if (isTaskgroup) {
                const label = container.querySelector(':scope > text.taskgroup-label');
                if (label) {
                    label.setAttribute('x', (newX + newWidth / 2).toFixed(3));
                    label.setAttribute('y', (originalY + collapsedLabelHeight).toFixed(3));
                }
            }

            reflowUp(container, delta);

            canvasGroup.querySelectorAll('.node, .taskgroup, .parallel-dag').forEach(el => {
                const uuid = el.id.replace(/^(node|taskgroup|parallel)-/, '');
                ctx.updateLines(uuid);
            });

            prevD = currD;

            const magnifierState = ctx.magnifierState;
            if (p < 1) {
                requestAnimationFrame(animateCollapse);
                /* repaint the magnifier if showing */
                if (magnifierState && magnifierState.isActive) {
                    magnifierState.updateContent();
                }
            } else {
                btn.dataset.collapsing = 'false';
                btn.dataset.collapsed = 'true';
                btn.textContent = '+';
                btn.style.height = '22px';
                btn.style.lineHeight = '22px';
                setContainerContentVisibility(container, false);

                if (isParallel) {
                    let pLabel = container.querySelector(':scope > text.parallel-collapsed-label');
                    if (!pLabel) {
                        pLabel = document.createElementNS('http://www.w3.org/2000/svg', 'text');
                        pLabel.setAttribute('class', 'label parallel-collapsed-label');
                        pLabel.setAttribute('text-anchor', 'middle');
                        pLabel.setAttribute('font-family', 'sans-serif');
                        pLabel.setAttribute('font-size', '11');
                        pLabel.textContent = 'Distributed sub-pipeline';
                        container.appendChild(pLabel);
                        const firstNodeLabel = container.querySelector('.node .label');
                        if (firstNodeLabel) {
                            pLabel.setAttribute('style', firstNodeLabel.getAttribute('style') || '');
                        }
                    }
                    if (pLabel) {
                        const finalX = parseFloat(rect.getAttribute('x'));
                        const finalWidth = parseFloat(rect.getAttribute('width'));
                        pLabel.setAttribute('x', (finalX + finalWidth / 2).toFixed(3));
                        pLabel.setAttribute('y', (originalY + collapsedLabelHeight).toFixed(3));
                    }
                }

                ctx.adjustSVGHeight();
                updateToolbarState();
                /* repaint the magnifier if showing */
                if (magnifierState && magnifierState.isActive) {
                    magnifierState.updateContent();
                }
                /* execute collapse callback if any */
                if (onComplete) onComplete();
            }
        }

        requestAnimationFrame(animateCollapse);
    };

    // Animate the expand of a collapsed taskgroup or parallel container.
    // Progressively reveals clipped content, resizes the container rect, and
    // recursively shifts all elements below downwards. Mirrors collapseElement.
    // duration (ms, default 300) controls animation speed;
    // batch operations pass 30 for faster feedback.
    ctx.expandElement = function(btn, onComplete, duration = 300) {
        if (btn.dataset.expanding === 'true' || btn.dataset.collapsed === 'false') {
            if (onComplete) onComplete();
            return;
        }
        btn.dataset.expanding = 'true';

        const container = btn.closest('.taskgroup') || btn.closest('.parallel-dag');
        if (!container) return;

        const isTaskgroup = container.classList.contains('taskgroup');
        const isParallel = container.classList.contains('parallel-dag');

        const maxPd = isTaskgroup ? parseFloat(container.dataset.maxParallelDepth || 0) : 0;
        const maxUnderlayOffset = maxPd > 0 && !container.dataset.parentTaskgroup ? 3.5 * maxPd : 0;
        const rects = container.querySelectorAll(':scope > rect, :scope > g.taskgroup-underlay-group > rect'); if (!rects.length) return; const rect = rects[0];
        const collapsedX = parseFloat(rect.getAttribute('x'));
        const collapsedY = parseFloat(rect.getAttribute('y'));
        const collapsedWidth = parseFloat(rect.getAttribute('width'));
        const collapsedHeight = parseFloat(rect.getAttribute('height'));

        if (isTaskgroup) {
            ctx.updateTaskgroupBounds(container.id);
        } else if (isParallel) {
            ctx.updateParallelBounds(container.id.replace('parallel-', ''));
        }

        const expandedX = parseFloat(rect.getAttribute('x'));
        const expandedY = parseFloat(rect.getAttribute('y'));
        const expandedWidth = parseFloat(rect.getAttribute('width'));
        const expandedHeight = parseFloat(rect.getAttribute('height'));

        rects.forEach(r => {
            r.setAttribute('x', collapsedX.toFixed(3));
            r.setAttribute('y', collapsedY.toFixed(3));
            r.setAttribute('width', collapsedWidth.toFixed(3));
            r.setAttribute('height', collapsedHeight.toFixed(3));
        });

        const clipId = 'clip-' + container.id;
        let clipPath = document.getElementById(clipId);
        if (!clipPath) {
            clipPath = document.createElementNS('http://www.w3.org/2000/svg', 'clipPath');
            clipPath.id = clipId;
            container.insertBefore(clipPath, container.firstChild);
            const newClipRect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
            clipPath.appendChild(newClipRect);
        }
        const clipRect = clipPath.querySelector('rect');
        container.setAttribute('clip-path', `url(#${clipId})`);

        setContainerContentVisibility(container, true);

        if (isParallel) {
            const pLabel = container.querySelector(':scope > text.parallel-collapsed-label');
            if (pLabel) pLabel.remove();
        }

        const startTime = performance.now();
        let prevD = 0;

        function animateExpand(time) {
            const elapsed = time - startTime;
            let p = Math.min(1, elapsed / duration);
            const ease = p < 0.5 ? 2 * p * p : 1 - Math.pow(-2 * p + 2, 2) / 2;

            const currD = ease * (expandedHeight - collapsedHeight);
            const currdW = ease * (expandedWidth - collapsedWidth);
            const currdX = ease * ((expandedWidth - collapsedWidth) / 2);
            const delta = currD - prevD;

            const newHeight = collapsedHeight + currD;
            const newWidth = collapsedWidth + currdW;
            const newX = collapsedX - currdX;

            rects.forEach(r => {
                r.setAttribute('height', newHeight.toFixed(3));
                r.setAttribute('width', newWidth.toFixed(3));
                r.setAttribute('x', newX.toFixed(3));
            });

            if (clipRect) {
                clipRect.setAttribute('x', (newX - borderStroke).toFixed(3));
                clipRect.setAttribute('y', (expandedY - borderStroke).toFixed(3));
                clipRect.setAttribute('width', (newWidth + 2 * borderStroke + maxUnderlayOffset).toFixed(3));
                clipRect.setAttribute('height', (newHeight + 2 * borderStroke + maxUnderlayOffset).toFixed(3));
                clipRect.setAttribute('rx', 6);
                clipRect.setAttribute('ry', 6);
            }

            const fo = container.querySelector(':scope > foreignObject.collapse-expand-fo');
            if (fo) {
                fo.setAttribute('x', newX.toFixed(3));
                fo.setAttribute('y', expandedY.toFixed(3));
                fo.setAttribute('width', newWidth.toFixed(3));
                fo.setAttribute('height', newHeight.toFixed(3));
            }

            if (isTaskgroup) {
                const label = container.querySelector(':scope > text.taskgroup-label');
                if (label) {
                    label.setAttribute('x', (newX + newWidth / 2).toFixed(3));
                    label.setAttribute('y', (expandedY + collapsedLabelHeight).toFixed(3));
                }
            }

            reflowUp(container, -delta);

            canvasGroup.querySelectorAll('.node, .taskgroup, .parallel-dag').forEach(el => {
                const uuid = el.id.replace(/^(node|taskgroup|parallel)-/, '');
                ctx.updateLines(uuid);
            });

            prevD = currD;

            const magnifierState = ctx.magnifierState;
            if (p < 1) {
                requestAnimationFrame(animateExpand);
                /* repaint the magnifier if showing */
                if (magnifierState && magnifierState.isActive) {
                    magnifierState.updateContent();
                }
            } else {
                btn.dataset.expanding = 'false';
                btn.dataset.collapsed = 'false';
                btn.textContent = '-';
                btn.style.height = '20px';
                btn.style.lineHeight = '20px';

                container.removeAttribute('clip-path');
                if (clipPath) clipPath.remove();

                ctx.adjustSVGHeight();
                updateToolbarState();
                /* repaint the magnifier if showing */
                if (magnifierState && magnifierState.isActive) {
                    magnifierState.updateContent();
                }
                /* execute expand callback if any */
                if (onComplete) onComplete();
            }
        }

        requestAnimationFrame(animateExpand);
    };

    // Shift all tooltips associated with an element (node, taskgroup, or parallel-dag)
    // by a given delta. Tooltips live outside the groups, so they must be moved manually.
    function shiftTooltipsForElement(el, delta) {
        if (Math.abs(delta) < 0.0001) return;
        if (el.classList.contains('node')) {
            const uuid = el.id.replace('node-', '');
            const tooltip = document.getElementById('tooltip-node-' + uuid);
            if (tooltip) {
                tooltip.setAttribute('y', (parseFloat(tooltip.getAttribute('y')) - delta).toFixed(3));
            }
        } else if (el.classList.contains('taskgroup')) {
            const uuid = el.id.replace('taskgroup-', '');
            const tooltip = document.getElementById('tooltip-taskgroup-' + uuid);
            if (tooltip) {
                tooltip.setAttribute('y', (parseFloat(tooltip.getAttribute('y')) - delta).toFixed(3));
            }
            ctx.splitIds(el.dataset.nodes).forEach(nodeId => {
                const nTooltip = document.getElementById('tooltip-node-' + nodeId);
                if (nTooltip) {
                    nTooltip.setAttribute('y', (parseFloat(nTooltip.getAttribute('y')) - delta).toFixed(3));
                }
            });
            ctx.splitIds(el.dataset.nestedTaskgroups).forEach(nestedTgId => {
                const nestedTgEl = document.getElementById('taskgroup-' + nestedTgId);
                if (nestedTgEl) shiftTooltipsForElement(nestedTgEl, delta);
            });
        } else if (el.classList.contains('parallel-dag')) {
            ctx.splitIds(el.dataset.containedNodes).forEach(nodeId => {
                const nTooltip = document.getElementById('tooltip-node-' + nodeId);
                if (nTooltip) {
                    nTooltip.setAttribute('y', (parseFloat(nTooltip.getAttribute('y')) - delta).toFixed(3));
                }
            });
            ctx.splitIds(el.dataset.containedTaskgroups).forEach(tgId => {
                const tgEl = document.getElementById('taskgroup-' + tgId);
                if (tgEl) shiftTooltipsForElement(tgEl, delta);
            });
        }
    }

    // Recursive helper to reshape ancestors and shift successors (up on collapse, down on expand).
    // Propagates both width and height changes up to canvas level via standard bounds
    // recomputation (same updateTaskgroupBounds / updateParallelBounds used by node-dragging).
    // Sibling shifting is gated on non-zero vertical delta; parent reshape is unconditional.
    function reflowUp(el, delta) {
        const parent = el.parentElement;
        if (!parent) return;

        if (Math.abs(delta) > 0.0001) {
            const elBottom = ctx.getBottom(el);
            Array.from(parent.children).forEach(sibling => {
                if (sibling === el) return;
                if (!sibling.classList.contains('node') && !sibling.classList.contains('taskgroup') && !sibling.classList.contains('parallel-dag')) return;

                const sibY = ctx.getAbsY(sibling);
                // Shift siblings that were strictly below the element's OLD bottom.
                // elBottom is the NEW bottom. Adding delta reconstructs the OLD bottom.
                // This prevents shifting siblings above the element when a large delta
                // occurs (e.g. fast 30ms animation completing in one frame), which
                // previously caused layout destruction by shifting ALL siblings upwards.
                if (sibY >= elBottom + delta - 0.1) {
                    const t = ctx.parseTranslate(sibling);
                    sibling.setAttribute('transform', `translate(${t.x.toFixed(3)},${(t.y - delta).toFixed(3)})`);
                    shiftTooltipsForElement(sibling, delta);
                }
            });
        }

        if (parent === canvasGroup) return;

        const pRect = parent.querySelector(':scope > rect.taskgroup-container, :scope > rect.parallel-container');
        if (!pRect) return;
        const oldW = parseFloat(pRect.getAttribute('width'));
        const oldH = parseFloat(pRect.getAttribute('height'));

        if (parent.classList.contains('taskgroup')) {
            ctx.updateTaskgroupBounds(parent.id);
        } else if (parent.classList.contains('parallel-dag')) {
            ctx.updateParallelBounds(parent.id.replace('parallel-', ''));
        } else {
            return;
        }

        const newW = parseFloat(pRect.getAttribute('width'));
        const newH = parseFloat(pRect.getAttribute('height'));
        const parentDelta = oldH - newH;

        if (Math.abs(parentDelta) > 0.0001 || Math.abs(oldW - newW) > 0.0001) {
            reflowUp(parent, parentDelta);
        }
    }

    /* Collapse-expand-btn toggling */
    canvasGroup.querySelectorAll('.collapse-expand-btn').forEach(el => {
        if (el.textContent === '-') {
            el.style.height = '20px';
            el.style.lineHeight = '20px';
        } else {
            el.style.height = '22px';
            el.style.lineHeight = '22px';
        }
        el.addEventListener('mousedown', (e) => { if (e.button === 0) e.stopPropagation(); });
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            if (opInProgress) return;
            opInProgress = true;
            if (el.textContent === '-') {
                el.textContent = '+';
                el.style.height = '20px';
                el.style.lineHeight = '20px';
                ctx.collapseElement(el, () => { opInProgress = false; });
            } else {
                el.textContent = '-';
                el.style.height = '20px';
                el.style.lineHeight = '20px';
                ctx.expandElement(el, () => { opInProgress = false; });
            }
        });
    });

    /* Floating glass toolbar: un-bevel on mousedown is delivered by CSS :active.
       Click handlers trigger batch collapse/expand or toggle magnifier. */
    svg.querySelectorAll('.dag-toolbar-item').forEach(el => {
        el.addEventListener('mousedown', (e) => { if (e.button === 0) e.stopPropagation(); });
        el.addEventListener('click', (e) => {
            e.stopPropagation();
            if (el.classList.contains('disabled')) return;
            if (el.id === 'collapse-all-' + config.id_prefix) {
                collapseAll();
            } else if (el.id === 'expand-all-' + config.id_prefix) {
                expandAll();
            }
        });
    });

    // Compute the DOM nesting depth of a container relative to the canvas group
    function getContainerDepth(el) {
        let depth = 0;
        let cur = el.parentElement;
        while (cur && cur !== canvasGroup) {
            if (cur.classList.contains('taskgroup') || cur.classList.contains('parallel-dag')) {
                depth++;
            }
            cur = cur.parentElement;
        }
        return depth;
    }

    // Collapse all containers from deep/nested up to canvas-level.
    // Uses fast 30ms animation duration; serialized via opInProgress.
    function collapseAll() {
        if (opInProgress) return;
        opInProgress = true;

        const btns = Array.from(canvasGroup.querySelectorAll('.collapse-expand-btn'))
            .filter(b => b.textContent === '-' && b.dataset.collapsing !== 'true');

        // Sort deepest first
        btns.sort((a, b) => {
            const cA = a.closest('.taskgroup, .parallel-dag');
            const cB = b.closest('.taskgroup, .parallel-dag');
            return getContainerDepth(cB) - getContainerDepth(cA);
        });

        let i = 0;
        function next() {
            if (i >= btns.length) {
                opInProgress = false;
                updateToolbarState();
                return;
            }
            const btn = btns[i++];
            ctx.collapseElement(btn, next, 30);
        }
        next();
    }

    // Expand all containers from canvas-level down to deep/nested.
    // Uses fast 30ms animation duration; serialized via opInProgress.
    function expandAll() {
        if (opInProgress) return;
        opInProgress = true;

        const btns = Array.from(canvasGroup.querySelectorAll('.collapse-expand-btn'))
            .filter(b => b.textContent === '+' && b.dataset.expanding !== 'true');

        // Sort shallowest first
        btns.sort((a, b) => {
            const cA = a.closest('.taskgroup, .parallel-dag');
            const cB = b.closest('.taskgroup, .parallel-dag');
            return getContainerDepth(cA) - getContainerDepth(cB);
        });

        let i = 0;
        function next() {
            if (i >= btns.length) {
                opInProgress = false;
                updateToolbarState();
                return;
            }
            const btn = btns[i++];
            ctx.expandElement(btn, next, 30);
        }
        next();
    }

    // Update toolbar state: hide collapse/expand if no containers exist,
    // disable buttons if all containers share the same state.
    function updateToolbarState() {
        const containers = canvasGroup.querySelectorAll('.taskgroup, .parallel-dag');
        const collapseExpandGroup = document.getElementById('collapse-expand-group-' + config.id_prefix);
        const collapseAllItem = document.getElementById('collapse-all-' + config.id_prefix);
        const expandAllItem = document.getElementById('expand-all-' + config.id_prefix);

        if (containers.length === 0) {
            if (collapseExpandGroup) collapseExpandGroup.style.display = 'none';
            return;
        }

        if (collapseExpandGroup) collapseExpandGroup.style.display = '';

        let allCollapsed = true;
        let allExpanded = true;

        containers.forEach(c => {
            const btn = c.querySelector(':scope > foreignObject.collapse-expand-fo > div.collapse-expand-btn');
            if (btn) {
                if (btn.dataset.collapsed === 'true') {
                    allExpanded = false;
                } else {
                    allCollapsed = false;
                }
            }
        });

        if (collapseAllItem) {
            collapseAllItem.classList.toggle('disabled', allCollapsed);
        }
        if (expandAllItem) {
            expandAllItem.classList.toggle('disabled', allExpanded);
        }
    }

    updateToolbarState();
}
