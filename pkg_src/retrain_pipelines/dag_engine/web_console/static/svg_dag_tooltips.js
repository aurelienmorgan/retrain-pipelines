/**
 * @file svg_dag_tooltips.js
 * @description Hover-show / leave-hide for node and taskgroup tooltips via DOM iteration.
 * Handles ESC keypress to hide all visible tooltips.
 * Exposes: None.
 * Dependencies: Core (ctx.svg), Magnifier (ctx.magnifierState at runtime).
 */
function initDagTooltips(ctx) {
    const { svg } = ctx;

    // Escape all visible tooltips except the given one, marking their
    // anchor elements as escaped so they do not reappear on re-hover
    function escapeAllVisibleTooltipsExcept(exceptId) {
        svg.querySelectorAll('[id^="tooltip-"]').forEach(t => {
            if (t.id !== exceptId && t.style.visibility === 'visible') {
                const id = t.id.replace('tooltip-node-', '').replace('tooltip-taskgroup-', '');
                const target = svg.querySelector('#node-' + id) || svg.querySelector('#taskgroup-' + id);
                if (target) {
                    target.dataset.tooltipEscaped = 'true';
                }
                t.style.visibility = 'hidden';
            }
        });
    }

    // Node tooltips
    svg.querySelectorAll('[id^="tooltip-node-"]').forEach(tooltip => {
        const id = tooltip.id.replace('tooltip-node-', '');
        const nodeEl = svg.querySelector('#node-' + id);
        if (!nodeEl) return;

        let hoverAnchor = false;
        let hoverTooltip = false;

        const updateVisibility = () => {
            if (nodeEl.dataset.tooltipEscaped === 'true') {
                tooltip.style.visibility = 'hidden';
                return;
            }
            tooltip.style.visibility = (hoverAnchor || hoverTooltip) ? 'visible' : 'hidden';
        };

        nodeEl.addEventListener('mouseenter', () => {
            escapeAllVisibleTooltipsExcept(tooltip.id);
            hoverAnchor = true;
            updateVisibility();
        });

        nodeEl.addEventListener('mouseleave', () => {
            hoverAnchor = false;
            nodeEl.dataset.tooltipEscaped = 'false';
            setTimeout(updateVisibility, 10);
        });

        tooltip.addEventListener('mouseenter', () => {
            hoverTooltip = true;
            updateVisibility();
        });

        tooltip.addEventListener('mouseleave', () => {
            hoverTooltip = false;
            updateVisibility();
        });

        tooltip.style.pointerEvents = 'auto';
    });

    // Taskgroup tooltips
    svg.querySelectorAll('[id^="tooltip-taskgroup-"]').forEach(tooltip => {
        const id = tooltip.id.replace('tooltip-taskgroup-', '');
        const taskgroupEl = svg.querySelector('#taskgroup-' + id);
        if (!taskgroupEl) return;

        let hoverAnchor = false;
        let hoverTooltip = false;

        const updateVisibility = () => {
            if (taskgroupEl.dataset.tooltipEscaped === 'true') {
                tooltip.style.visibility = 'hidden';
                return;
            }
            tooltip.style.visibility = (hoverAnchor || hoverTooltip) ? 'visible' : 'hidden';
        };

        taskgroupEl.addEventListener('mouseenter', () => {
            escapeAllVisibleTooltipsExcept(tooltip.id);
            hoverAnchor = true;
            updateVisibility();
        });

        taskgroupEl.addEventListener('mouseleave', (e) => {
            hoverAnchor = false;
            let isMovingInside = false;
            let related = e.relatedTarget;

            while (related && related !== document) {
                if (related.id && (related.id.startsWith('tooltip-node-') || related.id.startsWith('tooltip-taskgroup-'))) {
                    let targetId = related.id.replace('tooltip-node-', '').replace('tooltip-taskgroup-', '');
                    let targetEl = svg.querySelector('#node-' + targetId) || svg.querySelector('#taskgroup-' + targetId);
                    if (targetEl) {
                        let cur = targetEl;
                        while (cur && cur !== document) {
                            if (cur === taskgroupEl) {
                                isMovingInside = true;
                                break;
                            }
                            cur = cur.parentNode;
                        }
                    }
                    break;
                }
                related = related.parentNode;
            }

            if (!isMovingInside) {
                taskgroupEl.dataset.tooltipEscaped = 'false';
            }
            setTimeout(updateVisibility, 10);
        });

        tooltip.addEventListener('mouseenter', () => {
            hoverTooltip = true;
            updateVisibility();
        });

        tooltip.addEventListener('mouseleave', () => {
            hoverTooltip = false;
            updateVisibility();
        });

        tooltip.style.pointerEvents = 'auto';
    });

    // Hide all tooltips on ESC keypress and mark the target as escaped
    // Use capture phase to robustly intercept ESC in notebook environments
    window.addEventListener('keydown', (e) => {
        if (e.key === 'Escape') {
            let tooltipWasVisible = false;
            svg.querySelectorAll('[id^="tooltip-"]').forEach(t => {
                if (t.style.visibility === 'visible') {
                    tooltipWasVisible = true;
                    const id = t.id.replace('tooltip-node-', '').replace('tooltip-taskgroup-', '');
                    const target = svg.querySelector('#node-' + id) || svg.querySelector('#taskgroup-' + id);
                    if (target) {
                        target.dataset.tooltipEscaped = 'true';
                    }
                }
                t.style.visibility = 'hidden';
            });

            const magnifierState = ctx.magnifierState;
            if (magnifierState && magnifierState.isActive) {
                // Double-ESC hides the magnifier, but ONLY if no tooltip was visible
                // to consume the first ESC press.
                if (!tooltipWasVisible) {
                    const now = Date.now();
                    if (now - magnifierState.lastEscTime < 500) {
                        magnifierState.hide();
                        magnifierState.lastEscTime = 0;
                    } else {
                        magnifierState.lastEscTime = now;
                    }
                } else {
                    // A tooltip was visible; reset the double-ESC sequence
                    magnifierState.lastEscTime = 0;
                }
                /* repaint the magnifier if showing */
                magnifierState.updateContent();
            }
        }
    }, { capture: true });
}
