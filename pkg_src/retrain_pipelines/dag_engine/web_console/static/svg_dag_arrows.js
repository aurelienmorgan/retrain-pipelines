/**
 * @file svg_dag_arrows.js
 * @description Recomputes <line> endpoints after any element moves or resizes.
 * Exposes: updateLines.
 * Dependencies: Core (ctx.getOffset, ctx.parseTranslate, ctx.splitIds, ctx.config).
 */
function initDagArrows(ctx) {
    const { svg, canvasGroup, config } = ctx;

    ctx.updateLines = function(elementId) {
        // Helper to get position and size info of an element relative to a parent coordinate space
        function getLocalCoords(el, parentEl) {
            let localX = 0, localY = 0, width = 0, height = 0;

            if (el.classList.contains('node')) {
                const off = ctx.getOffset(el, parentEl);
                localX = off.x;
                localY = off.y;
                width = config.nodeWidth;
                height = config.nodeHeight;
            } else if (el.classList.contains('taskgroup') || el.classList.contains('parallel-dag')) {
                const rect = el.querySelector(':scope > rect.taskgroup-container')
                    || el.querySelector(':scope > rect.parallel-container');
                if (!rect) return null;
                const off = ctx.getOffset(el, parentEl);
                localX = parseFloat(rect.getAttribute('x')) + off.x;
                localY = parseFloat(rect.getAttribute('y')) + off.y;
                width = parseFloat(rect.getAttribute('width'));
                height = parseFloat(rect.getAttribute('height'));
            } else {
                return null;
            }

            return { x: localX, y: localY, width: width, height: height };
        }

        // Computes the horizontal attachment X for arrows: always the rect center
        // (middle of the bounding border), resolved into the parent's local
        // coordinate space. For taskgroups, dataset.mainChainX stores the same
        // rect-center value at render time and is kept in sync by
        // updateTaskgroupBounds; the clamp is a defensive no-op.
        function getLocalAttachX(el, info, parentEl) {
            let x = info.x + info.width / 2;
            if (el.classList.contains('taskgroup') && el.dataset.mainChainX !== undefined) {
                const off = ctx.getOffset(el, parentEl);
                x = parseFloat(el.dataset.mainChainX) + off.x;
                x = Math.max(info.x, Math.min(info.x + info.width, x));
            }
            return x;
        }

        const sourcePrefixes = [
            { prefix: `arrow-${elementId}-`, el: document.getElementById('taskgroup-' + elementId) || document.getElementById('node-' + elementId) || document.getElementById(elementId) },
            { prefix: `arrow-parallel-${elementId}-`, el: document.getElementById('parallel-' + elementId) }
        ];

        sourcePrefixes.forEach(src => {
            if (!src.el) return;

            canvasGroup.querySelectorAll(`line[id^="${src.prefix}"]`).forEach(line => {
                const parentEl = line.parentElement;
                const sourceInfo = getLocalCoords(src.el, parentEl);
                if (!sourceInfo) return;

                const targetId = line.id.slice(src.prefix.length);
                const targetEl =
                    document.getElementById('taskgroup-' + targetId) ||
                    document.getElementById('node-' + targetId) ||
                    document.getElementById(targetId);

                if (!targetEl) return;

                const targetInfo = getLocalCoords(targetEl, parentEl);
                if (!targetInfo) return;

                line.setAttribute('x1', getLocalAttachX(src.el, sourceInfo, parentEl));
                line.setAttribute('y1', sourceInfo.y + sourceInfo.height);
                line.setAttribute('x2', getLocalAttachX(targetEl, targetInfo, parentEl));
                line.setAttribute('y2', targetInfo.y);
            });
        });

        const targetSuffixes = [
            { suffix: `-${elementId}`, el: document.getElementById('taskgroup-' + elementId) || document.getElementById('node-' + elementId) || document.getElementById(elementId) },
            { suffix: `-parallel-${elementId}`, el: document.getElementById('parallel-' + elementId) }
        ];

        targetSuffixes.forEach(tgt => {
            if (!tgt.el) return;

            canvasGroup.querySelectorAll(`line[id$="${tgt.suffix}"]`).forEach(line => {
                const parentEl = line.parentElement;
                const targetInfo = getLocalCoords(tgt.el, parentEl);
                if (!targetInfo) return;

                const fullId = line.id.slice('arrow-'.length, -tgt.suffix.length);
                const srcEl =
                    document.getElementById('taskgroup-' + fullId) ||
                    document.getElementById('node-' + fullId) ||
                    document.getElementById(fullId);

                if (!srcEl) return;

                const sourceInfo = getLocalCoords(srcEl, parentEl);
                if (!sourceInfo) return;

                line.setAttribute('x1', getLocalAttachX(srcEl, sourceInfo, parentEl));
                line.setAttribute('y1', sourceInfo.y + sourceInfo.height);
                line.setAttribute('x2', getLocalAttachX(tgt.el, targetInfo, parentEl));
                line.setAttribute('y2', targetInfo.y);
            });
        });
    };
}
