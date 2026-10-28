/**
 * @file svg_dag_bounds.js
 * @description Recomputes bounding rectangles for taskgroups and parallel regions.
 * Exposes: updateTaskgroupBounds, updateParallelBounds, 
 * findAncestorParallelsForNode, findAncestorParallelsForTaskgroup.
 * Dependencies: Core (ctx.getOffset, ctx.splitIds, ctx.config).
 */
function initDagBounds(ctx) {
    const { canvasGroup, config } = ctx;

    ctx.updateTaskgroupBounds = function(taskgroupId) {
        const tgEl = document.getElementById(taskgroupId);
        if (!tgEl) return;
        // Preserve collapsed bounds: if already collapsed and not expanding, reflowing children must not alter bounds
        const tgBtn = tgEl.querySelector(':scope > foreignObject.collapse-expand-fo > div.collapse-expand-btn');
        if (tgBtn && tgBtn.dataset.collapsed === 'true' && tgBtn.dataset.expanding !== 'true') return;
        const off = ctx.getOffset(tgEl, canvasGroup); // Absolute offset of the taskgroup
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

        const nodes = ctx.splitIds(tgEl.dataset.nodes);
        nodes.forEach(nodeId => {
            const nodeEl = document.getElementById(`node-${nodeId}`);
            if (!nodeEl) return;
            const abs = ctx.getOffset(nodeEl, canvasGroup); // Absolute offset of node
            minX = Math.min(minX, abs.x);
            maxX = Math.max(maxX, abs.x + config.nodeWidth);
            minY = Math.min(minY, abs.y);
            maxY = Math.max(maxY, abs.y + config.nodeHeight);
        });

        const nestedTgs = ctx.splitIds(tgEl.dataset.nestedTaskgroups);
        nestedTgs.forEach(childTgId => {
            const childTgEl = document.getElementById(`taskgroup-${childTgId}`);
            if (!childTgEl) return;
            const childRect = childTgEl.querySelector(':scope > rect.taskgroup-container');
            if (!childRect) return;
            const childAbs = ctx.getOffset(childTgEl, canvasGroup); // Absolute offset of child taskgroup
            const cx = childAbs.x + parseFloat(childRect.getAttribute('x'));
            const cy = childAbs.y + parseFloat(childRect.getAttribute('y'));
            const cw = parseFloat(childRect.getAttribute('width'));
            const ch = parseFloat(childRect.getAttribute('height'));

            const child_min_x = cx + config.tgMarginX;
            const child_max_x = cx + cw - config.tgMarginX;
            const child_min_y = cy + config.tgLabelHeight + config.tgLabelMarginTop;
            const child_max_y = ch - config.tgLabelHeight - config.tgLabelMarginTop - config.tgMarginBottom + child_min_y;

            minX = Math.min(minX, child_min_x - config.tgMarginX);
            maxX = Math.max(maxX, child_max_x + config.tgMarginX);
            minY = Math.min(minY, child_min_y - (config.tgLabelHeight + config.tgLabelMarginTop));
            maxY = Math.max(maxY, child_max_y + config.tgMarginBottom);
        });

        if (minX === Infinity) return;

        const offsetX = minX - config.tgMarginX;
        const offsetY = minY - (config.tgLabelHeight + config.tgLabelMarginTop);
        const width = maxX - minX + 2 * config.tgMarginX;
        const height = maxY - minY + config.tgLabelHeight + config.tgLabelMarginTop + config.tgMarginBottom;

        // Rect attributes are local to the taskgroup: subtract the taskgroup's absolute offset
        // Safely target base rect and underlays without affecting nested parallel containers
        tgEl.querySelectorAll(':scope > rect, :scope > g.taskgroup-underlay-group > rect').forEach(r => {
            r.setAttribute('x', (offsetX - off.x).toFixed(3));
            r.setAttribute('y', (offsetY - off.y).toFixed(3));
            r.setAttribute('width', width.toFixed(3));
            r.setAttribute('height', height.toFixed(3));
        });

        tgEl.querySelectorAll(':scope > text').forEach(t => {
            t.setAttribute('x', (offsetX + width / 2 - off.x).toFixed(3));
            t.setAttribute('y', (minY - 10 - off.y).toFixed(3));
        });

        const tgToggleBtnFo = tgEl.querySelector(':scope > foreignObject.collapse-expand-fo');
        if (tgToggleBtnFo) {
            tgToggleBtnFo.setAttribute('x', (offsetX - off.x).toFixed(3));
            tgToggleBtnFo.setAttribute('y', (offsetY - off.y).toFixed(3));
            tgToggleBtnFo.setAttribute('width', width.toFixed(3));
            tgToggleBtnFo.setAttribute('height', height.toFixed(3));
        }

        const tgTooltip = document.getElementById('tooltip-' + taskgroupId);
        if (tgTooltip) {
            tgTooltip.setAttribute('x', minX.toFixed(3));
            tgTooltip.setAttribute('y', (offsetY + config.tooltipYOffset).toFixed(3));
        }

        // mainChainX stores the rect-center X (middle of the bounding border)
        // in the taskgroup's local coordinate space. This is the in/out arrow
        // attachment axis, consistent with the render-time center_x.
        tgEl.dataset.mainChainX = ((offsetX - off.x) + width / 2).toFixed(3);
    };

    ctx.updateParallelBounds = function(parallelId) {
        const pEl = document.getElementById('parallel-' + parallelId);
        if (!pEl) return;
        // Preserve collapsed bounds: if already collapsed and not expanding, reflowing children must not alter bounds
        const pBtn = pEl.querySelector(':scope > foreignObject.collapse-expand-fo > div.collapse-expand-btn');
        if (pBtn && pBtn.dataset.collapsed === 'true' && pBtn.dataset.expanding !== 'true') return;
        const mainRect = pEl.querySelector(':scope > rect.parallel-container');
        if (!mainRect) return;

        const off = ctx.getOffset(pEl, canvasGroup); // Absolute offset of the parallel container
        let minX = Infinity, maxX = -Infinity, minY = Infinity, maxY = -Infinity;

        const directNodes = ctx.splitIds(pEl.dataset.directNodes || pEl.dataset.containedNodes);
        const directTgs = ctx.splitIds(pEl.dataset.directTaskgroups || pEl.dataset.containedTaskgroups);
        const childParallels = ctx.splitIds(pEl.dataset.childParallels || pEl.dataset.containedParallels);

        const parallelLevel = parseFloat(pEl.dataset.parallelLevel || 1);
        const tgsInChildParallels = new Set();
        childParallels.forEach(npId => {
            const npEl = document.getElementById('parallel-' + npId);
            if (npEl) {
                ctx.splitIds(npEl.dataset.containedTaskgroups).forEach(t => tgsInChildParallels.add(t));
            }
        });

        directNodes.forEach(nodeId => {
            const nodeEl = document.getElementById('node-' + nodeId);
            if (!nodeEl) return;
            const nodeUnderlayOffset = 3 * nodeEl.querySelectorAll(':scope > g').length;
            const abs = ctx.getOffset(nodeEl, canvasGroup);
            minX = Math.min(minX, abs.x - nodeUnderlayOffset);
            maxX = Math.max(maxX, abs.x + config.nodeWidth + nodeUnderlayOffset);
            minY = Math.min(minY, abs.y);
            maxY = Math.max(maxY, abs.y + config.nodeHeight + nodeUnderlayOffset);
        });

        directTgs.filter(tgId => !tgsInChildParallels.has(tgId)).forEach(tgId => {
            const tgEl = document.getElementById('taskgroup-' + tgId);
            if (!tgEl) return;
            const tgRect = tgEl.querySelector(':scope > rect.taskgroup-container');
            if (!tgRect) return;
            const tgAbs = ctx.getOffset(tgEl, canvasGroup);
            const tx = tgAbs.x + parseFloat(tgRect.getAttribute('x'));
            const ty = tgAbs.y + parseFloat(tgRect.getAttribute('y'));
            const tw = parseFloat(tgRect.getAttribute('width'));
            const th = parseFloat(tgRect.getAttribute('height'));
            const tgUnderlayOffset = 3.5 * parseFloat(tgEl.dataset.maxParallelDepth || 0);
            minX = Math.min(minX, tx - tgUnderlayOffset);
            maxX = Math.max(maxX, tx + tw + tgUnderlayOffset);
            minY = Math.min(minY, ty);
            maxY = Math.max(maxY, ty + th);
        });

        childParallels.forEach(npId => {
            const npEl = document.getElementById('parallel-' + npId);
            if (!npEl) return;
            const npRect = npEl.querySelector(':scope > rect.parallel-container');
            if (!npRect) return;
            const npAbs = ctx.getOffset(npEl, canvasGroup);
            const nx = npAbs.x + parseFloat(npRect.getAttribute('x'));
            const ny = npAbs.y + parseFloat(npRect.getAttribute('y'));
            const nw = parseFloat(npRect.getAttribute('width'));
            const nh = parseFloat(npRect.getAttribute('height'));
            const npLevelMarginX = parseFloat(npEl.dataset.levelMarginX || config.parallelMarginX);
            minX = Math.min(minX, nx + npLevelMarginX);
            maxX = Math.max(maxX, nx + nw - npLevelMarginX);
            minY = Math.min(minY, ny + config.parallelMarginY);
            maxY = Math.max(maxY, ny + nh - config.parallelMarginY);
        });

        if (minX === Infinity) return;

        const levelMarginX = parseFloat(pEl.dataset.levelMarginX || config.parallelMarginX);
        mainRect.setAttribute('x', (minX - levelMarginX - off.x).toFixed(3));
        mainRect.setAttribute('y', (minY - config.parallelMarginY - off.y).toFixed(3));
        mainRect.setAttribute('width', ((maxX - minX) + 2 * levelMarginX).toFixed(3));
        mainRect.setAttribute('height', ((maxY - minY) + 2 * config.parallelMarginY).toFixed(3));

        const pToggleBtnFo = pEl.querySelector(':scope > foreignObject.collapse-expand-fo');
        if (pToggleBtnFo) {
            pToggleBtnFo.setAttribute('x', mainRect.getAttribute('x'));
            pToggleBtnFo.setAttribute('y', mainRect.getAttribute('y'));
            pToggleBtnFo.setAttribute('width', mainRect.getAttribute('width'));
            pToggleBtnFo.setAttribute('height', mainRect.getAttribute('height'));
        }
    };

    // Sort parallel IDs from innermost (most ancestors) to outermost
    function sortByNestingDepthDesc(parallelIds) {
        return parallelIds.sort((a, b) => {
            let depthA = 0, depthB = 0;
            let cur = a;
            while (ctx.parallelParentMap[cur]) { depthA++; cur = ctx.parallelParentMap[cur]; }
            cur = b;
            while (ctx.parallelParentMap[cur]) { depthB++; cur = ctx.parallelParentMap[cur]; }
            return depthB - depthA;
        });
    }

    // Find all parallels whose data-contained-nodes includes the given node
    ctx.findAncestorParallelsForNode = function(nodeUuid) {
        const containing = [];
        canvasGroup.querySelectorAll('.parallel-dag').forEach(pEl => {
            const contained = pEl.dataset.containedNodes
                ? pEl.dataset.containedNodes.split(',').filter(Boolean)
                : [];
            if (contained.includes(nodeUuid)) {
                containing.push(pEl.id.replace(/^parallel-/, ''));
            }
        });
        return sortByNestingDepthDesc(containing);
    };

    // Find all parallels whose data-contained-taskgroups includes the given taskgroup
    ctx.findAncestorParallelsForTaskgroup = function(tgUuid) {
        const containing = [];
        canvasGroup.querySelectorAll('.parallel-dag').forEach(pEl => {
            const containedTgs = pEl.dataset.containedTaskgroups
                ? pEl.dataset.containedTaskgroups.split(',').filter(Boolean)
                : [];
            if (containedTgs.includes(tgUuid)) {
                containing.push(pEl.id.replace(/^parallel-/, ''));
            }
        });
        return sortByNestingDepthDesc(containing);
    };
}
