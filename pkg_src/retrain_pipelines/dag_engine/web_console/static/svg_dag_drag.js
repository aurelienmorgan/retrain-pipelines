/**
 * @file svg_dag_drag.js
 * @description Element dragging (nodes, taskgroups, parallel regions) and tooltip sync.
 * Exposes: None.
 * Dependencies: Core (ctx.canvasGroup, ctx.config, ctx.state, ctx.updateTransform, 
 * ctx.getMouseSVGCoords, ctx.parseTranslate, ctx.getOffset, ctx.splitIds), 
 * Bounds (ctx.updateTaskgroupBounds, ctx.updateParallelBounds, 
 * ctx.findAncestorParallelsForNode, ctx.findAncestorParallelsForTaskgroup), 
 * Arrows (ctx.updateLines).
 */
function initDagDrag(ctx) {
    const { svg, canvasGroup, config } = ctx;

    let selected = null;
    let selectedTaskgroup = null;
    let selectedParallel = null;
    let startMouse = null;               // Mouse coordinates at drag start, used to calculate drag deltas (dx, dy) on mousemove
    let startPos = null;                 // Initial translate(x, y) of the dragged node, ensuring new positions are absolute rather than compounding deltas
    let taskgroupStartPositions = null;  // Taskgroup drag state: initial transform, tooltips (which live outside the <g>), and member node IDs to update attached lines
    let parallelStartPositions = null;   // Parallel drag state: initial transform, tooltips, and IDs of contained nodes/taskgroups/nested parallels to update attached lines

    // Collect tooltips (with start positions) of a taskgroup, its nested taskgroups
    // and all their member nodes: they live outside the group in tooltips-group
    // and must be moved manually when the group is dragged
    function collectTaskgroupTooltips(tgEl, tooltips, seen) {
        const tgTooltip = document.getElementById('tooltip-' + tgEl.id);
        if (tgTooltip && !seen.has(tgTooltip.id)) {
            seen.add(tgTooltip.id);
            tooltips.push({ el: tgTooltip, x: parseFloat(tgTooltip.getAttribute('x')), y: parseFloat(tgTooltip.getAttribute('y')) });
        }
        ctx.splitIds(tgEl.dataset.nodes).forEach(nodeId => {
            const nTooltip = document.getElementById('tooltip-node-' + nodeId);
            if (nTooltip && !seen.has(nTooltip.id)) {
                seen.add(nTooltip.id);
                tooltips.push({ el: nTooltip, x: parseFloat(nTooltip.getAttribute('x')), y: parseFloat(nTooltip.getAttribute('y')) });
            }
        });
        ctx.splitIds(tgEl.dataset.nestedTaskgroups).forEach(nestedTgId => {
            const nestedTgEl = document.getElementById('taskgroup-' + nestedTgId);
            if (nestedTgEl) collectTaskgroupTooltips(nestedTgEl, tooltips, seen);
        });
    }

    // Collect tooltips of all nodes and taskgroups contained in a parallel region
    function collectParallelTooltips(pEl, tooltips, seen) {
        ctx.splitIds(pEl.dataset.containedNodes).forEach(nodeId => {
            const nTooltip = document.getElementById('tooltip-node-' + nodeId);
            if (nTooltip && !seen.has(nTooltip.id)) {
                seen.add(nTooltip.id);
                tooltips.push({ el: nTooltip, x: parseFloat(nTooltip.getAttribute('x')), y: parseFloat(nTooltip.getAttribute('y')) });
            }
        });
        ctx.splitIds(pEl.dataset.containedTaskgroups).forEach(tgId => {
            const tgEl = document.getElementById('taskgroup-' + tgId);
            if (tgEl) collectTaskgroupTooltips(tgEl, tooltips, seen);
        });
    }

    // Collect the ids of all member nodes of a taskgroup and its nested taskgroups
    function collectTaskgroupNodeIds(tgEl, nodeIds) {
        ctx.splitIds(tgEl.dataset.nodes).forEach(nodeId => {
            if (!nodeIds.includes(nodeId)) nodeIds.push(nodeId);
        });
        ctx.splitIds(tgEl.dataset.nestedTaskgroups).forEach(nestedTgId => {
            const nestedTgEl = document.getElementById('taskgroup-' + nestedTgId);
            if (nestedTgEl) collectTaskgroupNodeIds(nestedTgEl, nodeIds);
        });
        return nodeIds;
    }

    // Resolve the outermost (top-level) taskgroup ancestor of a taskgroup element:
    // only top-level taskgroups are referenced by arrow ids
    function getTopLevelTaskgroupId(tgEl) {
        let cur = tgEl;
        while (cur) {
            const parentTgEl = cur.dataset.parentTaskgroup
                ? document.getElementById('taskgroup-' + cur.dataset.parentTaskgroup)
                : null;
            if (!parentTgEl) break;
            cur = parentTgEl;
        }
        return cur ? cur.id.replace(/^taskgroup-/, '') : null;
    }

    // Dragging logic
    canvasGroup.querySelectorAll('.node').forEach(node => {
        node.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            selected = node;
            startMouse = ctx.getMouseSVGCoords(e);
            startPos = ctx.parseTranslate(node);

            selected.style.cursor = 'grabbing';
            e.preventDefault();
            e.stopPropagation();
        });
    });

    canvasGroup.querySelectorAll('.taskgroup-container').forEach(container => {
        container.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            selectedTaskgroup = container.closest('.taskgroup');
            startMouse = ctx.getMouseSVGCoords(e);
            taskgroupStartPositions = {
                x: ctx.parseTranslate(selectedTaskgroup).x,
                y: ctx.parseTranslate(selectedTaskgroup).y,
                tooltips: [],
                nodeIds: []
            };
            collectTaskgroupTooltips(selectedTaskgroup, taskgroupStartPositions.tooltips, new Set());
            collectTaskgroupNodeIds(selectedTaskgroup, taskgroupStartPositions.nodeIds);

            selectedTaskgroup.style.cursor = 'grabbing';
            e.preventDefault();
            e.stopPropagation();
        });
    });

    canvasGroup.querySelectorAll('.parallel-container').forEach(container => {
        container.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            selectedParallel = container.closest('.parallel-dag');
            startMouse = ctx.getMouseSVGCoords(e);
            parallelStartPositions = {
                x: ctx.parseTranslate(selectedParallel).x,
                y: ctx.parseTranslate(selectedParallel).y,
                tooltips: [],
                nodeIds: ctx.splitIds(selectedParallel.dataset.containedNodes),
                tgIds: [],
                pIds: ctx.splitIds(selectedParallel.dataset.containedParallels)
            };
            collectParallelTooltips(selectedParallel, parallelStartPositions.tooltips, new Set());
            ctx.splitIds(selectedParallel.dataset.containedTaskgroups).forEach(tgId => {
                const tgEl = document.getElementById('taskgroup-' + tgId);
                const topLevelTgId = tgEl ? getTopLevelTaskgroupId(tgEl) : null;
                if (topLevelTgId && !parallelStartPositions.tgIds.includes(topLevelTgId)) {
                    parallelStartPositions.tgIds.push(topLevelTgId);
                }
            });

            selectedParallel.style.cursor = 'grabbing';
            e.preventDefault();
            e.stopPropagation();
        });
    });

    window.addEventListener('mousemove', (e) => {
        if (selected) {
            const coords = ctx.getMouseSVGCoords(e);
            const dx = coords.x - startMouse.x;
            const dy = coords.y - startMouse.y;
            const newX = startPos.x + dx;
            const newY = startPos.y + dy;

            selected.setAttribute('transform', `translate(${newX},${newY})`);
            const nodeUuid = selected.id.replace(/^node-/, '');
            ctx.updateLines(nodeUuid);

            const tooltip = document.getElementById('tooltip-node-' + nodeUuid);
            if (tooltip) {
                const abs = ctx.getOffset(selected, canvasGroup);
                tooltip.setAttribute('x', abs.x);
                tooltip.setAttribute('y', abs.y + config.tooltipYOffset);
            }

            let currentTgId = selected.dataset.taskgroup;
            while (currentTgId) {
                const currentTgEl = document.getElementById('taskgroup-' + currentTgId);
                if (!currentTgEl) break;
                ctx.updateTaskgroupBounds('taskgroup-' + currentTgId);
                ctx.updateLines(currentTgId);
                currentTgId = currentTgEl.dataset.parentTaskgroup || '';
            }

            ctx.findAncestorParallelsForNode(nodeUuid).forEach(pId => {
                ctx.updateParallelBounds(pId);
                ctx.updateLines(pId);
            });
        } else if (selectedTaskgroup) {
            const coords = ctx.getMouseSVGCoords(e);
            const dx = coords.x - startMouse.x;
            const dy = coords.y - startMouse.y;

            selectedTaskgroup.setAttribute('transform',
                `translate(${taskgroupStartPositions.x + dx},${taskgroupStartPositions.y + dy})`);

            taskgroupStartPositions.tooltips.forEach(t => {
                t.el.setAttribute('x', t.x + dx);
                t.el.setAttribute('y', t.y + dy);
            });

            const draggedTgUuid = selectedTaskgroup.id.replace(/^taskgroup-/, '');
            ctx.updateLines(draggedTgUuid);
            taskgroupStartPositions.nodeIds.forEach(nodeId => ctx.updateLines(nodeId));

            let currentTgEl = selectedTaskgroup;
            while (currentTgEl) {
                const parentTgId = currentTgEl.dataset.parentTaskgroup;
                if (!parentTgId) break;
                const parentTgEl = document.getElementById('taskgroup-' + parentTgId);
                if (!parentTgEl) break;
                ctx.updateTaskgroupBounds('taskgroup-' + parentTgId);
                ctx.updateLines(parentTgId);
                currentTgEl = parentTgEl;
            }

            ctx.findAncestorParallelsForTaskgroup(draggedTgUuid).forEach(pId => {
                ctx.updateParallelBounds(pId);
                ctx.updateLines(pId);
            });
        } else if (selectedParallel) {
            const coords = ctx.getMouseSVGCoords(e);
            const dx = coords.x - startMouse.x;
            const dy = coords.y - startMouse.y;

            selectedParallel.setAttribute('transform',
                `translate(${parallelStartPositions.x + dx},${parallelStartPositions.y + dy})`);

            parallelStartPositions.tooltips.forEach(t => {
                t.el.setAttribute('x', t.x + dx);
                t.el.setAttribute('y', t.y + dy);
            });

            const draggedPUuid = selectedParallel.id.replace(/^parallel-/, '');
            ctx.updateLines(draggedPUuid);
            parallelStartPositions.pIds.forEach(pId => ctx.updateLines(pId));
            parallelStartPositions.tgIds.forEach(tgId => ctx.updateLines(tgId));
            parallelStartPositions.nodeIds.forEach(nodeId => ctx.updateLines(nodeId));

            let curParentP = ctx.parallelParentMap[draggedPUuid];
            while (curParentP) {
                ctx.updateParallelBounds(curParentP);
                ctx.updateLines(curParentP);
                curParentP = ctx.parallelParentMap[curParentP];
            }
        }
    });

    window.addEventListener('mouseup', () => {
        if (selected) selected.style.cursor = 'grab';
        if (selectedTaskgroup) selectedTaskgroup.style.cursor = 'grab';
        if (selectedParallel) selectedParallel.style.cursor = 'grab';
        selected = null;
        selectedTaskgroup = null;
        selectedParallel = null;
        startMouse = null;
        startPos = null;
        taskgroupStartPositions = null;
        parallelStartPositions = null;
    });
}
