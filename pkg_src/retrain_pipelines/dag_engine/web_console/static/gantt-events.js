
/* ****************************************
* Events on collapsible grouped table     *
* with a gantt-timeline (column &) object *
**************************************** */

function getGlobalObjByName(globalObjectName) {
    /* ***************************************************
    * Use to retrieve the gantt-timeline object by name  *
    * (must be global var in the html page scope).       *
    *************************************************** */
    const obj = globalThis[globalObjectName];
    if (!obj) {
      throw new Error(`Global object ${globalObjectName} not found`);
    }
    return obj;
}

function getMaxVisibleLevel(bodyRows) {
    /* **************************
    * table-wise, how deep is   *
    * the deepest visible group *
    * (collapsed or not).       *
    ************************** */
    let maxLevel = 0;
    bodyRows.forEach((tr) => {
        if (
            !tr.classList.contains("hidden") &&
            tr.classList.contains("group-header")
        ) {
            maxLevel = Math.max(
                maxLevel, parseInt(tr.dataset.level));
        }
    });
    return maxLevel;
}

function reorderGroupChildren(ganttTimelineObj, groupHeaderRow) {
    /*
     * Reorder the direct children of a group so the child with the earliest
     * start-timestamp comes first.
     *
     * The discrepancy, in one sentence:
     * ganttInsert computes the insert position of a newly-created taskgroup from
     * the timestamp of the first-arriving task of that taskgroup, and never
     * re-evaluates that position when a later-arriving task of the same taskgroup
     * turns out to have an earlier timestamp; _organize_tasks doesn't have this
     * problem because it sees all tasks up-front, sorted.
     *
     * Note:
     * Consider a parent taskgroup T with two nested taskgroups, TB and TA.
     * The parent also contains a standalone task. Each nested taskgroup has two
     * tasks. Their start timestamps (relative to a common base) are:
     *
     *     +------------+--------------------------+
     *     | task       | start_timestamp (ms)     |
     *     +------------+--------------------------+
     *     | standalone | ...5924                  |
     *     | TB1        | ...6145                  |
     *     | TB2        | ...6162                  |
     *     | TA1        | ...6149                  |
     *     | TA2        | ...6144                  |
     *     +------------+--------------------------+
     *
     * During live-stream, SSE fires on task INSERT, in DB insertion order:
     *
     *     +----+------------+--------+------------------------------------------------+
     *     | ## | task       | ts     | event                                          |
     *     +----+------------+--------+------------------------------------------------+
     *     | 1  | standalone | ...5924|                                                |
     *     | 2  | TB1        | ...6145| -> creates TB as a child of T                  |
     *     | 3  | TA1        | ...6149| -> creates TA as a child of T.                 |
     *     |    |            |        |    At creation, ganttInsert computes           |
     *     |    |            |        |    group_index by comparing TA1's ts (6149)    |
     *     |    |            |        |    against TB1's ts (6145). Since 6149 > 6145, |
     *     |    |            |        |    TA is placed after TB.                      |
     *     | 4  | TB2        | ...6162| -> inserted inside the already-existing TB     |
     *     | 5  | TA2        | ...6144| -> inserted inside the already-existing TA.    |
     *     |    |            |        |    TA2's ts (6144) is earlier than TB1's ts    |
     *     |    |            |        |    (6145), so TA's true earliest task is now   |
     *     |    |            |        |    6144 — earlier than TB's 6145. But the code |
     *     |    |            |        |    only re-sorts TA's children, never TA's     |
     *     |    |            |        |    position relative to TB inside T.           |
     *     +----+------------+--------+------------------------------------------------+
     *
     * Result: live-stream leaves TB (earliest 6145) before
     * TA (earliest 6144).
     * Expected: TA (earliest 6144) before TB (earliest 6145).
     *
     * This function walks every ancestor taskgroup of the inserted task
     * (deepest first) and re-sorts that taskgroup's direct children by their
     * earliest descendant start-timestamp, so that TA is moved before TB as
     * soon as TA2 arrives, matching the post-execution order produced by
     * _organize_tasks.
     *
     * Direct children are identified by data-path: their path is the parent's
     * path + "." + a single segment (no further dots). For each child the
     * earliest start-timestamp across itself and all its descendants is
     * computed; children are sorted by that value ascending, using the numeric
     * child index as a tiebreaker for equal timestamps.
     *
     * No-op when the group has fewer than 2 children or is already in the
     * correct order.
     */
    if (!groupHeaderRow) return;
    const table = ganttTimelineObj.table;
    const colIdx = ganttTimelineObj.timelineColumnIndex;
    const prefix = groupHeaderRow.dataset.path + ".";
    const children = Array.from(
        table.querySelectorAll(`tr[data-path^="${prefix}"]`)
    ).filter(r => r.dataset.path.substring(prefix.length).indexOf('.') === -1);
    if (children.length <= 1) return;
    const earliestTs = row => {
        let min = null;
        const check = r => {
            if (r.classList.contains("group-header")) return;
            const c = r.cells[colIdx];
            if (c && c.dataset.startTimestamp) {
                const t = Number(c.dataset.startTimestamp);
                if (!isNaN(t) && (min === null || t < min)) min = t;
            }
        };
        check(row);
        table.querySelectorAll(`tr[data-path^="${row.dataset.path}."]`).forEach(check);
        return min;
    };
    const groups = children.map(c => ({
        row: c,
        desc: Array.from(table.querySelectorAll(`tr[data-path^="${c.dataset.path}."]`)),
        ts: earliestTs(c),
        idx: c.dataset.path.substring(prefix.length)
    }));
    groups.sort((a, b) => {
        const ta = a.ts === null ? Number.MAX_SAFE_INTEGER : a.ts;
        const tb = b.ts === null ? Number.MAX_SAFE_INTEGER : b.ts;
        return ta !== tb ? ta - tb : Number(a.idx) - Number(b.idx);
    });
    if (groups.every((g, i) => g.row === children[i])) return;
    console.warn(
        "reorder existing group children",
        "groupHeader=", groupHeaderRow.dataset.name,
        "path=" + groupHeaderRow.dataset.path,
        "id=" + groupHeaderRow.dataset.id,
        "reordered:", groups.map(g => ({
            name: g.row.dataset.name,
            id: g.row.dataset.id,
            fromIdx: g.idx,
            toIdx: groups.indexOf(g),
            earliestTs: g.ts
        }))
    );
    groups.forEach((g, i) => {
        if (g.idx === String(i)) return;
        const oldP = prefix + g.idx + ".";
        const newP = prefix + i + ".";
        g.row.dataset.path = prefix + i;
        g.desc.forEach(d => { d.dataset.path = newP + d.dataset.path.substring(oldP.length); });
    });
    let ref = groupHeaderRow.nextElementSibling;
    while (ref && ref.dataset.path && ref.dataset.path.startsWith(prefix))
        ref = ref.nextElementSibling;
    const tbody = table.querySelector('tbody');
    table.querySelectorAll(`tr[data-path^="${prefix}"]`).forEach(r => r.remove());
    groups.forEach(g => {
        tbody.insertBefore(g.row, ref);
        g.desc.forEach(d => tbody.insertBefore(d, ref));
    });
}


////////////////////////////////////////////////////////

function isHexColor(color) {
    /* true if color is hex code (#abc, #aabbcc), false otherwise */
    const yesNo = /^#([0-9A-F]{3}|[0-9A-F]{6})$/i.test(color);
    return yesNo;
}

function rgbaToHex(color) {
    /* from rgba to hex-code (dropping the transparency factor) */
    const match =
        color.match(/rgba?\s*\(\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)(?:\s*,\s*([\d.]+))?\s*\)/i);
    if (!match) {
        console.error(color, "isn't a recognized rgba string");
        return null;
    }
    let [r, g, b] = [match[1], match[2], match[3]].map(x => parseInt(x, 10));
    const hexCode = '#' + [r, g, b].map(x => x.toString(16).padStart(2, '0')).join('');
    return hexCode;
}

function hexToRgba(hex, alpha) {
    /* ***************************
    * Helper function to convert *
    * hex to rgba with alpha.    *
    *************************** */
    hex = hex.replace(/^#/, '');
    if (hex.length === 3) {
        hex = hex.split('').map(c => c + c).join('');
    }
    if (hex.length !== 6) {
        throw new Error('Invalid hex color length');
    }
    const r = parseInt(hex.slice(0, 2), 16);
    const g = parseInt(hex.slice(2, 4), 16);
    const b = parseInt(hex.slice(4, 6), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

function generateGradient(backgroundColor) {
    /* **************************************
    * Helper function to generate gradient. *
    ************************************** */
    if (!isHexColor(backgroundColor))
        backgroundColor = rgbaToHex(backgroundColor);

    return `linear-gradient(180deg, 
            ${hexToRgba(backgroundColor, 0.52)} 0%,
            ${hexToRgba(backgroundColor, 0.38)} 10%,
            ${hexToRgba(backgroundColor, 0.32)} 15%,
            ${hexToRgba(backgroundColor, 0.52)} 20%,
            ${hexToRgba(backgroundColor, 0.78)} 40%, 
            ${hexToRgba(backgroundColor, 0.63)} 60%,
            ${hexToRgba(backgroundColor, 0.55)} 80%,
            ${hexToRgba(backgroundColor, 0.44)} 100%)`;
}

function trapezoidalLabel(
    textContent,
    color, backgroundColor, borderColor, underlayColor,
    flipped = false
) {
    /* ******************************************
    * html for formatted trapezoidal shaped div *
    ****************************************** */
    const gradient = generateGradient(backgroundColor);
    const borderRadius = flipped ?
        '7.5px 7.5px 6px 6px' : '6px 6px 7.5px 7.5px';
    // adjust for 3D tilting leaving
    // empty top and bottom space
    const correctionMargin = flipped ?
        '-3.75px 0 -4px' :
        '-4px 0 -1.5px'; 
    const rotation = flipped ?
        'rotateX(-35deg)' : 'rotateX(35deg)';
    const justifyContent = flipped ?
        'justify-content: center;' : '';

    return "" +
        `<div class="shaped-label" style="
            position: relative;
            min-width: 93.75px;
            width: fit-content;
            padding: 0 6px 0 4.5px;
            height: 30px;
            display: flex;
            align-items: center;
            ${justifyContent}
            line-height: normal;
            color: ${color};
            font-size: 12px;
            font-weight: normal;
            text-shadow: 0 0.5625px 1.5px rgba(0,0,0,0.5);
            margin: ${correctionMargin};
            z-index: 0;
        ">
            ${textContent}
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: ${gradient};
                border-radius: ${borderRadius};
                border: 1.5px solid ${borderColor};
                box-shadow: 
                  inset 0 4.6875px 3.75px -2.8125px rgba(255,255,255,0.6),
                  inset 0 -2.8125px 3.75px -1.875px rgba(0,0,0,0.2),
                  0 2.34375px 4.6875px rgba(0,0,0,0.5);
                transform: perspective(150px) ${rotation};
                z-index: -1;
            "></div>
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: ${underlayColor};
                border-radius: ${borderRadius};
                transform: perspective(150px) ${rotation};
                z-index: -2;
            "></div>
        </div>`;
}

function rectangularLabel(
    textContent,
    color, backgroundColor, borderColor, underlayColor
) {
    /* ******************************************
    * html for formatted rectangular-shaped div *
    ****************************************** */
    const gradient = generateGradient(backgroundColor);

    return "" +
        `<div class="shaped-label" style="
            position: relative;
            min-width: 93.75px;
            width: fit-content;
            padding: 0 6px 0 4.5px;
            height: 25.5px;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: normal;
            color: ${color};
            font-size: 12px;
            font-weight: normal;
            text-shadow: 0 0.5625px 1.5px rgba(0,0,0,0.5);
            z-index: 0;
        ">
            ${textContent}
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: ${gradient};
                border-radius: 6px;
                border: 1.5px solid ${borderColor};
                box-shadow: 
                  inset 0 4.6875px 3.75px -2.8125px rgba(255,255,255,0.6),
                  inset 0 -2.8125px 3.75px -1.875px rgba(0,0,0,0.2),
                  0 2.34375px 4.6875px rgba(0,0,0,0.5);
                z-index: -1;
            "></div>
            <div style="
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: ${underlayColor};
                border-radius: 6px;
                z-index: -2;
            "></div>
        </div>`;
}


////////////////////////////////////////////////////////


function initFormat(ganttTimelineObjName) {
    /* ***************************************
    * Execute on a collapsible-grouped-table *
    * with a timeline column at load time.   *
    *************************************** */
    const ganttTimelineObj = getGlobalObjByName(ganttTimelineObjName);
    const tbody = ganttTimelineObj.table.querySelector('tbody');
    if (!tbody) return;

    const bodyRows = Array.from(tbody.querySelectorAll('tr'));

    // update right padding (based on max visible depth level)
    const maxLevel = getMaxVisibleLevel(bodyRows);
    tbody.style.setProperty('--max-visible-level', maxLevel);

    bodyRows.forEach((tr) => initTrFormat(ganttTimelineObj, tr));

    overrideLabels(ganttTimelineObj, bodyRows);
    ganttTimelineObj.refresh();

    /* set label-column length (table-layout: fixed) */
    const maxLabelLength = getLongestFirstColumnWidth(ganttTimelineObj.table);
    const firstCol = ganttTimelineObj.table.querySelector('colgroup col:first-child');
    firstCol.style.width = maxLabelLength + "px";
}

function initTrFormat(ganttTimelineObj, tr) {
    /* add event listener for timeline add */
    const timelineCell = tr.cells[ganttTimelineObj.timelineColumnIndex];
    timelineAddedObserver.observe(timelineCell, { childList: true });

    /* implement custom events init */
    if (tr.classList.contains("group-header")) {
        if (tr.classList.contains("collapsed")) {
            // header row of a collapsed group =>
            // set the start-timestamp/end-timestamp dataset attrs
            addSummaryTimestamps(ganttTimelineObj, tr);
        }

        if (tr.classList.contains("parallel-line")) {
            // header row of one of the spilt sub-DAG lines =>
            // apply odd/even background overlay to group
            const index = tr.dataset.path.split(".").at(-1);
            if (index%2) {
                ganttTimelineObj.table.querySelectorAll(
                    `tr[data-path="${tr.dataset.path}"], ` +
                    `tr[data-path^="${tr.dataset.path}."]`
                ).forEach(row => {
                    [...row.children].forEach(td => {
                        const bg = window.getComputedStyle(td).backgroundImage;

                        // Store original (i.e. if first pass on that cell)
                        if (!('storedBg' in td.dataset)) {
                            td.dataset.storedBg = td.style.backgroundImage || '';
                        }

                        td.style.backgroundImage = (
                            'linear-gradient(135deg,' +
                                            'rgba(255,255,255,0.3) 0%, ' +
                                            'rgba(248,249,250,0.3) 100%), ' +
                            `${bg !== 'none' ? bg : ''}`
                        ).replace(/,\s*$/, '');
                    });
                });
            }
        }
    } else if (tr.dataset.level === "0") {
        // top-level row (not part of a group)
        // we remove row-styling (set by collapsible-grouped-table
        // by default)
        tr.style.color = "";
        tr.style.background = "";
        tr.style.borderColor = "";
    } else {
        // non-header row (group row that is not the header)
    }
}

const timelineAddedObserver = new MutationObserver((mutationsList) => {
    for (const mutation of mutationsList) {
        if (mutation.type === 'childList') {
            mutation.addedNodes.forEach(node => {
                if (
                    node.nodeType === 1 &&
                    node.classList.contains('gantt-timeline-container')
                ) {
                    // timeline added event
                    const timelineCell = mutation.target;

                    if (timelineCell.dataset.startTimestamp) {
                        const bar = timelineCell.getElementsByClassName(
                            "gantt-timeline-bar")[0];
                        if (bar) {
                            /* add shine-hover layer */
                            const el = document.createElement("div");
                            el.className = "gantt-timeline-bar-hover-shine";
                            bar.insertBefore(el, bar.firstChild);

                            /* cascade row 'failed' class to timeline bar */
                            if (
                                timelineCell.closest('tr').classList.contains("failed")
                            ) {
                                bar.classList.add("failed");
                            }
                        }
                    }
                }
            });
        }
    }
});

function overrideLabels(ganttTimelineObj, bodyRows) {
    /* *****************************************
    * replaces default textNode                *
    * from default 'collapsible-grouped-table' *
    * with a formatted one in label cell.      *
    ***************************************** */
    bodyRows.forEach((tr) => {
        let trToUpdate = null;
        let shapedLabelHtmlString = null;
        if (tr.classList.contains("parallel-line")) {
            // case "split line of a distributed sub-pipeline"
            trToUpdate = tr;

            const groupStyle = JSON.parse(trToUpdate.dataset.groupStyle);
            shapedLabelHtmlString  = trapezoidalLabel(
                (
                    (trToUpdate.classList.contains("collapsed") ? "► " : "▼ ") +
                    trToUpdate.dataset.name
                ),
                groupStyle.color, groupStyle.background, groupStyle.border,
                groupStyle.labelUnderlay
            )

        } else if (tr.classList.contains("parallel-lines")) {
            // case of the "merging (last) task
            // of a distributed sub-pipeline"
            const targetPath = findLastVisibleChildOfGroup(
                ganttTimelineObj.table, tr.dataset.path
            );
            const targetIndentLevel =
                parseInt(
                    getComputedStyle(tr).getPropertyValue("--indent-level")
                ) + 1 ;
            trToUpdate = ganttTimelineObj.table.querySelector(
                `[data-path="${targetPath}"]:not(.parallel-line)` +
                `[style*="--indent-level: ${targetIndentLevel}"]`
            );
            if (!trToUpdate) {
                // case "merge task hasn't started yet"
                return; // skips currently iterated tr
            }

            const rowStyle = JSON.parse(trToUpdate.dataset.rowStyle);
            shapedLabelHtmlString  = trapezoidalLabel(
                trToUpdate.dataset.name,
                rowStyle.color, rowStyle.background, rowStyle.border,
                rowStyle.labelUnderlay,
                true
            )
        } else if (tr.classList.contains("taskgroup")) {
            // case "header row of a taskgoup"
        } else {
            // non-header row (can be top-level row
            // or group row that is not the header)
            trToUpdate = tr;

            const rowStyle = JSON.parse(trToUpdate.dataset.rowStyle);
            shapedLabelHtmlString  = rectangularLabel(
                trToUpdate.dataset.name,
                rowStyle.color, rowStyle.background, rowStyle.border,
                rowStyle.labelUnderlay
            )
        }
        if (trToUpdate) {
            // first, find original text-content
            // from default 'collapsible-grouped-table'
            let firstTextNode = null;
            for (let child of trToUpdate.cells[0].childNodes) {
                if (child.nodeType === 3) { // Text node
                    firstTextNode = child;
                    break;
                }
            }
            if (firstTextNode) {
                // then replace that node with shaped label
                const shapedLabel = document.createElement('div');
                shapedLabel.innerHTML = shapedLabelHtmlString;
                shapedLabel.classList.add("element-name");
                trToUpdate.cells[0].replaceChild(shapedLabel, firstTextNode);
            } else {
                // default didn't update with a textNode
                // (e.g. merge-task row of collapse/expand
                //  distributed sub-pipeline).
                // => no need to override.
            }
        }
    });
}

function toggleHeaderTimeline(ganttTimelineObjName, groupHeaderRow) {
    /* *********************************************
    * add/remove group-summary timeline            *
    * on group collapse/expand event respectively. *
    ********************************************* */
    const ganttTimelineObj = getGlobalObjByName(ganttTimelineObjName);

    if (!groupHeaderRow.classList.contains("collapsed")) {
        // group just expanded
        const headerTimelineCell =
            groupHeaderRow.cells[ganttTimelineObj.timelineColumnIndex];
        delete headerTimelineCell.dataset.startTimestamp;
        delete headerTimelineCell.dataset.endTimestamp;
        const oldTimeline =
            headerTimelineCell.querySelector('.gantt-timeline-container');
        oldTimeline.remove();
        groupHeaderRow.classList.remove("failed");
    } else {
        // group just collapsed
        addSummaryTimestamps(ganttTimelineObj, groupHeaderRow);
    }

    /* *************************
    * override default labels. *
    ************************* */
    overrideLabels(ganttTimelineObj, [groupHeaderRow]);

    /* ************************************
    * update whole-table right padding    *
    * (based on max visible depth level). *
    ************************************ */
    const tbody = ganttTimelineObj.table.querySelector('tbody');
    const bodyRows = Array.from(tbody.querySelectorAll('tr'));
    const maxDepth = getMaxVisibleLevel(bodyRows);
    tbody.style.setProperty('--max-visible-level', maxDepth);

    ganttTimelineObj.refresh();
}

function addSummaryTimestamps(ganttTimelineObj, groupHeaderRow) {
    const groupPath = groupHeaderRow.getAttribute('data-path');
    const headerTimelineCell =
        groupHeaderRow.cells[ganttTimelineObj.timelineColumnIndex];

    // Get all children of this group, all depths
    // and collect group timeline bounds
    const allRows = Array.from(
        ganttTimelineObj.table.querySelectorAll(
            `tr[data-path^="${groupPath}."]`)
    );

    var startTimestamp = Number.MAX_SAFE_INTEGER;
    var endTimestamp = -1;
    var anyChildFailed = false;
    allRows.forEach(row => {
        const timelineCell = row.cells[ganttTimelineObj.timelineColumnIndex];
        const start = (() => {
            const v = timelineCell.dataset.startTimestamp;
            return v && !isNaN(Number(v)) ? Number(v) : null 
        })();
        const end = (() => {
            const v = timelineCell.dataset.endTimestamp;
            return v && !isNaN(Number(v)) ? Number(v) : null
        })();

        if (start) {
            if (start < startTimestamp) {
                startTimestamp = start;
            }
            if (endTimestamp) {
                if (end) {
                    if (end > endTimestamp) {
                        endTimestamp = end;
                    }
                    if (row.classList.contains("failed"))
                        anyChildFailed = true;
                } else {
                    endTimestamp = null;
                }
            }
        }
    });

    headerTimelineCell.dataset.startTimestamp = startTimestamp;
    if (endTimestamp) {
        headerTimelineCell.dataset.endTimestamp = endTimestamp;
        if (anyChildFailed) {
            const tr = headerTimelineCell.closest('tr');
            tr.classList.add("failed");
        }
    } else {
        delete headerTimelineCell.dataset.endTimestamp;
    }
}

function getLongestFirstColumnWidth(table) {
    const rows = table.rows;
    let maxWidth = 0;
    const canvas = document.createElement('canvas');
    const context = canvas.getContext('2d');

    for (let i = 0; i < rows.length; i++) {
        const firstCell = rows[i].cells[0];
        const shapedLabelDiv = firstCell.querySelector('div.shaped-label');
            const textContent = (firstCell.textContent || firstCell.innerText).trim();
        if (shapedLabelDiv) {
            const style = getComputedStyle(shapedLabelDiv);
            context.font =  style.font;

            // Measure width (incl. padding)
            // and add letter spacing per character
            const baseWidth =
                context.measureText(textContent).width
                + (Number(style.paddingLeft.toString().replace('px', '')) | 0)
                + (Number(style.paddingRight.toString().replace('px', '')) | 0);
            const letterSpacing = parseFloat(style.letterSpacing) || 0;
            const lsWidth = letterSpacing * Math.max(0, textContent.length - 1);

            width = Math.max(
                baseWidth + lsWidth +
                (parseFloat(getComputedStyle(firstCell).paddingLeft) || 0),
                (Number(style.minWidth.toString().replace('px', '')) | 0)
            ) + 10;
        } else {
            // pure (non-shaped) text label (e.g. taskgroup name)
            const style = getComputedStyle(firstCell);
            context.font =  style.font;
            // Measure width and add letter spacing per character
            const baseWidth = context.measureText(textContent).width;
            const letterSpacing = parseFloat(style.letterSpacing) || 0;
            const lsWidth = letterSpacing * Math.max(0, textContent.length - 1);
            width = baseWidth + lsWidth +
                    (parseFloat(style.paddingLeft) || 0)
                    + 5;
        }
        maxWidth = Math.max(maxWidth, width);
    }

    return maxWidth;
}


////////////////////////////////////////////////////////


function ganttInsert(ganttTimelineObjName, payload, interBarsSpacing) {
    /* *
    * Insert a task in the Gantt chart component.
    * Takes care of group-header creations as needed.
    ** */
    const ganttTimelineObj = getGlobalObjByName(ganttTimelineObjName);
    const table_id = ganttTimelineObj.table.id;

    let group_header_row_id = null;
    let group_index = -1;

    const payloadStartTimestamp = new Date(payload.start_timestamp).getTime();
    const insertData = [
        {
            id: payload.id,
            cells: {
                name: {
                    value: payload.name,
                    attributes: {"uuid": payload.tasktype_uuid}
                },
                timeline: {
                    value: null,
                    attributes: {
                        "start_timestamp": payloadStartTimestamp
                    }
                }
            },
            callbacks: [`showDetailsModal(this);`],
            extraClasses: ["task"],
            style: payload.ui_css
        }
    ]
    const rowToStyleIds = [payload.id];

    // handle (potentially nested) parent taskgroup
    for (let taskgroup of payload.taskgroups_hierarchy) {
        const taskgroupGroupHeaderId = taskgroup.uuid + (
            payload.rank ? "." + JSON.stringify(payload.rank) : ""
        );
        const taskgroup_group_header_row =
            ganttTimelineObj.table.querySelector(
                `tr[data-id="${taskgroupGroupHeaderId}"]`);

        if (taskgroup_group_header_row) {
            // case "taskgroup has already been inserted"
            group_header_row_id = taskgroupGroupHeaderId;

            // determine insert index
            // within list of first-level child rows
            const prefix = taskgroup_group_header_row.dataset.path + ".";
            const groupRows = Array.from(
                    ganttTimelineObj.table.querySelectorAll(
                        `tr[data-path^="${prefix}"]`
                    )
                ).filter(
                    row => {
                        const suffix = row.dataset.path.substring(prefix.length);
                        // Accept only if suffix has no dots at all
                        return suffix.indexOf('.') === -1;
                    }
                );

            for (let i = 0; i < groupRows.length; i++) {
                const timelineCell =
                    groupRows[i].classList.contains("group-header") ?
                    groupRows[i].nextElementSibling.cells[ganttTimelineObj.timelineColumnIndex] :
                    groupRows[i].cells[ganttTimelineObj.timelineColumnIndex];
                const cellStartTs = Number(timelineCell.dataset.startTimestamp);

                if (cellStartTs > payloadStartTimestamp) {
                    group_index = i;
                    break;
                } else if (cellStartTs === payloadStartTimestamp) {
                    // Move forward to find extent of equal timestamp entries
                    let j = i;
                    while (j < groupRows.length) {
                        const nextCell = groupRows[j].classList.contains("group-header")
                            ? groupRows[j].nextElementSibling.cells[ganttTimelineObj.timelineColumnIndex]
                            : groupRows[j].cells[ganttTimelineObj.timelineColumnIndex];
                        if (Number(nextCell.dataset.startTimestamp) === payloadStartTimestamp) {
                            j++;
                        } else {
                            break;
                        }
                    }
                    // Insert payload by comparing IDs
                    // within block of items with equal timestamp
                    const payloadId = Number(payload.id);
                    group_index = i; // default to start of block
                    for (; group_index < j; group_index++) {
                        const cell = groupRows[group_index].classList.contains("group-header")
                            ? groupRows[group_index].nextElementSibling.cells[ganttTimelineObj.timelineColumnIndex]
                            : groupRows[group_index].cells[ganttTimelineObj.timelineColumnIndex];
                        const cellId = Number(cell.dataset.id);
                        if (payloadId < cellId) break;
                    }
                    // console.log("taskgroup", payload.name, "group_index -",
                    //             group_index, "groupRows", groupRows);
                    break;
                }
            }

            break;
        } else {
            // case "need to insert taskgroup row in addition to payload"
            const payloadItem = insertData[0];
            insertData[0] = {
                    id: taskgroupGroupHeaderId,
                    cells: {
                        name: {
                            value: taskgroup.name,
                            attributes: {}
                        },
                        timeline: {
                            value: null,
                            attributes: {}
                        }
                    },
                    callbacks: [
                        "toggleHeaderTimeline('execGanttTimelineObj', this);"
                    ],
                    extraClasses: ["taskgroup"],
                    children: [payloadItem],
                    style: taskgroup.ui_css
                }
        }
    }

    // handle (potentially deep) distributed sub-DAG
    if (payload.rank && !payload.merge_func) {
        if (payload.is_parallel) {
            // case "start of a new split line"
            const payloadItem = insertData[0];
            const payloadRankSuffix = "." + JSON.stringify(payload.rank);
            insertData[0] = {
                    id: payload.name + payloadRankSuffix,
                    cells: {
                        name: {
                            value: payload.name + payloadRankSuffix,
                            attributes: {}
                        },
                        timeline: {
                            value: null,
                            attributes: {}
                        }
                    },
                    callbacks: [
                        "toggleHeaderTimeline('execGanttTimelineObj', this);"
                    ],
                    extraClasses: ["parallel-line"],
                    children: [payloadItem],
                    style: payload.parent_ui_css.parallel_line
                }
            rowToStyleIds.push(payload.name + payloadRankSuffix);

            const subDagRankSuffix =
                payload.rank.length > 1 ?
                "." + JSON.stringify(
                          payload.rank.slice(0, payload.rank.length - 1)) :
                "";
            const subDagGroupHeaderRowId = 
                payload.name + subDagRankSuffix;
            const subDagGroupHeaderRow =
                ganttTimelineObj.table.querySelector(
                    `tr[data-id="${subDagGroupHeaderRowId}"]`);
            if (!subDagGroupHeaderRow) {
                // case "first line of a new distributed sub-DAG"
                const firstLineItem = insertData[0];
                insertData[0] = {
                        id: subDagGroupHeaderRowId,
                        cells: {
                            name: {
                                value: "Distributed sub-pipeline",
                                attributes: {}
                            },
                            timeline: {
                                value: null,
                                attributes: {}
                            }
                        },
                        callbacks: [
                            "toggleHeaderTimeline('execGanttTimelineObj', this);"
                        ],
                        extraClasses: ["parallel-lines"],
                        children: [firstLineItem],
                        style: payload.parent_ui_css.parallel_lines
                    }
                if (payload.rank.length > 1) {
                    // case "new sub-DAG is a deep sub-DAG"
                    const groupHeaderRow = [...ganttTimelineObj.table.querySelectorAll(
                            `.group-header.parallel-line[data-id$="${subDagRankSuffix}"]`
                        )].at(-1);
                    group_header_row_id = groupHeaderRow.dataset.id;
                    // TODO  -  determine insert index
                }
            } else {
                // case "append split-line to distributed sub-DAG"
                group_header_row_id = subDagGroupHeaderRowId;
                // determine insert index
                const prefix = subDagGroupHeaderRow.dataset.path + ".";
                  const existingSplitLinesArr = [...ganttTimelineObj.table.querySelectorAll(
                          `tr.group-header.parallel-line[data-path^="${prefix}"]`
                      )].filter(el => !el.dataset.path.slice(prefix.length).includes('.'));

                for (let i = 0; i < existingSplitLinesArr.length; i++) {
                    const groupRow = existingSplitLinesArr[i];

                    const parallelTaskRow = groupRow.nextElementSibling;
                    const timelineCell =
                        parallelTaskRow.cells[ganttTimelineObj.timelineColumnIndex];
                    const cellStartTs = Number(timelineCell.dataset.startTimestamp);

                    if (cellStartTs > payloadStartTimestamp) {
                        group_index = i;
                        break;
                    } else if (cellStartTs === payloadStartTimestamp) {
                        // Move forward to find extent of equal timestamp entries
                        let j = i;
                        while (j < existingSplitLinesArr.length) {
                            const nextCell = existingSplitLinesArr[j].classList.contains("group-header")
                                ? existingSplitLinesArr[j].nextElementSibling.cells[ganttTimelineObj.timelineColumnIndex]
                                : existingSplitLinesArr[j].cells[ganttTimelineObj.timelineColumnIndex];
                            if (Number(nextCell.dataset.startTimestamp) === payloadStartTimestamp) {
                                j++;
                            } else {
                                break;
                            }
                        }
                        // Insert payload by comparing IDs
                        // within block of items with equal timestamp
                        const payloadId = Number(payload.id);
                        group_index = i; // default to start of block
                        for (; group_index < j; group_index++) {
                            const cell = existingSplitLinesArr[group_index].classList.contains("group-header")
                                ? existingSplitLinesArr[group_index].nextElementSibling.cells[ganttTimelineObj.timelineColumnIndex]
                                : existingSplitLinesArr[group_index].cells[ganttTimelineObj.timelineColumnIndex];
                            const cellId = Number(cell.dataset.id);
                            if (payloadId < cellId) break;
                        }
                        // console.log("taskgroup", payload.name, "group_index -",
                        //             group_index, "existingSplitLinesArr", existingSplitLinesArr);
                        break;
                    }
                }
            }
        } else if (!group_header_row_id) {
            // case "not a task of an existing taskgroup"
            //  => split-line header is the last row with 'payload.rank' as id suffix
            const groupHeaderRow = [...document.querySelectorAll(
                    `.group-header.parallel-line[data-id$=".${JSON.stringify(payload.rank)}"]`
                )].at(-1);
            group_header_row_id = groupHeaderRow.dataset.id;
            // TODO  -  determine insert index
        }

    } else if (payload.merge_func) {
        // sub-DAG header is last row with below 2 conditions met :
        //  (1) => header 'level' equal to 'payload.rank' length
        const subDAGlevel = 2 * (
            payload.rank ? payload.rank.length : 0
        );
        //  (2) => header 'id' suffixed with 'payload.rank'
        const subDagRankCondition =
            payload.rank ? `[data-id$=".${JSON.stringify(payload.rank)}"]` : ""
        ;
        const groupHeaderRow = [...document.querySelectorAll(
                `.group-header.parallel-lines[data-level="${subDAGlevel}"]${subDagRankCondition}`
            )].at(-1);
        group_header_row_id = groupHeaderRow.dataset.id;
        // insert index remains the default "-1" here

        // trapezoidal shaped label to group tail
        rowToStyleIds.length = 0; // clear it
        rowToStyleIds.push(group_header_row_id);

    }

    // call collapsible-grouped-table insertAt method
    insertAt(
        table_id, group_header_row_id, group_index,
        insertData,
        interBarsSpacing
    )

    if (payload.taskgroups_hierarchy && payload.taskgroups_hierarchy.length >= 2) {
        for (let tg of payload.taskgroups_hierarchy) {
            const tgId = tg.uuid + (payload.rank ? "." + JSON.stringify(payload.rank) : "");
            const tgRow = ganttTimelineObj.table.querySelector(`tr[data-id="${tgId}"]`);
            if (tgRow) reorderGroupChildren(ganttTimelineObj, tgRow);
        }
    }

    // reset prior odd/even background overlay on groups
    // and apply back
    // (as inserts break prior odd/even distribution)
    if (
        (payload.rank && group_header_row_id)
        || payload.merge_func
    ) {
        // case "any element of not a first-level entirely new sub-DAG"
        // => get top-level path (path of parent element with level=0)
        const groupHeaderRow = ganttTimelineObj.table.querySelector(
                `.group-header[data-id$="${group_header_row_id}"]`
            );
        const allRelatedRows =
            ganttTimelineObj.table.querySelectorAll(
                `tr[data-path^="${groupHeaderRow.dataset.path.split('.')[0]}."]`
            );

        removeBgEvenOddOverlay(allRelatedRows);
        allRelatedRows.forEach((tr) => initTrFormat(ganttTimelineObj, tr));

    } else if (payload.is_parallel) {
        // case "first-level entirely new sub-DAG"
        const threeInsertedRows = ganttTimelineObj.table.querySelectorAll(
            `[data-id="${payload.name}"], ` +
            `[data-id="${payload.name + "." + JSON.stringify(payload.rank)}"], ` +
            `[data-id="${payload.id}"]`
        )
        threeInsertedRows.forEach((tr) => initTrFormat(ganttTimelineObj, tr));
    } else {
        // case "top-level, standard task"
        const tr = ganttTimelineObj.table.querySelector(`tr[data-id="${payload.id}"]`);
        initTrFormat(ganttTimelineObj, tr);
    }

    const tbody = ganttTimelineObj.table.querySelector('tbody');
    const bodyRows = Array.from(tbody.querySelectorAll('tr'));
    const maxLevel = getMaxVisibleLevel(bodyRows);
    tbody.style.setProperty('--max-visible-level', maxLevel);

    // custom Gantt-chart shaped-label
    const rowsToStyle = rowToStyleIds.map(rowId => 
        ganttTimelineObj.table.querySelector(`tr[data-id="${rowId}"]`)
    );
    overrideLabels(ganttTimelineObj, rowsToStyle);

    // update summary on potential (possibly deep) collapsed parent groups
    updateSummaryTimestamps(ganttTimelineObjName, payload.id)

    ganttTimelineObj.refresh();

    /* update label-column length (table-layout: fixed) */
    const maxLabelLength = getLongestFirstColumnWidth(ganttTimelineObj.table);
    const firstCol = ganttTimelineObj.table.querySelector('colgroup col:first-child');
    firstCol.style.width = maxLabelLength + "px";
}

function updateSummaryTimestamps(ganttTimelineObjName, childRowId) {
    const ganttTimelineObj = getGlobalObjByName(ganttTimelineObjName);

    const child_path = ganttTimelineObj.table.querySelector(
        `tr[data-id="${childRowId}"]`).dataset.path;

    const parts = child_path.split(".");

    for (let i = parts.length - 2 ; i >= 0 ; i--) {
        const parentGroupRow = ganttTimelineObj.table.querySelector(
            `tr.group-header[data-path="${parts.slice(0, i + 1).join(".")}"]`);
        if (parentGroupRow.classList.contains("collapsed")) {
            toggleHeaderTimeline(ganttTimelineObjName, parentGroupRow)
        }
    }
}

function removeBgEvenOddOverlay(rows) {
    [...rows].forEach(row => {
        [...row.children].forEach(td => {
            if ('storedBg' in td.dataset) {
                td.style.backgroundImage = td.dataset.storedBg;
                delete td.dataset.storedBg;
            }
        });
    });
}

function ganttUpdate(ganttTimelineObjName, payload, interBarsSpacing) {
    /* *
    * for tasks end-timestamp propagation.
    * fired by e.g. 'taskEnded' event listener
    ** */
    const ganttTimelineObj = getGlobalObjByName(ganttTimelineObjName);

    let taskRow = ganttTimelineObj.table.querySelector(
        `tr[data-id="${payload.id}"]`);

    if (!taskRow) {
        // start by inserting payload in Gantt-chart component
        // if, for any abnormal reason, "newTask" hadn't fired on it
        try {
            ganttInsert(ganttTimelineObjName, payload, interBarsSpacing);
        } catch (error) {
            console.error("Couldn't insert payload", payload);
            console.error(error);
            return;
        }
        // if it's still not there, give up
        taskRow = ganttTimelineObj.table.querySelector(
            `tr[data-id="${payload.id}"]`);
        if (!taskRow) return;
    }

    // update of end_timestamp
    const timelineCell = taskRow.cells[ganttTimelineObj.timelineColumnIndex];
    timelineCell.dataset.endTimestamp = new Date(payload.end_timestamp).getTime();

    // update of failed status, if applies
    if (payload.failed) {
        timelineCell.closest('tr').classList.add("failed")
        const bar = timelineCell.getElementsByClassName(
            "gantt-timeline-bar")[0];
        bar.classList.add("failed");
    }

    // update summary on potential collapsed parent groups
    updateSummaryTimestamps(ganttTimelineObjName, payload.id)

    ganttTimelineObj.refresh();
}
