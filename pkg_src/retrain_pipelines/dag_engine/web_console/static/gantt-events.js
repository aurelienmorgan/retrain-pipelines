
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


////////////////////////////////////////////////////////


function hexToRgba(hex, alpha) {
    /* ***************************
    * Helper function to convert *
    * hex to rgba with alpha.    *
    *************************** */
    const r = parseInt(hex.slice(1, 3), 16);
    const g = parseInt(hex.slice(3, 5), 16);
    const b = parseInt(hex.slice(5, 7), 16);
    return `rgba(${r}, ${g}, ${b}, ${alpha})`;
}

// Helper function to generate gradient
function generateGradient(backgroundColor) {
    /* **************************************
    * Helper function to generate gradient. *
    ************************************** */
    return `linear-gradient(180deg, 
            ${hexToRgba(backgroundColor, 0.48)} 0%,
            ${hexToRgba(backgroundColor, 0.42)} 20%,
            ${hexToRgba(backgroundColor, 0.98)} 40%, 
            ${hexToRgba(backgroundColor, 0.73)} 60%,
            ${hexToRgba(backgroundColor, 0.59)} 80%,
            ${hexToRgba(backgroundColor, 0.54)} 100%)`;
}

function trapezoidalLabel(
    textContent,
    color, backgroundColor, borderColor,
    flipped = false
) {
    /* ******************************************
    * html for formatted trapezoidal shaped div *
    ****************************************** */
    const gradient = generateGradient(backgroundColor);
    const borderRadius = flipped ?
        '10px 10px 8px 8px' : '8px 8px 10px 10px';
    // adjust for 3D tilting leaving
    // empty top and bottom space
    const correctionMargin = flipped ?
        '-5px 0 -8px' :
        '-8px 0 -2px'; 
    const rotation = flipped ?
        'rotateX(-35deg)' : 'rotateX(35deg)';
    const justifyContent = flipped ?
        'justify-content: center;' : '';

    return "" +
        `<div class="shaped-label" style="
            position: relative;
            min-width: 125px;
            width: fit-content;
            padding: 0 8px 0 6px;
            height: 40px;
            display: flex;
            align-items: center;
            ${justifyContent}
            line-height: normal;
            color: ${color};
            font-size: 16px;
            font-family: Robotto, Arial, sans-serif;
            letter-spacing: 0.5px;
            text-shadow: 0 0.75px 2px rgba(0,0,0,0.5);
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
                border: 2px solid ${borderColor};
                box-shadow: 
                  inset 0 6.25px 5px -3.75px rgba(255,255,255,0.6),
                  inset 0 -3.75px 5px -2.5px rgba(0,0,0,0.2),
                  0 3.125px 6.25px rgba(0,0,0,0.5);
                transform: perspective(200px) ${rotation};
                z-index: -1;
            "></div>
        </div>`;
}

function rectangularLabel(
    textContent,
    color, backgroundColor, borderColor
) {
    /* ******************************************
    * html for formatted rectangular-shaped div *
    ****************************************** */
    const gradient = generateGradient(backgroundColor);

    return "" +
        `<div class="shaped-label" style="
            position: relative;
            min-width: 125px;
            width: fit-content;
            padding: 0 8px 0 6px;
            height: 34px;
            display: flex;
            align-items: center;
            justify-content: center;
            line-height: normal;
            color: ${color};
            font-size: 16px;
            font-family: Robotto, Arial, sans-serif;
            letter-spacing: 0.5px;
            text-shadow: 0 0.75px 2px rgba(0,0,0,0.5);
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
                border-radius: 8px;
                border: 2px solid ${borderColor};
                box-shadow: 
                  inset 0 6.25px 5px -3.75px rgba(255,255,255,0.6),
                  inset 0 -3.75px 5px -2.5px rgba(0,0,0,0.2),
                  0 3.125px 6.25px rgba(0,0,0,0.5);
                z-index: -1;
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

    bodyRows.forEach((tr) => {
        if (tr.classList.contains("group-header")) {
            if (tr.classList.contains("collapsed")) {
                // header row of a collapsed group =>
                // set the start-timestamp/end-timestamp dataset attrs
                addSummaryTimestamps(ganttTimelineObj, tr);
            }

            if (tr.classList.contains("parallel-line")) {
                // header row of one of the spilt sub-DAG lines =>
                // apply odd/even backgroupd overlay to group
                const index = tr.dataset.path.split(".").at(-1);
                if (index%2) {
                    tbody.querySelectorAll(
                        `tr[data-path="${tr.dataset.path}"], ` +
                        `tr[data-path^="${tr.dataset.path}."]`
                    ).forEach(row => {
                        [...row.children].forEach(td => {
                            const bg = window.getComputedStyle(td).backgroundImage;
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
            // as the default behavior)
            tr.style.color = "";
            tr.style.background = "";
            tr.style.borderColor = "";
        } else {
            // non-header row (group row that is not the header)
        }
    });

    overrideLabels(ganttTimelineObj, bodyRows);
    ganttTimelineObj.refresh();
}

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

            shapedLabelHtmlString  = trapezoidalLabel(
                (
                    (trToUpdate.classList.contains("collapsed") ? "► " : "▼ ") +
                    trToUpdate.dataset.name
                ),
                // !!!!  TODO  -  handle group styling and defaults
                "#EAEAEA", "#00FF0D", "#FFD700"
            )

        } else if (tr.classList.contains("parallel-lines")) {
            // case of the "merging (last) task of a distributed sub-pipeline"
            const targetPath = findLastVisibleChildOfGroup(
                ganttTimelineObj.table, tr.dataset.path
            );
            trToUpdate = ganttTimelineObj.table.querySelector(
                `[data-path="${targetPath}"]`
            );

            shapedLabelHtmlString  = trapezoidalLabel(
                trToUpdate.dataset.name,
                // !!!!  TODO  -  handle group styling and defaults
                "#EAEAEA", "#C80043", "#FFD700",
                true
            )
        } else if (tr.classList.contains("taskgroup")) {
            // case "header row of a taskgoup"
        } else {
            // non-header row (can be top-level row
            // or group row that is not the header)
            trToUpdate = tr;

            shapedLabelHtmlString  = rectangularLabel(
                trToUpdate.dataset.name,
                // !!!!  TODO  -  handle group styling and defaults
                "#EAEAEA", "#00FF0D", "#FFD700"
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
    } else {
        // group just collapsed
        addSummaryTimestamps(ganttTimelineObj, groupHeaderRow);
    }

    /* *************************
    * override default labels. *
    ************************* */
    overrideLabels(ganttTimelineObj, [groupHeaderRow]);

    /* ************************************
    * update right padding                *
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
    allRows.forEach(row => {
        timelineCell = row.cells[ganttTimelineObj.timelineColumnIndex]
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
                } else {
                    endTimestamp = null;
                }
            }
        }
    });

    headerTimelineCell.dataset.startTimestamp = startTimestamp;
    if (endTimestamp)
        headerTimelineCell.dataset.endTimestamp = endTimestamp;    
}

