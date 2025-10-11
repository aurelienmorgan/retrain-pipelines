
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
                // header of one of the spilt sub-DAG lines =>
                // apply odd/even backgroupd overlay
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
        }
    });

    ganttTimelineObj.refresh();
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

