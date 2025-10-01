
const MAX_Z_INDEX = zIndex = 2147483647;
const defaultBottomPadding = 8; /* in px, must match with td CSS */
const barThickness = 3;         /* in px, nesting-bar width,
                                   must match with nesting-bar CSSs */

function setCookie(name, value) {
    document.cookie = name + "=" +
                      encodeURIComponent(value) +
                      ";path=/;max-age=31536000";
}

function getCookie(name) {
    const value = "; " + document.cookie;
    const parts = value.split("; " + name + "=");
    if (parts.length === 2)
        return decodeURIComponent(parts.pop().split(";").shift());
    return null;
}

function saveState() {
    const state = {};
    document.querySelectorAll('.group-header').forEach(row => {
        const path = row.getAttribute('data-path');
        state[path] = row.classList.contains('collapsed');
    });
    setCookie('tableState', JSON.stringify(state));
}

function hasCollapsedAncestor(path) {
    /* ****************************************
    * For any given row, whether or not it is *
    * part of a (even distant) parent         *
    * that is collapsed.                      *
    **************************************** */
    const parts = path.split('.');
    for (let i = parts.length - 1; i > 0; i--) {
        const ancestorPath = parts.slice(0, i).join('.');
        const ancestor = document.querySelector(`[data-path="${ancestorPath}"]`);
        if (ancestor && ancestor.classList.contains('collapsed')) {
            return true;
        }
    }
    return false;
}

function applyVisibility() {
    document.querySelectorAll('tbody tr').forEach(
        row => row.classList.remove('hidden'));
    document.querySelectorAll('[data-path]').forEach(row => {
        if (
            hasCollapsedAncestor(row.getAttribute('data-path'))
        ) {
            row.classList.add('hidden');
        }
    });
}

function loadState() {
    const stateStr = getCookie('tableState');
    if (!stateStr) {
        applyVisibility();
        return;
    }

    try {
        const state = JSON.parse(stateStr);
        document.querySelectorAll('.group-header').forEach(row => {
            const path = row.getAttribute('data-path');
            const isCollapsed = state[path] === true;
            row.classList.toggle('collapsed', isCollapsed);
            row.cells[0].textContent = (isCollapsed ? '► ' : '▼ ') +
                                        path + " - " +
                                        row.getAttribute('data-id');
        });
        applyVisibility();
    } catch (e) {
        applyVisibility();
    }
}

function isLastChildOfParent(path) {
    const parts = path.split('.');
    if (parts.length <= 1) return false;

    const parentPath = parts.slice(0, -1).join('.');
    const index = parseInt(parts[parts.length - 1]);

    let parentData = tableData;
    if (parentPath) {
        const indices = parentPath.split('.').map(Number);
        for (const idx of indices) {
            if (!parentData[idx] || !parentData[idx].children) return false;
            parentData = parentData[idx].children;
        }
    }

    result = index === parentData.length - 1;
    return result;
}

function getGroupStyleForPath(path) {
    const parts = path.split('.');
    for (let i = parts.length; i > 0; i--) {
        const candidatePath = parts.slice(0, i).join('.');
        const row = document.querySelector(`[data-path="${candidatePath}"]`);
        if (row && row.classList.contains('group-header')) {
            const styleJson = row.getAttribute('data-group-style');
            if (styleJson) return JSON.parse(styleJson);
        }
    }
    return null;
}

function findLastVisibleChildOfGroup(groupPath) {
    const groupRow = document.querySelector(`[data-path="${groupPath}"]`);
    if (!groupRow || !groupRow.classList.contains('group-header'))
        return null;

    const allDescendants =
        Array.from(document.querySelectorAll(`[data-path^="${groupPath}."]`));
    const directChildren = allDescendants.filter(row => {
        const path = row.getAttribute('data-path');
        const pathParts = path.split('.');
        const groupParts = groupPath.split('.');
        return pathParts.length ===
            groupParts.length + 1 && path.startsWith(groupPath);
    });

    if (directChildren.length === 0) return null;

    let lastChild = directChildren[directChildren.length - 1];
    let lastChildPath = lastChild.getAttribute('data-path');

    if (
        lastChild.classList.contains('group-header')
        && !lastChild.classList.contains('collapsed')
    ) {
        const deeperLastChild = findLastVisibleChildOfGroup(lastChildPath);
        if (deeperLastChild) {
            lastChildPath = deeperLastChild;
        }
    }

    return lastChildPath;
}

function toggleRow(path) {
    const row = document.querySelector(`[data-path="${path}"]`);
    if (!row) return;

    const isInitiallyCollapsed = row.classList.contains('collapsed');

    row.classList.toggle('collapsed', !isInitiallyCollapsed);
    applyVisibility();

    /* *****************************
    * cleanup of bottom bars       *
    * for the group being toggled. *
    ***************************** */
    // removing bars from previous state
    // and adjusting bottom padding back
    // (i.e. cleaning header
    //       if init state was collapsed,
    //       lastChild if it was expanded)
    var rowToClean;
    if (!isInitiallyCollapsed) {
        const lastChildPath = findLastVisibleChildOfGroup(path);
        rowToClean =
            document.querySelector(`[data-path="${lastChildPath}"]`);
    } else {
        rowToClean = row;
    }
    for (let i = 0; i < rowToClean.cells.length; i++) {
        const cell = rowToClean.cells[i];
        cell.querySelectorAll('.bottom-nesting-bar').forEach(bar => {
            bar.remove();
        });
        // force reset bottom padding to default
        // (avoid webbrowser rounding issues)
        cell.style.paddingBottom = `${defaultBottomPadding}px`;
    }
    /* ************************** */

    /* ******************************************
    * cleanup of left and right bars in corners *
    * for the group being toggled.              *
    ****************************************** */
    var groupLastVisibleRow;
    if (isInitiallyCollapsed) {
        const lastChildPath = findLastVisibleChildOfGroup(path);
        groupLastVisibleRow =
            document.querySelector(`[data-path="${lastChildPath}"]`);
    }
    row.querySelectorAll(
        'td:first-child .left-nesting-bar, ' +
        'td:last-child .right-nesting-bar'
    ).forEach(bar => bar.remove());
    if (isInitiallyCollapsed) {
        groupLastVisibleRow.querySelectorAll(
            'td:first-child .left-nesting-bar, ' +
            'td:last-child .right-nesting-bar'
        ).forEach(bar => bar.remove());
    }
    /* *************************************** */

    /* ***************************
    * top bars  and header arrow *
    *************************** */
    const existingTopBars = row.cells[0].querySelectorAll('.top-nesting-bar');

    row.cells[0].textContent =
        (!isInitiallyCollapsed ? '► ' : '▼ ') + path + " - " + row.getAttribute('data-id');

    existingTopBars.forEach(bar => row.cells[0].appendChild(bar));
    /* ************************ */

    /* **************************************
    * adding bottom bars for the            *
    * new state of the group being toggled. *
    ************************************** */
    // Find last visible row after toggle
    let targetRow;
    if (!isInitiallyCollapsed) { // Just expanded
        const lastChildPath = findLastVisibleChildOfGroup(path);
        targetRow = document.querySelector(`[data-path="${lastChildPath}"]`);
    } else { // Just collapsed
        targetRow = row;
    }

    // Get where bars need to be added
    const targetPath = targetRow.getAttribute('data-path');
    const targetParts = targetPath.split('.');
    // Add bottom bars for ALL ancestor groups ending at this target row
    for (let level = targetParts.length; level >= 1; level--) {
        const checkPath = targetParts.slice(0, level).join('.');

        if (isLastChildOfParent(checkPath)) {
            const parentPath = targetParts.slice(0, level - 1).join('.');
            const parentRow =
                document.querySelector(`[data-path="${parentPath}"]`);
            if (parentRow && parentRow.classList.contains('group-header')) {
                addBottomBar(parentRow, interBarsSpacing);
            }
        }
    }
    // Add bottom bar for the deepest child itself if it's a group
    if (targetRow.classList.contains('group-header')) {
        addBottomBar(targetRow, interBarsSpacing);
    }
    /* *********************************** */

    /* **************************************
    * adding left bars at corners for the   *
    * (and straight at header if expanding) *
    * new state of the group being toggled. *
    ************************************** */
    addLeftRightBars(row, interBarsSpacing);
    if (isInitiallyCollapsed) {
        addLeftRightBars(groupLastVisibleRow, interBarsSpacing);
    }
    /* *********************************** */

    saveState();
}

function isLastVisibleChild(groupPath, childPath) {
    const groupRow = document.querySelector(`[data-path="${groupPath}"]`);
    if (!groupRow || !groupRow.classList.contains('group-header')) {
        return false;
    }
    
    const isGroupCollapsed = groupRow.classList.contains('collapsed');
    
    if (isGroupCollapsed) {
        return childPath === groupPath;
    } else {
        const lastChildPath = findLastVisibleChildOfGroup(groupPath);
        return childPath === lastChildPath;
    }
}

function countParentGroupsEndingAt(lastRow) {
    const currentPath = lastRow.getAttribute('data-path');
    const pathParts = currentPath.split('.');
    
    let count = 0;
    
    // For each parent level, check if this row is the last child
    for (let level = pathParts.length - 1; level > 0; level--) {
        const parentPath = pathParts.slice(0, level).join('.');
        
        // Find parent group row
        const parentRow =
            document.querySelector(`tr[data-path="${parentPath}"]`);
        if (!parentRow) break;
        
        // Find all children of this parent at the next level
        const childLevel = level;
        const childRows = Array.from(
            document.querySelectorAll(`tr[data-level="${childLevel}"]`)
        ).filter(row => {
            const rowPath = row.getAttribute('data-path');
            return rowPath.startsWith(parentPath + '.');
        });
        
        // Check if current row is the last child
        const lastChild = childRows[childRows.length - 1];
        const currentSubPath = pathParts.slice(0, level + 1).join('.');
        
        if (
            lastChild &&
            lastChild.getAttribute('data-path') === currentSubPath
        ) {
            count++;
        } else {
            break;
        }
    }
    
    return count;
}

function applyGroupStyles(interBarsSpacing) {
    /* **************************
    * styling                   *
    * for each top-level rows. *
    ************************** */
    document.querySelectorAll(
        'tr[data-level="0"]:not(.group-header)'
    ).forEach(row => {
        const path = row.getAttribute('data-path');
        const item = getLeafByPath(tableData, path);
        if (item && item.style) {
            const { color, background, border } = item.style;
            row.style.cssText =
                `color: ${color}; background-color: ${background}; ` +
                `border-color: ${border}; ` +
                `--indent-level: ${row.getAttribute('data-level')}`;
        }
    });

    /* **************************
    * styling +                 *
    * left, right, and top bars *
    * for each group.           *
    ************************** */
    document.querySelectorAll(
        'table tbody tr:not([data-level="0"]), table tbody tr.group-header'
    ).forEach(
        row => {
            const path = row.getAttribute('data-path');
            const groupStyle = getGroupStyleForPath(path);
            // row styling
            if (groupStyle) {
                const { color, background, border } = groupStyle;
                row.style.cssText =
                    `color: ${color}; background-color: ${background};`+
                    `border-color: ${border};`+
                    `--indent-level: ${row.getAttribute('data-level')}`;
        }

        // row top bars
        const offset = barThickness + interBarsSpacing;
        if (row.classList.contains('group-header')) {
            const currentGroupStyle = getGroupStyleForPath(path);
            if (currentGroupStyle) {
                const level = parseInt(row.getAttribute('data-level'));
                const offset = barThickness + interBarsSpacing;
                const leftOffset = level * offset;
                const rightOffset = level * offset;

                for (let i = 0; i < row.cells.length; i++) {
                    const cell = row.cells[i];
                    const topBar = document.createElement('div');
                    topBar.className = 'top-nesting-bar';
                    topBar.style.backgroundColor = currentGroupStyle.border;
                    topBar.style.zIndex = MAX_Z_INDEX - level;

                    if (i === 0) {
                        topBar.style.left = `${leftOffset}px`;
                        topBar.style.right = '0';
                    } else if (i === row.cells.length - 1) {
                        topBar.style.left = '0';
                        topBar.style.right = `${rightOffset}px`;
                    } else {
                        topBar.style.left = '0';
                        topBar.style.right = '0';
                    }

                    cell.appendChild(topBar);
                }
            }
        }

        // row left & right bars
        addLeftRightBars(row, interBarsSpacing);
    });

    /* ****************************
    * bottom bars for each group. *
    **************************** */
    document.querySelectorAll('tbody tr.group-header').forEach(
        row => addBottomBar(row, interBarsSpacing));;
}

function addLeftRightBars(row, interBarsSpacing) {
    /*
    * All at once, accounting for all depths.
    */
    const path = row.getAttribute('data-path');
    const parts = path.split('.');
    const offset = barThickness + interBarsSpacing;
    const endingGroups = countParentGroupsEndingAt(row);

    for (let i = parts.length; i > 0; i--) {
        const candidatePath = parts.slice(0, i).join('.');
        const ancestorRow = document.querySelector(
            `[data-path="${candidatePath}"]`);
        const isCollapsed = ancestorRow.classList.contains('collapsed');

        if (ancestorRow && ancestorRow.classList.contains('group-header')) {
            const style = getGroupStyleForPath(candidatePath);
            if (style) {
                const level = i - 1;
                const groupsBelow = endingGroups - (parts.length - i);
                
                const leftBar = document.createElement('div');
                leftBar.className = 'left-nesting-bar';
                leftBar.style.left = `${level * offset}px`;
                leftBar.style.backgroundColor = style.border;
                leftBar.style.zIndex = MAX_Z_INDEX - level;
                
                if (isLastVisibleChild(candidatePath, path)) {
                    // how many groups are ending at this row
                    // that are deeper than candidatePath
                    leftBar.style.bottom =
                        `${Math.max(0, groupsBelow) * offset}px`;
                }

                row.cells[0].appendChild(leftBar);
                
                const rightBar = document.createElement('div');
                rightBar.className = 'right-nesting-bar';
                rightBar.style.right = `${level * offset}px`;
                rightBar.style.backgroundColor = style.border;
                rightBar.style.zIndex = MAX_Z_INDEX - level;
                
                if (isLastVisibleChild(candidatePath, path)) {
                    // how many groups are ending at this row
                    // that are deeper than candidatePath
                    rightBar.style.bottom =
                        `${Math.max(0, groupsBelow) * offset}px`;
                }
                
                row.cells[row.cells.length-1].appendChild(rightBar);
            }
        }
    }
}

function addBottomBar(row, interBarsSpacing) {
    /* *********************************************************
    * header row of the group for which to add a bottom line   *
    * the last row at which to add the bottom bar.             *
    * In case of nested groups and                             *
    * a sub-group is the last of children of parent group(s),  *
    * more than one bar shall be added (at the proper offset). *
    * Params:                                                  *
    *     - maxSubLevel(int):                                  *
    *           how many depth sub-groups there are            *
    *           withing the group having "row" as its header.  *
    ********************************************************* */
    const path = row.getAttribute('data-path');
    const isLastChildOfDirectParentGroup = isLastChildOfParent(path);
    const isCollapsed = row.classList.contains('collapsed');
    const groupStyle = getGroupStyleForPath(path);
    const offset = barThickness + interBarsSpacing;
    const level = parseInt(row.getAttribute('data-level'));
    const leftOffset = level * offset;
    const rightOffset = level * offset;

    const endingGroups = countParentGroupsEndingAt(row);

    if (isCollapsed) {
        if (groupStyle) {
            for (let i = 0; i < row.cells.length; i++) {
                const cell = row.cells[i];
                const bar = document.createElement('div');
                bar.className = 'bottom-nesting-bar';
                bar.style.backgroundColor = groupStyle.border;
                bar.style.zIndex = MAX_Z_INDEX - level;
                bar.style.bottom = `${endingGroups * offset}px`;
                
                if (i === 0) {
                    bar.style.left = `${leftOffset}px`;
                    bar.style.right = '0';
                } else if (i === row.cells.length - 1) {
                    bar.style.left = '0';
                    bar.style.right = `${rightOffset}px`;
                } else {
                    bar.style.left = '0';
                    bar.style.right = '0';
                }

                // add padding
                cell.style.paddingBottom = (
                        Number(getComputedStyle(cell)
                                  .paddingBottom.toString().replace('px', ''))
                        + endingGroups * (offset/2)
                        // no idea why this is the best value here
                    ) + "px";

                cell.appendChild(bar);
            }
        }
    } else {
        // expanded group
        if (groupStyle) {
            const lastChildPath = findLastVisibleChildOfGroup(path);
            const lastChildRow =
                document.querySelector(`[data-path="${lastChildPath}"]`);

            for (let i = 0; i < lastChildRow.cells.length; i++) {
                const cell = lastChildRow.cells[i];
                const bar = document.createElement('div');
                bar.className = 'bottom-nesting-bar';
                bar.style.backgroundColor = groupStyle.border;
                bar.style.zIndex = MAX_Z_INDEX - level;
                bar.style.bottom = `${endingGroups * offset}px`;
                
                if (i === 0) {
                    bar.style.left = `${leftOffset}px`;
                    bar.style.right = '0';
                } else if (i === lastChildRow.cells.length - 1) {
                    bar.style.left = '0';
                    bar.style.right = `${rightOffset}px`;
                } else {
                    bar.style.left = '0';
                    bar.style.right = '0';
                }

                // add padding
                cell.style.paddingBottom = (
                        Number(getComputedStyle(cell)
                                  .paddingBottom.toString().replace('px', ''))
                        + endingGroups * (offset/2)
                        // no idea why this is the best value here
                    ) + "px";

                cell.appendChild(bar);
            }
        }
    }
}

function getLeafByPath(data, pathStr) {
    const parts = pathStr.split('.').map(Number);
    let current = data;
    for (let i = 0; i < parts.length; i++) {
        if (Array.isArray(current) && current[parts[i]]) {
            if (i === parts.length - 1) return current[parts[i]];
            current = current[parts[i]].children || [];
        } else {
            return null;
        }
    }
    return null;
}

function renderRows(data, parentPath = "", level = 0, startIndex = 0) {
    let html = '';
    data.forEach((item, index) => {
        const path = parentPath ?
                    `${parentPath}.${startIndex + index}` :
                    `${startIndex + index}`;
        const hasChildren = item.children && item.children.length > 0;
        const isTopLevelRow = level === "0" && parentPath === "" && item.children;

        const idCell = (
            hasChildren
            ? `<td data-id="${item.id}">▼ ${path} - ${item.id}</td>`
            : `<td>${path}&nbsp;-&nbsp;${item.id}</td>`
        );

        const rowClass = (hasChildren ? 'group-header ' : '');
        const clickAttr =
            hasChildren ? `onclick="toggleRow('${path}')"` : '';
        const dataAttrs =
            `data-path="${path}" data-level="${level}" data-id="${item.id}"`;
        const extraAttrs = hasChildren && item.style 
            ? `data-group-style='${JSON.stringify(item.style)}'` 
            : '';

        html += `<tr class="${rowClass.trim()}" ${dataAttrs} ` +
                           `${clickAttr} ${extraAttrs}>` +
                `${idCell}<td>${item.name}</td><td>${item.description}</td>` +
                `<td>${item.value}</td></tr>`;

        if (hasChildren) {
            html += renderRows(item.children, path, level + 1);
        }
    });
    return html;
}

function init(tableId, tableData, interBarsSpacing) {
    var tbodyId = 'data-tbody';
    var table = document.getElementById(tableId);
    if (!table) return;
    var tbody = table.querySelector('tbody#' + tbodyId);
    if (!tbody) return;
    tbody.innerHTML = renderRows(tableData);
    loadState();
    applyGroupStyles(interBarsSpacing);
}

//////////////////////////////////////////////////////////////

function collapseAll() {
    /* ********************************************************
    * For each group G in table.children (top-level),         *
    * for each group g in G.children (depth 1),               *
    * recursive call collapsAll(header_row).                  *
    *    We start by collapsing deepest upward.               *
    *                                                         *
    * loop over children of current group (via data-path),    *
    * for each group, make recursive call                     *
    * after that's done for all chidren in current group,     *
    * before leaving method, trigger actual group collapse by *
    * programatically firing click event on header_row        *
    * if current group is expanded.                           *
    ******************************************************** */

    function collapseGroup(groupPath) {
        const groupRow =
            document.querySelector(`[data-path="${groupPath}"]`);
        if (
            !groupRow ||
            !groupRow.classList.contains('group-header')
        ) return;

        // Find all direct children groups
        const allRows = document.querySelectorAll('[data-path]');
        const directChildGroups = Array.from(allRows).filter(row => {
            const path = row.getAttribute('data-path');
            const pathParts = path.split('.');
            const groupParts = groupPath.split('.');
            return (
                row.classList.contains('group-header') &&
                pathParts.length === groupParts.length + 1 &&
                path.startsWith(groupPath + '.')
            );
        });

        // Recursively collapse children first (deepest first)
        directChildGroups.forEach(childRow => {
            const childPath = childRow.getAttribute('data-path');
            collapseGroup(childPath);
        });

        // After all children are collapsed,
        // collapse this group if expanded
        if (!groupRow.classList.contains('collapsed')) {
            toggleRow(groupPath);
        }
    }

    // Collapse each top-level group
    // (which will recursively collapse children)
    const topLevelGroups = Array.from(
        document.querySelectorAll('.group-header')
    ).filter(row => {
        const path = row.getAttribute('data-path');
        return !path.includes('.');
    });
    topLevelGroups.forEach(groupRow => {
        const path = groupRow.getAttribute('data-path');
        collapseGroup(path);
    });
}

function expandAll() {
    function expandGroup(groupPath) {
        const groupRow =
            document.querySelector(`[data-path="${groupPath}"]`);
        if (
            !groupRow ||
            !groupRow.classList.contains('group-header')
        ) return;

        // Expand this group first if collapsed
        if (groupRow.classList.contains('collapsed')) {
            toggleRow(groupPath);
        }

        // Find all direct children groups
        const allRows = document.querySelectorAll('[data-path]');
        const directChildGroups = Array.from(allRows).filter(row => {
            const path = row.getAttribute('data-path');
            const pathParts = path.split('.');
            const groupParts = groupPath.split('.');
            return (
                row.classList.contains('group-header') &&
                pathParts.length === groupParts.length + 1 &&
                path.startsWith(groupPath + '.')
            );
        });

        // Recursively expand children after parent is expanded
        directChildGroups.forEach(childRow => {
            const childPath = childRow.getAttribute('data-path');
            expandGroup(childPath);
        });
    }

    // Expand each top-level group
    // (which will recursively expand children)
    const topLevelGroups = Array.from(
        document.querySelectorAll('.group-header')
    ).filter(row => {
        const path = row.getAttribute('data-path');
        return !path.includes('.');
    });
    topLevelGroups.forEach(groupRow => {
        const path = groupRow.getAttribute('data-path');
        expandGroup(path);
    });
}

//////////////////////////////////////////////////////////////

function insertAt(
    table_id, group_header_row_id, group_index,
    data,
    interBarsSpacing
) {
    /* *************************************
    * Inserts "data", a list of rows,      *
    * at index "group_index" (zero-based)  *
    * of the group with header-row         *
    * of id "group_header_row_id"          *
    * (null if new row is table top-level) *
    ************************************* */
    const table = document.getElementById(table_id);
    if (!table) {
        console.error(`Table with id '${table_id}' not found.`);
        return;
    }
    if (!Number.isInteger(group_index)) {
        console.error("Invalid group_index:", group_index);
        return;
    }
    try {
        checkUnicityOfAllDepthsItems(table, data);
    } catch (error) {
        console.error(error);
        return;
    }

    //////////////////////////////////////////
    // retrieve rows in the targetted group //
    //////////////////////////////////////////
    var level = 0;
    var parentPath = "";
    var groupRows;
    if (group_header_row_id) {
        const parentRow =
            table.querySelector(`tr[data-id="${group_header_row_id}"]`);
        if (!parentRow.classList.contains('group-header')) {
            console.error(
                `row-id '${group_header_row_id}' ` +
                'is not that of a group-header but of a standard row.'
            );
            return;
        }
        if (!parentRow) {
            console.error("Invalid group_header_row_id:", group_header_row_id);
            return;
        }
        parentPath = parentRow.getAttribute('data-path');
        groupRows =
            Array.from(
                table.querySelectorAll(`tr[data-path^="${parentPath}."]`)
                    ).filter(row => {
                const dataPath = row.getAttribute('data-path');
                const suffix = dataPath.substring(parentPath.length + 1);
                return !suffix.includes('.');
            });
        level = parseInt(parentRow.getAttribute('data-level')) + 1;
    } else {
        // Top-level insertion
        groupRows = Array.from(
            table.querySelectorAll('tbody tr[data-level="0"]')
        );
    }

    if (group_index > groupRows.length) {
        console.error(
            `Can't insert at index ${group_index} `+
            `in a group of length ${groupRows.length}`);
        return;
    }
    // case "insert at last position (append) in group"
    if (group_index == -1) group_index = groupRows.length;
    //////////////////////////////////////////

    /////////////////////////////////
    // get anchor for insert after //
    /////////////////////////////////
    var insertAfterRow;
    if (group_index == 0) {
        if (group_header_row_id) {
            // insert at first position in group
            insertAfterRow = table.querySelector(`tr[data-id="${group_header_row_id}"]`);
        } else {
            // insert at first position in table
        }
    } else if (
        groupRows[group_index - 1].classList.contains("group-header")
    ) {
        // case "row before is a subgroup" =>
        // offset in table by rows-count (may have deep children)
        const subgroup_path =
            groupRows[0].dataset.path.lastIndexOf('.') === -1
                ? groupRows[0].dataset.path
                : groupRows[0].dataset.path.substring(
                    0, groupRows[0].dataset.path.lastIndexOf('.'));
        insertAfterRowPath = findLastVisibleChildOfGroup(subgroup_path);
        insertAfterRowPath = insertAfterRowPath ?
            insertAfterRowPath :
            groupRows[group_index-1].dataset.path; // if top-level insert
        insertAfterRow = table.querySelector(`tr[data-path="${insertAfterRowPath}"]`);
        if (
            insertAfterRow.classList.contains("group-header") &&
            insertAfterRow.classList.contains("collapsed")
        ) {
            insertAfterRow = Array.from(
                    table.querySelectorAll(
                        `tr[data-path^="${insertAfterRowPath}."]`)
                ).slice(-1)[0];
        }
    } else {
        // case "raw before is standard row"
        insertAfterRowPath = groupRows[group_index - 1].dataset.path;
        insertAfterRow = table.querySelector(`tr[data-path="${insertAfterRowPath}"]`);
    }
    /////////////////////////////////

    /////////////////////////////
    // insert raw html to DOM //
    /////////////////////////////
    rawHtml = renderRows(data, parentPath, level, group_index);
    const tbody = insertAfterRow ? insertAfterRow.parentNode : table.tBodies[0];
    if (insertAfterRow) {
      // Insert after existing row
      const nextSibling = insertAfterRow.nextSibling;
      if (nextSibling) {
        nextSibling.insertAdjacentHTML('beforebegin', rawHtml);
      } else {
        // insertAfterRow is last row
        tbody.insertAdjacentHTML('beforeend', rawHtml);
      }
    } else {
      // Insert at very start of tbody
      tbody.insertAdjacentHTML('afterbegin', rawHtml);
    }
    /////////////////////////////

    //////////////////////////////////////////
    // update pathes of following rows      //
    // (same level or deeper in that group) //
    //////////////////////////////////////////
    for (var i = group_index ; i <= groupRows.length - 1 ; i++) {
        incrementIndexes(
            table,
            groupRows[i].dataset.path,
            countAllDepthsItems(data)
        );
    }
    //////////////////////////////////////////

    // wipeout bars
    for (const row of table.rows) {
        const bars = row.querySelectorAll(
            '.left-nesting-bar, .right-nesting-bar, ' +
            '.top-nesting-bar, .bottom-nesting-bar'
        );
        bars.forEach(bar => {
            bar.remove();
        });
        for (const cell of row.cells) {
            cell.style.paddingBottom = `${defaultBottomPadding}px`;
        }
    }

    applyVisibility()
    // apply styling & restore bars
    applyGroupStyles(interBarsSpacing);

    saveState();
}

function checkUnicityOfAllDepthsItems(table, data) {
    /* ******************************************
    * Ensure that we don't insert any rows with *
    * id that already exists in target table.   *
    ****************************************** */

    // Collect all existing data-id values from the table
    const existingIds = new Set();
    const existingRows = table.querySelectorAll('[data-id]');
    existingRows.forEach(row => {
        const id = row.getAttribute('data-id');
        if (id) {
            existingIds.add(id);
        }
    });
    const duplicates = [];

    function traverse(items) {
        for (const item of items) {
            // Check if this item's id already exists in the table
            if (existingIds.has(item.id)) {
                duplicates.push(item.id);
            }
            // If item has children, traverse them recursively
            if (item.children && Array.isArray(item.children)) {
                traverse(item.children);
            }
        }
    }

    traverse(data);

    // If duplicates found, throw error
    // with all duplicate IDs
    if (duplicates.length > 0) {
        const error = new Error(
            `Duplicate ids found: ${duplicates.join(', ')}. ` +
            'These data-id values already exist in the table.'
        );
        throw error;
    }
}

function countAllDepthsItems(data) {
    let count = 0;

    function traverse(items) {
        for (const item of items) {
            count++; // Count the current item
            
            // If item has children, traverse them recursively
            if (item.children && Array.isArray(item.children)) {
                traverse(item.children);
            }
        }
    }

    traverse(data);
    return count;
}

function incrementIndexes(table, path, incrementValue) {
    /* *******************************************
    * "path" may be that of a group header       *
    * or a standard row.                         *
    * Increment prefix path by "incrementValue". *
    ******************************************* */

    // take last row with path
    // (the former one, the one that now
    //  requires to be incremented)
    const row = Array.from(
            table.querySelectorAll(`tr[data-path="${path}"]`)
        ).slice(-1)[0];

    // increment row path
    const old_index_value = parseInt(
        row.dataset.path.lastIndexOf('.') === -1
            ? row.dataset.path
            : row.dataset.path.substring(
                row.dataset.path.lastIndexOf('.') + 1)
    );

    const new_path =
        path.substring(0, path.length - String(old_index_value).length) +
        String(old_index_value + incrementValue)
    row.dataset.path = new_path;

    if (row.classList.contains('group-header')) {
        // case "row is a group header" =>
        // replace path prefix for all group rows
        // (all depths)
        table.querySelectorAll(
            `tr[data-path^="${path}."]`
        ).forEach(deepChildRow => {
            deepChildRow.dataset.path =
                new_path + deepChildRow.dataset.path.substring(path.length);
        });
    }
}










































































