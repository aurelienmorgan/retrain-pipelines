
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

function findLastChildOfGroup(groupPath) {
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
        lastChild.classList.contains('group-header') &&
        !lastChild.classList.contains('collapsed')
    ) {
        const deeperLastChild = findLastChildOfGroup(lastChildPath);
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
    const id = row.getAttribute('data-id');

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
        const lastChildPath = findLastChildOfGroup(path);
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
        const lastChildPath = findLastChildOfGroup(path);
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
        (!isInitiallyCollapsed ? '► ' : '▼ ') + path + " - " + id;

    existingTopBars.forEach(bar => row.cells[0].appendChild(bar));
    /* ************************ */

    /* **************************************
    * adding bottom bars for the            *
    * new state of the group being toggled. *
    ************************************** */
    // Find last visible row after toggle
    let targetRow;
    if (!isInitiallyCollapsed) { // Just expanded
        const lastChildPath = findLastChildOfGroup(path);
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
        const lastChildPath = findLastChildOfGroup(groupPath);
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
    * for each standalone rows. *
    ************************** */
    document.querySelectorAll('.standalone-row').forEach(row => {
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
    document.querySelectorAll('tbody tr:not(.standalone-row)').forEach(
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
            const lastChildPath = findLastChildOfGroup(path);
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

function renderRows(data, parentPath = "", level = 0) {
    let html = '';
    data.forEach((item, index) => {
        const path = parentPath ? `${parentPath}.${index}` : `${index}`;
        const hasChildren = item.children && item.children.length > 0;
        const isStandalone = item.standalone === true;

        const idCell = (
            hasChildren
            ? `<td data-id="${item.id}">▼ ${path} - ${item.id}</td>`
            : `<td>${path}&nbsp;-&nbsp;${item.id}</td>`
        );

        const rowClass = (hasChildren ? 'group-header ' : '') +
                         (isStandalone ? 'standalone-row' : '');
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

function insertRowUnder(table_id, row_id) {
    // Get the existing row by id
    const table = document.getElementById(table_id);
    const existingRow = table.querySelector(`tr[data-id="${row_id}"]`);
    if (!existingRow) {
        console.error(
            `row with data id ${row_id} not found ` +
                `in table with id '${table_id}'.`);
        return;
    }

    const newRow = document.createElement("tr");
    // Add some cells to the new row (example)
    for (let i = 0; i < existingRow.cells.length; i++) {
        const newCell = document.createElement("td");
        newCell.textContent = "New data 1";
        newRow.appendChild(newCell);
    }

    const tbody = existingRow.parentNode;
    tbody.insertBefore(newRow, existingRow.nextSibling);
}

function insertRowAt(table_id, row_id, index) {
    /* *
    * 'row_id' one of a group header
    * (null if new row is table top-level standalone)
    ** */
    // TODO, insert, manage (group) style and depth bars
    //       insert even if hidden (do not expand ancestors)
}










































































