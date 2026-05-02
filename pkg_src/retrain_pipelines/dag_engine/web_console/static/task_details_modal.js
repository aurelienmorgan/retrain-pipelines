
// ##################################################################### //

// Cookie helper functions
function setCookie(name, value) {
    document.cookie =
        name + "=" + encodeURIComponent(value) +
        ";path=/;max-age=31536000";
}

function getCookie(name) {
    const nameEQ = name + "=";
    const cookies = document.cookie.split(';');
    for (let i = 0; i < cookies.length; i++) {
        let cookie = cookies[i];
        while (cookie.charAt(0) === ' ') cookie = cookie.substring(1);
        if (cookie.indexOf(nameEQ) === 0) {
            const raw = cookie.substring(nameEQ.length);
            return decodeURIComponent(raw);
        }
    }
    return null;
}

/**
 * @typedef {Object} TaskEntry
 * @property {number} modifiedTimestamp - epoch
 * @property {number} viewPortTopLineIndex - int
 * @property {boolean} isAutoScroll - bool
 *
 * @typedef {Object.<string, TaskEntry>} TaskIdMap
 *
 * @typedef {Object.<string, TaskIdMap>} TaskTypeMap
 *
 * @typedef {Object} ActiveTabTypeMap
 * @property {Object.<string, {value: string, modifiedTimestamp: number}>} [taskIds]
 *
 * @typedef {Object.<string, ActiveTabTypeMap>} ActiveTabMap
 *
 * @typedef {Object} TasksCookie
 * @property {ActiveTabNameMap} activeTabName
 * @property {TaskTypeMap} traces
 * @property {Object} hosts - empty object for now
 * @property {Object} other - empty object for now
 *
 * cookie be shaped like:
 * ```
 * TasksCookie = {
 *   activeTabName: { [tasktypeUuid]: { [taskId]: {[value]: string, [modifiedTimestamp]: epoch} },
 *   traces: { [tasktypeUuid]: { [taskId]: TaskEntry } },
 *   hosts: {},
 *   other: {}
 * }
 * ```
 */

function getTasksCookie(execId) {
    const raw = getCookie(`tasks-details-${execId}`);
    if (!raw) {
        return { activeTabName: {},
                 traces: {}, hosts: {}, other: {} };
    }
    try {
        const parsed = JSON.parse(raw);
        // Ensure mandatory top-level keys exist
        return {
            activeTabName: parsed.activeTabName ?? {},
            traces: parsed.traces ?? {},
            hosts: parsed.hosts ?? {},
            other: parsed.other ?? {},
        };
    } catch {
        return { activeTabName: {}, traces: {}, hosts: {}, other: {} };
    }
}

function saveTasksCookie(execId, data) {
    setCookie(`tasks-details-${execId}`, JSON.stringify(data));
}

function upsertActiveTabNameCookieEntry(execId, tasktypeUuid, taskId, tabName) {
    /* *******************************************
    * Inserts or updates active tab name mapping *
    ******************************************* */
    const data = getTasksCookie(execId);
    if (!data.activeTabName) data.activeTabName = {};
    if (!data.activeTabName[tasktypeUuid]) data.activeTabName[tasktypeUuid] = {};

    data.activeTabName[tasktypeUuid][taskId] = {
        value: tabName,
        modifiedTimestamp: Date.now()
    };

    saveTasksCookie(execId, data);
}

function getActiveTabNameCookieEntry(execId, tasktypeUuid, taskId) {
    /* ********************************************
    * Gets active tab name for a given task.      *
    * Fallback: use latest modified entry in same *
    * tasktypeUuid if direct match not found.     *
    ******************************************** */
    const data = getTasksCookie(execId);
    const activeMap = data.activeTabName;
    if (!activeMap || !activeMap[tasktypeUuid]) return null;

    const direct = activeMap[tasktypeUuid][taskId];
    if (direct) return direct.value;

    // fallback to most recent in this task type
    const entries = Object.values(activeMap[tasktypeUuid]);
    if (!entries.length) return null;

    let latest = entries[0];
    for (const e of entries) {
        if (e.modifiedTimestamp > latest.modifiedTimestamp) {
            latest = e;
        }
    }
    return latest.value;
}

function upsertTaskTracesCookieEntry(execId, tasktypeUuid, taskId) {
    //console.log("upsertTaskTracesEntry(execId, tasktypeUuid, taskId)", "  -  ",
    //            execId, "  -  ", tasktypeUuid, "  -  ", taskId);
    const modalOverlay = document.getElementById("detailsModal");
    const tracesContainer = modalOverlay.querySelector(".traces-log-container");
    const tracesTabVisible = tracesContainer && !!(
        tracesContainer.offsetWidth ||
        tracesContainer.offsetHeight ||
        tracesContainer.getClientRects().length
    );

    if (tracesTabVisible) {
        // 2 possible cases :
        //   - traces-TAB showing on modal close
        //   - tab-switch away from traces-TAB
        // => "need to update cookies for the traces TAB"
        const wasAutoscroll = tracesContainer.classList.contains("autoscroll");

        let firstVisibleIndex = 0;
        if (!wasAutoscroll) {
            const lines =
                tracesContainer.querySelectorAll(".trace-line");
            const visibleRect = tracesContainer.getBoundingClientRect();
            if (visibleRect.height > 0) {
                const scrollTop = tracesContainer.scrollTop;
                for (let i = 0; i < lines.length; i++) {
                    if (lines[i].offsetTop >= scrollTop) {
                        firstVisibleIndex = i;
                        break;
                    }
                }
            } else {
                console.warn("traces tab not showing");
            }
        }

        const entry = {
            modifiedTimestamp: Date.now(),
            viewPortTopLineIndex: firstVisibleIndex,
            isAutoScroll: wasAutoscroll
        };

        // actual cookie upsert
        const data = getTasksCookie(execId);
        if (!data["traces"]) data["traces"] = {};
        if (!data["traces"][tasktypeUuid])
            data["traces"][tasktypeUuid] = {};
        data["traces"][tasktypeUuid][taskId] = entry;
        saveTasksCookie(execId, data);
    }
}

function getTaskTracesCookieEntry(execId, tasktypeUuid, taskId) {
    /* ************************
    * Gets task entry         *
    * from traces section.    *
    * Fallback to most recent *
    * in this tasktypeUuid.   *
    ************************ */
    const data = getTasksCookie(execId);
    const traces = data.traces;

    if (!traces || !traces[tasktypeUuid]) return null;

    // Direct match first
    const directEntry = traces[tasktypeUuid][taskId];
    if (directEntry) {
        return directEntry;
    }
    // most recent in tasktypeUuid
    const entries = Object.values(traces[tasktypeUuid]);
    if (entries.length === 0) {
        return null;
    }

    let latest = entries[0];
    for (const entry of entries) {
        if (entry.modifiedTimestamp > latest.modifiedTimestamp) {
            latest = entry;
        }
    }
    return latest;
}

// ##################################################################### //

function getGanttTable(row) {
    try {
        if (!row || typeof row !== 'object' || !row.nodeType) {
            throw new Error(
                'Invalid row: must be a valid DOM element');
        }
        const tables =
            document.querySelectorAll('table.gantt-table');
        if (tables.length === 0) {
            throw new Error(
                'No tables with class "gantt-table" found');
        }
        const table =
            Array.from(tables).find(t => t.contains(row));
        if (!table) {
            throw new Error(
                'No gantt-table contains the specified row');
        }
        return table;
    } catch (error) {
        console.error('getGanttTable failed:', error.message);
        throw error;
    }
}

function getEndpointPrefix() {
    const path = window.location.pathname;
    const lastSlashIndex = path.lastIndexOf('/');
    return path.substring(0, lastSlashIndex);
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

let isModalUp = false;
async function showDetailsModal(row, endpoint_prefix="") {
    if (document.getElementById("detailsModal") || isModalUp)
        return;

    isModalUp = true;

    // Load template
    const response = await fetch('task_details_modal.html');
    const templateHTML = await response.text();
    const tempDiv = document.createElement('div');
    tempDiv.innerHTML = templateHTML;
    const modalOverlay = tempDiv.firstElementChild;
    const modalContent = modalOverlay.querySelector('.modal-content');
    const closeButton = modalOverlay.querySelector('.modal-close');

    // Populate template
    const urlParams = new URLSearchParams(window.location.search);
    const execId = urlParams.get('id');
    modalOverlay.querySelector("#modal-exec-id").textContent =
        execId;
    const taskId = row.dataset.id;
    modalOverlay.querySelector("#modal-task-id").textContent =
        taskId;
    const tasktypeUuid = row.querySelector('td[data-uuid]').dataset.uuid;
    modalOverlay.querySelector("#modal-tasktype-uuid").textContent =
        tasktypeUuid;
    const ganttTable = getGanttTable(row);

    /* ***********
    * task label *
    *********** */
    const taskShapedLabels = []
    const parts = row.dataset.path.split(".");
    for (let i = 0 ; i <= parts.length - 2 ; i++) {
        const tmpRow = ganttTable.querySelector(
            `tr[data-path="${parts.slice(0, i + 1).join(".")}"]`);
        if (tmpRow.classList.contains("parallel-line")) {
            taskShapedLabels.push(
                tmpRow.querySelector('td .element-name').cloneNode(true)
            );
        }
    }
    taskShapedLabels.push(
        row.querySelector('td .element-name').cloneNode(true)
    );
    // console.log("taskShapedLabels", taskShapedLabels);
    const taskLabelContainer =
        modalOverlay.querySelector(".task-label-container");
    taskShapedLabels.forEach((taskShapedLabel, index) => {
        taskLabelContainer.appendChild(taskShapedLabel);

        if (index < taskShapedLabels.length - 1) {
            const separator = document.createElement('span');
            separator.textContent = ' / ';
            taskLabelContainer.appendChild(separator);
        }
    });
    /* ******** */

    /* *******************
    * tasktype docstring *
    ******************* */
    const docstringEndpointPrefix = getEndpointPrefix();
    const docstringEndpointUrl =
        (docstringEndpointPrefix ? `${docstringEndpointPrefix}` : "") +
        `/tasktype_docstring?uuid=${tasktypeUuid}`;
    const taskTypeDocstring =
        modalOverlay.querySelector("#tasktype-docstring");
    const docstringContentDiv =
        modalOverlay.querySelector("#tasktype-docstring .content");
    try {
        const response = await fetch(docstringEndpointUrl);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(
                (errorText || response.statusText || 'Unknown error') +
                ` [${tasktypeUuid}]`
            );
        }
        const docstring_str = await response.json();
        if (docstring_str) {
            docstringContentDiv.innerHTML = docstring_str.trim();
            taskTypeDocstring.classList.add("showing");

            waitForVisible('#detailsModal').then(() => {
                checkDocstringOverflow();
            }).catch(err => {
                console.warn(err);
            });
        }
    } catch (error) {
        docstringContentDiv.innerHTML =
            '<p class="error-message">' +
            `Error loading docstring: ${escapeHtml(error.message)}</p>`;
        console.error('Failed to load docstring:', error);
    }

    // TaskType docstring 'show-more'
    function checkDocstringOverflow() {
        const docstringContentDiv =
            document.querySelector("#tasktype-docstring .content");
        const showMore =
            document.getElementById("tasktype-docstring-show-more");
        // Calculating two lines of text height
        const lineHeight = parseFloat(getComputedStyle(docstringContentDiv).lineHeight);
        const maxTwoLinesHeight = lineHeight * 2;
        //console.log("checkDocstringOverflow", lineHeight, maxTwoLinesHeight, docstringContentDiv.scrollHeight);

        const overflows =
            docstringContentDiv.scrollHeight > maxTwoLinesHeight + 2;
        if (overflows) {
            if (showMore.style.display != "block") {
                showMore.style.display = "block";
                docstringContentDiv.classList.add("collapsed");
            }
        } else {
            showMore.style.display = "none";
        }

    }
    window.addEventListener('resize', checkDocstringOverflow);

    const showMoreTaskTypeDocstringBtn =
        modalOverlay.querySelector("#tasktype-docstring-show-more");

    showMoreTaskTypeDocstringBtn.addEventListener('click', () => {
        if (taskTypeDocstring.classList.contains('collapsed')) {
            // Expand
            taskTypeDocstring.classList.remove('collapsed');
            taskTypeDocstring.classList.add('expanded');
            showMoreTaskTypeDocstringBtn.innerText = 'Show less';
        } else {
            // Collapse
            taskTypeDocstring.scrollTop = 0;
            taskTypeDocstring.classList.remove('expanded');
            taskTypeDocstring.classList.add('collapsed');
            showMoreTaskTypeDocstringBtn.innerText = 'Show more';
        }
    });
    /* **************** */

    // Disable window scrolling on mousewheel
    function wheelModalScroll(e) {
        // Find the actual scrollable element
        let scrollableElement = e.target;
        while (scrollableElement && scrollableElement !== modalOverlay) {
            const overflowY = window.getComputedStyle(scrollableElement).overflowY;
            if (
                (overflowY === 'auto' || overflowY === 'scroll') &&
                scrollableElement.scrollHeight > scrollableElement.clientHeight
            ) {
                break;
            }
            scrollableElement = scrollableElement.parentElement;
        }

        if (scrollableElement && scrollableElement !== modalOverlay) {
            const delta = e.deltaY;
            const scrollTop = scrollableElement.scrollTop;
            const scrollHeight = scrollableElement.scrollHeight;
            const clientHeight = scrollableElement.clientHeight;
            
            const atTop = scrollTop === 0 && delta < 0;
            const atBottom = (scrollHeight - scrollTop - clientHeight) < 1 && delta > 0;

            if (atTop || atBottom) {
                e.preventDefault();
            }
        } else {
            e.preventDefault();
        }
    }
    modalOverlay.addEventListener('wheel', wheelModalScroll, { passive: false });

    // Disable window scrolling on arrows keypress
    let scrollInterval = null;
    function arrowModalScroll(e) {
        /*****************************************
        * pass to focused DIV to scroll.         *
        * Allows for smooth continuous scrolling *
        * on long keypress.                      *
        * Thus disallowing passing to            *
        * window scroll on hitting top/bottom    *
        * element bounds.                        *
        * NOTE: needs faking since tabindex "-1" *
        *       disables natively.               *
        *************************************** */
        if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
            // Skip native arrow handlers
            const active = document.activeElement;
            const tag = active.tagName;
            const type = active.getAttribute?.('type')?.toLowerCase() || '';
            if (tag === 'INPUT' && !['checkbox','radio','button','submit'].includes(type) ||
                tag === 'TEXTAREA' || tag === 'SELECT') {
                return;
            }

            e.preventDefault();  // Blocks window scroll
            if (scrollInterval) return;  // Already scrolling

            // Find nearest scrollable ancestor (Y-axis)
            function isScrollable(el) {
                if (!el) return false;
                const style = window.getComputedStyle(el);
                return (style.overflowY === 'auto' || style.overflowY === 'scroll') &&
                       el.scrollHeight > el.clientHeight;
            }
            let target = active;
            while (target && target !== document.documentElement && !isScrollable(target)) {
                target = target.parentElement;
            }
            target = target && isScrollable(target) ? target : modalOverlay;

            const direction = e.key === 'ArrowDown' ? 1 : -1;
            scrollInterval = setInterval(() => {
                target.scrollBy({ top: direction * 8, behavior: 'instant' });
                // Stop at bounds
                if (
                    (direction < 0 && target.scrollTop === 0) ||
                    (
                        direction > 0 &&
                        target.scrollHeight - target.clientHeight
                            === target.scrollTop
                    )
                ) {
                    clearInterval(scrollInterval);
                    scrollInterval = null;
                }
            }, 16);  // ~60fps fluid scroll
        }
    }

    // Stop on key release
    document.addEventListener('keyup', (e) => {
        if (e.key === 'ArrowUp' || e.key === 'ArrowDown') {
            if (scrollInterval) {
                clearInterval(scrollInterval);
                scrollInterval = null;
            }
        }
    }, { passive: false });
    // Listen on document for global prevention
    document.addEventListener('keydown', arrowModalScroll, { passive: false });

    function closeModal() {
        upsertTaskTracesCookieEntry(execId, tasktypeUuid, taskId);

        document.body.classList.remove('modal-open');
        modalOverlay.removeEventListener(
            'wheel', wheelModalScroll, { passive: false });
        modalOverlay.removeEventListener('keydown', handleEscKey);
        modalOverlay.remove();
        document.removeEventListener(
            'keydown', arrowModalScroll, { passive: false });
        document.removeEventListener('keydown', handleKeydown);
        window.removeEventListener('resize', checkDocstringOverflow);

        isModalUp = false;
    }
    closeButton.addEventListener('click', closeModal);

    const loadedTabs = {};

    const tabHeader = modalContent.querySelector('.tab-header');
    Array.from(tabHeader.children).forEach(tab => {
        tab.addEventListener(
            'click', () => switchTab(tab.textContent.toLowerCase().trim())
        );
    });

    const tabs = modalContent.querySelectorAll('.tab-content');
    function switchTab(tabName) {
        setCookie('detailsModalTab', tabName);
        upsertActiveTabNameCookieEntry(
            execId, tasktypeUuid, taskId, tabName
        );
        // button styling switch
        let previousActiveTabName = null;
        Array.from(tabHeader.children).forEach(tab => {
            if (tab.textContent.toLowerCase() === tabName) {
                tab.classList.add('active');
            } else if (tab.classList.contains('active')) {
                previousActiveTabName = tab.textContent.toLowerCase();
                tab.classList.remove('active');
            }
        });

        if (previousActiveTabName === "traces") {
            // store traces scroll-state cookie for the task
            upsertTaskTracesCookieEntry(execId, tasktypeUuid, taskId);
        }

        // hide inactive tab
        Array.from(tabs).forEach(tab =>
            tab.classList.remove('active'));
        // show active tab
        const selectedTabContent =
            document.getElementById(`tab-${tabName}`);
        selectedTabContent.classList.add('active');
        // load active tab if shown for the first time
        if (!loadedTabs[tabName]) {
            loadedTabs[tabName] = true;
            loadTabContent(tabName, selectedTabContent, taskId);
        } else {
            if (tabName === "traces") {
                const tracesContainer =
                    modalContent.querySelector(".traces-log-container");
                if (
                    tracesContainer &&
                    tracesContainer.classList.contains("autoscroll")
                ) {
                    // ta handle the case
                    // traces-TAB was not showing while autoscroll 
                    // AND received streamed traces
                    tracesContainer.scrollTop = tracesContainer.scrollHeight;
                }
            }
        }
    }

    async function loadTabContent(tabName, container, taskId) {
        if (tabName === 'traces') {
            loadTracesTabContent(
                container, taskId
            ).then(({ tracesContainer }) => {
                const entry =
                    getTaskTracesCookieEntry(execId, tasktypeUuid, taskId);
                //console.log("taskTracesCookieEntry", entry);

                if (entry) {
                    if (entry.isAutoScroll) {
                        // DO NOTHING here
                        // (already handled by mutation observer)
                    } else {
                        firstVisibleIndex = entry.viewPortTopLineIndex;
                        //console.log("firstVisibleIndex", firstVisibleIndex);
                        const lines =
                            tracesContainer.querySelectorAll(
                                ".trace-line");
                        const target = lines[firstVisibleIndex];
                        if (target) {
                            tracesContainer.scrollTop =
                                target.offsetTop -
                                    tracesContainer.offsetTop;
                        }
                    }
                }
            }).catch(err => {
                console.error('loadTracesTabContent failed:', err);
            });

        } else {
            container.innerHTML = `
                <div class="loading">
                    <div class="spinner"></div>
                    <div class="message">Loading...</div>
                </div>
            `;
            await new Promise(resolve => setTimeout(resolve, 500));
            container.innerHTML = `
                <h3>Other Information</h3>
                <p>Content for the "${tabName}" tab.</p>
                <p class="task-id">TaskId ID: ${taskId}</p>
            `;
        }
    }

    document.body.appendChild(modalOverlay);
    document.body.classList.add('modal-open');

    modalOverlay.addEventListener('click', (e) => {
        if (e.target === modalOverlay) closeModal();
    });

    // ESC keypress
    function handleEscKey(e) {
        if (e.key === 'Escape') {
            closeModal();
        }
    }
    modalOverlay.addEventListener('keydown', handleEscKey);

    // TAB keypress
    function handleKeydown(e) {
        // focus trap, avoid focus leaving modal and going to page
        if (e.key === 'Tab') {
            const els = Array.from(
                modalOverlay.querySelectorAll(
                    'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
                )).filter(el => el.offsetParent !== null);
            const first = els[0];
            const last  = els[els.length - 1];
            if (e.shiftKey) {
                if (
                    document.activeElement === first ||
                    document.activeElement === modalContent
                ) {
                    e.preventDefault();
                    last.focus();
                }
            } else {
                if (document.activeElement === last) {
                    e.preventDefault();
                    first.focus();
                }
            }
        }
    }
    document.addEventListener('keydown', handleKeydown);
    modalContent.setAttribute('tabindex', '-1');
    modalContent.focus();
    function firstShiftTrap(e) {
        // on modal show event, block Shift+Tab
        // from passing focus back to page
        if (e.key === 'Tab' && e.shiftKey) {
            e.preventDefault();
            const els = modalOverlay.querySelectorAll(
                'button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'
            );
            const last = els[els.length - 1];
            last.focus();
            modalContent.removeEventListener('keydown', firstShiftTrap);
        }
    }
    modalContent.addEventListener('keydown', firstShiftTrap);
    modalContent.addEventListener('focusin', (event) => {
        // if content got focus (if any element got focused), drop 
        modalContent.removeEventListener('keydown', firstShiftTrap);
    });
    // prevent focus leaving the modal (e.g. on footer click)
    modalOverlay.addEventListener('focusout', (e) => {
        const leavingElement = e.target;
        // If focus is going outside the modal
        if (!modalOverlay.contains(e.relatedTarget)) {
            e.preventDefault();
            // force focus back to the element
            // that was losing if
            leavingElement.focus();
        }
    });

    // Get saved tab from cookie
    const savedTab =
        getActiveTabNameCookieEntry(execId, tasktypeUuid, taskId)
        || getCookie('detailsModalTab') || 'traces';
    switchTab(savedTab);
}

let ansiUpInstance = null;
function getAnsiUp() {
    if (ansiUpInstance) return ansiUpInstance;
    if (typeof AnsiUp === "undefined")
        return { ansi_to_html: (text) => escapeHtml(text) };
    ansiUpInstance = new AnsiUp();
    ansiUpInstance.use_classes = false;
    ansiUpInstance.url_whitelist = {};
    return ansiUpInstance;
}

function setupTracesContainer(tracesContainer) {
    /* ************************************
    * establishes listeners and observers *
    * (for scrolling & resizing).         *
    ************************************ */
    tracesContainer.scrollTop = tracesContainer.scrollHeight;

    const logsContainerObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
            const hasScrollbar =
                entry.target.scrollHeight > entry.target.clientHeight;
            const toolbar =
                tracesContainer.parentElement.querySelector(
                    ".traces-log-toolbar");
            toolbar.classList.toggle("showing", hasScrollbar);

            const prevOverflowY = tracesContainer.style.overflowY;
            tracesContainer.style.overflowY = "hidden";
            void tracesContainer.offsetHeight;
            tracesContainer.style.overflowY = prevOverflowY || "";

            if (tracesContainer.classList.contains("autoscroll")) {
                // logLine.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
                scrollLogs(1, true);
            }

        }
    });
    logsContainerObserver.observe(tracesContainer);

    document.addEventListener('keydown', (e) => {
        const btn = document.activeElement;
        if (!btn.classList.contains('toolbar-button')) return;

        if (e.key === ' ' || e.key === 'Enter') {
            e.preventDefault();
            btn.classList.add('active-sim');
            btn.click();
        }
    });

    document.addEventListener('keyup', (e) => {
        if (e.key === ' ' || e.key === 'Enter') {
            const btn = document.activeElement;
            if (btn.classList.contains('toolbar-button')) {
                btn.classList.remove('active-sim');
            }
        }
    });

    const THRESHOLD_PX = 4;
    function toggleAutoscroll() {
        const paddingBottom =
            (Number(
                window.getComputedStyle(tracesContainer)
                      .paddingBottom
                      .toString()
                      .replace("px", "")
            ) | 0);

        const atBottom =
            tracesContainer.scrollTop + tracesContainer.clientHeight >=
            tracesContainer.scrollHeight -
            paddingBottom - THRESHOLD_PX;

        tracesContainer.classList.toggle("autoscroll", atBottom);
    }
    toggleAutoscroll();
    tracesContainer.addEventListener('scroll', toggleAutoscroll);

    const autoscrollObserver = new MutationObserver(entries => {
        for (let entry of entries) {
            if (entry.target.classList.contains("autoscroll")) {
                entry.target.scrollTop = entry.target.scrollHeight;
            }
        }
    });
    autoscrollObserver.observe(
        tracesContainer, { childList: true, subtree: true });
}

async function loadTracesTabContent(container, taskId) {
    /* ************************************************
    *                                                 *
    * Usage:                                          *
    * loadTracesTabContent(container, taskId)         *
    *     .then(({ tracesContainer }) => {            *
    *         if (tracesContainer) {                  *
    *             // do stuff, eg autoscroll..        *
    *         }                                       *
    *     })                                          *
    *     .catch(err => {                             *
    *         console.error(                          *
    *           "loadTracesTabContent failed:", err); *
    *     });                                         *
    ************************************************ */
    container.innerHTML += `
        <div class="loading">
            <div class="spinner"></div>
            <div class="message">Loading...</div>
        </div>
    `;
    const endpoint_prefix = getEndpointPrefix();
    const endpoint_url = (endpoint_prefix ? `${endpoint_prefix}` : '') +
        `/task_traces?task_id=${taskId}`;

    try {
        const response = await fetch(endpoint_url);
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(errorText || response.statusText || 'Unknown error');
        }
        const data = await response.json();

        if (data && data.length > 0) {
            let logHTML = '<div class="traces-log-container" tabindex="-1">';

            data.forEach((taskTrace) => {
                const richContent = processAnsiHyperlinks(taskTrace.content);
                logHTML +=
                    '<div class="trace-line" tabindex="-1" ' +
                          `data-trace-id="${taskTrace.id}" ` +
                          `data-trace-timestamp="${taskTrace.timestamp}" ` +
                          `data-trace-microsec="${taskTrace.microsec}" ` +
                          `data-trace-microsec-idx="${taskTrace.microsec_idx}" ` +
                    '>' +
                    '<span class="trace-content' +
                                  (taskTrace.is_err ? ' error-trace' : '') +
                                '">' + richContent +
                    '</span></div>';
            });
            logHTML += '</div>';

            container.querySelector(".loading").remove();
            container.innerHTML += logHTML;

            const tracesContainer =
                container.querySelector('.traces-log-container');
            setupTracesContainer(tracesContainer);

            // Explicit resolve value
            return {
                tracesContainer
            };
        } else {
            container.querySelector(".loading").remove();
            container.innerHTML +=
                '<p class="no-traces">No traces found for this task.</p>';
            return { tracesContainer: null };
        }
    } catch (error) {
        container.innerHTML =
            '<p class="error-message" style="padding: 100px 50px;">' +
            `Error loading traces: ${escapeHtml(error.message)}</p>`;
        console.error('Failed to load traces:', error);
        // Re-throw so .catch() works on caller side
        throw error;
    }
}

function scrollLogs(direction, max) {
    /* **************************
    * scrolling-toolbar buttons *
    ************************** */
    const tracesContainer = document.querySelector('.traces-log-container');
    if (!tracesContainer) return;

    const isDown = direction === 1;

    // Internal callback function
    function onScrollComplete() {
        // Callback executed when scroll reaches bottom
        console.log("Scroll finished or bottom reached");
    }

    if (max && isDown) {
        // While loop with scrollBy callback chaining
        function scrollStep() {
            console.log("scrollStep");
            const remaining = tracesContainer.scrollHeight - 
                             tracesContainer.clientHeight - 
                             tracesContainer.scrollTop;
            
            if (remaining > 0) {
                tracesContainer.scrollBy({
                    top: remaining,
                    behavior: 'smooth'
                });

                // Recheck new bottom after anim,
                // in case new traces have been
                // received in-between
                const THRESHOLD_PX = 4; // small tolerance
                const check = () => {
                    const newRemaining = tracesContainer.scrollHeight - 
                                         tracesContainer.clientHeight - 
                                         tracesContainer.scrollTop;
                    if (newRemaining > THRESHOLD_PX) {
                        setTimeout(scrollStep, 150); // recursion / looping
                    } else {
                        onScrollComplete();
                    }
                };
                requestAnimationFrame(check);
            } else {
                onScrollComplete();
            }
        }
        scrollStep();
    } else if (max) {
        tracesContainer.scrollTo({ top: 0, behavior: 'smooth' });
    } else {
        const delta = isDown ?
            tracesContainer.clientHeight : -tracesContainer.clientHeight;
        if (!isDown) {
            // remove class (if present) early
            // or interfeers if when items are being streamed
            tracesContainer.classList.remove("autoscroll");
        }
        tracesContainer.scrollBy({ top: delta, behavior: 'smooth' });
    }
}

function waitForVisible(selector, interval = 100) {
    return new Promise((resolve, reject) => {
        const start = Date.now();
        const timer = setInterval(() => {
            const el = document.querySelector(selector);
            const visible = el && !!(
                el.offsetWidth || el.offsetHeight
                || el.getClientRects().length
            );
            if (visible) {
                clearInterval(timer);
                resolve(el);
            } else if (
                !el
            ) {
                clearInterval(timer);
                reject(new Error(
                    "Promise interrupted while waiting for "
                    + selector
                ));
            }
        }, interval);
    });
}

function insertTraceToTable(taskTrace) {
    // console.log("insertTraceToTable ENTER");
    // console.table(taskTrace);
    const existingModal = document.getElementById("detailsModal");
    if (
        !existingModal
    ) {
        // if modal not showing
        return;
    }
    const taskId =
        parseInt(document.getElementById("modal-task-id").textContent, 10);

    const delay = function (delayMs) {
        return new Promise(function (resolve) {
            setTimeout(resolve, delayMs);
        });
    };

    if (taskTrace.task_id != taskId) {
        // if modal showing on another task
        return;
    }
    // console.log("taskId", taskId, "taskTrace.id", taskTrace.id);

    let tracesContainer =
        existingModal.querySelector(".traces-log-container");
    if (!tracesContainer) {
        // if traces-TAB not initialized yet
        const tabHeaders = Array.from(
            existingModal.querySelector(".tab-header").children
        );
        for (let i = 0; i < tabHeaders.length; i++) {
            const tab = tabHeaders[i];
            if (tab.classList.contains("active")) {
                if (tab.textContent.toLowerCase() != "traces") {
                    // if traces-TAB is not showing
                    return;
                } else {
                    const selectedTabContent =
                        document.getElementById("tab-traces");
                    if (selectedTabContent.querySelector(".loading")) {
                        // case "traces-TAB init state being loaded"
                        /* ************************************
                        * BEWARE :                            *
                        *   This will lead to treating events *
                        *   twice if they occur in the middle *
                        *   of the initial response.          *
                        *   (which we handle and,             *
                        *    if we don't wait, the init state *
                        *    may not be loaded yet)           *
                        ************************************ */
                        setTimeout(
                            () => {insertTraceToTable(taskTrace);},
                            50
                        );
                        return;
                    } else {
                        // case "streamed trace is first trace"
                        let logHTML =
                            '<div class="traces-log-container" tabindex="-1">';
                        logHTML += '</div>';
                        selectedTabContent.querySelector(".no-traces").remove();
                        selectedTabContent.innerHTML += logHTML;
                        tracesContainer =
                            existingModal.querySelector(".traces-log-container");
                        setupTracesContainer(tracesContainer);
                    }
                }
                break;
            }
        }
    }
    // console.log("insertTraceToTable ENTER");

    if (tracesContainer.querySelector(
        `div[data-trace-id="${taskTrace.id}"]`
    )) {
        const error = new Error(
            `Duplicate id found: ${taskTrace.id}. ` +
            'This data-trace-id value already exists ' +
            'in the "tracesContainer" DIV.'
        );
        console.warn(error.message, error.stack);
        return;
    }

    const richContent = processAnsiHyperlinks(taskTrace.content);
    const logLine = document.createElement('div');
    logLine.className = 'trace-line';
    logLine.tabIndex = -1;
    logLine.dataset.traceId = taskTrace.id;
    logLine.dataset.traceTimestamp = taskTrace.timestamp;
    logLine.dataset.traceMicrosec = Math.min(
        Number.parseInt(taskTrace.microsec, 10) || 0
    );
    logLine.dataset.traceMicrosecIdx =
        Number.parseInt(taskTrace.microsec_idx, 10) || 0;
    logLine.innerHTML =
        '<span class="trace-content' +
                      (taskTrace.is_err ? ' error-trace' : '') +
                    '">' + richContent +
        '</span>';

    // insert at proper index
    const newTimestamp   = parseInt(taskTrace.timestamp, 10);
    const newMicrosec    = parseInt(taskTrace.microsec, 10);
    const newMicrosecIdx = parseInt(taskTrace.microsec_idx, 10);
    const newId          = parseInt(taskTrace.id, 10);
    const existingLines = Array.from(tracesContainer.querySelectorAll('.trace-line'));
    let insertIndex = existingLines.length;
    for (let i = 0; i < existingLines.length; i++) {
        const line = existingLines[i];
        const lineTimestamp   = parseInt(line.dataset.traceTimestamp, 10);
        const lineMicrosec    = parseInt(line.dataset.traceMicrosec, 10);
        const lineMicrosecIdx = parseInt(line.dataset.traceMicrosecIdx, 10);
        const lineId          = parseInt(line.dataset.traceId, 10);
        if (
            newTimestamp < lineTimestamp ||
            (newTimestamp === lineTimestamp && newMicrosec < lineMicrosec) ||
            (newTimestamp === lineTimestamp && newMicrosec === lineMicrosec &&
             newMicrosecIdx < lineMicrosecIdx) ||
            (newTimestamp === lineTimestamp && newMicrosec === lineMicrosec &&
             newMicrosecIdx === lineMicrosecIdx && newId < lineId)
        ) {
            insertIndex = i;
            break;
        }
    }
    if (insertIndex === existingLines.length) {
        tracesContainer.appendChild(logLine);
    } else {
        tracesContainer.insertBefore(logLine, existingLines[insertIndex]);
    }

/*
    if (tracesContainer.classList.contains("autoscroll")) {
        // logLine.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
        scrollLogs(1, true);
    }
*/

}

function processAnsiHyperlinks(content) {
    const links = [];
    let index = 0;
    const stripControl = s => s.replace(/[\x00-\x1F\x7F]/g, '');
    const stripSgr = s => s.replace(/\x1b\[[0-9;]*m/g, '');
    const osc8Regex =
        /\x1b\]8;[^;]*;([^\x1b\x07]+?)(?:\x1b\\|\x07)([\s\S]*?)(?:\x1b\]8;;(?:\x1b\\|\x07))/g;

    content = content.replace(osc8Regex, (full, url, text) => {
        const cleanUrl = stripControl(url);
        const cleanText = stripControl(stripSgr(text));
        const filename = cleanUrl.replace(/^file:\/\//, '').split('/').pop();
        const colonPos = cleanText.lastIndexOf(':');
        const shortName = colonPos >= 0 ?
            filename + cleanText.substring(colonPos) : filename;

        links.push({ url: cleanUrl, text: shortName });
        return `XLINK${index++}X`;
    });

    let html = getAnsiUp().ansi_to_html(content);

    links.forEach((link, i) => {
        html = html.replace(`XLINK${i}X`,
            `<span class="trace-link-span"
                   onclick="showFullPath(this, '${link.url}')"
                   tabindex="0"
                   onkeydown="
                        if (
                            event.key === ' ' ||
                            event.key === 'Spacebar' ||
                            event.key === 'Enter'
                        ) {
                            event.preventDefault();
                            this.click();
                        }
                   "
             >${link.text}</span>`);
    });

    return html;
}

function showFullPath(el, path) {
    document.querySelectorAll('.full-path, .copied-toast').forEach(n => n.remove());

    const displayPath = path.replace(/^file:\/\//, '');
    const modalContent = document.querySelector('.modal-content');

    const fullDiv = document.createElement('div');
    fullDiv.className = 'full-path';
    fullDiv.textContent = displayPath;
    fullDiv.setAttribute('tabindex', '-1');
    fullDiv.style.userSelect = 'none';
    modalContent.appendChild(fullDiv);

    const rect = el.getBoundingClientRect();
    const parentRect = modalContent.getBoundingClientRect();
    fullDiv.style.top = `${rect.bottom - parentRect.top + 5}px`;
    fullDiv.style.left = `${rect.left - parentRect.left}px`;

    let timeoutId;
    const fadeOut = () => {
        fullDiv.style.transition = 'opacity 0.7s ease-out';
        fullDiv.style.opacity = '0';
        setTimeout(cleanup, 700);
    };

    timeoutId = setTimeout(fadeOut, 1500);

    const showToast = () => {
        const toast = document.createElement('div');
        toast.className = 'copied-toast';
        toast.textContent = 'Copied!';
        toast.style.top = '20px';
        toast.style.left = '50%';
        toast.style.transform = 'translateX(-50%)';
        toast.style.pointerEvents = 'none';
        modalContent.appendChild(toast);
        setTimeout(() => toast.remove(), 2000);
    };

    const trigger = (e) => {
        if (
            e.type === 'click' ||
            (e.type === 'keydown' && (e.key === ' ' || e.key === 'Enter'))
        ) {
            e.preventDefault();
            e.stopPropagation();
            navigator.clipboard.writeText(displayPath).then(showToast);
        }
        el.focus();
    };

    const onEnter = () => clearTimeout(timeoutId);

    fullDiv.addEventListener('mouseenter', onEnter);
    fullDiv.addEventListener('mouseleave', fadeOut);
    fullDiv.addEventListener('click', trigger);
    el.addEventListener('keydown', trigger);

    const cleanup = () => {
        clearTimeout(timeoutId);
        fullDiv.removeEventListener('click', trigger);
        fullDiv.removeEventListener('mouseenter', onEnter);
        fullDiv.removeEventListener('mouseleave', fadeOut);
        el.removeEventListener('keydown', trigger);
        fullDiv.remove();
    };

    el.focus();
}

