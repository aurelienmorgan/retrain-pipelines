/* **
* GanttTimeline - A reusable library to add
* Gantt-style timeline visualization to any HTML table
* 
* Usage:
*   const timeline = new GanttTimeline('myTableId', 'myTimelineColumnHeader');
* 
* Timeline cells should have data-start-timestamp and
* data-end-timestamp attributes with UNIX TIMESTAMPS (milliseconds):
*   <td data-start-timestamp="1704096000000" data-end-timestamp="1704441600000"></td>
* 
* For ongoing tasks without end time, omit data-end-timestamp and it will use current time:
*   <td data-start-timestamp="1704096000000"></td>
*
* NOTE : sets a unique dataset.id attribute on rows if none is already present.
** */

class GanttTimeline {
    constructor(tableId, timelineColumnHeader = 'Timeline') {
        this.tableId = tableId;
        this.timelineColumnHeader = timelineColumnHeader;
        this.table = document.getElementById(tableId);

        if (!this.table) {
            throw new Error(`Table with id "${tableId}" not found`);
        }

        this.timelineColumnIndex = null;
        this.rowsMap = new Map();
        this.globalStart = null;
        this.globalEnd = null;
        this.isUpdating = false;
        this.autoRefreshInterval = null;
        this.hasOngoingTasks = false;

        this.init();
    }

    init() {
        this.findTimelineColumn();
        this.parseRows();
        this.render();
        this.attachMutationObserver();
        this.startAutoRefresh();
    }

    // Generate a unique ID for rows
    generateRowId() {
        return 'row-' + Date.now() + '-' + Math.random().toString(36).substr(2, 9);
    }

    findTimelineColumn() {
        const headerRow = this.table.querySelector('thead tr');
        if (!headerRow) {
            throw new Error('Table must have a <thead> with header row');
        }

        const headers = Array.from(headerRow.querySelectorAll('th'));

        headers.forEach((th, index) => {
            const headerText = th.textContent.trim();
            if (headerText === this.timelineColumnHeader) {
                this.timelineColumnIndex = index;
            }
        });

        if (this.timelineColumnIndex === null) {
            throw new Error(`Timeline column "${this.timelineColumnHeader}" not found`);
        }
    }

    parseRows() {
        const tbody = this.table.querySelector('tbody');
        if (!tbody) return;

        this.rowsMap.clear();
        this.hasOngoingTasks = false;
        const bodyRows = Array.from(tbody.querySelectorAll('tr'));

        bodyRows.forEach((tr) => {
            // Set row ID if not present
            if (!tr.dataset.id) {
                tr.dataset.id = this.generateRowId();
            }

            const cells = tr.querySelectorAll('td');
            const timelineCell = cells[this.timelineColumnIndex];

            if (timelineCell) {
                const startTimestamp = timelineCell.dataset.startTimestamp;
                const endTimestamp = timelineCell.dataset.endTimestamp;

                if (startTimestamp) {
                    const start = parseInt(startTimestamp, 10);
                    let end = null;
                    let isOngoing = false;

                    if (endTimestamp) {
                        end = parseInt(endTimestamp, 10);
                    } else {
                        // No end timestamp - use current time
                        end = Date.now();
                        isOngoing = true;
                        this.hasOngoingTasks = true;
                    }

                    if (!isNaN(start) && !isNaN(end)) {
                        this.rowsMap.set(tr.dataset.id, {
                            id: tr.dataset.id,
                            element: tr,
                            timelineCell,
                            start: start,
                            end: end,
                            isOngoing: isOngoing
                        });
                    }
                }
            }
        });

        this.updateGlobalBounds();
    }

    updateGlobalBounds() {
        if (this.rowsMap.size === 0) {
            this.globalStart = null;
            this.globalEnd = null;
            return;
        }

        const rows = Array.from(this.rowsMap.values());
        this.globalStart = Math.min(...rows.map(r => r.start));
        this.globalEnd = Math.max(...rows.map(r => r.end));
    }

    calculateBarPosition(row) {
        if (!this.globalStart || !this.globalEnd) return { left: 0, width: 100 };

        const totalDuration = this.globalEnd - this.globalStart;
        if (totalDuration === 0) return { left: 0, width: 100 };

        const taskStart = row.start - this.globalStart;
        const taskDuration = row.end - row.start;

        const leftPercent = (taskStart / totalDuration) * 100;
        const widthPercent = (taskDuration / totalDuration) * 100;

        return { left: leftPercent, width: widthPercent };
    }

    formatTimestamp(timestamp) {
        const date = new Date(timestamp);
        return date.toLocaleString('en-US', { 
            month: 'short', 
            day: 'numeric', 
            hour: '2-digit', 
            minute: '2-digit' 
        });
    }

    formatTimestampForInput(timestamp) {
        const date = new Date(timestamp);
        const year = date.getFullYear();
        const month = String(date.getMonth() + 1).padStart(2, '0');
        const day = String(date.getDate()).padStart(2, '0');
        const hours = String(date.getHours()).padStart(2, '0');
        const minutes = String(date.getMinutes()).padStart(2, '0');
        return `${year}-${month}-${day}T${hours}:${minutes}`;
    }

    formatEpochDiff(startMs, endMs) {
        let remaining = endMs - startMs;

        const hours = Math.floor(remaining / 3600000);
        remaining %= 3600000;

        const minutes = Math.floor(remaining / 60000);
        remaining %= 60000;

        const seconds = Math.floor(remaining / 1000);
        remaining %= 1000;

        // fractional milliseconds - pad to 6 digits (milliseconds + 3 zeros for microseconds)
        const fractional = remaining.toString().padStart(3, '0') + '000';

        return `${hours}:${minutes.toString().padStart(2, '0')}:`+ 
               `${seconds.toString().padStart(2, '0')}.${fractional}`;
    }

    getRowLabel(row) {
        const cells = Array.from(row.element.querySelectorAll('td'));
        for (let i = 0; i < cells.length; i++) {
            if (i !== this.timelineColumnIndex) {
                const text = cells[i].textContent.trim();
                if (text) return text;
            }
        }
        return '';
    }

    renderRow(row) {
        // Update end time for ongoing tasks
        if (row.isOngoing) {
            row.end = Date.now();
        }

        const { left, width } = this.calculateBarPosition(row);
        const label = this.getRowLabel(row);
        const showLabel = width > 15;
        const ongoingClass = row.isOngoing ? 'ongoing' : '';

        row.timelineCell.classList.add("gantt-timeline-cell");

        // remove any timeline from previous rendering
        // (it might need updating, e.g. if major bounds changed)
        const oldTimeleineElement = row.timelineCell.querySelector('.gantt-timeline-container');
        if (oldTimeleineElement) oldTimeleineElement.remove();
        // create and prepend to content
        // (to preserve any unrelated content)
        const newTimeleineElement = document.createElement('div');
        newTimeleineElement.className = 'gantt-timeline-container';
        newTimeleineElement.innerHTML =
            `<div class="gantt-timeline-bar ${ongoingClass}" ` +
                 `style="left: ${left}%; width: ${width}%;"` +
            '>' +
                (showLabel ? label : '') +
            '</div>'
        ;
        row.timelineCell.prepend(newTimeleineElement);

        row.timelineCell.title =
            `${this.formatTimestamp(row.start)} → ` +
            `${row.isOngoing ? 'NOW' : this.formatTimestamp(row.end)} ` +
            `(${this.formatEpochDiff(row.start, row.end)})`
    }

    render() {
        if (this.isUpdating) return;

        // Update global bounds before rendering (for ongoing tasks)
        if (this.hasOngoingTasks) {
            this.rowsMap.forEach(row => {
                if (row.isOngoing) {
                    row.end = Date.now();
                }
            });
            this.updateGlobalBounds();
        }

        this.rowsMap.forEach(row => this.renderRow(row));
    }

    editStart(rowId) {
        const row = this.rowsMap.get(rowId);
        if (!row) return;

        const newStart = prompt('Enter new start time (YYYY-MM-DDTHH:MM):', this.formatTimestampForInput(row.start));
        if (newStart === null) return;

        const newTimestamp = new Date(newStart).getTime();
        if (isNaN(newTimestamp)) {
            alert('Invalid date format');
            return;
        }

        // For ongoing tasks, check against current time
        const effectiveEnd = row.isOngoing ? Date.now() : row.end;
        if (newTimestamp >= effectiveEnd) {
            alert('Start time must be before ' + (row.isOngoing ? 'current time' : 'end time'));
            return;
        }

        this.isUpdating = true;

        // Update the DOM
        row.timelineCell.setAttribute('data-start-timestamp', newTimestamp.toString());

        // Update the row object
        row.start = newTimestamp;

        // Recalculate bounds based on updated row objects
        if (this.hasOngoingTasks) {
            this.rowsMap.forEach(r => {
                if (r.isOngoing) {
                    r.end = Date.now();
                }
            });
        }
        this.updateGlobalBounds();

        // Re-render ALL rows (because bounds changed, all bars need repositioning)
        this.rowsMap.forEach(r => this.renderRow(r));

        this.isUpdating = false;
    }

    editEnd(rowId) {
        const row = this.rowsMap.get(rowId);
        if (!row) return;

        const defaultEnd = row.isOngoing ? this.formatTimestampForInput(Date.now()) : this.formatTimestampForInput(row.end);
        const newEnd = prompt('Enter end time (YYYY-MM-DDTHH:MM):', defaultEnd);
        if (newEnd === null) return;

        const newTimestamp = new Date(newEnd).getTime();
        if (isNaN(newTimestamp)) {
            alert('Invalid date format');
            return;
        }
        if (newTimestamp <= row.start) {
            alert('End time must be after start time');
            return;
        }

        this.isUpdating = true;

        // Update the DOM
        row.timelineCell.setAttribute('data-end-timestamp', newTimestamp.toString());

        // Update the row object
        row.end = newTimestamp;
        row.isOngoing = false; // No longer ongoing once end is set

        // Check if we still have ongoing tasks
        this.hasOngoingTasks = Array.from(this.rowsMap.values()).some(r => r.isOngoing);

        // Recalculate bounds
        this.updateGlobalBounds();

        // Re-render ALL rows
        this.rowsMap.forEach(r => this.renderRow(r));

        this.isUpdating = false;
    }

    deleteRow(rowId) {
        const row = this.rowsMap.get(rowId);
        if (!row) return;

        if (confirm('Are you sure you want to delete this row?')) {
            this.isUpdating = true;

            // Remove from DOM
            row.element.remove();

            // Re-parse all rows (this will exclude the deleted row)
            this.parseRows();

            // Re-render all remaining rows with updated bounds
            this.rowsMap.forEach(r => this.renderRow(r));

            this.isUpdating = false;
        }
    }

    refresh() {
        if (this.isUpdating) return;
        this.parseRows();
        this.render();
    }

    startAutoRefresh() {
        // Refresh every second if there are ongoing tasks
        this.autoRefreshInterval = setInterval(() => {
            if (this.hasOngoingTasks && !this.isUpdating) {
                this.render();
            }
        }, 1000);
    }

    stopAutoRefresh() {
        if (this.autoRefreshInterval) {
            clearInterval(this.autoRefreshInterval);
            this.autoRefreshInterval = null;
        }
    }

    attachMutationObserver() {
        const tbody = this.table.querySelector('tbody');
        if (!tbody) return;

        const observer = new MutationObserver((mutations) => {
            if (this.isUpdating) return;

            let shouldRefresh = false;
            mutations.forEach(mutation => {
                if (mutation.type === 'childList' && mutation.addedNodes.length > 0) {
                    shouldRefresh = true;
                }
            });

            if (shouldRefresh) {
                this.refresh();
            }
        });

        observer.observe(tbody, {
            childList: true
        });

        this.observer = observer;
    }

    destroy() {
        if (this.observer) {
            this.observer.disconnect();
        }
        this.stopAutoRefresh();
    }
}

if (typeof module !== 'undefined' && module.exports) {
    module.exports = GanttTimeline;
}