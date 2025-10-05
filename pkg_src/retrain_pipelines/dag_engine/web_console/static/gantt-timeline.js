/**
 * GanttTimeline - A reusable library to add Gantt-style timeline visualization to any HTML table
 * 
 * Usage:
 *   const timeline = new GanttTimeline('myTableId', 'Timeline');
 * 
 * Timeline cells should have data-start-timestamp and data-end-timestamp attributes with UNIX TIMESTAMPS (milliseconds):
 *   <td data-start-timestamp="1704096000000" data-end-timestamp="1704441600000"></td>
 * 
 * For ongoing tasks without end time, omit data-end-timestamp and it will use current time:
 *   <td data-start-timestamp="1704096000000"></td>
 */

class GanttTimeline {
    constructor(tableId, timelineColumnHeader = 'Timeline') {
        this.tableId = tableId;
        this.timelineColumnHeader = timelineColumnHeader;
        this.table = document.getElementById(tableId);
        
        if (!this.table) {
            throw new Error(`Table with id "${tableId}" not found`);
        }
        
        this.timelineColumnIndex = null;
        this.rows = [];
        this.globalStart = null;
        this.globalEnd = null;
        this.isUpdating = false;
        this.autoRefreshInterval = null;
        this.hasOngoingTasks = false;
        
        this.init();
    }
    
    init() {
        this.injectStyles();
        this.findTimelineColumn();
        this.parseRows();
        this.render();
        this.attachMutationObserver();
        this.startAutoRefresh();
    }
    
    injectStyles() {
        if (document.getElementById('gantt-timeline-styles')) return;
        
        const style = document.createElement('style');
        style.id = 'gantt-timeline-styles';
        style.textContent = `
            .gantt-timeline-cell {
                position: relative;
                min-height: 50px;
                background: linear-gradient(to right, #f9f9f9 0%, #f9f9f9 100%);
                padding: 5px !important;
                min-width: 250px;
                vertical-align: middle;
            }
            
            .gantt-timeline-container {
                position: relative;
                width: 100%;
                height: 35px;
                background: #e8e8e8;
                border-radius: 4px;
                overflow: visible;
            }
            
            .gantt-timeline-bar {
                position: absolute;
                height: 100%;
                top: 0;
                border-radius: 4px;
                display: flex;
                align-items: center;
                justify-content: center;
                color: white;
                font-size: 11px;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0,0,0,0.2);
                transition: all 0.3s ease;
                cursor: pointer;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            }
            
            .gantt-timeline-bar.ongoing {
                background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                animation: pulse 2s ease-in-out infinite;
            }
            
            @keyframes pulse {
                0%, 100% { opacity: 1; }
                50% { opacity: 0.8; }
            }
            
            .gantt-timeline-bar:hover {
                box-shadow: 0 3px 6px rgba(0,0,0,0.3);
                filter: brightness(1.1);
            }
            
            .gantt-timeline-controls {
                display: flex;
                gap: 5px;
                margin-top: 8px;
                justify-content: center;
            }
            
            .gantt-action-btn {
                padding: 4px 10px;
                font-size: 11px;
                cursor: pointer;
                border: none;
                border-radius: 3px;
                transition: all 0.2s;
                white-space: nowrap;
            }
            
            .gantt-btn-edit-start {
                background-color: #2196F3;
                color: white;
            }
            
            .gantt-btn-edit-start:hover {
                background-color: #0b7dda;
            }
            
            .gantt-btn-edit-end {
                background-color: #4CAF50;
                color: white;
            }
            
            .gantt-btn-edit-end:hover {
                background-color: #45a049;
            }
            
            .gantt-btn-delete {
                background-color: #f44336;
                color: white;
            }
            
            .gantt-btn-delete:hover {
                background-color: #da190b;
            }
            
            .gantt-timeline-info {
                font-size: 10px;
                color: #666;
                text-align: center;
                margin-top: 3px;
            }
            
            .gantt-timeline-info.ongoing {
                color: #f5576c;
                font-weight: bold;
            }
        `;
        document.head.appendChild(style);
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
        
        this.rows = [];
        this.hasOngoingTasks = false;
        const bodyRows = Array.from(tbody.querySelectorAll('tr'));
        
        bodyRows.forEach((tr, index) => {
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
                        this.rows.push({
                            id: `row-${index}-${Date.now()}`,
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
        if (this.rows.length === 0) {
            this.globalStart = null;
            this.globalEnd = null;
            return;
        }
        
        this.globalStart = Math.min(...this.rows.map(r => r.start));
        this.globalEnd = Math.max(...this.rows.map(r => r.end));
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
        
        row.timelineCell.className = 'gantt-timeline-cell';
        row.timelineCell.innerHTML = `
            <div class="gantt-timeline-container">
                <div class="gantt-timeline-bar ${ongoingClass}" style="left: ${left}%; width: ${width}%;">
                    ${showLabel ? label : ''}
                </div>
            </div>
            <div class="gantt-timeline-info ${ongoingClass}">
                ${this.formatTimestamp(row.start)} → ${row.isOngoing ? 'NOW' : this.formatTimestamp(row.end)}
            </div>
            <div class="gantt-timeline-controls">
                <button class="gantt-action-btn gantt-btn-edit-start">Change Start</button>
                <button class="gantt-action-btn gantt-btn-edit-end">${row.isOngoing ? 'Set End' : 'Change End'}</button>
                <button class="gantt-action-btn gantt-btn-delete">Delete</button>
            </div>
        `;
        
        const editStartBtn = row.timelineCell.querySelector('.gantt-btn-edit-start');
        const editEndBtn = row.timelineCell.querySelector('.gantt-btn-edit-end');
        const deleteBtn = row.timelineCell.querySelector('.gantt-btn-delete');
        
        editStartBtn.onclick = () => this.editStart(row);
        editEndBtn.onclick = () => this.editEnd(row);
        deleteBtn.onclick = () => this.deleteRow(row);
    }
    
    render() {
        if (this.isUpdating) return;
        
        // Update global bounds before rendering (for ongoing tasks)
        if (this.hasOngoingTasks) {
            this.rows.forEach(row => {
                if (row.isOngoing) {
                    row.end = Date.now();
                }
            });
            this.updateGlobalBounds();
        }
        
        this.rows.forEach(row => this.renderRow(row));
    }
    
    editStart(row) {
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
            this.rows.forEach(r => {
                if (r.isOngoing) {
                    r.end = Date.now();
                }
            });
        }
        this.updateGlobalBounds();
        
        // Re-render ALL rows (because bounds changed, all bars need repositioning)
        this.rows.forEach(r => this.renderRow(r));
        
        this.isUpdating = false;
    }
    
    editEnd(row) {
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
        this.hasOngoingTasks = this.rows.some(r => r.isOngoing);
        
        // Recalculate bounds
        this.updateGlobalBounds();
        
        // Re-render ALL rows
        this.rows.forEach(r => this.renderRow(r));
        
        this.isUpdating = false;
    }
    
    deleteRow(row) {
        if (confirm('Are you sure you want to delete this row?')) {
            this.isUpdating = true;
            
            // Remove from DOM
            row.element.remove();
            
            // Re-parse all rows (this will exclude the deleted row)
            this.parseRows();
            
            // Re-render all remaining rows with updated bounds
            this.rows.forEach(r => this.renderRow(r));
            
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