
export function attachDateTimePicker(divId, {COOKIE_PREFIX = ''} = {}) {
    const container = document.getElementById(divId);
    if (!container) {
        console.error("No picker container with id ${divId} found.");
        return;
    }

    // Prevent double-initialization
    if (container.__datetime_picker_initialized) return;
    container.__datetime_picker_initialized = true;

    // Local state for each picker
    const now = new Date();
    let currentMonth = now.getMonth();
    let currentYear = now.getFullYear();
    let selectedDate = null;
    let selectedTime = null;
    let isOpen = false;
    let focusedDay = null; // Track which day has keyboard focus

    const months = [
        'January', 'February', 'March', 'April', 'May', 'June',
        'July', 'August', 'September', 'October', 'November', 'December'
    ];
    // pattern to match date, optionally with [time with optional AMPM]
    const dateTimeRegex = /^(\d{4}-\d{2}-\d{2})(?:\s+(\d{1,2}:\d{2})(\s*(AM|PM))?)?$/i;

    // Inject global style only once (per page)
    if (!document.getElementById('dtp-style')) {
        const style = document.createElement('style');
        style.id = 'dtp-style';
        style.textContent = `
            .datetime-picker {
                position: relative;
                display: flex;
                align-items: baseline;
                line-height: 1;
            }
            .datetime-input {
                height: 18px; text-align: center;
                min-width: 144px; width: 144px;
                transition: width 0.4s ease, padding 0.4s ease,
                    opacity 0.4s ease;
                transform-origin: right; box-sizing: border-box;
                margin-left: 5px; margin-right: 8px;
                padding: 0 6px;
                border: 1px solid rgba(180,200,230,0.5);
                border-radius: 6px; font-size: 13px; color: #4d0066;
                background: linear-gradient(135deg,
                    rgba(230,240,255,0.7) 0%,
                    rgba(200,220,255,0.6) 100%);
                box-shadow:
                    0 0 12px 3px var(--shadow-color),
                    0 1px 3px rgba(0,0,0,0.06),
                    inset 0 1px 0 rgba(255,255,255,0.7);
                backdrop-filter: blur(1.5px); outline: none;
            }
            .datetime-input::placeholder {
                font-style: italic;
                font-size: 10pt;
                text-align: left;
            }
            .datetime-input-unselected {
                font-style: italic; color: #222 !important;
            }
            .datetime-input-selected-red {
                font-style: italic; color: red !important;
            }

            /* Date-Time Popup Container */
            .datetime-popup {
                background: linear-gradient(135deg,
                    rgba(230,240,255,0.7) 40%,
                    rgba(200,220,255,0.6) 100%);
                border: 1px solid rgba(180,200,230,0.45);
                border-radius: 4px; /* Preserve original radius */
                box-shadow: 0 2px 14px rgba(60,30,102,0.12);
                padding: 6px; /* Keep original padding */
                backdrop-filter: blur(2px);
                color: #4d0066; /* Purple text */
                font-family: 'Roboto', sans-serif;
                font-size: 13px;
            }
            .datetime-popup::backdrop {
                background: transparent !important;
            }

            /* Calendar Header */
            .calendar-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 3px;
                color: #4d0066;
                font-weight: bold;
            }

            /* Navigation Arrows */
            .prev-month, .next-month {
                background: linear-gradient(135deg,
                    rgba(255,255,255,0.8), rgba(230,240,255,0.6));
                border: 1px solid rgba(180,200,230,0.5);
                border-radius: 6px;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1),
                    inset 0 1px 0 rgba(255,255,255,0.7);
                cursor: pointer;
                padding: 2px 5px;
                transition: box-shadow 0.2s ease;
                color: #4d0066;
                font-weight: bold;
                line-height: 1;
                font-size: 13px;
                font-family: 'Roboto', sans-serif;
            }

            .prev-month:hover, .next-month:hover:not(:disabled) {
                box-shadow: 0 2px 12px rgba(77, 0, 102, 0.6);
            }

            .prev-month:focus, .next-month:focus {
                outline: 2px solid #4d0066;
                outline-offset: 1px;
            }

            .next-month:disabled {
                color: #888;
                cursor: not-allowed;
                background: transparent;
                border: none;
                cursor: not-allowed;
                pointer-events: none;
            }

            /* Month and Year Label */
            .month-year {
                font-size: 12px;
                font-weight: bold;
                color: #4d0066;
            }

            /* Weekdays */
            .weekdays {
                display: grid;
                grid-template-columns: repeat(7, 1fr);
                gap: 2px;
            }
            .weekdays div {
                text-align: center;
                font-weight: bold;
                font-size: 10px;
                color: #7a5299;
                white-space: nowrap; /* Prevent breaking */
                padding: 0;
                margin: 0;
            }

            /* Days */
            .days-container {
                margin: 3px 0px;
                display: grid;
                grid-template-columns: repeat(7, 1fr);
                gap: 2px;
            }
            .day {
                padding: 5px;
                font-size: 10px;
                text-align: center;
                cursor: pointer;
                border-radius: 6px;
                color: #4d0066;
                background: linear-gradient(135deg,
                    rgba(163,101,203,0.55),
                    rgba(196,168,213,0.45)
                );
                border: 1px solid rgba(112, 51, 152, 0.5);
                box-shadow: 0 1px 3px rgba(0,0,0,0.06),
                    inset 0 1px 0 rgba(255,255,255,0.7);
                transition: none;
                user-select: none;
            } 
            .day:hover:not(.disabled):not(.selected) {
                background: linear-gradient(135deg,
                    rgba(160,110,200,0.8),
                    rgba(210,180,240,0.9)
                );
            }
            .day.selected {
                background: #4d0066;
                color: white;
                border-color: #3c004d;
                box-shadow: 0 0 8px rgba(77, 0, 102, 0.8);
            }
            .day.keyboard-focused {
                outline: 2px solid #7a5299;
                outline-offset: 0px;
            }
            .day.keyboard-focused:focus {
                outline: 2px solid #4d0066;
            }
            .day.disabled {
                color: #b3a7bf;
                cursor: not-allowed;
                background: transparent;
                border: none;
            }
            .day.disabled:hover {
                background: none;
            }
            .day.empty {
                background: transparent;
                border: 1px ridge rgba(112, 51, 152, 0.2);
                box-shadow: 0 1px 3px rgba(0,0,0,0.06),
                    inset 0 1px 0 rgba(255,255,255,0.3);
            }

            /* Time Picker Container */
            .time-picker { margin: 0px 0px 6px;
                           display: flex; align-items: center;
                           justify-content: center; gap: 2px; }

            .custom-number-input {
                display: inline-block;
                position: relative;
                width: 28px;
                vertical-align: middle;
            }
            .custom-number-input input {
                width: 100%;
                padding: 1px 9px 1px 0px; /* right padding = arrow area */
                border: 1px solid #ddd;
                border-radius: 3px;
                text-align: center;
                font-size: small;
                box-sizing: border-box;
                transition: border 0.15s;
                background: rgba(255, 255, 255, 0.3);
            }
            .custom-number-input:hover input {
                border: 1.5px solid #703398;
            }
            .custom-number-input input:focus {
                border: 1.5px solid #703398;
                outline: none;
                box-shadow: 0 0 5px 2px rgba(112, 51, 152, 0.5);
            }

            .custom-arrows {
                position: absolute;
                right: 5px;
                top: 50%;
                transform: translateY(-50%);
                width: 4px;
                height: 20px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                z-index: 2;
            }
            .custom-arrow-btn {
                height: 8px;
                background: none;
                border: none;
                padding: 0;
                margin: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                color: #888;
                transition: background 0.2s, color 0.2s;
            }
            .custom-number-input:hover .custom-arrow-btn {
                color: #703398;
            }
            .custom-arrow-btn:hover {
                color: #b266cc;
                background: #f4e9ef;
            }
            .custom-arrow-icon {
                width: 0;
                height: 0;
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                display: block;
            }
            .custom-arrow-up .custom-arrow-icon {
                border-bottom: 4px solid currentColor;
            }
            .custom-arrow-down .custom-arrow-icon {
                border-top: 4px solid currentColor;
            }

            /* Custom AMPM select styling */
            .custom-ampm-wrap {
                position: relative;
                display: inline-block;
                width: 35px;
                vertical-align: middle;
            }
            .ampm-select {
                width: 100%;
                padding: 1px 8px 1.5px 2px;
                margin: 0px 0px 0px 2px;
                border: 1px solid #ddd;
                border-radius: 3px;
                font-size: small;
                line-height: normal;
                appearance: none;
                -webkit-appearance: none;
                -moz-appearance: none;
                background: rgba(255, 255, 255, 0.3);
                box-sizing: border-box;
                transition: border 0.15s;
                cursor: pointer;
            }
            .custom-ampm-wrap:hover .ampm-select,
            .ampm-select:focus {
                border: 1.5px solid #703398;
                outline: none;
                box-shadow: 0 0 5px 2px rgba(112, 51, 152, 0.5);
            }
            .custom-ampm-wrap::after {
                content: '';
                pointer-events: none;
                position: absolute;
                top: 50%;
                right: 2.5px;
                transform: translateY(-2px);
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 4px solid #888;
                width: 0;
                height: 0;
                transition: border-top-color 0.2s;
            }
            .custom-ampm-wrap:hover::after,
            .ampm-select:focus + .custom-ampm-wrap::after {
                border-top-color: #703398;
            }
            .ampm-select:active + .custom-ampm-wrap::after {
                border-top-color: #b266cc;
            }

            /* Picker Action Buttons Container */
            .picker-actions {
                display: flex;
                gap: 6px;
                justify-content: flex-end;
            }

            /* Confirm and Cancel Buttons */
            .confirm-btn {
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 6px;
                border: 1px solid rgba(112, 51, 152, 0.5);
                background: linear-gradient(135deg,
                    rgba(112, 51, 152, 0.7),
                    rgba(150, 100, 180, 0.6)
                );
                color: #4d0066; /* matching purple text color */
                cursor: pointer;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06),
                    inset 0 1px 0 rgba(255,255,255,0.7);
                transition: none;
                outline: none;
            }
            .cancel-btn {
                font-size: 11px;
                padding: 4px 8px;
                border-radius: 6px;
                border: 1px solid rgba(180,200,230,0.5);
                background: linear-gradient(135deg,
                    rgba(230,240,255,0.7), rgba(200,220,255,0.6));
                color: #4d0066;
                cursor: pointer;
                box-shadow: 0 1px 3px rgba(0,0,0,0.06),
                    inset 0 1px 0 rgba(255,255,255,0.7);
                transition: background 0.2s ease,
                    color 0.2s ease, box-shadow 0.3s ease;
                outline: none;
            }
            .confirm-btn:focus, .cancel-btn:focus {
                outline: none;
                box-shadow:
                    0 0 8px 2px rgba(77, 0, 102, 0.7),
                    inset 0 1px 0 rgba(255,255,255,0.7),
                    0 1px 3px rgba(0,0,0,0.06);
            }
            .confirm-btn.outline, .cancel-btn.outline {
                box-shadow:
                    0 0 4px 1px rgba(77, 0, 102, 0.4),
                    inset 0 1px 0 rgba(255,255,255,0.7);
            }
            .confirm-btn:hover {
                background: #4d0066;
                color: white;
                border-color: #330046;
                box-shadow: 0 0 12px rgba(77, 0, 102, 0.9);
            }
            .cancel-btn:hover {
                background: #7a5299;
                color: white;
                border-color: #5c3d7e;
                box-shadow: 0 0 8px rgba(154, 102, 179, 0.7);
            }
        `;
        document.head.appendChild(style);
    }

    // Helper: returns HTML for a custom number input (for hour/minute)
    function customNumberInputHTML(name, min, max, placeholder) {
        return `
        <span class="custom-number-input" data-name="${name}" tabindex="-1"
              onclick="if(!event.target.closest('button')) this.querySelector('input').focus();">
            <input type="text" inputmode="numeric" pattern="[0-9]*" class="${name}-input" 
                   aria-label="${name}" value="" min="${min}" max="${max}"
                   placeholder="${placeholder}" autocomplete="off">
            <span class="custom-arrows">
                <button tabindex="-1" type="button" class="custom-arrow-btn custom-arrow-up" aria-label="Increase ${name}">
                    <span class="custom-arrow-icon"></span>
                </button>
                <button tabindex="-1" type="button" class="custom-arrow-btn custom-arrow-down" aria-label="Decrease ${name}">
                    <span class="custom-arrow-icon"></span>
                </button>
            </span>
        </span>
        `;
    }

    container.innerHTML = `
        <div class="datetime-picker">
            <input type="text" class="datetime-input" spellcheck="false"
                   placeholder="Pick or type">
            <input type="hidden" id="${divId}-selected" />
            <dialog class="datetime-popup">
                <div class="calendar-header">
                    <button class="prev-month">&lt;</button>
                    <span class="month-year">${months[currentMonth]} ${currentYear}</span>
                    <button class="next-month">&gt;</button>
                </div>
                <div class="calendar-grid">
                    <div class="weekdays">
                        <div>Sun</div><div>Mon</div><div>Tue</div><div>Wed</div>
                        <div>Thu</div><div>Fri</div><div>Sat</div>
                    </div>
                    <div class="days-container"></div>
                </div>
                <div class="time-picker">
                    ${customNumberInputHTML('hour', 1, 12, 'hh')}
                    <span style="font-size: x-small;">:</span>
                    ${customNumberInputHTML('minute', 0, 59, 'mm')}
                    <span class="custom-ampm-wrap">
                        <select class="ampm-select" aria-label="AM/PM"
                                onkeydown="
                              if (event.key === 'ArrowUp' || event.key === 'ArrowDown') {
                                event.preventDefault();
                                const options = Array.from(this.options);
                                const currentIndex = this.selectedIndex;
                                let newIndex;

                                if (event.key === 'ArrowUp') {
                                  newIndex = (currentIndex === 0) ? options.length - 1 : currentIndex - 1;
                                } else {  // ArrowDown
                                  newIndex = (currentIndex === options.length - 1) ? 0 : currentIndex + 1;
                                }
                                this.selectedIndex = newIndex;
                                this.dispatchEvent(new Event('change')); // trigger change event if needed
                              }
                            "
                        >
                            <option value="PM">PM</option>
                            <option value="AM">AM</option>
                        </select>
                    </span>
                    <style>
                      .hour-input::placeholder, .minute-input::placeholder {
                          font-size: 7.5pt;
                          font-style: normal;
                          text-align: center;
                      }
                    </style>
                </div>
                <div class="picker-actions">
                    <button class="confirm-btn">OK</button>
                    <button class="cancel-btn">Cancel</button>
                </div>
            </dialog>
        </div>
    `;

    const input = container.querySelector('.datetime-input');
    const hiddenInput = document.getElementById(`${divId}-selected`);
    const popup = container.querySelector('.datetime-popup');
    const prevMonth = container.querySelector('.prev-month');
    const nextMonth = container.querySelector('.next-month');
    const monthYear = container.querySelector('.month-year');
    const daysContainer = container.querySelector('.days-container');
    const hourInput = container.querySelector('.hour-input');
    const minuteInput = container.querySelector('.minute-input');
    const ampmSelect = container.querySelector('.ampm-select');
    const confirmBtn = container.querySelector('.confirm-btn');
    const cancelBtn = container.querySelector('.cancel-btn');

    function getDaysInMonth(year, month) {
        return new Date(year, month + 1, 0).getDate();
    }

    function moveFocus(direction) {
        const daysInMonth = getDaysInMonth(currentYear, currentMonth);

        if (!focusedDay) {
            focusedDay = selectedDate &&
                selectedDate.getFullYear() === currentYear &&
                selectedDate.getMonth() === currentMonth ?
                selectedDate.getDate() : 1;
        }

        let newDay = focusedDay;

        switch (direction) {
            case 'left':
                newDay = focusedDay > 1 ? focusedDay - 1 : daysInMonth;
                break;
            case 'right':
                newDay = focusedDay < daysInMonth ? focusedDay + 1 : 1;
                break;
            case 'up':
                newDay = focusedDay - 7;
                if (newDay < 1) {
                    currentMonth--;
                    if (currentMonth < 0) {
                        currentMonth = 11;
                        currentYear--;
                    }
                    const prevMonthDays = getDaysInMonth(currentYear, currentMonth);
                    newDay = prevMonthDays + newDay;
                    renderCalendar();
                }
                break;
            case 'down':
                newDay = focusedDay + 7;
                if (newDay > daysInMonth) {
                    currentMonth++;
                    if (currentMonth > 11) {
                        currentMonth = 0;
                        currentYear++;
                    }
                    newDay = newDay - daysInMonth;
                    renderCalendar();
                }
                break;
        }

        focusedDay = Math.max(1, Math.min(newDay, getDaysInMonth(currentYear, currentMonth)));
        updateKeyboardFocus();
    }

    function updateKeyboardFocus() {
        container.querySelectorAll('.day').forEach(day => {
            day.classList.remove('keyboard-focused');
            day.removeAttribute('tabindex');
        });

        if (focusedDay) {
            const dayElement = Array.from(container.querySelectorAll('.day')).find(day =>
                !day.classList.contains('empty') && parseInt(day.textContent) === focusedDay
            );
            if (dayElement && !dayElement.classList.contains('disabled')) {
                dayElement.classList.add('keyboard-focused');
                dayElement.setAttribute('tabindex', '0');
                dayElement.focus();
            }
        }
    }

    function renderCalendar() {
        monthYear.textContent = `${months[currentMonth]} ${currentYear}`;
        const firstDay = new Date(currentYear, currentMonth, 1).getDay();
        const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
        const today = new Date();
        today.setHours(23, 59, 59, 999);

        if (
            currentYear > today.getFullYear() ||
            (currentYear === today.getFullYear() && currentMonth >= today.getMonth())
        ) {
            nextMonth.setAttribute('disabled', 'disabled');
        } else {
            nextMonth.removeAttribute('disabled');
        }

        daysContainer.innerHTML = '';

        for (let i = 0; i < firstDay; i++) {
            const emptyDay = document.createElement('div');
            emptyDay.className = 'day empty';
            daysContainer.appendChild(emptyDay);
        }

        for (let day = 1; day <= daysInMonth; day++) {
            const dayElement = document.createElement('div');
            dayElement.className = 'day';
            dayElement.textContent = day;

            const currentDate = new Date(currentYear, currentMonth, day);
            if (currentDate > today) {
                dayElement.classList.add('disabled');
            } else {
                dayElement.addEventListener('mouseup', (e) => {
                    e.stopPropagation();
                    selectDate(day);
                });
                dayElement.addEventListener('keydown', (e) => {
                    switch (e.key) {
                        case 'ArrowLeft':
                            e.preventDefault();
                            moveFocus('left');
                            break;
                        case 'ArrowRight':
                            e.preventDefault();
                            moveFocus('right');
                            break;
                        case 'ArrowUp':
                            e.preventDefault();
                            moveFocus('up');
                            break;
                        case 'ArrowDown':
                            e.preventDefault();
                            moveFocus('down');
                            break;
                        case 'Enter':
                        case ' ':
                            e.preventDefault();
                            selectDate(day);
                            break;
                    }
                });
            }

            if (selectedDate &&
                selectedDate.getFullYear() === currentYear &&
                selectedDate.getMonth() === currentMonth &&
                selectedDate.getDate() === day) {
                dayElement.classList.add('selected');
            }

            daysContainer.appendChild(dayElement);
        }
        if (
            !focusedDay &&
            today.getFullYear() === currentYear &&
            today.getMonth() === currentMonth
        ) {
            // if no valid date is in 'input',
            // focus 'today' (do not select)
            focusedDay = today.getDate();
        }

        if (focusedDay && focusedDay <= daysInMonth) {
            updateKeyboardFocus();
        }
    }

    function selectDate(day) {
        selectedDate = new Date(currentYear, currentMonth, day);
        focusedDay = day;
        renderCalendar();
    }

    function getInputDate() {
        const value = input.value.trim();
        if (!value) return [null, null, null];

        const match = value.match(dateTimeRegex);
        if (match) {
            const datePart = match[1];
            const timePart = match[2];
            const ampmPart = match[3];

            const [year, month, day] = datePart.split('-').map(Number);
            const date = new Date(year, month - 1, day); // Client-local midnight
            if (isNaN(date.getTime())) return [null, null, null, null];

            if (timePart) {
                let [hours, minutes] = timePart.split(':').map(Number);
                let ampm;

                if (ampmPart) {
                    ampm = ampmPart.toUpperCase().trim();
                    if (ampm === 'PM' && hours < 12) hours += 12;
                    if (ampm === 'AM' && hours === 12) hours = 0;
                } else {
                    ampm = hours >= 12 ? 'PM' : 'AM';
                }
                // Convert to 12h format
                let hour12 = hours % 12;
                if (hour12 === 0) hour12 = 12;

                return [date, hour12, minutes, ampm];
            } else {
                return [date, null, null, null];
            }
        } else {
            return [null, null, null, null];
        }
    }

    function parseInputDate() {
        const [date, hours, minutes, ampm] = getInputDate();
        if (!date) return;

        selectedDate = date;
        currentMonth = date.getMonth();
        currentYear = date.getFullYear();
        focusedDay = date.getDate();
        renderCalendar();

        if (hours !== null && minutes !== null) {
            hourInput.value = hours.toString().padStart(2, '0');
            minuteInput.value = minutes.toString().padStart(2, '0');
            ampmSelect.value = ampm.toUpperCase();
        }
    }

    function saveState() {
        const state = {
            date: selectedDate ? selectedDate.toISOString() : null,
            time: selectedTime
        };
        document.cookie = `${COOKIE_PREFIX}${divId}=${JSON.stringify(state)}; expires=${new Date(Date.now()+30*24*60*60*1000).toUTCString()}; path=/`;
    }

    function loadState() {
        const cookies = document.cookie.split(';');
        const cookie = cookies.find(c => c.trim().startsWith(`${COOKIE_PREFIX}${divId}=`));
        if (cookie) {
            try {
                const state = JSON.parse(cookie.split('=')[1]);
                const date = new Date(state.date);
                let dateStr = ``;
                if (state.date) {
                    selectedDate = date;
                    currentMonth = selectedDate.getMonth();
                    currentYear = selectedDate.getFullYear();
                    focusedDay = selectedDate.getDate();

                    const y = selectedDate.getFullYear();
                    const m = (selectedDate.getMonth() + 1).toString().padStart(2, '0');
                    const d = selectedDate.getDate().toString().padStart(2, '0');
                    dateStr = `${y}-${m}-${d}`;
                }
                let selectedTime = '';
                if (state.time) {
                    selectedTime = state.time;
                    const match = state.time.match(/^(\d{1,2}):(\d{2}) ?(AM|PM)?$/i);
                    if (match) {
                        hourInput.value = match[1].padStart(2, '0');
                        minuteInput.value = match[2].padStart(2, '0');
                        if (match[3]) ampmSelect.value = match[3].toUpperCase();

                        selectedTime = `${hourInput.value}:${minuteInput.value} ${ampmSelect.value}`;
                    }
                }
                input.value = `${dateStr} ${selectedTime}`.trim();
                savedSelection = {start: 0, end: input.value.length};
                hiddenInput.value = input.value;
                renderCalendar();
            } catch (e) {
                console.error('Error loading state:', e);
            }
        }
    }

    ///////////////////////////////////////////////////////////////////
    // Position dialog at the bottom-left corner of the input
    // handle window resize events
    function getCachedDialogLeftTop(popup) {
        // Function to get cached dialog rect left/top as numbers
        // (works out-of-the-box only once out of 2, so we cache it)
        let left = popup.getAttribute('data-initial-left');
        let top = popup.getAttribute('data-initial-top');

        if (left !== null && top !== null) {
            return { left: parseFloat(left), top: parseFloat(top) };
        }
        return null;
    }

    function clearDialogCache(popup) {
        popup.removeAttribute('data-initial-left');
        popup.removeAttribute('data-initial-top');
    }

    function cacheAndPositionDialog(popup, input) {
        if (!getCachedDialogLeftTop(popup)) {
            const dialogRect = popup.getBoundingClientRect();

            // dialog not consistently shows at viewport center
            // so, we take the systematic approach to
            // first adjust against that
            const viewportWidth = window.innerWidth;
            const viewportHeight = window.innerHeight;
            const centeredLeft = (viewportWidth - dialogRect.width) / 2;
            const centeredTop = (viewportHeight - dialogRect.height) / 2;

            popup.setAttribute('data-initial-left', centeredLeft);
            popup.setAttribute('data-initial-top', centeredTop);
            popup._associatedInput = input;
        }

        const popupRect = getCachedDialogLeftTop(popup);
        const inputRect = input.getBoundingClientRect();
        const offsetX = inputRect.left - popupRect.left;
        const offsetY = inputRect.bottom - popupRect.top;
        popup.style.transform = `translate(${offsetX}px, ${offsetY}px)`;
    }

    window.addEventListener('resize', () => {
        document.querySelectorAll('dialog[data-initial-left]').forEach(dialog => {
            clearDialogCache(dialog);
            if (dialog.open && dialog._associatedInput) {
                cacheAndPositionDialog(dialog, dialog._associatedInput);

                const popupRect = getCachedDialogLeftTop(dialog);
                const inputRect = dialog._associatedInput.getBoundingClientRect();
                const offsetX = inputRect.left - popupRect.left;
                const offsetY = inputRect.bottom - popupRect.top;
                dialog.style.transform = `translate(${offsetX}px, ${offsetY}px)`;
            }
        });
    });

    function openPopupDialog() {
        // Scrollbar compensation
        const scrollbarWidth = window.innerWidth - document.documentElement.clientWidth;
        const scrollbarHeight = window.innerHeight - document.documentElement.clientHeight;
        document.body.style.paddingRight = `${scrollbarWidth}px`;
        document.body.style.paddingBottom = `${scrollbarHeight}px`;
        document.body.style.overflow = 'hidden';

        popup.showModal();
    }
    popup.addEventListener('close', () => {
        document.body.style.paddingRight = '';
        document.body.style.paddingBottom = '';
        document.body.style.overflow = 'auto';
    });

    input.addEventListener('click', (e) => {
        e.preventDefault();
        openPopupDialog();
        cacheAndPositionDialog(popup, input);
        isOpen = true;
    });

    input.addEventListener('keydown', (e) => {
        if ((e.key === 'ArrowDown' || e.key === 'Down') && !isOpen) {
            openPopupDialog();
            cacheAndPositionDialog(popup, input);
            isOpen = true;
            e.preventDefault();
        }
    });
    // popup resize (some month are on a 5-rows grid, others on a 6-rows grid)
    const resizeObserver = new ResizeObserver(entries => {
        for (let entry of entries) {
            // entry.contentRect has the new size
            const dialog = entry.target;
            clearDialogCache(dialog);
            if (dialog.open && dialog._associatedInput) {
                cacheAndPositionDialog(dialog, dialog._associatedInput);
                const popupRect = getCachedDialogLeftTop(dialog);
                const inputRect = dialog._associatedInput.getBoundingClientRect();
                const offsetX = inputRect.left - popupRect.left;
                const offsetY = inputRect.bottom - popupRect.top;
                dialog.style.transform = `translate(${offsetX}px, ${offsetY}px)`;
            }
        }
    });
    resizeObserver.observe(popup);
    ///////////////////////////////////////////////////////////////////

    ///////////////////////////////////////////////////////////////////
    // 3-states with regex valid on user-inputed datetime sting
    input.addEventListener('input', () => {
      input.classList.add('datetime-input-unselected');
      input.classList.remove('datetime-input-selected-red');
    });

    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            const value = (input.textContent || input.value || "").trim();
            if (value === "") {
                // reset, not selecting any datetime
                hiddenInput.value = null;
                hourInput.value = "";
                minuteInput.value = "";
                ampmSelect.value = 'PM';
                selectedDate = null;
                selectedTime = null;
                saveState();
                const today = new Date();
                currentYear = today.getFullYear();
                currentMonth = today.getMonth();
                focusedDay = today.getDate();
                renderCalendar();
                return;
            }

            const match = value.match(dateTimeRegex);
            if (!match) {
                input.classList.add('datetime-input-selected-red');
                input.classList.remove('datetime-input-unselected');
            } else {
                input.classList.remove('datetime-input-unselected');
                input.classList.remove('datetime-input-selected-red');

                const datePart = match[1];
                const timePart = match[2];
                let ampmPart = match[3];

                let hours = null;
                let minutes = null;
                let ampm = null;
                if (timePart) {
                    const timeParts = timePart.split(':');
                    hours = parseInt(timeParts[0], 10);
                    if (hours >= 12) {
                        hours -= 12;
                        ampmPart = ' PM';
                    }
                    minutes = timeParts[1] ? parseInt(timeParts[1], 10) : 0;
                    if (ampmPart) {
                        ampm = ampmPart.trim();
                    } else {
                        ampm = 'AM'
                    }
                    hourInput.value = hours.toString().padStart(2, '0');
                    minuteInput.value = minutes.toString().padStart(2, '0');
                    ampmSelect.value = ampm.toUpperCase();
                    selectedTime = `${hourInput.value}:${minuteInput.value} ${ampmSelect.value}`;
                } else {
                    hourInput.value = "";
                    minuteInput.value = "";
                    ampmSelect.value = 'PM';
                    selectedTime = "";
                }
                selectedDate = new Date(datePart);
                currentMonth = selectedDate.getMonth();
                currentYear = selectedDate.getFullYear();
                focusedDay = selectedDate.getDate();

                input.value = `${datePart} ${selectedTime}`.trim();
                hiddenInput.value = input.value; // for external access to selected datetime
                saveState();
                renderCalendar();
            }
        }
    });
    ///////////////////////////////////////////////////////////////////

    let savedSelection = { start: 0, end: 0 };
    input.addEventListener('blur', () => {
        savedSelection = {
            start: input.selectionStart,
            end: input.selectionEnd
        };
    });
    input.addEventListener('focus', () => {
        input.setSelectionRange(savedSelection.start, savedSelection.end);
    });

    ///////////////////////////////////////////////////////////////////

    function clamp(val, min, max) {
        val = parseInt(val);
        if (isNaN(val)) return min;
        if (val < min) return min;
        if (val > max) return max;
        return val;
    }

    function setCustomInputValue(input, val, min, max) {
        val = clamp(val, min, max);
        input.value = String(val).padStart(2, '0');
    }

    function attachCustomNumberInputEvents(wrapper, min, max) {
        const input = wrapper.querySelector('input');
        const up = wrapper.querySelector('.custom-arrow-up');
        const dn = wrapper.querySelector('.custom-arrow-down');

        input.addEventListener('input', e => {
            let v = e.target.value.replace(/[^0-9]/g, '');
            if (v.trim() === '') return;

            v = clamp(v, min, max);
            if (isNaN(v)) v = min;
            input.value = String(v).padStart(2, '0');
        });

        input.addEventListener('blur', e => {
            let v = e.target.value.replace(/[^0-9]/g, '');
            if (v.trim() === '') {
                input.value = '';
                return;
            }
            v = clamp(v, min, max);
            input.value = String(v).padStart(2, '0');
        });

        up.addEventListener('mousedown', e => {
            e.preventDefault();
            e.stopPropagation();
            setCustomInputValue(input, parseInt(input.value || min) + 1, min, max);
            input.dispatchEvent(new Event('input', { bubbles: true }));
        });

        dn.addEventListener('mousedown', e => {
            e.preventDefault();
            e.stopPropagation();
            setCustomInputValue(input, parseInt(input.value || min) - 1, min, max);
            input.dispatchEvent(new Event('input', { bubbles: true }));
        });

        input.addEventListener('keydown', e => {
            if (e.key === "ArrowUp") {
                setCustomInputValue(input, parseInt(input.value || min) + 1, min, max);
                e.preventDefault();
            }
            if (e.key === "ArrowDown") {
                setCustomInputValue(input, parseInt(input.value || min) - 1, min, max);
                e.preventDefault();
            }
        });

//        input.addEventListener('focus', () => {
//            input.select();
//            input.setSelectionRange(input.selectionStart, input.selectionEnd);
//        });
        input.addEventListener('mousedown', e => {
            e.preventDefault();
            e.stopPropagation()
        });
        up.addEventListener('mousedown', e => e.stopPropagation());
        dn.addEventListener('mousedown', e => e.stopPropagation());
    }

    container.querySelectorAll('.custom-number-input').forEach(wrapper => {
        if (wrapper.dataset.name === 'hour') attachCustomNumberInputEvents(wrapper, 1, 12);
        if (wrapper.dataset.name === 'minute') attachCustomNumberInputEvents(wrapper, 0, 59);
    });

    ampmSelect.addEventListener('mousedown', e => e.stopPropagation());

    // Prev / Next month
    function createMonthChangeHandler(delta) {
      return function handleMonthChange(e) {
        e.stopPropagation();
        currentMonth += delta;
        if (currentMonth > 11) {
          currentMonth = 0;
          currentYear++;
        } else if (currentMonth < 0) {
          currentMonth = 11;
          currentYear--;
        }
        focusedDay = Math.min(focusedDay || 1,
                              getDaysInMonth(currentYear, currentMonth));
        renderCalendar();
      };
    }
    const handlePrevMonth = createMonthChangeHandler(-1);
    const handleNextMonth = createMonthChangeHandler(1);
    prevMonth.addEventListener('click', handlePrevMonth);
    nextMonth.addEventListener('click', handleNextMonth);
    [prevMonth, nextMonth].forEach((btn, idx) => {
      btn.addEventListener('keydown', (e) => {
        if (e.code === 'Space' || e.code === 'Enter'
            || e.key === ' ' || e.key === 'Enter'
        ) {
          e.preventDefault();
          if (btn === prevMonth) {
            handlePrevMonth(e);
          } else if (btn === nextMonth) {
            handleNextMonth(e);
          }
        }
      });
    });

    // Popup inner elements focus switcher
    popup.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            const focusableElements = popup.querySelectorAll(
                'button:not([tabindex="-1"]), [tabindex="0"], input, select');
            const firstElement = focusableElements[0];
            const lastElement = focusableElements[focusableElements.length - 1];

            if (e.shiftKey) {
                if (document.activeElement === firstElement) {
                    e.preventDefault();
                    lastElement.focus();
                }
            } else {
                if (document.activeElement === lastElement) {
                    e.preventDefault();
                    firstElement.focus();
                }
            }
        }
    });

    // OK/Cancel buttons
    function confirmAction(e) {
        e.stopPropagation();
        if (selectedDate) {
            // in case user clicked to change current calendar month
            // AFTER he/she selected the day, we re-render the calendar grid
            currentYear = selectedDate.getFullYear();
            currentMonth = selectedDate.getMonth();
            focusedDay = selectedDate.getDate();
            renderCalendar();

            // handle the time
            const hours = hourInput.value || '12';
            const minutes = minuteInput.value || '00';
            const ampm = ampmSelect.value;

            selectedTime = `${hours.padStart(2, '0')}:${minutes.padStart(2, '0')} ${ampm}`;

            const y = selectedDate.getFullYear();
            const m = (selectedDate.getMonth() + 1).toString().padStart(2, '0');
            const d = selectedDate.getDate().toString().padStart(2, '0');
            const dateStr = `${y}-${m}-${d}`;

            // save & close popup dialog
            input.value = `${dateStr} ${selectedTime}`;

            saveState();
            popup.close();
            isOpen = false;
            input.focus();
        }
    }
    function cancelAction(e) {
        e.stopPropagation();
        const [date, hours, minutes, ampm] = getInputDate();
        if (date === null) {
            selectedDate = null;
            const today = new Date();
            currentYear = today.getFullYear();
            currentMonth = today.getMonth();
            focusedDay = today.getDate();
            renderCalendar();
            hourInput.value = null;
            minuteInput.value = null;
            ampmSelect.value = 'PM';
        } else {
            // DON'T keep current view (year/month page)
            // and with regards to the day,
            // don't even keep focus and don't select
            // (contrary to what we do on "click outside" below)
            selectedDate = date;
            focusedDay = selectedDate;
            currentMonth = selectedDate.getMonth();
            currentYear = selectedDate.getFullYear();

            hourInput.value = (hours ? hours.toString().padStart(2, '0') : null);
            minuteInput.value = (minutes ? minutes.toString().padStart(2, '0') : null);
            ampmSelect.value = (ampm ? ampm.toUpperCase() : 'PM');
        }
        renderCalendar();
        popup.close();
        isOpen = false;
        input.focus();
    }
    confirmBtn.addEventListener('click', confirmAction);
    cancelBtn.addEventListener('click', cancelAction);
    // Add keydown event for the spacebar to both buttons
    function handleKeyDown(e) {
        if (e.code === 'Space' || e.key === ' ') {
            e.preventDefault(); // Prevent scrolling
            // Dispatch the same action based on the focused element
            if (e.target === confirmBtn) {
                confirmAction(e);
            } else if (e.target === cancelBtn) {
                cancelAction(e);
            }
        }
    }
    confirmBtn.addEventListener('keydown', handleKeyDown);
    cancelBtn.addEventListener('keydown', handleKeyDown);

    // popup focus lost (click poutside => hide)
    popup.addEventListener('click', (e) => {
        // hide dialog on backdrop-clicked event
        e.stopPropagation();
        const popupRect = popup.getBoundingClientRect();
        const isClickInside = (
            e.clientX >= popupRect.left &&
            e.clientX <= popupRect.right &&
            e.clientY >= popupRect.top &&
            e.clientY <= popupRect.bottom
        );

        if (!isClickInside) {
            // click on the dialog backdrop
            e.preventDefault();
            e.stopPropagation();
            const [date, hours, minutes, ampm] = getInputDate();
            if (date === null) {
                selectedDate = null;
            } else {
                // keep current view (year/month page
                // just with regards to the day,
                // keep focus but don't select
                selectedDate = date;
            }
            renderCalendar();
            popup.close();
            isOpen = false;
            input.focus();
        }
    });

    popup.addEventListener('cancel', (e) => {
        event.preventDefault();
        if (isOpen) {
            popup.close();
            isOpen = false;
            input.focus();
        }
    });

    renderCalendar();
    loadState();
}

