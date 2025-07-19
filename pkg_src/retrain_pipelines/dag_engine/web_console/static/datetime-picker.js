export function attachDateTimePicker(divId) {
    const container = document.getElementById(divId);
    if (!container) return;

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

    // Inject global style only once (per page)
    if (!document.getElementById('dtp-style')) {
        const style = document.createElement('style');
        style.id = 'dtp-style';
        style.textContent = `
            .datetime-picker {
                position: relative;
                display: inline-block;
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
                box-shadow: 0 1px 3px rgba(0,0,0,0.06),
                    inset 0 1px 0 rgba(255,255,255,0.7);
                backdrop-filter: blur(1.5px); outline: none;
            }
            .datetime-popup {
                position: absolute;
                top: 100%;
                left: 0;
                background: white;
                border: 1px solid #ddd;
                border-radius: 4px;
                box-shadow: 0 2px 10px rgba(0,0,0,0.1);
                z-index: 1000;
                padding: 15px;
                min-width: 280px;
            }
            .calendar-header {
                display: flex;
                justify-content: space-between;
                align-items: center;
                margin-bottom: 10px;
            }
            .prev-month, .next-month {
                background: none;
                border: none;
                cursor: pointer;
                font-size: 18px;
                padding: 5px;
                border-radius: 3px;
            }
            .prev-month:focus, .next-month:focus {
                outline: 2px solid #007bff;
                outline-offset: 1px;
            }
            .month-year { font-weight: bold; }
            .weekdays {
                display: grid;
                grid-template-columns: repeat(7, 1fr);
                gap: 2px;
                margin-bottom: 5px;
            }
            .weekdays div {
                text-align: center;
                font-weight: bold;
                padding: 5px;
                font-size: 12px;
            }
            .days-container { display: grid; grid-template-columns: repeat(7, 1fr); gap: 2px; }
            .day { padding: 8px; text-align: center; cursor: pointer; border-radius: 3px; }
            .day:hover { background: #f0f0f0; }
            .day.selected { background: #007bff; color: white; }
            .day.keyboard-focused { outline: 2px solid #007bff; outline-offset: 1px; }
            .day.disabled { color: #ccc; cursor: not-allowed; }
            .day.disabled:hover { background: none; }
            .time-picker { margin: 15px 0; display: flex; align-items: center; justify-content: center; gap: 5px; }

            .custom-number-input {
                display: inline-block;
                position: relative;
                width: 38px; /* less wide */
                vertical-align: middle;
            }
            .custom-number-input input {
                width: 100%;
                padding: 5px 14px 5px 5px; /* right padding = arrow area */
                border: 1px solid #ddd;
                border-radius: 3px;
                text-align: center;
                font-size: 15px;
                box-sizing: border-box;
                transition: border 0.15s;
            }
            .custom-number-input:hover input {
                border: 1.5px solid #1976d2;
            }
            .custom-number-input input:focus {
                border: 1.5px solid #1976d2;
                outline: none;
            }
            .custom-arrows {
                position: absolute;
                right: 2px;
                top: 50%;
                transform: translateY(-50%);
                width: 12px;
                height: 22px;
                display: flex;
                flex-direction: column;
                justify-content: center;
                z-index: 2;
            }
            .custom-arrow-btn {
                width: 12px;
                height: 9px;
                background: none;
                border: none;
                padding: 0;
                margin: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                cursor: pointer;
                color: #999;
                transition: background 0.2s, color 0.2s;
            }
            .custom-number-input:hover .custom-arrow-btn {
                color: #1976d2;
            }
            .custom-arrow-btn:hover {
                color: #e91e63;
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
                width: 56px;
                vertical-align: middle;
            }
            .ampm-select {
                width: 100%;
                padding: 5px 18px 5px 5px;
                border: 1px solid #ddd;
                border-radius: 3px;
                font-size: 15px;
                appearance: none;
                -webkit-appearance: none;
                -moz-appearance: none;
                background: transparent;
                box-sizing: border-box;
                transition: border 0.15s;
                cursor: pointer;
            }
            .custom-ampm-wrap:hover .ampm-select,
            .ampm-select:focus {
                border: 1.5px solid #1976d2;
                outline: none;
            }
            .custom-ampm-wrap::after {
                content: '';
                pointer-events: none;
                position: absolute;
                top: 50%;
                right: 8px;
                transform: translateY(-2px);
                border-left: 3px solid transparent;
                border-right: 3px solid transparent;
                border-top: 4px solid #999;
                width: 0;
                height: 0;
                transition: border-top-color 0.2s;
            }
            .custom-ampm-wrap:hover::after,
            .ampm-select:focus + .custom-ampm-wrap::after {
                border-top-color: #1976d2;
            }
            .ampm-select:active + .custom-ampm-wrap::after {
                border-top-color: #e91e63;
            }

            .picker-actions { display: flex; gap: 10px; justify-content: flex-end; }
            .confirm-btn, .cancel-btn { padding: 6px 12px; border: none; border-radius: 3px; cursor: pointer; }
            .confirm-btn { background: #007bff; color: white; }
            .cancel-btn { background: #6c757d; color: white; }
        `;
        document.head.appendChild(style);
    }

    // Helper: returns HTML for a custom number input (for hour/minute)
    function customNumberInputHTML(name, min, max, placeholder) {
        return `
        <span class="custom-number-input" data-name="${name}" tabindex="-1">
            <input type="text" inputmode="numeric" pattern="[0-9]*" class="${name}-input" 
                   aria-label="${name}" autocomplete="off"
                   value="" min="${min}" max="${max}" placeholder="${placeholder}">
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
            <input type="text" class="datetime-input" placeholder="Select date and time" autocomplete="off">
            <div class="datetime-popup" style="display: none;">
                <div class="calendar-header">
                    <button class="prev-month">&lt;</button>
                    <span class="month-year">${months[currentMonth]} ${currentYear}</span>
                    <button class="next-month">&gt;</button>
                </div>
                <div class="calendar-grid">
                    <div class="weekdays">
                        <div>Sun</div><div>Mon</div><div>Tue</div><div>Wed</div><div>Thu</div><div>Fri</div><div>Sat</div>
                    </div>
                    <div class="days-container"></div>
                </div>
                <div class="time-picker">
                    ${customNumberInputHTML('hour', 1, 12, 'HH')}
                    <span>:</span>
                    ${customNumberInputHTML('minute', 0, 59, 'MM')}
                    <span class="custom-ampm-wrap">
                        <select class="ampm-select" aria-label="AM/PM">
                            <option value="AM">AM</option>
                            <option value="PM">PM</option>
                        </select>
                    </span>
                </div>
                <div class="picker-actions">
                    <button class="confirm-btn">OK</button>
                    <button class="cancel-btn">Cancel</button>
                </div>
            </div>
        </div>
    `;

    const input = container.querySelector('.datetime-input');
    const popup = container.querySelector('.datetime-popup');
    const monthYear = container.querySelector('.month-year');
    const daysContainer = container.querySelector('.days-container');
    const hourInput = container.querySelector('.hour-input');
    const minuteInput = container.querySelector('.minute-input');
    const ampmSelect = container.querySelector('.ampm-select');

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
                    monthYear.textContent = `${months[currentMonth]} ${currentYear}`;
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
                    monthYear.textContent = `${months[currentMonth]} ${currentYear}`;
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
        const firstDay = new Date(currentYear, currentMonth, 1).getDay();
        const daysInMonth = new Date(currentYear, currentMonth + 1, 0).getDate();
        const today = new Date();
        today.setHours(23, 59, 59, 999);

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

        if (focusedDay && focusedDay <= daysInMonth) {
            updateKeyboardFocus();
        }
    }

    function selectDate(day) {
        selectedDate = new Date(currentYear, currentMonth, day);
        focusedDay = day;
        renderCalendar();
    }

    function parseInputDate() {
        const value = input.value.trim();
        if (!value) return;

        const dateTimeRegex = /(\d{4}-\d{2}-\d{2})\s*(\d{1,2}:\d{2})\s*(AM|PM)?/i;
        const match = value.match(dateTimeRegex);

        if (match) {
            const datePart = match[1];
            const timePart = match[2];
            const ampmPart = match[3];

            try {
                const [year, month, day] = datePart.split('-').map(Number);
                const date = new Date(year, month - 1, day); // Client-local midnight
                if (!isNaN(date.getTime())) {
                    selectedDate = date;
                    currentMonth = date.getMonth();
                    currentYear = date.getFullYear();
                    focusedDay = date.getDate();
                    monthYear.textContent = `${months[currentMonth]} ${currentYear}`;
                    renderCalendar();
                }

                if (timePart) {
                    const [hours, minutes] = timePart.split(':');
                    let hour12 = parseInt(hours);
                    if (hour12 === 0) hour12 = 12;
                    if (hour12 > 12) hour12 = hour12 - 12;

                    hourInput.value = hour12.toString().padStart(2, '0');
                    minuteInput.value = minutes.padStart(2, '0');

                    if (ampmPart) {
                        ampmSelect.value = ampmPart.toUpperCase();
                    } else {
                        ampmSelect.value = parseInt(hours) >= 12 ? 'PM' : 'AM';
                    }
                }
            } catch (e) {
                console.error('Invalid date format');
            }
        }
    }

    function saveState() {
        const state = {
            date: selectedDate ? selectedDate.toISOString() : null,
            time: selectedTime,
            inputValue: input.value
        };
        document.cookie = `${divId}_dateTimePicker=${JSON.stringify(state)}; expires=${new Date(Date.now()+30*24*60*60*1000).toUTCString()}; path=/`;
    }

    function loadState() {
        const cookies = document.cookie.split(';');
        const cookie = cookies.find(c => c.trim().startsWith(`${divId}_dateTimePicker=`));
        if (cookie) {
            try {
                const state = JSON.parse(cookie.split('=')[1]);
                if (state.date) {
                    selectedDate = new Date(state.date);
                    currentMonth = selectedDate.getMonth();
                    currentYear = selectedDate.getFullYear();
                    focusedDay = selectedDate.getDate();
                    monthYear.textContent = `${months[currentMonth]} ${currentYear}`;
                }
                if (state.time) {
                    selectedTime = state.time;
                    const match = state.time.match(/^(\d{1,2}):(\d{2}) ?(AM|PM)?$/i);
                    if (match) {
                        hourInput.value = match[1].padStart(2, '0');
                        minuteInput.value = match[2].padStart(2, '0');
                        if (match[3]) ampmSelect.value = match[3].toUpperCase();
                    }
                }
                if (state.inputValue) {
                    input.value = state.inputValue;
                }
                renderCalendar();
            } catch (e) {
                console.error('Error loading state:', e);
            }
        }
    }

    input.addEventListener('click', () => {
        popup.style.display = 'block';
        isOpen = true;
    });

    input.addEventListener('blur', () => {
        parseInputDate();
        saveState();
    });

    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') {
            parseInputDate();
            saveState();
        }
    });

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
            v = clamp(v, min, max);
            if (isNaN(v)) v = min;
            input.value = String(v).padStart(2, '0');
        });

        input.addEventListener('blur', e => {
            let v = clamp(e.target.value, min, max);
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

        input.addEventListener('focus', () => { input.select(); });
        input.addEventListener('mousedown', e => e.stopPropagation());
        up.addEventListener('mousedown', e => e.stopPropagation());
        dn.addEventListener('mousedown', e => e.stopPropagation());
    }

    container.querySelectorAll('.custom-number-input').forEach(wrapper => {
        if (wrapper.dataset.name === 'hour') attachCustomNumberInputEvents(wrapper, 1, 12);
        if (wrapper.dataset.name === 'minute') attachCustomNumberInputEvents(wrapper, 0, 59);
    });

    ampmSelect.addEventListener('mousedown', e => e.stopPropagation());

    container.querySelector('.prev-month').addEventListener('click', (e) => {
        e.stopPropagation();
        currentMonth--;
        if (currentMonth < 0) {
            currentMonth = 11;
            currentYear--;
        }
        monthYear.textContent = `${months[currentMonth]} ${currentYear}`;
        focusedDay = Math.min(focusedDay || 1, getDaysInMonth(currentYear, currentMonth));
        renderCalendar();
    });

    container.querySelector('.next-month').addEventListener('click', (e) => {
        e.stopPropagation();
        currentMonth++;
        if (currentMonth > 11) {
            currentMonth = 0;
            currentYear++;
        }
        monthYear.textContent = `${months[currentMonth]} ${currentYear}`;
        focusedDay = Math.min(focusedDay || 1, getDaysInMonth(currentYear, currentMonth));
        renderCalendar();
    });

    popup.addEventListener('keydown', (e) => {
        if (e.key === 'Tab') {
            const focusableElements = popup.querySelectorAll('button:not([tabindex="-1"]), [tabindex="0"], input, select');
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

    container.querySelector('.confirm-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        if (selectedDate) {
            const hours = hourInput.value || '12';
            const minutes = minuteInput.value || '00';
            const ampm = ampmSelect.value;

            selectedTime = `${hours.padStart(2, '0')}:${minutes.padStart(2, '0')} ${ampm}`;

            const y = selectedDate.getFullYear();
            const m = (selectedDate.getMonth() + 1).toString().padStart(2, '0');
            const d = selectedDate.getDate().toString().padStart(2, '0');
            const dateStr = `${y}-${m}-${d}`;

            input.value = `${dateStr} ${selectedTime}`;

            saveState();
            popup.style.display = 'none';
            isOpen = false;
        }
    });

    container.querySelector('.cancel-btn').addEventListener('click', (e) => {
        e.stopPropagation();
        popup.style.display = 'none';
        isOpen = false;
    });

    popup.addEventListener('click', (e) => {
        e.stopPropagation();
    });

    document.addEventListener('mousedown', (e) => {
        if (!container.contains(e.target)) {
            popup.style.display = 'none';
            isOpen = false;
        }
    });

    input.addEventListener('keydown', (e) => {
        if ((e.key === 'ArrowDown' || e.key === 'Down') && !isOpen) {
            popup.style.display = 'block';
            isOpen = true;
            e.preventDefault();
        }
        if (isOpen && e.key === 'Escape') {
            popup.style.display = 'none';
            isOpen = false;
            input.blur();
        }
    });

    renderCalendar();
    loadState();
}