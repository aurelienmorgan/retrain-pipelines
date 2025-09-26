
/*
expects '#glass-overlay' DOM element
and var names '--background-normal' & '--background-hover'

All list items must have "wavy-list-item" as one of its css classes. Those are the elements tha are animated.
The body of those elements shall have "wavy-list-item-body" as one of its classes. Those are the subelements (or the element itself) the color of which is "transitioned" on hover.
The list-items container (the list DOM containber element) has a css class named 'wavy-items-list'.

Note regarding borders :
We apply border only if the css var exists ('--border-normal' & '--border-hover').
In js/css, we can only check against emty string.
*/

const applyWaveEffect = (entry, entries, i) => {
    const clamp = (v, min, max) => Math.max(min, Math.min(max, v));
    const rect = entry.getBoundingClientRect();

    const scaleFocusedBase = 1.05;
    const scaleNeighborMaxBase = 1.03;
    const marginTopMax = 3;

    // Get container width and clamp scales
    // so scaled width does not exceed container width
    const container = entry.closest('.wavy-items-list');
    if (!container) return;

    const containerWidth = container.clientWidth;
    const entryWidth = entry.offsetWidth;

    // Clamp scales to prevent overflow:
    const scaleFocused = Math.min(scaleFocusedBase, containerWidth / entryWidth);
    const scaleNeighborMax = Math.min(scaleNeighborMaxBase, containerWidth / entryWidth);

    // Store original margin-bottom on all entries if not stored yet
    entries.forEach(e => {
        if (e._originalMarginBottom === undefined) {
            const style = window.getComputedStyle(e);
            e._originalMarginBottom = style.marginBottom;
        }
    });

    // Focused entry styles
    entry.style.transition = 'transform 0.2s ease, margin-top 0.2s ease, background 0.3s ease';
    entry.style.transform = `scale(${scaleFocused})`;
    entry.style.marginTop = '4px';
    entry.style.marginBottom = entry._originalMarginBottom; // restore original margin-bottom
    entry.style.zIndex = '10';

    const overlay = entry.querySelector('#glass-overlay');
    if (overlay) {
        overlay.style.transition = 'opacity 0.3s ease';
        overlay.style.opacity = '0';
    }

    if (!entry.classList.contains('wavy-list-item-body')) {
        const entryBody = entry.querySelector('.wavy-list-item-body');
        entryBody.style.transition =
            'transform 0.2s ease, margin-top 0.2s ease, background 0.3s ease, border 0.3s ease';
        entryBody.style.transform = ''; //`scale(${scaleFocused})`;
        entryBody.style.background =
            window.getComputedStyle(entryBody).getPropertyValue('--background-hover');
        entryBody.style.color =
            window.getComputedStyle(entryBody).getPropertyValue('--color-hover');
        const borderColor = window.getComputedStyle(entryBody).getPropertyValue('--border-hover');
        if (borderColor) {
            entryBody.style.border = `1px solid ${borderColor.trim()}`;
        }
    } else {
        entry.style.background = window.getComputedStyle(entry).getPropertyValue('--background-hover');
        entry.style.color = window.getComputedStyle(entry).getPropertyValue('--color-hover');
        const borderColor = window.getComputedStyle(entry).getPropertyValue('--border-hover');
        if (borderColor) {
            entry.style.border = `1px solid ${borderColor.trim()}`;
        }
    }

    const mouseMoveHandler = (ev) => {
        const y = ev.clientY;
        const relativeY = (y - rect.top) / rect.height;

        const before = entries[i - 1];
        const after = entries[i + 1];

        if (before && relativeY <= 0.66) {
            const intensity = 1 - clamp(relativeY / 0.66, 0, 1);
            before.style.transition =
                'transform 0.2s ease, margin-top 0.2s ease, background 0.3s ease, border 0.3s ease';

            // Clamp neighbor scale as well:
            const neighborScale = 1 + (scaleNeighborMax - 1) * intensity;
            before.style.transform = `scale(${neighborScale})`;
            before.style.marginTop = `${marginTopMax * intensity}px`;
            before.style.marginBottom =
                before._originalMarginBottom; // original margin-bottom restored
            before.style.zIndex = '5';

            const beforeOverlay = before.querySelector('#glass-overlay');
            if (beforeOverlay) {
                beforeOverlay.style.transition = 'opacity 0.3s ease';
                beforeOverlay.style.opacity = `${1 - intensity}`;
            }

            if (!before.classList.contains('wavy-list-item-body')) {
                const beforeBody = before.querySelector('.wavy-list-item-body');
                beforeBody.style.transition =
                    'transform 0.2s ease, margin-top 0.2s ease, background 0.3s ease, border 0.3s ease';
                beforeBody.style.transform = ''; //`scale(${neighborScale})`;
                beforeBody.style.background =
                    window.getComputedStyle(beforeBody).getPropertyValue('--background-hover');
                const borderColor = window.getComputedStyle(beforeBody).getPropertyValue('--border-hover');
                if (borderColor) {
                    beforeBody.style.border = `1px solid ${borderColor.trim()}`;
                }
            } else {
                before.style.background =
                    window.getComputedStyle(before).getPropertyValue('--background-hover');
                const borderColor = window.getComputedStyle(before).getPropertyValue('--border-hover');
                if (borderColor) {
                    before.style.border = `1px solid ${borderColor.trim()}`;
                }
            }
        } else if (before) {
            before.style.transform = '';
            before.style.marginTop = '';
            before.style.marginBottom = before._originalMarginBottom;
            before.style.zIndex = '1';

            const beforeOverlay = before.querySelector('#glass-overlay');
            if (beforeOverlay) beforeOverlay.style.opacity = '1';

            if (!before.classList.contains('wavy-list-item-body')) {
                const beforeBody = before.querySelector('.wavy-list-item-body');
                beforeBody.style.transform = '';
                beforeBody.style.background =
                    window.getComputedStyle(beforeBody).getPropertyValue('--background-normal');
                const borderColor = window.getComputedStyle(beforeBody).getPropertyValue('--border-normal');
                if (borderColor) {
                    beforeBody.style.border = `1px solid ${borderColor.trim()}`;
                }
            } else {
                before.style.background =
                    window.getComputedStyle(before).getPropertyValue('--background-normal');
                const borderColor = window.getComputedStyle(before).getPropertyValue('--border-normal');
                if (borderColor) {
                    before.style.border = `1px solid ${borderColor.trim()}`;
                }
            }
        }

        if (after && relativeY >= 0.33) {
            const intensity = clamp((relativeY - 0.33) / 0.66, 0, 1);
            after.style.transition =
                'transform 0.2s ease, margin-top 0.2s ease, background 0.3s ease, border 0.3s ease';

            // Clamp neighbor scale as well:
            const neighborScale = 1 + (scaleNeighborMax - 1) * intensity;
            after.style.transform = `scale(${neighborScale})`;
            after.style.marginTop = `${marginTopMax * intensity}px`;
            after.style.marginBottom =
                after._originalMarginBottom; // original margin-bottom restored
            after.style.zIndex = '5';

            const afterOverlay = after.querySelector('#glass-overlay');
            if (afterOverlay) {
                afterOverlay.style.transition = 'opacity 0.3s ease';
                afterOverlay.style.opacity = `${1 - intensity}`;
            }

            if (!after.classList.contains('wavy-list-item-body')) {
                const afterBody = after.querySelector('.wavy-list-item-body');
                afterBody.style.transition =
                    'transform 0.2s ease, margin-top 0.2s ease, background 0.3s ease, border 0.3s ease';
                afterBody.style.transform = ''; //`scale(${neighborScale})`;
                afterBody.style.background =
                    window.getComputedStyle(afterBody).getPropertyValue('--background-hover');
                const borderColor = window.getComputedStyle(afterBody).getPropertyValue('--border-hover');
                if (borderColor) {
                    afterBody.style.border = `1px solid ${borderColor.trim()}`;
                }
            } else {
                after.style.background =
                    window.getComputedStyle(after).getPropertyValue('--background-hover');
                const borderColor = window.getComputedStyle(after).getPropertyValue('--border-hover');
                if (borderColor) {
                    after.style.border = `1px solid ${borderColor.trim()}`;
                }
            }
        } else if (after) {
            after.style.transform = '';
            after.style.marginTop = '';
            after.style.marginBottom = after._originalMarginBottom;
            after.style.zIndex = '1';

            const afterOverlay = after.querySelector('#glass-overlay');
            if (afterOverlay) afterOverlay.style.opacity = '1';

            if (!after.classList.contains('wavy-list-item-body')) {
                const afterBody = after.querySelector('.wavy-list-item-body');
                afterBody.style.transform = '';
                afterBody.style.background =
                    window.getComputedStyle(afterBody).getPropertyValue('--background-normal');
                const borderColor = window.getComputedStyle(afterBody).getPropertyValue('--border-normal');
                if (borderColor) {
                    afterBody.style.border = `1px solid ${borderColor.trim()}`;
                }
            } else {
                after.style.background =
                    window.getComputedStyle(after).getPropertyValue('--background-normal');
                const borderColor = window.getComputedStyle(after).getPropertyValue('--border-normal');
                if (borderColor) {
                    after.style.border = `1px solid ${borderColor.trim()}`;
                }
            }
        }
    };

    const cleanup = () => {
        entries.forEach(e => {
            // Reset only transform, margin, zIndex — keep padding & background & overlay untouched
            e.style.transform = '';
            e.style.marginTop = '';
            e.style.marginBottom = e._originalMarginBottom;
            e.style.zIndex = '1';

            // Restore glass overlay opacity
            const overlay = e.querySelector('#glass-overlay');
            if (overlay) overlay.style.opacity = '1';

            // Restore background color to normal state
            if (!e.classList.contains('wavy-list-item-body')) {
                const eBody = e.querySelector('.wavy-list-item-body');
                eBody.style.transform = '';
                eBody.style.background =
                    window.getComputedStyle(eBody).getPropertyValue('--background-normal');
                eBody.style.color = window.getComputedStyle(eBody).getPropertyValue('--color-normal');
                const borderColor = window.getComputedStyle(eBody).getPropertyValue('--border-normal');
                if (borderColor) {
                    eBody.style.border = `1px solid ${borderColor.trim()}`;
                }
            } else {
                e.style.background = window.getComputedStyle(e).getPropertyValue('--background-normal');
                e.style.color = window.getComputedStyle(e).getPropertyValue('--color-normal');
                const borderColor = window.getComputedStyle(e).getPropertyValue('--border-normal');
                if (borderColor) {
                    e.style.border = `1px solid ${borderColor.trim()}`;
                }
            }
        });
        document.removeEventListener('mousemove', mouseMoveHandler);
    };

    document.addEventListener('mousemove', mouseMoveHandler);
    entry.addEventListener('mouseleave', cleanup, { once: true });
};

const enhanceEntry = (entry) => {
    if (entry.dataset.enhanced === "1") return;
    entry.dataset.enhanced = "1";

    entry.addEventListener('mouseenter', () => {
        const entries = Array.from(document.querySelectorAll(".wavy-list-item"));
        const i = entries.indexOf(entry);
        if (i !== -1) {
            applyWaveEffect(entry, entries, i);
        }
    });
};

