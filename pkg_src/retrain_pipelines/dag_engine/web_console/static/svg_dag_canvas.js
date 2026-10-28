/**
 * @file svg_dag_canvas.js
 * @description Canvas pan/zoom, DAG centering, and SVG height adjustment.
 * Exposes: adjustSVGHeight, centerDAG, getDAGBBox.
 * Dependencies: Core (ctx.svg, ctx.canvasGroup, ctx.config, ctx.state, 
 * ctx.updateTransform), Magnifier (ctx.magnifierState at runtime).
 */
function initDagCanvas(ctx) {
    const { svg, canvasGroup, config } = ctx;

    // Calculates the full rendered DAG bounding box, including all margins and
    // underlays, by hiding the tooltips group temporarily to avoid factoring in
    // their detached bounds.
    function getDAGBBox() {
        const tooltipsGroup = canvasGroup.querySelector('[id="tooltips-group"]');
        let originalDisplay = '';
        if (tooltipsGroup) {
            originalDisplay = tooltipsGroup.style.display;
            tooltipsGroup.style.display = 'none';
        }
        const bbox = canvasGroup.getBBox();
        if (tooltipsGroup) {
            tooltipsGroup.style.display = originalDisplay;
        }
        return bbox;
    }
    ctx.getDAGBBox = getDAGBBox;

    function adjustSVGHeight() {
        if (canvasGroup.querySelectorAll('.node').length === 0) return;

        const bbox = getDAGBBox();
        const dagHeight = bbox.height;
        const newHeight = (dagHeight + 2 * -config.tooltipYOffset) * ctx.state.currentTransform.scale;

        svg.setAttribute('height', newHeight);
    }
    ctx.adjustSVGHeight = adjustSVGHeight;

    // Center the DAG in the SVG viewport. When shrinkOverflowingWidth is true
    // and the rendered DAG exceeds the canvas width, the zoom scale is reduced
    // to fit before centering (height is recomputed to match the new scale).
    // A fixed 10px left/right margin is preserved when shrinking to fit.
    function centerDAG(shrinkOverflowingWidth = false) {
        if (canvasGroup.querySelectorAll('.node').length === 0) return;

        const bbox = getDAGBBox();
        if (bbox.width === 0) return;

        const contentWidth = bbox.width;
        const contentCenterX = bbox.x + contentWidth / 2;
        const minY = bbox.y;

        // Use computed style width to account for CSS (e.g., 90vw, borders, etc.)
        const computedWidthStr = window.getComputedStyle(svg).width;
        const canvasWidth = parseFloat(computedWidthStr) || svg.getBoundingClientRect().width;
        const viewportCenterX = canvasWidth / 2;

        // Shrink to fit when the rendered DAG overflows the canvas width,
        // preserving a fixed 10px margin on both left and right sides.
        if (shrinkOverflowingWidth && canvasWidth > 0 && contentWidth > 0) {
            const sideMargin = 10;
            const maxContentWidth = canvasWidth - 2 * sideMargin;
            const scaledWidth = contentWidth * ctx.state.currentTransform.scale;
            if (scaledWidth > maxContentWidth) {
                ctx.state.currentTransform.scale = maxContentWidth / contentWidth;
                adjustSVGHeight();
            }
        }

        ctx.state.currentTransform.x = viewportCenterX - contentCenterX * ctx.state.currentTransform.scale;
        const desiredTopOffset = -config.tooltipYOffset * ctx.state.currentTransform.scale;
        ctx.state.currentTransform.y = desiredTopOffset - minY * ctx.state.currentTransform.scale;

        ctx.updateTransform();
    }
    ctx.centerDAG = centerDAG;

    adjustSVGHeight();
    centerDAG(true);
    canvasGroup.style.visibility = 'visible';
    window.addEventListener('resize', () => {
        centerDAG();
        const magnifierState = ctx.magnifierState;
        if (magnifierState && magnifierState.isActive) {
            // Adjust magnifier size on resize if it's active
            // Note: The original code called updateMagnifierSize which is in magnifier module.
            // For simplicity here we trigger updateContent which handles layout.
            if (magnifierState.updateContent) {
                magnifierState.updateContent();
            }
        }
    });
    const intersectionObserver = new IntersectionObserver((entries) => {
        entries.forEach(entry => {
            if (entry.isIntersecting) {
                centerDAG();
            }
        });
    }, { threshold: 0.01 });
    intersectionObserver.observe(canvasGroup);

    /* ****************
    * Canvas-dragging *
    **************** */
    let canvasDrag = false;
    let canvasStartMouse = null;
    let canvasStartTransform = {x: 0, y: 0};

    svg.addEventListener('mousedown', (e) => {
        if (e.button === 1 || (e.button === 0 && e.target === svg)) {
            canvasDrag = true;
            const pt = svg.createSVGPoint();
            pt.x = e.clientX;
            pt.y = e.clientY;
            canvasStartMouse = pt.matrixTransform(svg.getScreenCTM().inverse());

            canvasStartTransform = {
                x: ctx.state.currentTransform.x,
                y: ctx.state.currentTransform.y
            };
            svg.classList.add('dragging-canvas');
            e.preventDefault();
        }
    });

    window.addEventListener('mousemove', (e) => {
        if (canvasDrag) {
            const pt = svg.createSVGPoint();
            pt.x = e.clientX;
            pt.y = e.clientY;
            const coords = pt.matrixTransform(svg.getScreenCTM().inverse());

            const dx = coords.x - canvasStartMouse.x;
            const dy = coords.y - canvasStartMouse.y;

            ctx.state.currentTransform.x = canvasStartTransform.x + dx;
            ctx.state.currentTransform.y = canvasStartTransform.y + dy;

            ctx.updateTransform();
        }
    });

    window.addEventListener('mouseup', () => {
        if (canvasDrag) {
            svg.classList.remove('dragging-canvas');
        }
        canvasDrag = false;
        canvasStartMouse = null;
    });

    /* ***************
    * Canvas-zooming *
    *************** */
    svg.addEventListener('wheel', (e) => {
        if (!e.ctrlKey) return;
        e.preventDefault();

        const pt = svg.createSVGPoint();
        pt.x = e.clientX;
        pt.y = e.clientY;
        const svgPoint = pt.matrixTransform(svg.getScreenCTM().inverse());

        const zoomFactor = e.deltaY < 0 ? 1.1 : 0.9;
        const newScale = Math.max(0.1, Math.min(5, ctx.state.currentTransform.scale * zoomFactor));

        const canvasPointBefore = {
            x: (svgPoint.x - ctx.state.currentTransform.x) / ctx.state.currentTransform.scale,
            y: (svgPoint.y - ctx.state.currentTransform.y) / ctx.state.currentTransform.scale
        };

        ctx.state.currentTransform.scale = newScale;

        ctx.state.currentTransform.x = svgPoint.x - canvasPointBefore.x * ctx.state.currentTransform.scale;
        ctx.state.currentTransform.y = svgPoint.y - canvasPointBefore.y * ctx.state.currentTransform.scale;

        ctx.updateTransform();

        /* repaint the magnifier if showing */
        const magnifierState = ctx.magnifierState;
        if (magnifierState && magnifierState.isActive) {
            magnifierState.updateContent();
        }
    });
}
