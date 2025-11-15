"""
Searchable PDF generation with invisible text layer.

Creates searchable PDFs by adding an invisible text layer over the original
image-based PDF, enabling text selection, search, and copy-paste functionality.

FUNDAMENTAL SOLUTION (macOS + PyMuPDF):
- Embed CJK Unicode font with ToUnicode mapping
- NFC normalization + line-level text reconstruction
- Coordinate transformation with rotation support
- Minimum font size 3pt (not 1pt)
- Line-based insert_textbox (NOT character-level insert_text)
"""

from pathlib import Path
from typing import List, Optional, Tuple, Dict
from collections import defaultdict

import fitz  # PyMuPDF

from util.coords import BoundingBox
from util.geom import PageGeom, rect_from_norm, rect_from_px, baseline_from_rect
from util.logging import get_logger
from util.font_loader import load_cjk_font
from postproc.unicode_norm import normalize_text, merge_hangul_by_gap
from data.ocr_result import OCRResult

logger = get_logger(__name__)


def build_searchable_pdf(
    input_pdf_path: str,
    output_pdf_path: Optional[str] = None,
    prefer_cjk_fonts: Optional[List[str]] = None,
    debug_overlay: bool = False,
    min_fontsize_pt: float = 3.0,
) -> str:
    """
    Helper function to build searchable PDF from scanned PDF using Document AI.

    Args:
        input_pdf_path: Path to input PDF file
        output_pdf_path: Path to output PDF (default: <name>_searchable.pdf)
        prefer_cjk_fonts: Optional list of CJK font paths to prioritize
        debug_overlay: If True, render visible gray text for debugging
        min_fontsize_pt: Minimum font size in points (default: 3.0)

    Returns:
        Path to generated searchable PDF

    Example:
        >>> out = build_searchable_pdf(
        ...     "scanned.pdf",
        ...     prefer_cjk_fonts=["/System/Library/Fonts/AppleSDGothicNeo.ttc"],
        ...     debug_overlay=False,
        ...     min_fontsize_pt=3.0
        ... )
    """
    from pathlib import Path
    from connectors.docai_client import DocumentAIClient
    from preproc.pdf_loader import PDFLoader
    from data.ocr_result import OCRResult, PageResult, Block, BoundingBox
    from data.block_types import BlockType
    import io

    input_path = Path(input_pdf_path)

    if output_pdf_path is None:
        output_path = input_path.parent / f"{input_path.stem}_searchable.pdf"
    else:
        output_path = Path(output_pdf_path)

    logger.info(f"Building searchable PDF: {input_path} -> {output_path}")

    # Initialize clients
    docai_client = DocumentAIClient()
    pdf_loader = PDFLoader(dpi=300)

    # Load PDF pages
    logger.info("Loading PDF pages...")
    page_images = pdf_loader.load_pages_parallel(
        input_path,
        max_workers=4
    )
    logger.info(f"Loaded {len(page_images)} pages")

    # Process each page with Document AI
    all_page_results = []

    for page_idx, page_image in enumerate(page_images, 1):
        logger.info(f"Processing page {page_idx}/{len(page_images)}...")

        # Convert to bytes
        img_bytes = io.BytesIO()
        page_image.image.save(img_bytes, format='PNG')
        img_bytes.seek(0)

        # Process with DocAI OCR
        result = docai_client.process_and_extract(
            input_data=img_bytes.read(),
            processor_type='ocr',
            mime_type='image/png'
        )

        if 'error' in result:
            logger.error(f"DocAI error on page {page_idx}: {result['error']}")
            continue

        # Convert to PageResult
        blocks = []
        if result.get('pages'):
            page_data = result['pages'][0]
            for block_data in page_data.get('blocks', []):
                bbox_norm = block_data.get('bbox_norm', [0, 0, 0, 0])

                # Convert normalized coords to pixel coords (for generic consumers)
                x = bbox_norm[0] * page_image.width
                y = bbox_norm[1] * page_image.height
                width = bbox_norm[2] * page_image.width
                height = bbox_norm[3] * page_image.height

                bbox = BoundingBox(
                    x=x,
                    y=y,
                    width=width,
                    height=height,
                    page=page_idx - 1,
                )
                block = Block(
                    text=block_data.get('text', ''),
                    confidence=block_data.get('conf', 1.0),
                    block_type=BlockType.PARAGRAPH,
                    bbox=bbox,
                    metadata={
                        # Preserve original normalized DocAI coordinates (0..1, x,y,w,h)
                        "bbox_norm": bbox_norm,
                        "coord_type": "docai_normalized",
                    },
                )
                blocks.append(block)

        page_result = PageResult(
            page_number=page_idx,
            blocks=blocks,
            width=page_image.width,
            height=page_image.height,
        )
        all_page_results.append(page_result)

    # Create OCR result
    ocr_result = OCRResult(
        pages=all_page_results,
        doc_id=str(input_path),
    )

    # Generate searchable PDF
    logger.info("Generating searchable PDF...")
    pdf_generator = SearchablePDFGenerator(
        use_cjk_font=True,
        prefer_font_paths=prefer_cjk_fonts,
        debug_overlay=debug_overlay,
        min_font_size=min_fontsize_pt,
    )

    actual_output = pdf_generator.create_from_ocr_result(
        input_pdf_path=input_path,
        output_pdf_path=output_path,
        ocr_result=ocr_result,
    )

    logger.info(f"Searchable PDF created: {actual_output}")
    return str(actual_output)


class SearchablePDFGenerator:
    """
    Generate searchable PDFs with invisible text layer.

    Uses fundamental solution:
    - CJK font embedding with ToUnicode
    - Line-level text reconstruction
    - Proper coordinate transformation
    - Minimum 3pt font size
    """

    def __init__(
        self,
        text_opacity: int = 0,
        min_font_size: float = 3.0,  # Changed from 1.0 to 3.0
        max_font_size: float = 24.0,  # Reduced from 72.0 to prevent oversized text
        use_cjk_font: bool = True,
        prefer_font_paths: Optional[List[str]] = None,
        debug_overlay: bool = False,
        background_image_scale: float = 2.0,
        background_image_format: str = "jpeg",
        background_image_quality: int = 75,
    ):
        """
        Initialize generator.

        Args:
            text_opacity: Text opacity (0-255, 0=invisible for searchable PDF)
            min_font_size: Minimum font size in points (default: 3.0pt)
            max_font_size: Maximum font size in points
            use_cjk_font: If True, try to load and embed CJK font
            prefer_font_paths: Optional list of font paths to prioritize
            debug_overlay: If True, render visible gray text for verification
            background_image_scale: Scale factor for rendering original page (e.g., 1.0, 1.5, 2.0)
            background_image_format: Background image encoding ('jpeg' or 'raw')
            background_image_quality: JPEG quality (1-100) when using 'jpeg'
        """
        self.text_opacity = max(0, min(255, text_opacity))
        self.min_font_size = max(3.0, min_font_size)  # Enforce minimum 3.0pt
        self.max_font_size = max_font_size
        self.use_cjk_font = use_cjk_font
        self.prefer_font_paths = prefer_font_paths
        self.debug_overlay = debug_overlay
        self._fontname = None  # Embedded font name
        self.background_image_scale = max(0.5, float(background_image_scale))
        self.background_image_format = background_image_format.lower()
        self.background_image_quality = int(background_image_quality)

    def create_from_ocr_result(
        self,
        input_pdf_path: Path,
        output_pdf_path: Path,
        ocr_result: OCRResult,
    ) -> Path:
        """
        Create searchable PDF from OCR result.

        Args:
            input_pdf_path: Path to original (image-based) PDF
            output_pdf_path: Path to output PDF
            ocr_result: OCR result with pages and blocks

        Returns:
            Path to created searchable PDF
        """
        try:
            logger.info(f"Creating searchable PDF: {output_pdf_path}")

            # Open original PDF
            doc = fitz.open(input_pdf_path)

            # Create a new clean PDF document
            clean_doc = fitz.open()

            # Font will be embedded after first page is created
            self._fontname = None

            # Process each page
            for page_idx, page_result in enumerate(ocr_result.pages):
                if page_idx >= len(doc):
                    logger.warning(f"Page {page_idx} exceeds PDF page count")
                    break

                orig_page = doc[page_idx]
                pdf_width = orig_page.rect.width
                pdf_height = orig_page.rect.height

                # Create clean page by rendering original to image
                # This removes ALL text including corrupted text
                mat = fitz.Matrix(self.background_image_scale, self.background_image_scale)
                pix = orig_page.get_pixmap(matrix=mat, alpha=False)
                render_width, render_height = pix.width, pix.height

                # Create new page in clean document
                new_page = clean_doc.new_page(width=pdf_width, height=pdf_height)

                # Insert rendered image with optional compression
                if self.background_image_format == "jpeg":
                    # PyMuPDF 1.26.x uses 'jpg_quality' keyword for JPEG quality
                    img_bytes = pix.tobytes("jpeg", jpg_quality=self.background_image_quality)
                    new_page.insert_image(new_page.rect, stream=img_bytes)
                else:
                    new_page.insert_image(new_page.rect, pixmap=pix)

                logger.debug(f"Created clean image-only page {page_idx}")

                # STEP 1: Embed CJK font on first page
                if page_idx == 0 and self._fontname is None:
                    if self.use_cjk_font:
                        self._fontname = load_cjk_font(clean_doc, self.prefer_font_paths)
                        logger.info(f"Embedded CJK font: {self._fontname}")
                    else:
                        self._fontname = "helv"  # Fallback to built-in
                        logger.warning("Using built-in font (no CJK support)")

                # IMPORTANT:
                # - OCR was performed on image with size (page_result.width, page_result.height)
                # - Background image for the new PDF page was rendered as pix (render_width, render_height)
                # - Our geom.PageGeom must ALWAYS use the same image size as the background render
                #   to keep coordinate systems perfectly aligned.
                # - Therefore we will scale OCR pixel bboxes to the rendered image size and
                #   pass render_width/render_height into the geom system.
                img_width = page_result.width
                img_height = page_result.height

                logger.debug(
                    f"Page {page_idx}: PDF=({pdf_width:.1f}, {pdf_height:.1f}), "
                    f"OCR Image=({img_width}, {img_height}), "
                    f"Rendered=({render_width}, {render_height})"
                )

                # Get page rotation (0, 90, 180, 270)
                rotation = orig_page.rotation

                # STEP 2-6: Add text layer with line-level reconstruction
                self._add_text_layer_line_based(
                    new_page,
                    page_result.blocks,
                    img_width,
                    img_height,
                    pdf_width,
                    pdf_height,
                    rotation,
                    render_width,
                    render_height,
                )

            # STEP 8: Save with optimal settings
            output_pdf_path.parent.mkdir(parents=True, exist_ok=True)
            clean_doc.save(
                output_pdf_path,
                garbage=4,      # Maximum garbage collection
                deflate=True,   # Compress streams
                clean=True,     # Clean up unused objects
            )
            clean_doc.close()
            doc.close()

            logger.info(f"Searchable PDF saved to {output_pdf_path}")
            return output_pdf_path

        except Exception as e:
            logger.error(f"Failed to create searchable PDF: {e}")
            raise

    def _add_text_layer_line_based(
        self,
        page: fitz.Page,
        blocks: List,
        ocr_img_width: int,
        ocr_img_height: int,
        pdf_width: float,
        pdf_height: float,
        rotation: int = 0,
        render_width: Optional[int] = None,
        render_height: Optional[int] = None,
    ):
        """
        Add text layer to page using line-level reconstruction.

        FUNDAMENTAL SOLUTION:
        1. Group blocks into lines (Y-axis grouping)
        2. Sort blocks within each line (X-axis sorting)
        3. Merge Korean characters using gap analysis (line-level)
        4. Insert each line once using CJK-capable font (no per-character insert)

        Args:
            page: PyMuPDF page object
            blocks: List of text blocks (from OCR)
            ocr_img_width: OCR input image width in pixels
            ocr_img_height: OCR input image height in pixels
            pdf_width: PDF page width in points
            pdf_height: PDF page height in points
            rotation: Page rotation (0, 90, 180, 270)
            render_width: Background image width used in PDF (e.g., pix.width)
            render_height: Background image height used in PDF (e.g., pix.height)
        """
        if not blocks:
            return

        # If render size is not provided, fall back to OCR image size
        # (keeps backward compatibility for tests that don't use a separate render step)
        if render_width is None:
            render_width = ocr_img_width
        if render_height is None:
            render_height = ocr_img_height

        # Scale factors from OCR image → rendered background image
        # This ensures that normalized coordinates stay identical while
        # PageGeom.img_w/h is tied to the rendered image size.
        if ocr_img_width and ocr_img_height:
            scale_x = render_width / ocr_img_width
            scale_y = render_height / ocr_img_height
        else:
            scale_x = scale_y = 1.0

        # 페이지 기하학 정보 생성
        geom = PageGeom(
            pdf_w=pdf_width,
            pdf_h=pdf_height,
            rotation_deg=rotation,
            img_w=render_width,
            img_h=render_height,
        )

        logger.debug(
            f"PageGeom: PDF {geom.pdf_w}x{geom.pdf_h}, "
            f"IMG_OCR {ocr_img_width}x{ocr_img_height}, "
            f"IMG_RENDER {geom.img_w}x{geom.img_h}, "
            f"scale=({scale_x:.4f}, {scale_y:.4f}), "
            f"rot={geom.rotation_deg}°"
        )

        # STEP 2: Sort blocks by reading order (Y then X)
        sorted_blocks = sorted(blocks, key=lambda b: (b.bbox.y, b.bbox.x))

        # Debug: Log first 3 blocks with their OCR image coordinates
        logger.debug("First 3 blocks (OCR image coords):")
        for i, block in enumerate(sorted_blocks[:3]):
            logger.debug(
                f"  Block {i}: '{block.text[:30]}' at "
                f"img_y={block.bbox.y:.1f}, img_height={block.bbox.height:.1f}"
            )

        # CRITICAL: 각 OCR 블록을 독립적인 라인으로 처리 (그룹핑 비활성화)
        # Document AI가 이미 적절한 단위로 블록을 분리했으므로
        lines = [[block] for block in sorted_blocks]

        logger.debug(f"Processing {len(blocks)} blocks as {len(lines)} separate lines")

        # Process each line
        for line_idx, line_blocks in enumerate(lines):
            # Sort blocks within line by X-coordinate (left to right)
            line_blocks = sorted(line_blocks, key=lambda b: b.bbox.x)

            logger.debug(f"Line {line_idx}: {len(line_blocks)} blocks")

            # STEP 3: Create spans for word reconstruction
            spans = []
            for block in line_blocks:
                text = block.text
                if not text or not text.strip():
                    continue

                # 통합 변환 함수 사용:
                # - DocAI normalized(0..1)이 metadata에 있으면 rect_from_norm 사용
                # - 그렇지 않으면 픽셀 bbox를 스케일링 후 rect_from_px 사용
                rect = self._to_pdf_rect(
                    block=block,
                    geom=geom,
                    scale_x=scale_x,
                    scale_y=scale_y,
                )

                # Debug: Log transformation for first 3 lines
                if line_idx < 3:
                    logger.debug(
                        f"  '{text[:20]}': "
                        f"→ PDF({rect.x0:.1f},{rect.y0:.1f},{rect.x1:.1f},{rect.y1:.1f})"
                    )

                spans.append((text, rect.x0, rect.x1, rect))

            if not spans:
                continue

            # Calculate line bounding box
            line_x0 = min(s[3].x0 for s in spans)
            line_y0 = min(s[3].y0 for s in spans)
            line_x1 = max(s[3].x1 for s in spans)
            line_y1 = max(s[3].y1 for s in spans)

            line_rect = fitz.Rect(line_x0, line_y0, line_x1, line_y1)

            # STEP 5: Calculate font size and baseline
            rect_height = line_rect.height
            # 적절한 폰트 크기 계산 (bbox 높이의 ~82%)
            fontsize = max(round(rect_height * 0.82, 1), self.min_font_size)
            fontsize = min(fontsize, min(18.0, self.max_font_size))  # Cap at 18pt for body text

            # Baseline 계산 (새로운 geom 시스템 사용)
            baseline_y = baseline_from_rect(line_rect, fontsize)

            # Debug: Log line rect and baseline for first 5 lines
            if line_idx < 5:
                logger.debug(f"  Line {line_idx}:")
                logger.debug(f"    PDF rect: ({line_rect.x0:.1f}, {line_rect.y0:.1f}, {line_rect.x1:.1f}, {line_rect.y1:.1f})")
                logger.debug(f"    Height: {rect_height:.1f}pt, Font: {fontsize:.1f}pt")
                logger.debug(f"    baseline_y (PDF): {baseline_y:.1f} → tw_y (TextWriter): {pdf_height - baseline_y:.1f}")

            # STEP 3: Merge Korean characters intelligently
            # Extract (text, x0, x1) for gap analysis
            span_tuples = [(s[0], s[1], s[2]) for s in spans]
            gap_thresh_pt = fontsize * 0.25

            # STEP 4: Apply NFC normalization + word reconstruction
            merged_text = merge_hangul_by_gap(span_tuples, gap_thresh_pt)
            merged_text = normalize_text(merged_text)

            if not merged_text or not merged_text.strip():
                continue

            # STEP 6: Insert using insert_textbox (line-level)
            # Create rect for textbox (slightly expanded for safety)
            textbox_rect = fitz.Rect(
                line_rect.x0 - 0.5,
                baseline_y - fontsize,
                line_rect.x1 + 0.5,
                baseline_y + fontsize * 1.2
            )

            try:
                # Insert text using TextWriter for better Unicode support
                # insert_textbox with fontfile doesn't create proper ToUnicode mapping
                tw = fitz.TextWriter(page.rect)

                # CRITICAL FIX: TextWriter uses INVERTED Y-axis!
                # insert_text: Y=0 at bottom, Y=pdf_height at top (standard PDF)
                # TextWriter.append: Y=0 at top, Y=pdf_height at bottom (INVERTED!)
                tw_y = pdf_height - baseline_y

                # Create Font object for CJK support
                if self._fontname and (self._fontname.startswith('/') or self._fontname.endswith(('.ttf', '.otf', '.ttc'))):
                    # Use Font object for proper ToUnicode mapping
                    try:
                        font_obj = fitz.Font("CJKFont", fontfile=self._fontname)
                        tw.append(
                            (textbox_rect.x0, tw_y),
                            merged_text + " ",
                            font=font_obj,
                            fontsize=fontsize,
                        )
                    except Exception as font_err:
                        logger.warning(f"Failed to create Font object: {font_err}, using built-in")
                        tw.append(
                            (textbox_rect.x0, tw_y),
                            merged_text + " ",
                            fontsize=fontsize,
                        )
                else:
                    # Use built-in font
                    tw.append(
                        (textbox_rect.x0, tw_y),
                        merged_text + " ",
                        fontsize=fontsize,
                    )

                # Write to page with appropriate render mode
                if self.debug_overlay:
                    # Debug mode: visible text + green line bbox for visual alignment
                    page.draw_rect(
                        line_rect,
                        color=(0, 1, 0),
                        width=0.5,
                        overlay=True,
                    )
                    tw.write_text(page, render_mode=0, color=(0.7, 0.2, 0.2))
                else:
                    # Production mode: invisible text
                    tw.write_text(page, render_mode=3, color=(0, 0, 0))
                rc = 1  # Success

            except Exception as e:
                logger.error(f"Failed to insert textbox: {e}")
                logger.error(f"Text: {merged_text[:50]}...")
                import traceback
                logger.error(traceback.format_exc())

        logger.debug(f"Added {len(lines)} text lines to page")

    def _to_pdf_rect(
        self,
        block,
        geom: PageGeom,
        scale_x: float,
        scale_y: float,
    ) -> fitz.Rect:
        """
        통합 좌표 변환:
        - DocAI normalizedVertices(0..1) → rect_from_norm
        - 픽셀 bbox(xywh_px) → rect_from_px (스케일링 포함)

        DocAI 케이스:
            block.metadata['bbox_norm'] = [x, y, w, h]  # 0..1

        기타 엔진/내부 OCR:
            block.bbox.x/y/width/height 가 픽셀 좌표
        """
        # 1) DocAI normalized(0..1) 우선 사용
        meta = getattr(block, "metadata", {}) or {}
        norm = None

        if "normalized" in meta:
            norm = meta["normalized"]  # 기대 형식: [nx0, ny0, nx1, ny1]
        elif "bbox_norm" in meta:
            # repo 기본 포맷: [x, y, w, h] in 0..1
            bx, by, bw, bh = meta["bbox_norm"]
            norm = [bx, by, bx + bw, by + bh]

        if norm is not None:
            nx0, ny0, nx1, ny1 = norm
            return rect_from_norm(nx0, ny0, nx1, ny1, geom)

        # 2) 픽셀 bbox 경로 (엔진/내부 OCR 공통)
        bbox = block.bbox

        # OCR 픽셀 좌표 → 렌더 이미지 좌표로 스케일링
        scaled_x = bbox.x * scale_x
        scaled_y = bbox.y * scale_y
        scaled_w = bbox.width * scale_x
        scaled_h = bbox.height * scale_y

        # 렌더 이미지 픽셀 좌표 → PDF 좌표 변환
        return rect_from_px(scaled_x, scaled_y, scaled_w, scaled_h, geom)

    def _group_blocks_into_lines(
        self,
        blocks: List,
        img_height: float,
        line_height_thresh: float = 0.5,
    ) -> List[List]:
        """
        Group blocks into lines based on Y-coordinate proximity.

        Args:
            blocks: Sorted list of blocks (by Y coordinate)
            img_height: Image height (for normalization)
            line_height_thresh: Threshold ratio for line grouping

        Returns:
            List of line groups, each containing blocks
        """
        if not blocks:
            return []

        lines = []
        current_line = [blocks[0]]
        current_y_center = blocks[0].bbox.y + blocks[0].bbox.height / 2

        for block in blocks[1:]:
            block_y_center = block.bbox.y + block.bbox.height / 2
            block_height = block.bbox.height

            # Calculate Y distance
            y_dist = abs(block_y_center - current_y_center)

            # CRITICAL FIX: Very strict threshold to prevent grouping separate lines
            # Blocks on the same line should have nearly identical Y centers (within 3 pixels)
            thresh = 3.0  # Fixed 3-pixel threshold

            if y_dist < thresh:
                # Same line
                current_line.append(block)
                # Update center (weighted average)
                current_y_center = sum(
                    b.bbox.y + b.bbox.height / 2 for b in current_line
                ) / len(current_line)
            else:
                # New line
                lines.append(current_line)
                current_line = [block]
                current_y_center = block_y_center

        # Add last line
        if current_line:
            lines.append(current_line)

        return lines

    # Legacy method for backward compatibility
    def create_searchable_pdf(
        self,
        input_pdf_path: Path,
        output_pdf_path: Optional[Path] = None,
        text_blocks_per_page: Optional[List[List]] = None,
        image_size_per_page: Optional[List[Tuple[int, int]]] = None,
    ) -> Path:
        """
        DEPRECATED: Use create_from_ocr_result instead.

        Create searchable PDF from original PDF and text blocks.

        Args:
            input_pdf_path: Path to original PDF
            output_pdf_path: Path to output PDF
            text_blocks_per_page: List of text blocks for each page
            image_size_per_page: List of (width, height) for each page

        Returns:
            Path to created searchable PDF
        """
        # Generate output path if not provided
        if output_pdf_path is None:
            stem = input_pdf_path.stem
            suffix = input_pdf_path.suffix
            output_pdf_path = input_pdf_path.parent / f"{stem}_searchable{suffix}"

        # Convert to OCRResult format
        from data.ocr_result import PageResult

        pages = []
        for page_idx, blocks in enumerate(text_blocks_per_page or []):
            if image_size_per_page and page_idx < len(image_size_per_page):
                width, height = image_size_per_page[page_idx]
            else:
                width, height = 595, 842  # Default A4 size

            page_result = PageResult(
                page_number=page_idx + 1,
                blocks=blocks,
                width=width,
                height=height,
            )
            pages.append(page_result)

        ocr_result = OCRResult(
            pages=pages,
            doc_id=str(input_pdf_path),
        )

        return self.create_from_ocr_result(
            input_pdf_path,
            output_pdf_path,
            ocr_result,
        )
