"""Vector engine: re-render a PDF page directly at the target resolution."""

from __future__ import annotations

import numpy as np
import pypdfium2 as pdfium

from crisp.types import Page, Raster, Target


class VectorEngine:
    name = "vector"

    def upscale(self, page: Page, target: Target) -> Raster:
        out_w, out_h = target.out_size
        pdf = pdfium.PdfDocument(page.source)
        try:
            bitmap = pdf[page.index].render(
                scale=out_w / page.size_pt[0],
                fill_color=(255, 255, 255, 255),
                may_draw_forms=True,
            )
            arr = np.asarray(bitmap.to_pil().convert("RGB"), dtype=np.float32) / 255.0
        finally:
            pdf.close()
        # pdfium rounds sizes up, so it can be a pixel off; crop or edge-pad to exact size.
        arr = arr[:out_h, :out_w]
        pad_h, pad_w = out_h - arr.shape[0], out_w - arr.shape[1]
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w), (0, 0)), mode="edge")
        return Raster(np.ascontiguousarray(arr), None)
