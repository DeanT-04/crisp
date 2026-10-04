# crisp — design spec

**Date:** 2026-10-04
**Status:** Draft, awaiting review
**Scope of this spec:** the v1 engine and CLI. The desktop app and the MCP server are separate projects with their own specs, built later on top of this engine.

---

## 1. Purpose and agreed understanding

**What the user asked for.** A free tool that increases the resolution of images without AI. A user drops in one image or many and chooses how far to upscale. The driving use case: 2D technical drawings whose dimensions and annotations become unreadable when zoomed. The goal is quality good enough to give away instead of paying for commercial tools.

**Decisions made during brainstorming.**

| Topic | Decision |
|---|---|
| Inputs | A mix: PDFs exported from CAD, PNG/JPG exports and screenshots, scans and photos of paper drawings |
| Outputs | PNG at a chosen scale by default; SVG/PDF vector output as an option |
| Interface | Engine + CLI first (this spec). Desktop app is phase 2, MCP is phase 3 |
| AI boundary | No machine learning inside the tool. AI may be used **only** in the benchmark, as a comparator and as a text-legibility judge |
| Language | Python 3.12 + NumPy/OpenCV, Numba for custom hot loops |
| Photos | In scope as a solid fallback only, not tuned to compete with AI |
| Repo | Public, MIT, `DeanT-04/crisp` |

**Core principles.**

1. **Rebuild, don't stretch.** Line art is reconstructed (edges, lines, arcs) rather than interpolated pixel by pixel.
2. **Never invent content.** Every output must be consistent with its input (see §6). A crisp wrong number is worse than a blurry right one.
3. **Measure, don't eyeball.** Quality claims come from the benchmark (§7), not from screenshots.
4. **Pixels, not DPI.** DPI is metadata. crisp changes pixel counts and updates DPI so the physical print size stays the same.

**Known hard limit.** Information that is gone cannot be recovered. Text around 5 px tall with fully merged strokes will not become reliably readable. The benchmark will publish where the limit sits.

---

## 2. Architecture

```
file(s) ─► Intake ─► Analyze ─► Restore ─► Upscale ─► Faithfulness ─► Write
                        │                    ├─ vector re-render   (PDF with vectors)
                        │                    ├─ line-art rebuild   (drawings, text, scans)
                        │                    ├─ continuous-tone    (photos, shading)
                        │                    └─ baseline (Lanczos) (safe fallback)
                        └─ decides clean-up steps and engine routing per image/region
```

Every unit has one job, a typed interface, and can be tested alone. The CLI, the future desktop app and the future MCP server are all thin callers of the same library API.

### 2.1 Core data types

- **`Page`**: one image to process. Holds pixels (`float32`, H×W×C, range 0–1) *or* a handle to a vector PDF page, plus metadata: source path, page index, input DPI (may be unknown), `has_vectors`, `source_format`.
- **`Analysis`**: what Analyze measured. Region label map (`lineart` / `tone`), blur estimate (PSF), noise level, JPEG blockiness score, skew angle, ink/paper palette.
- **`Target`**: the requested output size, resolved from `--scale`, `--dpi` or `--size` into a scale factor and output DPI.
- **`Result`**: output pixels (and optional vector data), engine used, faithfulness score, warnings.

### 2.2 Library API (what the CLI calls)

```python
crisp.process(path, target, options) -> list[Result]   # one Result per page
crisp.process_batch(paths, target, options) -> BatchSummary
```

---

## 3. Units

| Unit | File | Job | Depends on |
|---|---|---|---|
| Intake | `intake.py` | Open PNG/JPG/TIFF/BMP/WebP/PDF; split PDFs into pages; detect vector content; extract embedded images from image-only PDFs | Pillow, pypdfium2 |
| Analyze | `analyze.py` | Measure only, never modify (§4) | `psf.py`, OpenCV, NumPy |
| Restore | `restore.py` | Fix only what Analyze found (§5) | OpenCV, scikit-image |
| Engines | `engines/*.py` | `upscale(page, analysis, target) -> pixels` (§6) | see each engine |
| Blur measurement | `psf.py` | Slanted-edge PSF estimation | OpenCV, NumPy |
| Geometry | `geometry.py` | Line and arc detection, fitting, snapping | OpenCV, NumPy |
| Faithfulness | `faithful.py` | Consistency check and fallback decision (§6.5) | NumPy |
| Vectorize | `vectorize.py` | Trace clean line art to SVG; wrap as PDF | vtracer |
| Write | `write.py` | Save PNG with correct DPI; save SVG/PDF; never overwrite originals | Pillow, pypng |
| Tiling | `tiling.py` | Split huge jobs into overlapping tiles and blend them | NumPy |
| Batch | `batch.py` | Parallel processing, per-file isolation, summary | `concurrent.futures` |
| CLI | `cli.py` | Argument parsing, progress, exit codes | typer, rich |

### Dependencies and licences

All runtime dependencies are permissive so the project can stay MIT:

| Package | Licence | Why this one |
|---|---|---|
| numpy | BSD-3 | Arrays |
| opencv-python-headless | Apache-2.0 | Fast filters, line detection, resampling |
| scikit-image | BSD-3 | Metrics, restoration helpers |
| numba | BSD-2 | Compiles custom pixel loops to native speed |
| Pillow | HPND (MIT-style) | Image I/O, DPI metadata |
| pypng | MIT | Row-by-row writing of very large PNGs |
| pypdfium2 | Apache-2.0 / BSD-3 | PDF rendering. Chosen over PyMuPDF, which is AGPL |
| vtracer | MIT | Vector tracing. Chosen over potrace, which is GPL |
| typer, rich | MIT | CLI and progress display |

Benchmark-only (never shipped in the package): Real-ESRGAN ncnn-vulkan binary (BSD-3), Tesseract via pytesseract (Apache-2.0).

---

## 4. Analyze

Analyze runs on every raster page and returns an `Analysis`. It changes nothing.

- **Content classification.** The image is split into 64×64 tiles. Each tile is scored with classical features: histogram bimodality (Otsu separability), number of distinct colour clusters, edge density, how strongly gradient directions cluster at 0°/90°, and flat-area ratio. Simple thresholds, tuned on the benchmark set, label tiles `lineart` or `tone`. The label map is smoothed with morphological operations to give clean regions. An image is "mixed" when both labels cover meaningful area.
- **Blur (PSF).** See §6.2 step 1. Measured on line-art regions only.
- **Noise.** Robust noise estimate from the median absolute deviation of the finest wavelet band.
- **JPEG damage.** Blockiness score: strength of discontinuities on the 8×8 grid versus off-grid. Only computed when the source was a JPEG or the grid signal is present.
- **Skew.** Dominant angle of long straight lines (line segment detection). Only reported when |angle| > 0.1°.
- **Palette.** Ink and paper colours via colour clustering (k ≤ 8) on line-art regions.

The `--mode lineart|photo` CLI option overrides classification for the whole image.

---

## 5. Restore

Each step runs **only** when Analyze flagged the problem. A clean export passes through untouched.

| Problem | Fix |
|---|---|
| JPEG blocks/ringing | Grid-aware deblocking filter, then light edge-preserving smoothing |
| Noise | Non-local means denoising, strength set from the measured noise level |
| Skew | Rotate by the measured angle (high-quality resampling, background-coloured fill) |
| Uneven lighting (scans/photos) | Estimate the background with a large morphological closing, then divide it out |

Order: deblock → denoise → lighting → deskew. Restore records each step it applied so the summary and `--verbose` output can show it.

---

## 6. Upscale engines

All engines share one interface: `upscale(page, analysis, target) -> pixels`. That makes them swappable and directly comparable in the benchmark.

### 6.1 Vector re-render (PDFs with vector content)

Render the page with pdfium at the target DPI. No interpolation, no guessing. Intake decides `has_vectors` by inspecting page objects. A page is **image-only** when image objects cover at least 90% of the page and path/text objects cover under 1%. Its largest image is extracted at native resolution and treated as raster. Every other page is **vector** and rendered directly. Embedded images on a vector page are rendered by pdfium as they are; upscaling them separately is out of scope for v1.

### 6.2 Line-art rebuild (the main engine)

Operates on line-art regions. Steps:

1. **Measure the blur.** Find straight high-contrast edges (line segment detection). Sample the intensity profile across each edge at sub-pixel positions (the ISO 12233 slanted-edge method), derive the edge spread function, differentiate it into the line spread function, and fit a PSF (Gaussian σ, with an option for a generalised Gaussian). Take a robust median across edges. If too few edges qualify, use a conservative default and record a warning.
2. **Undo the blur.** Richardson–Lucy deconvolution with total-variation regularisation, using the measured PSF and a bounded number of iterations. This separates strokes that blur had merged.
3. **Build coverage maps.** For each ink colour in the palette, compute a fractional coverage map (0 = paper, 1 = ink).
4. **Redraw edges at the target size.** Upscale each coverage map with a smooth interpolant, then apply a smooth step centred on 0.5 whose width is one *output* pixel. This places every edge at its sub-pixel position and gives clean anti-aliasing at any scale, while preserving line thickness.
5. **Straighten lines and arcs.** Fit straight segments and circular arcs to edge contours (line segment detection plus RANSAC circle fitting). Snap an edge to its fitted shape **only** when the RMS fit error is below 0.25 input pixels. Anything else is left as measured.
6. **Protect faint strokes.** Detect thin, low-contrast ridges with a Hessian-based ridge filter (faint lines, small text). These keep a local contrast stretch instead of the hard step from step 4, so they are not erased.
7. **Recolour.** Composite the coverage maps over the paper colour using the palette.

### 6.3 Continuous-tone (photos and shading)

1. Convert sRGB to linear light.
2. **Edge-directed interpolation**: Directional Cubic Convolution Interpolation (DCCI) in 2× steps, which interpolates along edges rather than across them. Non-power-of-two factors finish with a Lanczos resample.
3. **Iterative back-projection**: repeatedly downscale the estimate with the acquisition model, compare it with the input, and push the difference back up. This enforces consistency and recovers some sharpness.
4. **Adaptive sharpening** with halo clamping (the overshoot is limited to the local min/max).
5. Convert back to sRGB.

### 6.4 Baseline (Lanczos)

Plain Lanczos-3 resampling in linear light. Used as the safe fallback and as a benchmark comparator.

### 6.5 Mixed images

Each region goes to its engine. Results are blended across a feathered mask about 8 output pixels wide at region borders.

### 6.6 Faithfulness check and fallback

For every raster output: apply the acquisition model (the measured PSF, then area downscaling by the scale factor) and compare the result with the restored input. The pass threshold is a PSNR set from benchmark data, starting at 35 dB, checked per region.

- **Pass:** keep the result.
- **Fail:** redo that region with the baseline engine plus back-projection, and record a warning naming the region and the score.

Vector re-render output is exempt because it involves no estimation.

### 6.7 Output size resolution

- `--scale F`: output = input × F. Any F > 1, up to 16.
- `--dpi D`: requires known input DPI. Scale = D ÷ input DPI. Unknown input DPI → clear error suggesting `--scale`. For vector PDFs, render directly at D.
- `--size PAPER@D` (A0–A5, Letter, Tabloid): scale so the image fits the paper at D DPI, keeping aspect ratio.
- Output DPI metadata = input DPI × scale (raster), or D (PDF/`--size`). If input DPI is unknown and `--scale` is used, DPI metadata is left unset.

---

## 7. Benchmark

The benchmark is a first-class part of the project, built in M1 before the main engine, so every change can be measured.

### 7.1 Datasets

1. **Synthetic drawings** (generated, committed generator, not committed images): random compositions of lines, arcs, circles, hatching, dimension lines with arrows, and dimension text from 4 px to 40 px tall. Rendered with Pillow at 4× supersampling to give a perfect ground truth. Text in DejaVu Sans (bundled, permissive licence). Text content and position are recorded for legibility scoring.
2. **Real drawings**: US patent drawings (public domain, high-resolution black-and-white TIFFs), fetched by a script from a committed list of document numbers. The user's own drawings go in `bench/data/private/`, which is git-ignored and never uploaded.
3. **Photos**: standard super-resolution research sets (Set5, Set14, BSD100, Urban100), fetched by a script, used locally only.

### 7.2 Degradation

Ground truth → optional Gaussian blur (σ 0.5–2.0) → area downscale (2×, 4×, 8×) → optional JPEG (quality 50–90) → optional noise → optional scan simulation (small rotation, uneven lighting). Each test case has a fixed seed, so runs are reproducible.

### 7.3 Metrics

| Metric | Measures | Applies to |
|---|---|---|
| Text legibility | % of dimension labels read exactly by Tesseract (judge only) | Synthetic drawings |
| Edge accuracy | Mean distance between output and ground-truth edges (chamfer), and stroke-width error | Drawings |
| PSNR, SSIM | Standard pixel and structure fidelity | All |
| Speed | Seconds per image, megapixels per second | All |

### 7.4 Comparators

Nearest neighbour, bicubic, Lanczos-3, Real-ESRGAN `x4plus` and `x4plus-anime` (the line-art model). Real-ESRGAN is optional: if the binary isn't present, its columns show N/A.

### 7.5 Targets for v1

- **Drawings at 2×/4×/8×:** beat bicubic and Lanczos on every metric.
- **Drawings at 4×:** match or beat the better Real-ESRGAN model on text legibility and edge accuracy.
- **Photos at 2×/4×:** PSNR and SSIM ≥ Lanczos.
- **Speed:** 4× of a 2-megapixel drawing in ≤ 10 s on the development machine (tracked, optimised in M5).

### 7.6 Output and regression

`crisp bench` writes an HTML report (score tables plus zoomed side-by-side crops) to `bench/reports/` and opens it. `bench/results/baseline.json` (committed) stores the last accepted scores. Any metric that drops by more than its tolerance is flagged. CI runs `crisp bench --quick` (small synthetic set; PSNR, SSIM and edge accuracy only; no Tesseract or Real-ESRGAN) as a regression gate.

---

## 8. CLI

```
crisp INPUT... [--scale F | --dpi D | --size PAPER@D]
               [--out DIR] [--vector svg|pdf] [--mode auto|lineart|photo]
               [--jobs N] [--recursive] [--overwrite] [--verbose]
crisp bench [--quick] [--no-open]
```

- `INPUT` can be files, folders or glob patterns. Folders are scanned one level deep unless `--recursive`.
- The default is `--scale 2`.
- Output goes next to each source as `<name>@<scale>x.png` (`<name>-p<page>@<scale>x.png` for PDF pages), or into `--out`. **Source files are never overwritten.** Existing crisp outputs are skipped unless `--overwrite`.
- `--vector` also writes an SVG/PDF for line-art pages. Pages classified as tone skip it with a note.
- `--jobs` defaults to the number of CPU cores.
- `--verbose` prints each file's analysis: label mix, blur σ, noise, JPEG score, skew, restore steps, engine, faithfulness score.
- Progress bar via rich. End-of-run summary table: done / skipped / failed / fell back, each with a reason.
- **Exit codes:** 0 = all succeeded, 1 = some files failed, 2 = usage error.

---

## 9. Error handling

| Situation | Behaviour |
|---|---|
| Corrupt, unsupported or password-protected file | Skip, record the reason, continue the batch |
| Any exception while processing a file | Caught per file in the batch worker, recorded with the message, batch continues |
| Output would be very large | Estimate before starting; switch to tiling automatically (see Tiling below) |
| Not enough disk space for the estimated output | Refuse that file with a clear message before doing work |
| `--dpi` with unknown input DPI | Usage error suggesting `--scale` |
| Too few edges to measure blur | Use the default PSF, record a warning |
| Faithfulness check fails | Fall back to baseline for that region, record a warning (§6.6) |

### Tiling

When the estimated peak working memory for a page exceeds 25% of available RAM, it is processed in 1024×1024 input-pixel tiles with 32-pixel overlap, blended with a feathered window. Output is assembled in a disk-backed array (`numpy.memmap`) and written row by row with pypng. Analyze still runs on a downscaled whole-page view so the PSF and palette stay consistent across tiles.

---

## 10. Testing

- **Test-driven development** for all units, with pytest.
- **Ground-truth unit tests** on synthetic inputs, for example:
  - A perfect edge blurred by a known σ: `psf.py` recovers σ within 5%.
  - A wobbly line within tolerance snaps straight; a genuinely curved edge does not.
  - Synthetic line-art and photo tiles are classified correctly.
  - A 300-DPI input at `--scale 4` writes a 1200-DPI PNG.
  - Image-only and vector PDFs are detected correctly.
- **Property test across all engines:** output passes the faithfulness check on clean synthetic inputs.
- **CLI tests:** never overwrites a source; batch continues past a corrupt file; exit codes are correct.
- **CI** (GitHub Actions, Ubuntu + Windows): ruff, pytest, `crisp bench --quick`.

---

## 11. Project layout and tooling

```
crisp/
├── src/crisp/
│   ├── __init__.py        # public API: process, process_batch
│   ├── intake.py
│   ├── analyze.py
│   ├── restore.py
│   ├── psf.py
│   ├── geometry.py
│   ├── faithful.py
│   ├── engines/
│   │   ├── base.py        # Engine protocol
│   │   ├── vector.py
│   │   ├── lineart.py
│   │   ├── tone.py
│   │   └── baseline.py
│   ├── vectorize.py
│   ├── write.py
│   ├── tiling.py
│   ├── batch.py
│   └── cli.py
├── bench/
│   ├── generate.py  degrade.py  metrics.py  run.py  report.py  fetch.py
│   └── results/baseline.json
├── tests/
├── docs/
├── pyproject.toml
├── README.md  PROMPT-BANNER.md  LICENSE
```

Tooling: Python 3.12, `uv` for environments and dependencies, ruff for lint and format, pytest. Each feature goes on its own branch with a Conventional Commit PR, merged only after the user approves.

---

## 12. Milestones

| Milestone | Delivers | Done when |
|---|---|---|
| **M1 — Foundation** | Intake, vector PDF re-render, write, baseline engine, batch + CLI, benchmark harness (synthetic generator, degradation, metrics, comparators, report), CI | `crisp drawing.png --scale 4` works end to end; `crisp bench` produces a report with baseline numbers |
| **M2 — Line-art engine** | `psf.py`, deconvolution, coverage rebuild, geometry snapping, faint-stroke protection, Analyze classification, faithfulness check + fallback | Line-art engine beats Lanczos on all drawing metrics at 2×/4×/8× |
| **M3 — Restore** | Deblocking, denoising, deskew, lighting correction | Degraded-scan benchmark cases improve over M2 with no regression on clean cases |
| **M4 — Tone + mixed + tiling** | Continuous-tone engine, region routing and blending, tiling | Photo targets met; a 16× job on a large scan completes within memory limits |
| **M5 — Vector export + v1.0** | vtracer SVG/PDF export, speed optimisation, docs, release | All §7.5 targets met or honestly reported; v1.0 tagged |

---

## 13. Out of scope for v1

Desktop app, MCP server, text recognition inside the tool, GPU acceleration, photo-specific tuning, video, colour correction beyond lighting flattening, output formats other than PNG/SVG/PDF.

---

## 14. Risks

| Risk | Mitigation |
|---|---|
| Blur measurement fails on images with few straight edges | Default PSF plus warning; benchmark tracks how often it happens |
| Deconvolution ringing or noise amplification | TV regularisation, bounded iterations, faithfulness check catches failures |
| Classifier sends a region to the wrong engine | `--mode` override; benchmark reports per-class classification accuracy |
| Small text below the recoverable limit | Documented limit; benchmark publishes legibility by text height |
| Python too slow on large batches | Numba for hot loops, parallel batch processing, profiling in M5 |
| Real-ESRGAN beats crisp on legibility at 8× | Targets are set at 4×; 8× results reported honestly either way |
