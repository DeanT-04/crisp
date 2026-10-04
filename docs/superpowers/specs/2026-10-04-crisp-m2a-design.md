# crisp M2a — line-art engine core and benchmark repair

**Date:** 2026-10-04
**Status:** Draft, awaiting review
**Parent spec:** [2026-10-04-crisp-design.md](2026-10-04-crisp-design.md) (milestone M2, §6.2). M2 is split into three sub-projects, each with its own spec, plan and PR. This spec covers **M2a only**. M2b (line and arc snapping, faint-stroke protection) and M2c (per-region classifier, mixed-image routing, full faithfulness check) come afterwards.

---

## 1. Purpose and agreed understanding

**Goal.** Make crisp measurably sharper than standard resizing on ink-on-paper drawings, with no machine learning, and prove it with a benchmark whose metrics mean something.

**Decisions made during brainstorming.**

| Topic | Decision |
|---|---|
| Scope | M2a only: benchmark repair, blur measurement, solver, engine, routing. M2b and M2c are separate. |
| Benchmark | Repaired first, as the first task of M2a (§3) |
| Algorithm | Model-based reconstruction (approach 2): find the large image that, blurred by the measured blur and shrunk, reproduces the input. This replaces spec §6.2 steps 2–4 (deconvolve, interpolate, redraw) with one joint step. Steps 1 (blur measurement) and 7 (recolour) are kept. |
| Fallback algorithm | If approach 2 does not beat the pipeline in parent spec §6.2 on the benchmark, switch to that pipeline (approach 1). |
| Engine selection | Conservative auto-detection plus `--mode auto\|lineart\|photo`. The whole-image test is replaced by the per-region classifier in M2c. |
| Colours | One ink colour on one paper colour. Several ink colours stay on the baseline engine. |
| Done means | Strict: crisp beats Lanczos on every drawing metric on the clean, blur and JPEG presets at 2×, 4× and 8× (§3.4) |

**Principles carried over.** Never invent content; measure rather than eyeball; pixels, not DPI.

**Known limits (to be stated in the README).** Several ink colours, motion blur, transparency, and text too small to recover (about 5 px tall with fully merged strokes).

---

## 2. Findings from the M1 baseline that shape this work

- On blurred inputs every method scores about the same (PSNR 19.6 vs 19.5 at 2×). Lanczos cannot undo blur; this is the gap the engine must close.
- Scan-preset PSNR is about 16 for every method, because the input is rotated 0.5° but compared with the unrotated original. It cannot show any improvement.
- `edge_error` is unusable: it is degenerate on blurred 8× cases (nearest-neighbour looks best) and a method with no edges looks better than one with a few.
- Legibility reads 0.5–0.7 even for nearest-neighbour, from 6 scenes and Tesseract noise.
- The M1 `crisp` method (Lanczos in linear light) is level with Lanczos on clean input and slightly ahead on JPEG.

---

## 3. Benchmark repair (first task)

### 3.1 Edge F-score replaces edge error

Canny edge maps (thresholds 50/150, as now) for output and ground truth. Precision is the fraction of output edge pixels within the tolerance of a ground-truth edge pixel, recall the reverse, and the metric is their harmonic mean. Tolerance is `max(1, scale / 4)` output pixels. The result is always finite: 1.0 when both maps are empty, 0.0 when exactly one is empty. Higher is better. Aggregation, baseline comparison and the report are updated; `edge_error` is removed.

### 3.2 Fair scan comparison

For presets with rotation or lighting, the reference image is the ground truth with the same rotation and lighting applied (not noise). M2a is then judged on sharpness; deskew is judged in M3.

### 3.3 Speed and scene count

- Ground-truth-derived data (edge map, skeleton, stroke width) is computed once per case and shared by all methods.
- Tesseract calls run in a thread pool.
- Target: 24 scenes in about one hour. If that is not reached, `FULL` uses 12 scenes and the ruling is recorded.
- A new `DEV` config uses seeds 100–111 for tuning. `FULL` and `QUICK` keep seeds from 0. Config gains a `seed_offset` field.

### 3.4 Gate

`crisp bench --gate` runs `FULL`, prints a pass/fail table of crisp against Lanczos for every (scale, preset, metric), and exits 1 if any required cell fails.

- **Required:** clean, blur and JPEG presets at 2×, 4×, 8×.
- **Metrics that must win:** PSNR, SSIM, edge F-score, stroke-width error (lower is better).
- **"Win" means** a better mean and a better result on at least two thirds of the scenes (paired by scene).
- **Legibility:** no worse, meaning the mean is within 0.02 of Lanczos.
- **Scan preset:** reported against the aligned reference, not required.

### 3.5 Baselines

All baselines are regenerated because the metrics change, and the README is updated.

---

## 4. Engine

### 4.1 Units

| Unit | File | Job | Depends on |
|---|---|---|---|
| Two-tone test | `lineart_check.py` | Decide whether a raster is clearly ink on paper; find ink and paper colours | numpy, OpenCV |
| Blur measurement | `psf.py` | Measure optical blur from straight edges | OpenCV, numpy, scipy.optimize or an equivalent fit |
| Solver | `reconstruct.py` | Model-based reconstruction | OpenCV, numpy |
| Engine | `engines/lineart.py` | `LineArtEngine` with the standard `Engine` interface | the three above |
| Routing | `pipeline.py` | Choose engine, record notes, fall back | engines |
| CLI | `cli.py` | `--mode auto\|lineart\|photo` | pipeline |

If a general optimiser is needed beyond numpy/OpenCV, `scipy` (BSD-3) is added as a dependency.

### 4.2 Two-tone test (`lineart_check.py`)

`check_line_art(raster: Raster) -> LineArtCheck`, with `LineArtCheck(is_line_art, ink, paper, ink_fraction, reason)`. A page passes only when all hold:

- Its pixels split into two clusters (Otsu on luminance), paper covering at least 55% of the area and ink between 0.2% and 40%.
- At least 98% of pixels lie within a small distance of the colour line segment between the ink and paper cluster means. This accepts black on white or blue on cream, and rejects red over black.
- Fewer than 3% of pixels are mid-tones that are not adjacent to an edge (this rejects photos and shaded areas).
- There is no alpha channel.

### 4.3 Blur measurement (`psf.py`)

`measure_psf(coverage: np.ndarray) -> Psf`, with `Psf(sigma, n_edges, spread, reliable)`.

1. Find straight segments with OpenCV's line segment detector and keep those at least 20 px long, isolated (no other edge within ±6 px across), with contrast above half the ink–paper range.
2. For each segment, sample brightness across the edge at sub-pixel steps and fit a step edge blurred by a Gaussian. The fit gives that edge's composite blur σ.
3. `sigma_measured` is the median over edges and `spread` is the median absolute deviation divided by the median.
4. The sampling box has known variance 1/12, so the optical blur is `sqrt(max(sigma_measured² − 1/12, 0))`.
5. `reliable` is false when fewer than 8 edges qualify or `spread` is above 0.3.

Isotropic Gaussian blur only.

### 4.4 Solver (`reconstruct.py`)

`reconstruct(coverage, scale, sigma, noise, params) -> np.ndarray` (output-size coverage, 0–1).

- **Space.** Coverage `c = (paper − L) / (paper − ink)`, clipped to 0–1, computed on encoded values (not linear light). The exported-image blur this engine targets mostly comes from software that resizes encoded values.
- **Model.** The forward operator is a Gaussian blur of σ_out = `scale × sigma_optical` at output resolution, followed by an average over `scale × scale` blocks.
- **Objective.** `½‖D·B·x − y‖² + λ·TV(x)` with a smoothed (Charbonnier) total variation. λ scales with the measured noise level (median absolute deviation of the finest wavelet band) and is tuned on `DEV`.
- **Method.** Fixed-count gradient steps starting from a Lanczos enlargement of `y`, with `x` clamped to 0–1 after every step. The back-projection of the data residual is the transpose of the forward operator (replicate, then the same Gaussian).
- **Edge steepening (optional).** A final smooth step that pushes values towards 0 or 1, kept only if `DEV` results show it helps, and never applied where local contrast is below 0.35. Faint-stroke protection proper is M2b.
- **Residual.** The solver returns the RMS of `D·B·x − y` for the fallback test.
- **Cost.** An 8× solve of a 512×384 image is about 12 megapixels. Seconds per image are reported by the benchmark. If the cost is unusable, coarse-to-fine solving (2×, then 4×, then 8×, warm-started) is the planned remedy. Speed work otherwise stays in M5.

### 4.5 Recolour

Output pixels are `paper + (ink − paper) × c`, per channel, from the ink and paper colours the two-tone test found.

### 4.6 Tuning discipline

Penalty strength, iteration count and whether steepening is used are tuned on `DEV` scenes. The gate runs on different scenes (`FULL`), so a pass cannot come from overfitting.

---

## 5. Routing, fallbacks and errors

**`--mode auto` (default).** The raster must pass the two-tone test and `measure_psf` must return `reliable`. Otherwise the baseline engine is used and a note is recorded (for example "baseline: not line art" or "baseline: blur not measurable"). `PageResult` gains `note: str | None`; `--verbose` prints it.

**`--mode lineart`.** Forces the engine. If the two-tone test fails, ink and paper are the darkest and brightest 2% medians. If the blur cannot be measured, σ_optical = 0.7 and a note says so.

**`--mode photo`.** Forces the baseline engine.

**Vector PDF pages** still use the vector engine.

**Residual check.** If the solver's residual RMS exceeds 0.1 coverage units, or any output value is non-finite, the page falls back to the baseline engine with a note. This is a small version of the faithfulness check in M2c.

**Memory.** The solver holds more working copies than the baseline, so `check_resources` takes the engine's copy count (baseline 4, line-art 6). The engine is chosen before the guard runs.

---

## 6. Testing

Test-driven, pytest.

- **`psf.py`:** recovers a known blur within 5% on synthetic edges at several angles; returns `reliable=False` on blank images, curve-only drawings and noise.
- **`lineart_check.py`:** synthetic drawings pass; noise, gradients, photo-like images, red-over-black drawings and images with alpha fail.
- **`reconstruct.py`:** shrinking the result back down reproduces the input within a stated tolerance; a perfect edge stays sharp (transition within 1.5 output px); a 1 px line is not lost; non-finite input to the solver triggers the fallback path.
- **Routing and CLI:** each `--mode` selects the intended engine; notes appear in `--verbose` output; a forced run on a photo does not crash.
- **Metrics:** edge F-score is finite on empty inputs, 1.0 on identical edge maps, drops as an edge is displaced beyond the tolerance; the reference for the scan preset is aligned.
- **Gate:** `compare` and the gate function are tested with synthetic aggregates for each pass and fail case.
- **Regression:** CI continues to run `crisp bench --quick` against the regenerated quick baseline.

---

## 7. Work order

1. Benchmark repair, `DEV` config, gate command, regenerated baselines.
2. Two-tone test.
3. Blur measurement.
4. Solver.
5. Engine, routing, CLI `--mode`, `PageResult.note`, memory guard update.
6. Tuning on `DEV`.
7. Gate run on `FULL`, README, final baselines.

If the gate is not met, the actual numbers and failing metrics are reported. Tuning on the gate scenes is not allowed. If the pipeline in parent spec §6.2 passes where approach 2 does not, that pipeline is used instead, and this spec is amended.

---

## 8. Out of scope for M2a

Line and arc snapping, faint-stroke protection beyond the steepening guard, per-region classification, mixed-image blending, JPEG deblocking, denoising and deskew (M3), tiling (M4), vector export (M5), several ink colours, transparency, anisotropic or motion blur.

---

## 9. Risks

| Risk | Mitigation |
|---|---|
| Few straight edges, so blur cannot be measured | Falls back to the baseline engine with a note; the benchmark counts how often it happens |
| The solver is slow on large outputs | Report seconds per image; coarse-to-fine solving is the planned remedy |
| Penalty tuned too strongly erases thin lines | Stroke-width error is a gate metric; steepening never touches low-contrast strokes |
| The gate cannot be met on JPEG input without M3 deblocking | Noise-scaled penalty first; if it still fails, report it and ask whether to narrow the gate |
| Tesseract noise hides a real legibility change | 24 scenes, and the gate asks only for "no worse" |
| Whole-image test sends a drawing to the baseline engine | Safe by design; `--mode lineart` overrides; M2c adds per-region detection |
