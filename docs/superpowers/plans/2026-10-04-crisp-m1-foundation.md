# crisp M1 — Foundation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** `crisp drawing.png --scale 4` works end to end (images and PDFs, single files and folders), and `crisp bench` produces an HTML report with baseline scores that every later milestone is measured against.

**Architecture:** A `src/`-layout Python package. Intake turns files into `Page`s. Target resolution turns `--scale/--dpi/--size` into an exact output size. One of two engines runs: `vector` (pdfium re-render) or `baseline` (Lanczos-3 in linear light). Write saves the PNG atomically with corrected DPI. A process-pool batch runner and a Typer CLI sit on top. The benchmark lives in `crisp.bench`. It generates synthetic drawings with known ground truth, degrades them, runs every method, scores them, and writes a report.

**Tech Stack:** Python 3.12, uv, numpy, Pillow ≥10.1, opencv-python-headless, scikit-image, pypdfium2 (5.x API), typer, rich, psutil. Dev: pytest, ruff, reportlab (test PDFs only). Bench group: pytesseract.

**Spec:** `docs/superpowers/specs/2026-10-04-crisp-design.md` (milestone M1, §12)

## Global Constraints

- Python `>=3.12`, package `crisp` in `src/crisp/`, console entry point `crisp = "crisp.cli:main"`.
- Runtime dependencies must be permissive (MIT/BSD/Apache/HPND). Never PyMuPDF (AGPL) or potrace (GPL).
- No machine learning anywhere in `src/crisp/` outside `src/crisp/bench/`. Tesseract and Real-ESRGAN are optional, benchmark-only, and imported or located lazily.
- Pixels are `float32`, range 0–1, shape H×W×C with C ∈ {1, 3}. Alpha is a separate H×W array or `None`.
- Scale must be `> 1` and `<= 16`.
- Source files are never overwritten. Output name: `<stem>[-p<n>]@<label>.png` (`n` is 1-based, present only for multi-page sources).
- Output DPI = input DPI × scale for `--scale`; = D for `--dpi D` and `--size PAPER@D`; unset when the input DPI is unknown.
- Exit codes: 0 = no file failed, 1 = at least one file failed, 2 = usage error.
- Memory guard: refuse a page whose estimated working memory exceeds 25% of available RAM.
- Windows is the primary platform: use `pathlib` everywhere; batch workers must be top-level functions (spawn-safe).
- Work on branch `feat/m1-foundation`. Conventional Commit messages, each ending with `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- Run every command through `uv run` from the repo root.

### Decisions this plan adds to the spec

1. Benchmark code lives in `src/crisp/bench/` so the installed `crisp bench` command can import it. Data, reports and results stay under `bench/` at the repo root (`bench/results/baseline.json` is committed).
2. Synthetic text uses Pillow's built-in scalable font (`ImageFont.load_default(size=…)`, Aileron Regular, CC0) instead of bundling DejaVu Sans. No download, no extra licence file.
3. Synthetic text heights (4–40 px) and blur σ are measured in **degraded-input** pixels.
4. Label suffix for `--dpi D` is `<D>dpi`; for `--size A3@300` it is `A3-300dpi`.
5. A vector PDF page's reference resolution is 72 DPI (PDF points), so `--scale 4` renders it at 288 DPI.
6. `--dpi` on an image whose DPI is unknown fails **that file** (with a hint to use `--scale`), not the whole batch. DPI values below 20 count as unknown.
7. Corrupt, unsupported and password-protected files count as **failed** (exit 1). "Skipped" means the output already exists.
8. Pages too large for memory are refused in M1. Tiling arrives in M4.
9. Multi-page TIFFs are split into pages like PDFs. Transparency is preserved. Output is always 8-bit PNG in M1.
10. CLI dispatch: if the first argument is `bench`, run the benchmark app; otherwise run the upscale command. `crisp bench` also gains `--update-baseline` and `--root PATH`.
11. Deferred to later plans: `--mode` (M2), `--vector` (M5), patent-drawing and private-drawing datasets (M2), photo datasets (M4).

## Review Focus

1. **Re-running on a folder that already contains crisp outputs.** Folder scans must ignore files whose stem matches the crisp output pattern, so `a@4x.png` doesn't become `a@4x@4x.png`. Explicitly named files are still processed. *(test in Task 9)*
2. **`--recursive --out` with the same filename in different subfolders.** Outputs mirror the subfolder structure under `--out`, so nothing collides. *(tests in Tasks 7 and 9)*
3. **A run killed mid-write.** No truncated PNG may be left at the final path, otherwise the next run would skip it as "output exists". *(test in Task 7)*
4. **Phone photo of a drawing with EXIF rotation.** Output comes out upright, matching what image viewers show. *(test in Task 2)*
5. **`crisp *.png` in PowerShell or cmd, where the shell doesn't expand globs.** crisp expands the pattern itself. *(test in Task 9)*

---

### Task 1: Project scaffold, colour conversion, core types

**Files:**
- Create: `pyproject.toml`, `src/crisp/__init__.py`, `src/crisp/color.py`, `src/crisp/types.py`, `tests/__init__.py`, `tests/test_color.py`

**Interfaces:**
- Produces: `crisp.__version__ == "0.1.0"`; `srgb_to_linear(x: np.ndarray) -> np.ndarray`, `linear_to_srgb(x: np.ndarray) -> np.ndarray`; all types below (later tasks import them from `crisp.types`).

- [ ] **Step 1: Create the branch and `pyproject.toml`**

```bash
git switch main && git pull && git switch -c feat/m1-foundation
```

```toml
[project]
name = "crisp"
version = "0.1.0"
description = "Free, open-source, AI-free upscaler for technical drawings, scans and documents."
readme = "README.md"
license = "MIT"
requires-python = ">=3.12"
dependencies = [
  "numpy>=2.0",
  "pillow>=10.1",
  "opencv-python-headless>=4.10",
  "scikit-image>=0.24",
  "pypdfium2>=5.0",
  "typer>=0.12",
  "rich>=13.7",
  "psutil>=5.9",
]

[project.scripts]
crisp = "crisp.cli:main"

[dependency-groups]
dev = ["pytest>=8.2", "ruff>=0.6", "reportlab>=4.2"]
bench = ["pytesseract>=0.3.10"]

[tool.uv]
default-groups = ["dev", "bench"]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/crisp"]

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.ruff.lint]
select = ["E", "F", "I", "UP", "B", "SIM"]

[tool.pytest.ini_options]
testpaths = ["tests"]
```

- [ ] **Step 2: Write the failing tests** (`tests/test_color.py`)

```python
def test_roundtrip():
    x = np.linspace(0, 1, 1001, dtype=np.float32)
    assert np.allclose(linear_to_srgb(srgb_to_linear(x)), x, atol=1e-5)


def test_known_values():
    assert float(srgb_to_linear(np.float32(0.5))) == pytest.approx(0.21404, abs=1e-4)
    assert float(srgb_to_linear(np.float32(0.04045))) == pytest.approx(0.0031308, abs=1e-6)
```

- [ ] **Step 3: Run `uv sync`, then `uv run pytest tests/test_color.py -v`.** Expected: FAIL (ImportError).

- [ ] **Step 4: Implement `color.py` (IEC 61966-2-1 piecewise curves, keep `float32`) and `types.py`:**

```python
@dataclass(eq=False)
class Raster:
    pixels: np.ndarray  # float32 HxWxC, C in {1,3}, 0-1
    alpha: np.ndarray | None  # float32 HxW, 0-1


@dataclass(eq=False)
class Page:
    source: Path
    index: int  # 0-based
    page_count: int
    raster: Raster | None  # None for vector PDF pages
    dpi: float | None
    has_vectors: bool
    size_pt: tuple[float, float] | None  # PDF page size in points, vector pages only
    source_format: str  # "png" | "jpeg" | "tiff" | "bmp" | "webp" | "pdf"

    @property
    def size_px(self) -> tuple[int, int]: ...  # (w, h); vector pages: round(size_pt) at 72 DPI


@dataclass(frozen=True)
class TargetSpec:
    kind: Literal["scale", "dpi", "size"]
    scale: float | None = None
    dpi: int | None = None
    paper: str | None = None  # "A0".."A5", "Letter", "Tabloid"


@dataclass(frozen=True)
class Target:
    scale: float
    out_size: tuple[int, int]  # (w, h)
    out_dpi: float | None
    label: str  # "4x", "1.5x", "600dpi", "A3-300dpi"


@dataclass(frozen=True)
class Options:
    out_dir: Path | None = None
    overwrite: bool = False
    protected: frozenset[Path] = frozenset()  # resolved input paths that must never be written


@dataclass
class PageResult:
    page: int
    status: Literal["done", "skipped", "failed"]
    output: Path | None = None
    engine: str | None = None  # "baseline" | "vector"
    target: Target | None = None
    reason: str | None = None
    seconds: float = 0.0


@dataclass
class FileOutcome:
    source: Path
    pages: list[PageResult] = field(default_factory=list)
    error: str | None = None  # file-level failure (could not open)

    @property
    def status(self) -> Literal["done", "skipped", "failed"]: ...

    # failed if error or any page failed; skipped if every page skipped; else done


@dataclass
class BatchSummary:
    outcomes: list[FileOutcome]

    def counts(self) -> dict[str, int]: ...  # keys "done", "skipped", "failed", always present
    @property
    def exit_code(self) -> int: ...  # 1 if any outcome failed, else 0


class CrispError(Exception): ...  # message is shown to the user as-is


class UnsupportedInput(CrispError): ...


class TargetError(CrispError): ...


class ResourceError(CrispError): ...
```

- [ ] **Step 5: Verify.** `uv run pytest -v` → PASS. `uv run ruff check . && uv run ruff format --check .` → clean. `uv run python -c "from PIL import ImageFont; print(type(ImageFont.load_default(size=40)).__name__)"` → `FreeTypeFont`.

- [ ] **Step 6: Commit** `pyproject.toml uv.lock src tests` with message `chore: scaffold package with colour conversion and core types`.

---

### Task 2: Intake — raster images

**Files:**
- Create: `src/crisp/intake.py`, `tests/conftest.py`, `tests/test_intake_raster.py`

**Interfaces:**
- Consumes: `Page`, `Raster`, `UnsupportedInput` (Task 1)
- Produces: `SUPPORTED_EXTENSIONS: frozenset[str]` = `{".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".pdf"}`; `open_file(path: Path) -> list[Page]` (raises `UnsupportedInput`). `tests/conftest.py` fixture `save_image(tmp_path)` → `fn(name: str, img: PIL.Image.Image, **save_kwargs) -> Path`.

- [ ] **Step 1: Write the failing tests**

```python
def test_rgb_png(save_image):
    p = save_image("a.png", Image.new("RGB", (4, 3), (255, 0, 0)), dpi=(300, 300))
    [page] = open_file(p)
    assert page.raster.pixels.shape == (3, 4, 3) and page.raster.pixels.dtype == np.float32
    assert page.raster.alpha is None and page.dpi == pytest.approx(300, abs=0.01)
    assert (page.index, page.page_count, page.has_vectors, page.source_format) == (
        0,
        1,
        False,
        "png",
    )
    assert page.size_px == (4, 3)


def test_grayscale_single_channel(save_image):
    [page] = open_file(save_image("g.png", Image.new("L", (5, 5), 128)))
    assert page.raster.pixels.shape == (5, 5, 1)


def test_16bit_scaled(save_image):
    img = Image.fromarray(np.full((2, 2), 65535, np.uint16))
    [page] = open_file(save_image("h.png", img))
    assert page.raster.pixels.max() == pytest.approx(1.0)


def test_alpha_preserved(save_image):
    [page] = open_file(save_image("t.png", Image.new("RGBA", (3, 3), (0, 0, 0, 128))))
    assert page.raster.pixels.shape == (3, 3, 3)
    assert page.raster.alpha == pytest.approx(np.full((3, 3), 128 / 255), abs=1e-6)


def test_exif_rotation_applied(save_image):  # Review Focus 4
    exif = Image.Exif()
    exif[0x0112] = 6
    [page] = open_file(save_image("r.jpg", Image.new("RGB", (4, 2)), exif=exif))
    assert page.size_px == (2, 4)


def test_dpi_missing_or_tiny_is_unknown(save_image):
    assert open_file(save_image("n.png", Image.new("L", (2, 2))))[0].dpi is None
    assert open_file(save_image("s.png", Image.new("L", (2, 2)), dpi=(1, 1)))[0].dpi is None


def test_multipage_tiff(tmp_path):
    frames = [Image.new("L", (4, 4), v) for v in (0, 100, 200)]
    p = tmp_path / "m.tif"
    frames[0].save(p, save_all=True, append_images=frames[1:])
    pages = open_file(p)
    assert [pg.index for pg in pages] == [0, 1, 2] and all(pg.page_count == 3 for pg in pages)


def test_corrupt(tmp_path):
    (p := tmp_path / "x.png").write_bytes(b"not an image")
    with pytest.raises(UnsupportedInput, match="could not read"):
        open_file(p)


def test_unsupported_extension(tmp_path):
    (p := tmp_path / "x.gif").write_bytes(b"GIF89a")
    with pytest.raises(UnsupportedInput, match="unsupported file type"):
        open_file(p)
```

- [ ] **Step 2: Run `uv run pytest tests/test_intake_raster.py -v`.** Expected: FAIL (ImportError).

- [ ] **Step 3: Implement `open_file` for raster formats.** Dispatch on lower-cased suffix (`.pdf` raises `NotImplementedError` until Task 3). Iterate frames with `ImageSequence.Iterator`, apply `ImageOps.exif_transpose` per frame, then normalise modes. `1`/`L`/`I;16*`/`I` become 1 channel (16-bit and `I` divided by 65535 and clipped). `LA`/`PA`/`RGBA`, and `P` with transparency, split off alpha. Everything else is converted to `RGB`. DPI is `info["dpi"][0]` when ≥ 20, else `None`. Wrap Pillow errors as `UnsupportedInput(f"could not read image: {e}")`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(intake): open raster images as normalised pages`.

---

### Task 3: Intake — PDFs

**Files:**
- Modify: `src/crisp/intake.py`, `tests/conftest.py`
- Create: `tests/test_intake_pdf.py`

**Interfaces:**
- Consumes: `open_file` (Task 2)
- Produces: `open_file` now handles `.pdf`. Conftest fixtures (via reportlab): `vector_pdf(tmp_path) -> Path` (A4, lines + text, 1 page), `image_pdf(tmp_path) -> Path` (one 400×300 pt page fully covered by an 800×600 px RGB image), `two_page_pdf(tmp_path) -> Path` (2 vector pages), `encrypted_pdf(tmp_path) -> Path` (`canvas.Canvas(..., encrypt="secret")`).

- [ ] **Step 1: Write the failing tests**

```python
def test_vector_page(vector_pdf):
    [page] = open_file(vector_pdf)
    assert page.has_vectors and page.raster is None and page.dpi == 72.0
    assert page.size_pt == pytest.approx((595.28, 841.89), abs=0.1)
    assert page.size_px == (595, 842) and page.source_format == "pdf"


def test_image_only_page(image_pdf):
    [page] = open_file(image_pdf)
    assert not page.has_vectors and page.raster.pixels.shape == (600, 800, 3)
    assert page.dpi == pytest.approx(144, abs=0.5)


def test_pages_split(two_page_pdf):
    pages = open_file(two_page_pdf)
    assert [p.index for p in pages] == [0, 1] and pages[1].page_count == 2


def test_password_protected(encrypted_pdf):
    with pytest.raises(UnsupportedInput, match="password-protected"):
        open_file(encrypted_pdf)
```

- [ ] **Step 2: Run `uv run pytest tests/test_intake_pdf.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement PDF intake with pypdfium2 5.x.** Use `pdfium.PdfDocument(path)`, where a `PdfiumError` mentioning "password" becomes `UnsupportedInput("password-protected PDF")`; zero pages becomes `UnsupportedInput("PDF has no pages")`. Then `page.get_size()` and `page.get_objects(max_depth=15)`. Sum the clamped bounding-box area of each object, using `obj.get_bounds()`, by type (`pdfium.raw.FPDF_PAGEOBJ_IMAGE` vs `FPDF_PAGEOBJ_PATH`/`FPDF_PAGEOBJ_TEXT`). Apply spec §6.1: image coverage ≥ 90% **and** path+text coverage < 1% → image-only. For image-only pages, take the largest image via `get_bitmap(render=False).to_pil()`, normalise it with the Task 2 code path, and set DPI = image px width ÷ (bounds width pt ÷ 72). Any other page is vector: `raster=None`, `dpi=72.0`, `size_pt` set.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(intake): split PDFs and detect vector vs image-only pages`.

---

### Task 4: Target resolution

**Files:**
- Create: `src/crisp/target.py`, `tests/test_target.py`; add `make_page(w, h, dpi=None, channels=3, vector=False)` helper fixture to `tests/conftest.py` (`vector=True` ignores `w`/`h` and returns an A4 vector page: `size_pt=(595.28, 841.89)`, `dpi=72.0`, `raster=None`)

**Interfaces:**
- Consumes: `Page`, `TargetSpec`, `Target`, `TargetError`
- Produces: `MAX_SCALE = 16.0`; `PAPER_SIZES_MM` (A0 841×1189, A1 594×841, A2 420×594, A3 297×420, A4 210×297, A5 148×210, Letter 215.9×279.4, Tabloid 279.4×431.8, as portrait (w, h)); `parse_target(scale: float | None, dpi: int | None, size: str | None) -> TargetSpec` (raises `ValueError` with user-facing text); `resolve_target(spec: TargetSpec, page: Page) -> Target`.

- [ ] **Step 1: Write the failing tests**

```python
def test_parse():
    assert parse_target(None, None, None) == TargetSpec("scale", scale=2.0)
    assert parse_target(None, None, "a3@300") == TargetSpec("size", dpi=300, paper="A3")
    with pytest.raises(ValueError, match="only one of"):
        parse_target(4, 300, None)
    with pytest.raises(ValueError, match="greater than 1"):
        parse_target(1.0, None, None)
    with pytest.raises(ValueError, match="greater than 1"):
        parse_target(17, None, None)
    with pytest.raises(ValueError, match="A3@300"):
        parse_target(None, None, "B5@300")


def test_scale(make_page):
    t = resolve_target(TargetSpec("scale", scale=4), make_page(100, 50, dpi=300))
    assert (t.out_size, t.out_dpi, t.label) == ((400, 200), 1200, "4x")
    assert resolve_target(TargetSpec("scale", scale=1.5), make_page(10, 10)).label == "1.5x"
    assert resolve_target(TargetSpec("scale", scale=2), make_page(10, 10)).out_dpi is None


def test_dpi(make_page):
    t = resolve_target(TargetSpec("dpi", dpi=600), make_page(100, 50, dpi=150))
    assert (t.scale, t.out_size, t.out_dpi, t.label) == (4.0, (400, 200), 600, "600dpi")
    with pytest.raises(TargetError, match="--scale"):
        resolve_target(TargetSpec("dpi", dpi=600), make_page(10, 10))
    with pytest.raises(TargetError, match="already at or above"):
        resolve_target(TargetSpec("dpi", dpi=300), make_page(10, 10, dpi=600))
    with pytest.raises(TargetError, match="16×"):
        resolve_target(TargetSpec("dpi", dpi=600), make_page(10, 10, dpi=20))


def test_size_matches_orientation(make_page):
    t = resolve_target(TargetSpec("size", dpi=300, paper="A4"), make_page(1000, 600))
    assert (t.out_size, t.out_dpi, t.label) == ((3508, 2105), 300, "A4-300dpi")


def test_vector_page(make_page):
    t = resolve_target(TargetSpec("scale", scale=2), make_page(0, 0, vector=True))  # A4 points
    assert (t.out_size, t.out_dpi) == ((1191, 1684), 144)
```

- [ ] **Step 2: Run `uv run pytest tests/test_target.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement.** Out size = `max(1, round(dim * scale))` using the size of `page.size_pt` for vector pages and `page.size_px` otherwise. For `size`, rotate the paper to landscape when the page is wider than tall; paper px = mm ÷ 25.4 × D; scale = min of the two ratios. Error texts: `"input DPI unknown; use --scale instead"`, `"already at or above the requested size (would need {s:.2f}×)"`, `"would need {s:.1f}×, above the 16× limit"`. Parse texts: `"Use only one of --scale, --dpi or --size."`, `"--scale must be greater than 1 and at most 16."`, `"--size must look like A3@300 (paper: A0–A5, Letter, Tabloid)."`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(target): resolve --scale, --dpi and --size into output sizes`.

---

### Task 5: Baseline engine (Lanczos-3, linear light)

**Files:**
- Create: `src/crisp/engines/__init__.py`, `src/crisp/engines/base.py`, `src/crisp/engines/baseline.py`, `tests/test_engine_baseline.py`

**Interfaces:**
- Consumes: `Page`, `Target`, `Raster`, colour functions
- Produces: `class Engine(Protocol): name: str; def upscale(self, page: Page, target: Target) -> Raster`; `lanczos_resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray` (H×W×C float32 in, no colour conversion, unclipped); `BaselineEngine` with `name = "baseline"`.

- [ ] **Step 1: Write the failing tests**

```python
def up(arr, scale, alpha=None):
    page = Page(Path("x.png"), 0, 1, Raster(arr, alpha), None, False, None, "png")
    return BaselineEngine().upscale(page, resolve_target(TargetSpec("scale", scale=scale), page))


def test_shape_and_range():
    out = up(np.random.default_rng(0).random((8, 10, 3), dtype=np.float32), 4)
    assert out.pixels.shape == (32, 40, 3) and out.pixels.dtype == np.float32
    assert 0 <= out.pixels.min() and out.pixels.max() <= 1


def test_constant_stays_constant():
    assert np.allclose(up(np.full((6, 6, 1), 0.5, np.float32), 3).pixels, 0.5, atol=1e-3)


def test_resamples_in_linear_light():
    checker = (np.indices((16, 16)).sum(0) % 2).astype(np.float32)[..., None]
    out = up(checker, 2).pixels
    assert srgb_to_linear(out).mean() == pytest.approx(0.5, abs=0.02)


def test_alpha_has_no_dark_fringe():
    rgb = np.zeros((8, 8, 3), np.float32)
    rgb[:, :4] = 1.0
    alpha = np.zeros((8, 8), np.float32)
    alpha[:, :4] = 1.0
    out = up(rgb, 4, alpha)
    assert out.alpha is not None and out.pixels[out.alpha > 0.01].min() >= 0.98
```

- [ ] **Step 2: Run `uv run pytest tests/test_engine_baseline.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement.** `lanczos_resize` resizes each channel with Pillow mode `"F"` and `Image.Resampling.LANCZOS` (Lanczos-3). `BaselineEngine.upscale`: convert sRGB → linear, premultiply by alpha, resize colour and alpha, unpremultiply where alpha > 1e-6, convert linear → sRGB, clip to [0, 1], cast to `float32`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(engines): add baseline Lanczos-3 engine in linear light`.

---

### Task 6: Vector engine (PDF re-render)

**Files:**
- Create: `src/crisp/engines/vector.py`, `tests/test_engine_vector.py`

**Interfaces:**
- Consumes: `vector_pdf` fixture (Task 3), `Engine` protocol (Task 5)
- Produces: `VectorEngine` with `name = "vector"`.

- [ ] **Step 1: Write the failing test**

```python
def test_renders_vector_page_at_exact_size(vector_pdf):
    [page] = open_file(vector_pdf)
    target = resolve_target(TargetSpec("scale", scale=2), page)
    out = VectorEngine().upscale(page, target)
    assert out.pixels.shape == (1684, 1191, 3) and out.alpha is None
    assert out.pixels.min() < 0.2 and np.median(out.pixels) > 0.9
```

- [ ] **Step 2: Run `uv run pytest tests/test_engine_vector.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement.** Open `page.source` with pypdfium2, call `pdf[page.index].render(scale=target.out_size[0] / page.size_pt[0], fill_color=(255, 255, 255, 255), may_draw_forms=True).to_pil().convert("RGB")`, then crop or edge-pad to exactly `target.out_size` (pdfium rounding can differ by 1 px). Return a `float32` array ÷ 255.

- [ ] **Step 4: Run the test.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(engines): re-render vector PDF pages with pdfium`.

---

### Task 7: Writer and output naming

**Files:**
- Create: `src/crisp/write.py`, `tests/test_write.py`

**Interfaces:**
- Consumes: `Raster`
- Produces: `OUTPUT_PATTERN: re.Pattern` (matches crisp output **stems**); `output_path(source: Path, page_index: int, page_count: int, label: str, out_dir: Path | None, rel_parent: Path = Path()) -> Path`; `write_png(raster: Raster, path: Path, dpi: float | None) -> None` (creates parent folders, atomic).

- [ ] **Step 1: Write the failing tests**

```python
def test_output_path():
    assert output_path(Path("d/a.png"), 0, 1, "4x", None) == Path("d/a@4x.png")
    assert output_path(Path("d/plan.pdf"), 2, 5, "600dpi", Path("out")) == Path(
        "out/plan-p3@600dpi.png"
    )
    assert output_path(Path("s/x/a.png"), 0, 1, "2x", Path("o"), Path("x")) == Path(
        "o/x/a@2x.png"
    )  # RF2


def test_output_pattern():
    for stem in ("a@4x", "a@1.5x", "plan-p3@600dpi", "a@A3-300dpi"):
        assert OUTPUT_PATTERN.search(stem)
    for stem in ("a", "me@home", "a@4"):
        assert not OUTPUT_PATTERN.search(stem)


@pytest.mark.parametrize(
    "channels,alpha,mode", [(3, False, "RGB"), (1, False, "L"), (3, True, "RGBA"), (1, True, "LA")]
)
def test_write_modes_and_dpi(tmp_path, channels, alpha, mode):
    r = Raster(
        np.full((4, 5, channels), 0.5, np.float32), np.ones((4, 5), np.float32) if alpha else None
    )
    write_png(r, p := tmp_path / "sub" / "o.png", 1200)
    img = Image.open(p)
    assert (img.mode, img.size) == (mode, (5, 4)) and img.info["dpi"][0] == pytest.approx(
        1200, abs=0.01
    )


def test_no_dpi(tmp_path):
    write_png(Raster(np.zeros((2, 2, 1), np.float32), None), p := tmp_path / "o.png", None)
    assert "dpi" not in Image.open(p).info


def test_failed_write_leaves_nothing(tmp_path, monkeypatch):  # RF3
    def broken_save(self, fp, *a, **k):
        Path(fp).write_bytes(b"partial")
        raise OSError("disk full")

    monkeypatch.setattr(Image.Image, "save", broken_save)
    with pytest.raises(OSError):
        write_png(Raster(np.zeros((2, 2, 1), np.float32), None), tmp_path / "o.png", None)
    assert list(tmp_path.iterdir()) == []
```

- [ ] **Step 2: Run `uv run pytest tests/test_write.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement.** Pattern: `r"@(\d+(\.\d+)?x|\d+dpi|(A[0-5]|Letter|Tabloid)-\d+dpi)$"`, case-insensitive. `write_png` converts with `np.round(x * 255)` to `uint8`, picks the mode from channels and alpha, saves to a temp **path** in the target folder (`<name>.<random>.tmp`, passed to `Image.save` as a path, not a file object), then `os.replace` into place. On any exception it deletes the temp file and re-raises. Pass `dpi=(d, d)` only when not `None`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(write): atomic PNG writer with DPI and output naming`.

---

### Task 8: Pipeline and resource guard

**Files:**
- Create: `src/crisp/resources.py`, `src/crisp/pipeline.py`, `tests/test_resources.py`, `tests/test_pipeline.py`
- Modify: `src/crisp/__init__.py` (export `process`, `TargetSpec`, `Options`)

**Interfaces:**
- Consumes: Tasks 2–7
- Produces: `check_resources(target: Target, channels: int, out_dir: Path, available_ram: int | None = None, free_disk: int | None = None) -> None` (raises `ResourceError`; `None` means "measure with psutil / shutil.disk_usage on the nearest existing parent"); `upscale_page(page: Page, target: Target) -> tuple[Raster, str]` (raster + engine name; used by the benchmark); `process(path: Path, spec: TargetSpec, options: Options = Options(), rel_parent: Path = Path()) -> FileOutcome` (never raises for per-file problems).

- [ ] **Step 1: Write the failing tests**

```python
# test_resources.py — working ≈ w*h*C*4 bytes*4 copies = 48 MB; PNG ≈ w*h*C/4 = 0.75 MB
T1000 = Target(4.0, (1000, 1000), None, "4x")


def test_ok(tmp_path):
    check_resources(T1000, 3, tmp_path, available_ram=10**9, free_disk=10**12)


def test_memory(tmp_path):
    with pytest.raises(ResourceError, match="memory"):
        check_resources(T1000, 3, tmp_path, available_ram=10**8, free_disk=10**12)


def test_disk(tmp_path):
    with pytest.raises(ResourceError, match="disk"):
        check_resources(T1000, 3, tmp_path, available_ram=10**12, free_disk=100)


# test_pipeline.py
def test_png_300dpi_scale4_writes_1200dpi(save_image):
    src = save_image("a.png", Image.new("RGB", (10, 8), "white"), dpi=(300, 300))
    out = process(src, TargetSpec("scale", scale=4))
    [pr] = out.pages
    assert (out.status, pr.engine) == ("done", "baseline")
    img = Image.open(pr.output)
    assert img.size == (40, 32) and img.info["dpi"][0] == pytest.approx(1200, abs=0.01)


def test_vector_pdf_uses_vector_engine(vector_pdf):
    assert process(vector_pdf, TargetSpec("scale", scale=2)).pages[0].engine == "vector"


def test_existing_output_skipped_unless_overwrite(save_image):
    src = save_image("a.png", Image.new("L", (4, 4)))
    process(src, TargetSpec("scale", scale=2))
    assert process(src, TargetSpec("scale", scale=2)).pages[0].reason == "output exists"
    assert process(src, TargetSpec("scale", scale=2), Options(overwrite=True)).status == "done"


def test_corrupt_file(tmp_path):
    (p := tmp_path / "x.png").write_bytes(b"junk")
    out = process(p, TargetSpec("scale", scale=2))
    assert out.status == "failed" and "could not read" in out.error


def test_never_overwrites_an_input(save_image):
    src = save_image("a.png", Image.new("L", (4, 4)))
    other = save_image("a@2x.png", Image.new("L", (9, 9)))
    before = other.read_bytes()
    out = process(
        src,
        TargetSpec("scale", scale=2),
        Options(overwrite=True, protected=frozenset({other.resolve()})),
    )
    assert out.pages[0].reason == "would overwrite an input file" and other.read_bytes() == before


def test_dpi_unknown_fails_file(save_image):
    out = process(save_image("a.png", Image.new("L", (4, 4))), TargetSpec("dpi", dpi=600))
    assert out.status == "failed" and "--scale" in out.pages[0].reason


def test_unexpected_error_is_contained(save_image, monkeypatch):
    monkeypatch.setattr(
        BaselineEngine, "upscale", lambda *a: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    out = process(save_image("a.png", Image.new("L", (4, 4))), TargetSpec("scale", scale=2))
    assert out.pages[0].reason == "unexpected error: RuntimeError: boom"
```

- [ ] **Step 2: Run `uv run pytest tests/test_resources.py tests/test_pipeline.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement `resources.py`.** Messages: `"needs about {x:.1f} GB of memory, more than 25% of the {y:.1f} GB available; very large outputs need tiling, which arrives in a later version"` and `"needs about {x:.1f} GB of disk space but only {y:.1f} GB is free"`.

- [ ] **Step 4: Implement `pipeline.py`.** `upscale_page` picks `VectorEngine` when `page.has_vectors`, else `BaselineEngine`. Per page, `process` runs in this order: resolve target → compute output path → protected check (`output.resolve()` in `options.protected` or equal to `path.resolve()` → failed `"would overwrite an input file"`) → existing-output check (skipped `"output exists"`) → `check_resources` → `upscale_page` → `write_png`. Time each page with `time.perf_counter`. Catch `CrispError` as `reason=str(e)` and any other `Exception` as `reason=f"unexpected error: {type(e).__name__}: {e}"`.

- [ ] **Step 5: Run the tests.** Expected: PASS.

- [ ] **Step 6: Commit** with message `feat(pipeline): process files end to end with resource guard`.

---

### Task 9: Batch runner and input discovery

**Files:**
- Create: `src/crisp/batch.py`, `tests/test_batch.py`
- Modify: `src/crisp/__init__.py` (export `process_batch`)

**Interfaces:**
- Consumes: `process`, `SUPPORTED_EXTENSIONS`, `OUTPUT_PATTERN`
- Produces: `@dataclass(frozen=True) class InputItem: path: Path; rel_parent: Path`; `discover_inputs(paths: list[Path], recursive: bool = False) -> list[InputItem]` (raises `FileNotFoundError` for a path that neither exists nor matches a glob); `process_batch(items: list[InputItem], spec: TargetSpec, options: Options = Options(), jobs: int = 1, on_done: Callable[[FileOutcome], None] | None = None) -> BatchSummary`.

- [ ] **Step 1: Write the failing tests**

```python
@pytest.fixture
def tree(tmp_path):
    for rel in ("a.png", "b.JPG", "sub/c.png", "a@2x.png"):
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        Image.new("L", (4, 4)).save(
            tmp_path / rel, format="PNG" if rel.lower().endswith("png") else "JPEG"
        )
    (tmp_path / "notes.txt").write_text("x")
    return tmp_path


def names(items):
    return sorted(i.path.name for i in items)


def test_folder_scan_one_level_skips_crisp_outputs(tree):  # RF1
    assert names(discover_inputs([tree])) == ["a.png", "b.JPG"]


def test_explicit_crisp_output_still_processed(tree):
    assert names(discover_inputs([tree / "a@2x.png"])) == ["a@2x.png"]


def test_recursive_sets_rel_parent(tree):
    items = discover_inputs([tree], recursive=True)
    assert {i.path.name: i.rel_parent for i in items}["c.png"] == Path("sub")


def test_glob_expanded_by_crisp(tree):  # RF5 (and RF1 for globs)
    assert names(discover_inputs([tree / "*.png"])) == ["a.png"]


def test_missing_and_duplicates(tree):
    with pytest.raises(FileNotFoundError):
        discover_inputs([tree / "nope.png"])
    assert len(discover_inputs([tree / "a.png", tree / "a.png"])) == 1


@pytest.mark.parametrize("jobs", [1, 2])
def test_batch_isolates_failures(tree, jobs):
    (tree / "bad.png").write_bytes(b"junk")
    items = discover_inputs([tree / "a.png", tree / "bad.png", tree / "b.JPG"])
    seen = []
    s = process_batch(items, TargetSpec("scale", scale=2), jobs=jobs, on_done=seen.append)
    assert [o.source.name for o in s.outcomes] == ["a.png", "bad.png", "b.JPG"]
    assert (
        s.counts() == {"done": 2, "skipped": 0, "failed": 1} and s.exit_code == 1 and len(seen) == 3
    )


def test_recursive_out_mirrors_folders(tmp_path):  # RF2
    for d in ("x", "y"):
        (tmp_path / "in" / d).mkdir(parents=True)
        Image.new("L", (4, 4)).save(tmp_path / "in" / d / "a.png")
    items = discover_inputs([tmp_path / "in"], recursive=True)
    process_batch(items, TargetSpec("scale", scale=2), Options(out_dir=tmp_path / "o"))
    assert (tmp_path / "o/x/a@2x.png").exists() and (tmp_path / "o/y/a@2x.png").exists()
```

- [ ] **Step 2: Run `uv run pytest tests/test_batch.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement.** Discovery keeps argument order. An existing file is included as is. An existing dir contributes files with a supported suffix (case-insensitive), via `rglob` if recursive. Otherwise, if the string contains `*?[`, use `glob.glob`, keeping supported suffixes only. Folder scans and glob matches skip stems that match `OUTPUT_PATTERN` and are sorted within themselves. Dedupe by resolved path, keeping the first occurrence. `process_batch` sets `options.protected` to all resolved input paths (`dataclasses.replace`). With `jobs == 1` it runs inline; otherwise it uses a `ProcessPoolExecutor` with a top-level worker. A future that raises (e.g. `BrokenProcessPool`) becomes `FileOutcome(source, error=f"worker crashed: {e}")`. Return outcomes in input order and call `on_done` as each finishes.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(batch): discover inputs and process them in parallel`.

---

### Task 10: CLI — upscale command

**Files:**
- Create: `src/crisp/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: `parse_target`, `discover_inputs`, `process_batch`
- Produces: `upscale_app: typer.Typer` (single command: `inputs: list[Path]`, `--scale FLOAT`, `--dpi INT`, `--size TEXT`, `--out PATH`, `--jobs INT` (default `os.cpu_count()`), `--recursive`, `--overwrite`, `--verbose`); `main(argv: list[str] | None = None) -> None` (runs `upscale_app`; Task 17 adds `bench` dispatch).

- [ ] **Step 1: Write the failing tests**

```python
invoke = lambda args: CliRunner().invoke(upscale_app, args)


def test_upscales(save_image):
    src = save_image("a.png", Image.new("L", (4, 4)))
    r = invoke([str(src), "--scale", "4", "--jobs", "1"])
    assert r.exit_code == 0 and (src.parent / "a@4x.png").exists()


def test_conflicting_size_options(save_image):
    r = invoke([str(save_image("a.png", Image.new("L", (4, 4)))), "--scale", "4", "--dpi", "300"])
    assert r.exit_code == 2 and "only one of" in r.output


def test_bad_size(save_image):
    assert (
        invoke([str(save_image("a.png", Image.new("L", (4, 4)))), "--size", "B9@300"]).exit_code
        == 2
    )


def test_empty_folder(tmp_path):
    r = invoke([str(tmp_path)])
    assert r.exit_code == 2 and "No supported images found" in r.output


def test_missing_path(tmp_path):
    assert invoke([str(tmp_path / "nope.png")]).exit_code == 2


def test_failure_reported_with_exit_1(save_image, tmp_path):
    good = save_image("a.png", Image.new("L", (4, 4)))
    (bad := tmp_path / "bad.png").write_bytes(b"junk")
    r = invoke([str(good), str(bad), "--jobs", "1"])
    assert r.exit_code == 1 and "bad.png" in r.output and "could not read" in r.output
```

- [ ] **Step 2: Run `uv run pytest tests/test_cli.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement.** Convert `ValueError` from `parse_target` into `typer.BadParameter` (exit 2), and `FileNotFoundError` from discovery into exit 2 with `"Path not found: …"`. No items → print `"No supported images found."` and exit 2. Show a rich progress bar over files. With `--verbose`, print one line per page: source, page, engine, `target.label`, output path, seconds. At the end, print a summary table with Done/Skipped/Failed counts, then each failed file with its reason as a plain line (`soft_wrap=True`, not a table cell, so reasons are never truncated). Exit with `summary.exit_code`.

- [ ] **Step 4: Run the tests, then smoke-test by hand:** `uv run crisp tests --scale 2 --out <scratch dir>` should print the summary. Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(cli): add crisp upscale command with summary and exit codes`.

---

### Task 11: Benchmark — synthetic scenes

**Files:**
- Create: `src/crisp/bench/__init__.py`, `src/crisp/bench/scene.py`, `tests/bench/__init__.py`, `tests/bench/test_scene.py`

**Interfaces:**
- Produces (all frozen dataclasses, coordinates in **input pixels**): `Line(x0, y0, x1, y1, width)`, `Arc(cx, cy, r, start_deg, end_deg, width)`, `Text(x, y, height, string)`, `Scene(width: int, height: int, lines: tuple[Line, ...], arcs: tuple[Arc, ...], texts: tuple[Text, ...])`, `TextLabel(string: str, height_in: float, box: tuple[int, int, int, int])` (box in output px); `TEXT_HEIGHTS = (4, 6, 8, 12, 16, 24, 32, 40)`; `random_scene(seed: int, width: int, height: int) -> Scene` (one `Text` for each height `h` in `TEXT_HEIGHTS` with `h <= height / 4`); `render_scene(scene: Scene, scale: int, supersample: int = 4) -> tuple[np.ndarray, list[TextLabel]]` (H×W×1 float32 at `scale` × input size).

- [ ] **Step 1: Write the failing tests**

```python
def test_deterministic():
    a, _ = render_scene(random_scene(7, 256, 192), 2)
    b, _ = render_scene(random_scene(7, 256, 192), 2)
    assert np.array_equal(a, b)


def test_contents():
    s = random_scene(1, 256, 192)
    assert sorted(t.height for t in s.texts) == list(TEXT_HEIGHTS)
    assert all(t.string and set(t.string) <= set("0123456789.xRM") for t in s.texts)
    assert len(s.lines) >= 6 and len(s.arcs) >= 2


def test_render_shape_ink_and_boxes():
    img, labels = render_scene(random_scene(3, 256, 192), 4)
    assert img.shape == (768, 1024, 1) and img.dtype == np.float32
    assert 0.01 < (img < 0.5).mean() < 0.4
    assert len(labels) == len(TEXT_HEIGHTS)
    for i, a in enumerate(labels):
        x0, y0, x1, y1 = a.box
        assert 0 <= x0 < x1 <= 1024 and 0 <= y0 < y1 <= 768
        for b in labels[i + 1 :]:
            assert (
                a.box[2] <= b.box[0]
                or b.box[2] <= a.box[0]
                or a.box[3] <= b.box[1]
                or b.box[3] <= a.box[1]
            )
```

- [ ] **Step 2: Run `uv run pytest tests/bench/test_scene.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement.** Use `numpy.random.default_rng(seed)`. Scene contents: a border rectangle; at least 4 more lines (horizontal, vertical and angled); 2–4 circles or arcs; one hatched rectangle (45° lines clipped to the rectangle analytically); 2 dimension lines (shaft plus 2 short arrowhead strokes per end); one `Text` per `TEXT_HEIGHTS` entry, with strings drawn from the formats `n`, `n.d`, `Rn`, `Mnxp`, `axb` (digits only otherwise). Line widths are uniform in 0.5–2.0. Place text by rejection sampling against already-placed boxes (up to 200 tries per label, drawing a new string each try; heights ≥ 32 use at most 4 characters), measuring with `ImageFont.load_default(size=…).getbbox`. Draw lines and arcs first, then for each text a white filled box behind it, so no line crosses a label and the legibility judge is never confused by the ground truth itself. Render: `L` canvas at `scale × supersample`, black ink on white, `round(width × k)` (min 1) stroke widths, then `Image.Resampling.BOX` down by `supersample`. Label boxes come from `draw.textbbox` divided by `supersample`.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(bench): generate synthetic drawings with known ground truth`.

---

### Task 12: Benchmark — degradation

**Files:**
- Create: `src/crisp/bench/degrade.py`, `tests/bench/test_degrade.py`

**Interfaces:**
- Produces: `Degradation(name: str, blur_sigma: float = 0.0, jpeg_quality: int | None = None, noise_sigma: float = 0.0, rotate_deg: float = 0.0, lighting: float = 0.0)` (frozen); `PRESETS: dict[str, Degradation]` = `clean` (none), `blur` (σ 1.0), `jpeg` (σ 0.5, quality 60), `scan` (σ 0.8, noise 0.02, rotate 0.5°, lighting 0.15); `degrade(gt: np.ndarray, scale: int, d: Degradation, seed: int) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests**

```python
GT = render_scene(random_scene(0, 128, 96), 4)[0]


def test_shape_range_dtype():
    for d in PRESETS.values():
        low = degrade(GT, 4, d, seed=1)
        assert (
            low.shape == (96, 128, 1)
            and low.dtype == np.float32
            and 0 <= low.min()
            and low.max() <= 1
        )


def test_clean_is_area_downscale_of_constant():
    assert np.allclose(
        degrade(np.full((40, 40, 1), 0.3, np.float32), 4, PRESETS["clean"], 0), 0.3, atol=1e-6
    )


def test_seeded():
    s = PRESETS["scan"]
    assert np.array_equal(degrade(GT, 4, s, 5), degrade(GT, 4, s, 5))
    assert not np.array_equal(degrade(GT, 4, s, 5), degrade(GT, 4, s, 6))


def test_blur_removes_high_frequencies():
    hf = lambda x: np.abs(np.diff(x, axis=1)).mean()
    assert hf(degrade(GT, 4, PRESETS["blur"], 0)) < hf(degrade(GT, 4, PRESETS["clean"], 0))
```

- [ ] **Step 2: Run `uv run pytest tests/bench/test_degrade.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement in spec §7.2 order.** Gaussian blur with σ × scale on the ground truth (`cv2.GaussianBlur`, ksize 0) → `cv2.resize(INTER_AREA)` to `(W // scale, H // scale)` → JPEG encode/decode via Pillow at the given quality → Gaussian noise from `default_rng(seed)` → rotation (`cv2.warpAffine` about the centre, `BORDER_REPLICATE`) → lighting multiply by `1 - lighting × (r / r_max)²` → clip, cast to `float32`, keep the channel axis.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(bench): add seeded degradation presets`.

---

### Task 13: Benchmark — metrics

**Files:**
- Create: `src/crisp/bench/metrics.py`, `tests/bench/test_metrics.py`

**Interfaces:**
- Consumes: `TextLabel` (Task 11)
- Produces: `psnr(out, gt) -> float`; `ssim(out, gt) -> float`; `edge_error(out, gt) -> float` (symmetric chamfer distance in output px, `nan` if either edge set is empty); `stroke_width_error(out, gt) -> float` (relative, `nan` if no ink); `tesseract_available() -> bool`; `legibility(out: np.ndarray, labels: list[TextLabel]) -> dict[float, float] | None` (height_in → fraction exactly read; `None` when Tesseract is unavailable). All take H×W×C float32 arrays of equal shape.

- [ ] **Step 1: Write the failing tests**

```python
def bar(x, width, size=(64, 64)):
    img = np.ones((*size, 1), np.float32)
    img[:, x : x + width] = 0
    return img


def test_psnr_ssim():
    a = np.full((16, 16, 1), 0.5, np.float32)
    assert psnr(a, a + 0.1) == pytest.approx(20.0, abs=1e-3)
    assert ssim(bar(20, 4), bar(20, 4)) == pytest.approx(1.0)


def test_edge_error():
    assert edge_error(bar(20, 4), bar(20, 4)) == 0
    assert edge_error(bar(23, 4), bar(20, 4)) == pytest.approx(3, abs=0.5)


def test_stroke_width_error():
    assert stroke_width_error(bar(20, 6), bar(20, 6)) == pytest.approx(0, abs=1e-6)
    assert stroke_width_error(bar(20, 30), bar(20, 20)) == pytest.approx(0.5, abs=0.1)


def test_legibility_none_without_tesseract(monkeypatch):
    monkeypatch.setattr(metrics, "tesseract_available", lambda: False)
    assert metrics.legibility(bar(0, 1), []) is None


@pytest.mark.skipif(not tesseract_available(), reason="Tesseract not installed")
def test_legibility_reads_large_text():
    scene = Scene(200, 80, (), (), (Text(20, 20, 40, "1234"),))
    img, labels = render_scene(scene, 2)
    assert legibility(img, labels) == {40.0: 1.0}
    assert legibility(np.ones_like(img), labels) == {40.0: 0.0}
```

- [ ] **Step 2: Run `uv run pytest tests/bench/test_metrics.py -v`.** Expected: FAIL (the Tesseract test may skip).

- [ ] **Step 3: Implement.** PSNR and SSIM come from `skimage.metrics` with `data_range=1` (SSIM on the squeezed 2-D image when C = 1, `channel_axis=-1` otherwise). Edge error: `cv2.Canny` on the 8-bit grey image (thresholds 50/150), then `cv2.distanceTransform` of each image's inverted edge map, averaged both ways. Stroke width: Otsu-binarise each image, `skimage.morphology.skeletonize`, take mean width = mean(2 × distance transform at skeleton pixels), error = `|w_out − w_gt| / w_gt`. Legibility: import `pytesseract` lazily. Crop each box padded by 25% of its height. Read with `--psm 7 -c tessedit_char_whitelist=0123456789.xRM`. Strip whitespace and compare exactly. Group results by `height_in`.

- [ ] **Step 4: Run the tests.** Optionally `scoop install tesseract` first so the legibility test runs. Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(bench): add PSNR, SSIM, edge, stroke-width and legibility metrics`.

---

### Task 14: Benchmark — comparison methods

**Files:**
- Create: `src/crisp/bench/methods.py`, `tests/bench/test_methods.py`

**Interfaces:**
- Consumes: `upscale_page` (Task 8), `lanczos_resize` (Task 5)
- Produces: `Method = Callable[[np.ndarray, int], np.ndarray]`; `class MethodUnavailable(Exception)`; `nearest`, `bicubic`, `lanczos` (Lanczos-3 in sRGB, i.e. "what most software does"), `crisp_method` (the real pipeline routing via `upscale_page` on an in-memory `Page`); `passes_needed(scale: int) -> int` (number of 4× Real-ESRGAN passes); `realesrgan(model: str, root: Path) -> Method | None` (`None` when `root/bench/tools/realesrgan-ncnn-vulkan[.exe]` is missing); `available_methods(root: Path, ai: bool) -> dict[str, Method]` (order: nearest, bicubic, lanczos, crisp, then `realesrgan-x4plus` and `realesrgan-x4plus-anime` when `ai` and present).

- [ ] **Step 1: Write the failing tests**

```python
IMGS = [np.random.default_rng(0).random((12, 10, c), dtype=np.float32) for c in (1, 3)]


@pytest.mark.parametrize("method", [nearest, bicubic, lanczos, crisp_method])
@pytest.mark.parametrize("scale", [2, 4, 8])
def test_shapes(method, scale):
    for img in IMGS:
        out = method(img, scale)
        assert (
            out.shape == (12 * scale, 10 * scale, img.shape[2])
            and 0 <= out.min()
            and out.max() <= 1
        )


def test_crisp_method_uses_pipeline():
    page = Page(Path("<bench>"), 0, 1, Raster(IMGS[1], None), None, False, None, "png")
    expected, engine = upscale_page(page, resolve_target(TargetSpec("scale", scale=2), page))
    assert engine == "baseline" and np.allclose(crisp_method(IMGS[1], 2), expected.pixels)


def test_passes_needed():
    assert [passes_needed(s) for s in (2, 4, 8, 16)] == [1, 1, 2, 2]


def test_realesrgan_absent(tmp_path):
    assert realesrgan("realesrgan-x4plus", tmp_path) is None
    assert list(available_methods(tmp_path, ai=True)) == ["nearest", "bicubic", "lanczos", "crisp"]
```

- [ ] **Step 2: Run `uv run pytest tests/bench/test_methods.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement.** `nearest`/`bicubic` use `cv2.resize` with `INTER_NEAREST`/`INTER_CUBIC`, keeping the channel axis and clipping. The Real-ESRGAN wrapper converts grey to RGB, then runs `<exe> -i in.png -o out.png -n <model> -s 4` `passes_needed(scale)` times in a temp dir. It area-downscales to the exact output size, converts back to grey when the input was grey, and raises `MethodUnavailable(stderr)` on a non-zero exit (e.g. no Vulkan GPU). Downloading the binary is **not** part of this task. It is an optional manual step described in Task 18.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(bench): add comparator methods including optional Real-ESRGAN`.

---

### Task 15: Benchmark — runner, aggregation and baseline comparison

**Files:**
- Create: `src/crisp/bench/run.py`, `src/crisp/bench/baseline.py`, `tests/bench/test_run.py`, `tests/helpers.py` (holds `TINY`, shared by Tasks 15–17)

**Interfaces:**
- Consumes: Tasks 11–14
- Produces:
  - `Config(name: str, scenes: int, size: tuple[int, int], scales: tuple[int, ...], presets: tuple[str, ...], ocr: bool, ai: bool)` (frozen); `QUICK = Config("quick", 4, (256, 192), (2, 4), ("clean", "blur"), ocr=False, ai=False)`; `FULL = Config("full", 24, (512, 384), (2, 4, 8), ("clean", "blur", "jpeg", "scan"), ocr=True, ai=True)`.
  - `Case(scene_seed: int, scale: int, preset: str)`; `build_cases(cfg) -> list[Case]` (seeds `0..scenes-1`).
  - `CaseResult(case, method, psnr, ssim, edge_error, stroke_width_error, legibility: dict[float, float] | None, seconds)`.
  - `Sample(case, crops: dict[str, np.ndarray])`: crops around the smallest text label, keys `"ground truth"`, `"input"` (nearest-upscaled), then method names. One sample for scene 0 per (scale, preset).
  - `BenchRun(config, results: list[CaseResult], samples: list[Sample], unavailable: dict[str, str])`.
  - `run_benchmark(cfg: Config, root: Path, on_case: Callable[[int, int], None] | None = None) -> BenchRun`.
  - `aggregate(run: BenchRun) -> dict[str, float]`: keys `"{scale}/{preset}/{method}/{metric}"` (metrics `psnr`, `ssim`, `edge_error`, `stroke_width_error`, `seconds`, `legibility`, and `legibility@{h}` per text height), values are means ignoring `nan`.
  - `baseline.py`: `TOLERANCES = {"psnr": 0.1, "ssim": 0.002, "legibility": 0.01, "stroke_width_error": 0.01}` (absolute), with `edge_error` at 5% relative; `Regression(key, baseline, current)`; `load_baseline(path) -> dict[str, dict[str, float]]` (`{}` if missing); `save_baseline(path, config_name, agg) -> None` (keeps other config sections); `compare(agg, baseline_section, method="crisp") -> list[Regression]` (ignores `seconds` and keys missing on either side).

- [ ] **Step 1: Write the failing tests**

```python
# tests/helpers.py
TINY = Config("tiny", 1, (128, 96), (2,), ("clean",), ocr=False, ai=False)

# tests/bench/test_run.py  (from tests.helpers import TINY)


def test_build_cases():
    cases = build_cases(QUICK)
    assert len(cases) == 16 and cases == build_cases(QUICK)


def test_run_and_aggregate(tmp_path):
    run = run_benchmark(TINY, tmp_path)
    assert {r.method for r in run.results} == {"nearest", "bicubic", "lanczos", "crisp"}
    agg = aggregate(run)
    assert agg["2/clean/lanczos/psnr"] > agg["2/clean/nearest/psnr"]
    assert math.isfinite(agg["2/clean/crisp/psnr"]) and "2/clean/crisp/legibility" not in agg
    assert set(run.samples[0].crops) >= {"ground truth", "input", "crisp"}


def test_compare_directions():
    base = {"2/clean/crisp/psnr": 30.0, "2/clean/crisp/edge_error": 1.0}
    assert [
        r.key for r in compare({"2/clean/crisp/psnr": 29.8, "2/clean/crisp/edge_error": 1.0}, base)
    ] == ["2/clean/crisp/psnr"]
    assert compare({"2/clean/crisp/psnr": 29.95, "2/clean/crisp/edge_error": 1.04}, base) == []
    assert [
        r.key for r in compare({"2/clean/crisp/psnr": 30.0, "2/clean/crisp/edge_error": 1.06}, base)
    ] == ["2/clean/crisp/edge_error"]


def test_baseline_roundtrip_keeps_other_sections(tmp_path):
    p = tmp_path / "b.json"
    save_baseline(p, "full", {"k": 1.0})
    save_baseline(p, "quick", {"k": 2.0})
    assert load_baseline(p) == {"full": {"k": 1.0}, "quick": {"k": 2.0}}
```

- [ ] **Step 2: Run `uv run pytest tests/bench/test_run.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement.** Degrade seed = `scene_seed * 1000 + scale * 10 + presets.index(preset)`. Legibility runs only when `cfg.ocr and tesseract_available()`. On the first `MethodUnavailable`, a method is dropped for the rest of the run and recorded in `unavailable`. Crops are 256×256 output px (clamped) centred on the smallest-height label. `save_baseline` writes sorted, indented JSON.

- [ ] **Step 4: Run the tests.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(bench): run cases, aggregate scores and compare to baseline`.

---

### Task 16: Benchmark — HTML report

**Files:**
- Create: `src/crisp/bench/report.py`, `tests/bench/test_report.py`

**Interfaces:**
- Consumes: `BenchRun`, `aggregate`, `Regression` (Task 15)
- Produces: `write_report(run: BenchRun, agg: dict[str, float], regressions: list[Regression], path: Path) -> Path` (one self-contained HTML file, no external assets).

- [ ] **Step 1: Write the failing test**

```python
def test_report(tmp_path):
    run = run_benchmark(TINY, tmp_path)
    agg = aggregate(run)
    reg = [Regression("2/clean/crisp/psnr", 30.0, 29.0)]
    html = write_report(run, agg, reg, tmp_path / "r" / "report.html").read_text(encoding="utf-8")
    for needle in (
        "<table",
        "nearest",
        "bicubic",
        "lanczos",
        "crisp",
        "data:image/png;base64,",
        "N/A",
        "Regressions",
        "2/clean/crisp/psnr",
    ):
        assert needle in html
```

- [ ] **Step 2: Run `uv run pytest tests/bench/test_report.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement with the standard library only** (`html.escape`, f-strings, `base64`). Structure: header (config name, date, crisp version, unavailable methods with reasons) → regressions banner when the list is non-empty → one table per (scale, preset) with rows for methods and columns PSNR ↑, SSIM ↑, edge error ↓, stroke-width error ↓, legibility ↑, s/img, best value per column in bold, `N/A` where missing → legibility-by-text-height table when present → crop grid per sample (`image-rendering: pixelated`). Style it with the banner palette from `PROMPT-BANNER.md` (navy background `#0E1A2B`, cream text `#FFF6E5`, gold headings `#F6B93B`, mint for best values `#3DDC97`, red for regressions `#E8473F`).

- [ ] **Step 4: Run the test.** Expected: PASS.

- [ ] **Step 5: Commit** with message `feat(bench): write self-contained HTML report`.

---

### Task 17: CLI — `crisp bench`

**Files:**
- Modify: `src/crisp/cli.py`, `tests/test_cli.py`

**Interfaces:**
- Consumes: Tasks 15–16
- Produces: `bench_app: typer.Typer` with options `--quick`, `--no-open`, `--update-baseline`, `--root PATH` (default `.`); `main(argv)` dispatches to `bench_app` when `argv[0] == "bench"`. Report path: `<root>/bench/reports/<YYYYmmdd-HHMMSS>-<config>/report.html`. Baseline: `<root>/bench/results/baseline.json`.

- [ ] **Step 1: Write the failing tests.** Add fixture `tiny_quick` to `tests/conftest.py`: `monkeypatch.setattr(crisp.bench.run, "QUICK", TINY)`.

```python
def test_bench_writes_report_and_baseline(tmp_path, tiny_quick):
    assert (
        main_exit(["bench", "--quick", "--no-open", "--update-baseline", "--root", str(tmp_path)])
        == 0
    )
    assert list((tmp_path / "bench/reports").glob("*-tiny/report.html"))
    assert "tiny" in json.loads((tmp_path / "bench/results/baseline.json").read_text())


def test_bench_exit_1_on_regression(tmp_path, tiny_quick, monkeypatch):
    args = ["bench", "--quick", "--no-open", "--root", str(tmp_path)]
    assert main_exit(args + ["--update-baseline"]) == 0
    monkeypatch.setattr(cli, "compare", lambda *a, **k: [Regression("k", 1.0, 0.0)])
    assert main_exit(args) == 1


def test_main_dispatches_upscale(save_image):
    src = save_image("a.png", Image.new("L", (4, 4)))
    assert main_exit([str(src), "--jobs", "1"]) == 0 and (src.parent / "a@2x.png").exists()
```

(`main_exit(argv)` calls `cli.main(argv)` and returns the `SystemExit` code.) Import conventions the tests rely on: `cli.py` does `from crisp.bench import run` and reads `run.QUICK` / `run.FULL` at call time, and does `from crisp.bench.baseline import compare` so `cli.compare` can be patched.

- [ ] **Step 2: Run `uv run pytest tests/test_cli.py -v`.** Expected: FAIL.

- [ ] **Step 3: Implement.** Run with a rich progress bar driven by `on_case` → `aggregate` → `compare` against the config's baseline section → `write_report` → print a short console table (crisp vs lanczos PSNR and edge error per scale/preset, plus any regressions). Open the report with `webbrowser.open(path.as_uri())` unless `--no-open`. With `--update-baseline`, call `save_baseline` and exit 0. Otherwise exit 1 if there are regressions, else 0. With no baseline yet, print `"No baseline yet; run with --update-baseline to create one."` and exit 0.

- [ ] **Step 4: Run the full suite.** `uv run pytest -v` → PASS.

- [ ] **Step 5: Commit** with message `feat(cli): add crisp bench command`.

---

### Task 18: CI, committed baseline, README status

**Files:**
- Create: `.github/workflows/ci.yml`, `bench/results/baseline.json` (generated)
- Modify: `README.md`

**Interfaces:**
- Consumes: everything above.

- [ ] **Step 1: Write the workflow.** Trigger on `push` to `main` and on `pull_request`. Matrix `os: [ubuntu-latest, windows-latest]`. Steps: `actions/checkout@v4` → `astral-sh/setup-uv@v6` with `python-version: "3.12"` → `uv sync` → `uv run ruff check .` → `uv run ruff format --check .` → `uv run pytest -v` → `uv run crisp bench --quick --no-open`.

- [ ] **Step 2: Generate the baselines locally.** `uv run crisp bench --quick --no-open --update-baseline`, then `uv run crisp bench --update-baseline`. For the full run, optional tools are used if present. Tesseract: `scoop install tesseract`. Real-ESRGAN: **ask the user before downloading** `realesrgan-ncnn-vulkan` (xinntao/Real-ESRGAN GitHub release) into `bench/tools/`. Open the full report and check the numbers look sane: lanczos/bicubic beat nearest on PSNR, and legibility rises with text height.

- [ ] **Step 3: Update `README.md` (pro-readme rules, no overclaiming).** Add the CI badge. Change the status to `v0.1 — foundation: CLI and benchmark work; only the baseline (Lanczos) engine exists, so output is not yet sharper than standard resizing`. Replace "Using it (planned)" with a real Quick start (`uv tool install git+https://github.com/DeanT-04/crisp`, then the working commands), keeping `--vector` marked as planned. Tick roadmap M1. Update Features statuses: Vector PDF re-render, Batch CLI and Benchmark report → `Available`.

- [ ] **Step 4: Verify.** `uv run pytest -v`, `uv run ruff check .`, `uv run crisp bench --quick --no-open` → all pass (exit 0, no regressions against the committed baseline). Push the branch, open the PR `feat: M1 foundation — CLI, PDF re-render, baseline engine, benchmark`, and confirm CI is green on both operating systems.

- [ ] **Step 5: Commit** `.github bench/results/baseline.json README.md` with message `ci: add test and quick-benchmark workflow, commit baseline scores`.
