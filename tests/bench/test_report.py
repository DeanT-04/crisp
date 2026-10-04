from crisp.bench.baseline import Regression
from crisp.bench.report import write_report
from crisp.bench.run import aggregate, run_benchmark
from tests.helpers import TINY


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
