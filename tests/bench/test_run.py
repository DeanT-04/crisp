import math

from crisp.bench.baseline import Regression, compare, load_baseline, save_baseline
from crisp.bench.run import QUICK, aggregate, build_cases, run_benchmark
from tests.helpers import TINY


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
    worse_psnr = {"2/clean/crisp/psnr": 29.8, "2/clean/crisp/edge_error": 1.0}
    assert [r.key for r in compare(worse_psnr, base)] == ["2/clean/crisp/psnr"]
    within = {"2/clean/crisp/psnr": 29.95, "2/clean/crisp/edge_error": 1.04}
    assert compare(within, base) == []
    worse_edge = {"2/clean/crisp/psnr": 30.0, "2/clean/crisp/edge_error": 1.06}
    assert [r.key for r in compare(worse_edge, base)] == ["2/clean/crisp/edge_error"]


def test_compare_ignores_other_methods_seconds_and_missing_keys():
    base = {"2/clean/lanczos/psnr": 30.0, "2/clean/crisp/seconds": 1.0, "2/clean/crisp/ssim": 0.9}
    assert compare({"2/clean/lanczos/psnr": 10.0, "2/clean/crisp/seconds": 99.0}, base) == []
    assert Regression("k", 1.0, 0.5).key == "k"


def test_baseline_roundtrip_keeps_other_sections(tmp_path):
    p = tmp_path / "b.json"
    save_baseline(p, "full", {"k": 1.0})
    save_baseline(p, "quick", {"k": 2.0})
    assert load_baseline(p) == {"full": {"k": 1.0}, "quick": {"k": 2.0}}
    assert load_baseline(tmp_path / "missing.json") == {}
