import pytest

from crisp.resources import check_resources
from crisp.types import ResourceError, Target

# working ≈ w*h*C*4 bytes*4 copies = 48 MB; PNG ≈ w*h*C/4 = 0.75 MB
T1000 = Target(4.0, (1000, 1000), None, "4x")


def test_ok(tmp_path):
    check_resources(T1000, 3, tmp_path, available_ram=10**9, free_disk=10**12)


def test_memory(tmp_path):
    with pytest.raises(ResourceError, match="memory"):
        check_resources(T1000, 3, tmp_path, available_ram=10**8, free_disk=10**12)


def test_disk(tmp_path):
    with pytest.raises(ResourceError, match="disk"):
        check_resources(T1000, 3, tmp_path, available_ram=10**12, free_disk=100)
