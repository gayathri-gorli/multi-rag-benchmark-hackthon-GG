import math

import pytest

from core.graph_viz import project


def test_no_rotation_keeps_front_hemisphere_in_front():
    p = project({"a": (0.0, 0.0, 1.0), "b": (0.0, 0.0, -1.0)}, 0, 0)
    assert p["a"] == pytest.approx((0, 0, 1)) and p["b"][2] < 0


def test_rotation_preserves_points_on_unit_sphere():
    pts = {"a": (0.6, 0.0, 0.8), "b": (0.0, 1.0, 0.0), "c": (-0.48, -0.6, 0.64)}
    for az, el in ((0, 0), (90, 0), (30, 20), (200, -70)):
        for x, y, z in project(pts, az, el).values():
            assert x * x + y * y + z * z == pytest.approx(1.0)


def test_azimuth_180_flips_front_and_back():
    p = project({"a": (0.0, 0.0, 1.0)}, 180, 0)
    assert p["a"][2] == pytest.approx(-1.0)
