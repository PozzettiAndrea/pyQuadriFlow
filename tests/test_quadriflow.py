"""Tests for pyquadriflow: QuadriFlow quad-dominant remeshing."""

import numpy as np
import pytest


# ── Basic Remeshing ──────────────────────────────────────────────────


def test_quadriflow_basic(icosphere):
    """Test basic quad remeshing round-trip."""
    import pyquadriflow

    verts, faces = icosphere
    v_out, f_out = pyquadriflow.quadriflow_remesh(verts, faces, target_faces=100)

    assert isinstance(v_out, np.ndarray)
    assert isinstance(f_out, np.ndarray)
    assert v_out.ndim == 2
    assert f_out.ndim == 2
    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 4
    assert len(v_out) > 0
    assert len(f_out) > 0


def test_quadriflow_output_is_quads(icosphere):
    """Verify output faces are quads (4 vertices each)."""
    import pyquadriflow

    verts, faces = icosphere
    v_out, f_out = pyquadriflow.quadriflow_remesh(verts, faces, target_faces=100)

    assert f_out.shape[1] == 4
    # All face indices should be non-negative (valid quads)
    assert np.all(f_out >= 0)


def test_quadriflow_preserves_scale(icosphere):
    """Test that remeshing roughly preserves the bounding box."""
    import pyquadriflow

    verts, faces = icosphere
    v_out, f_out = pyquadriflow.quadriflow_remesh(verts, faces, target_faces=100)

    original_extent = np.max(np.abs(verts))
    remeshed_extent = np.max(np.abs(v_out))
    assert abs(remeshed_extent - original_extent) < 0.3


def test_quadriflow_seed_reproducibility(icosphere):
    """Test that same seed produces same output."""
    import pyquadriflow

    verts, faces = icosphere

    v1, f1 = pyquadriflow.quadriflow_remesh(verts, faces, target_faces=100, seed=42)
    v2, f2 = pyquadriflow.quadriflow_remesh(verts, faces, target_faces=100, seed=42)

    np.testing.assert_array_equal(v1, v2)
    np.testing.assert_array_equal(f1, f2)


def test_quadriflow_cube(cube):
    """Test remeshing a subdivided cube."""
    import pyquadriflow

    verts, faces = cube
    v_out, f_out = pyquadriflow.quadriflow_remesh(verts, faces, target_faces=50)

    assert v_out.shape[1] == 3
    assert f_out.shape[1] == 4
    assert len(v_out) > 0


# ── Input Validation ─────────────────────────────────────────────────


def test_quadriflow_input_validation_vertex_shape():
    """Test that invalid vertex shape raises error."""
    import pyquadriflow

    bad_verts = np.zeros((10, 2), dtype=np.float64)
    good_faces = np.zeros((5, 3), dtype=np.int32)

    with pytest.raises((ValueError, RuntimeError)):
        pyquadriflow.quadriflow_remesh(bad_verts, good_faces, target_faces=10)


def test_quadriflow_input_validation_face_shape():
    """Test that non-triangle face input raises error."""
    import pyquadriflow

    good_verts = np.zeros((10, 3), dtype=np.float64)
    bad_faces = np.zeros((5, 4), dtype=np.int32)

    with pytest.raises((ValueError, RuntimeError)):
        pyquadriflow.quadriflow_remesh(good_verts, bad_faces, target_faces=10)


def test_quadriflow_input_validation_empty():
    """Test that empty mesh raises error."""
    import pyquadriflow

    with pytest.raises((ValueError, RuntimeError)):
        pyquadriflow.quadriflow_remesh(
            np.zeros((0, 3), dtype=np.float64),
            np.zeros((0, 3), dtype=np.int32),
            target_faces=10,
        )


def test_quadriflow_input_validation_target_faces():
    """Test that non-positive target_faces raises error."""
    import pyquadriflow

    verts = np.array([
        [-1, -1, -1], [1, -1, -1], [1, 1, -1], [-1, 1, -1],
        [-1, -1,  1], [1, -1,  1], [1, 1,  1], [-1, 1,  1],
    ], dtype=np.float64)
    faces = np.array([
        [0, 1, 2], [0, 2, 3], [4, 6, 5], [4, 7, 6],
        [0, 5, 1], [0, 4, 5], [2, 7, 3], [2, 6, 7],
        [0, 3, 7], [0, 7, 4], [1, 5, 6], [1, 6, 2],
    ], dtype=np.int32)

    with pytest.raises((ValueError, RuntimeError)):
        pyquadriflow.quadriflow_remesh(verts, faces, target_faces=0)

    with pytest.raises((ValueError, RuntimeError)):
        pyquadriflow.quadriflow_remesh(verts, faces, target_faces=-5)
