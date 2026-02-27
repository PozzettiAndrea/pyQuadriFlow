"""QuadriFlow quad-dominant remeshing wrapper."""

import numpy as np
from numpy.typing import NDArray

from pyquadriflow._pyquadriflow import quadriflow_remesh as _quadriflow_remesh


def quadriflow_remesh(
    vertices: NDArray[np.float64],
    faces: NDArray[np.int32],
    target_faces: int,
    *,
    seed: int = 0,
    preserve_sharp: bool = False,
    preserve_boundary: bool = False,
    adaptive_scale: bool = False,
    aggressive_sat: bool = False,
    minimum_cost_flow: bool = False,
) -> tuple[NDArray[np.float64], NDArray[np.int32]]:
    """Quad-dominant remeshing using QuadriFlow.

    Takes a triangle mesh and produces a quad-dominant mesh with
    approximately the requested number of faces.

    Parameters
    ----------
    vertices : ndarray, shape (N, 3)
        Input triangle mesh vertex positions.
    faces : ndarray, shape (M, 3)
        Input triangle mesh face indices (0-based).
    target_faces : int
        Target number of quad faces in the output.
    seed : int, default 0
        Random seed for reproducibility.
    preserve_sharp : bool, default False
        Preserve sharp features during remeshing.
    preserve_boundary : bool, default False
        Preserve mesh boundary edges.
    adaptive_scale : bool, default False
        Use adaptive scale for quad sizing.
    aggressive_sat : bool, default False
        Use aggressive SAT solver.
    minimum_cost_flow : bool, default False
        Use minimum cost flow solver.

    Returns
    -------
    vertices : ndarray, shape (K, 3), dtype float64
        Output quad mesh vertex positions.
    faces : ndarray, shape (L, 4), dtype int32
        Output quad mesh face indices (0-based).

    Examples
    --------
    >>> import numpy as np
    >>> import pyquadriflow
    >>> # Load a triangle mesh (e.g. via trimesh)
    >>> v_quad, f_quad = pyquadriflow.quadriflow_remesh(vertices, faces, target_faces=500)
    >>> print(f"Quads: {f_quad.shape[0]}, vertices: {v_quad.shape[0]}")
    """
    v = np.ascontiguousarray(vertices, dtype=np.float64)
    f = np.ascontiguousarray(faces, dtype=np.int32)

    if v.ndim != 2 or v.shape[1] != 3:
        raise ValueError(f"vertices must have shape (N, 3), got {v.shape}")
    if f.ndim != 2 or f.shape[1] != 3:
        raise ValueError(f"faces must have shape (M, 3), got {f.shape}")
    if len(v) == 0 or len(f) == 0:
        raise ValueError("Input mesh is empty")
    if target_faces <= 0:
        raise ValueError(f"target_faces must be positive, got {target_faces}")

    return _quadriflow_remesh(
        v, f, target_faces,
        seed=seed,
        preserve_sharp=preserve_sharp,
        preserve_boundary=preserve_boundary,
        adaptive_scale=adaptive_scale,
        aggressive_sat=aggressive_sat,
        minimum_cost_flow=minimum_cost_flow,
    )
