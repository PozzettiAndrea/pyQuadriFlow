"""
pyquadriflow — Python bindings for QuadriFlow quad-dominant remeshing.

Functions
---------
quadriflow_remesh
    Quad-dominant remeshing from a triangle mesh.
"""

from pyquadriflow.quadriflow import quadriflow_remesh

__version__ = "0.2.0"
__all__ = ["quadriflow_remesh"]
