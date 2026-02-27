// Python bindings for QuadriFlow quad-dominant remeshing via nanobind.
//
// This file ONLY includes nanobind and the pipeline wrapper header.
// All QuadriFlow/Eigen headers are isolated in pipeline.cpp.

#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <cstring>

#include "array_support.h"
#include "pipeline.h"

namespace nb = nanobind;

static nb::tuple py_quadriflow_remesh(
    const NDArray<const double, 2> vertices,
    const NDArray<const int, 2> faces,
    int target_faces,
    int seed,
    bool preserve_sharp,
    bool preserve_boundary,
    bool adaptive_scale,
    bool aggressive_sat,
    bool minimum_cost_flow
) {
    if (vertices.shape(1) != 3) {
        throw std::runtime_error("vertices must have shape (N, 3)");
    }
    if (faces.shape(1) != 3) {
        throw std::runtime_error("faces must have shape (M, 3) — input must be a triangle mesh");
    }
    if (vertices.shape(0) == 0 || faces.shape(0) == 0) {
        throw std::runtime_error("Input mesh is empty");
    }

    QuadriFlowResult result = run_quadriflow(
        vertices.data(), static_cast<int>(vertices.shape(0)),
        faces.data(), static_cast<int>(faces.shape(0)),
        target_faces, seed,
        preserve_sharp, preserve_boundary,
        adaptive_scale, aggressive_sat, minimum_cost_flow
    );

    // Convert result to numpy arrays
    NDArray<double, 2> verts_arr = MakeNDArray<double, 2>(
        {result.num_vertices, 3});
    std::memcpy(verts_arr.data(), result.vertices.data(),
        result.num_vertices * 3 * sizeof(double));

    NDArray<int, 2> faces_arr = MakeNDArray<int, 2>(
        {result.num_faces, 4});
    std::memcpy(faces_arr.data(), result.faces.data(),
        result.num_faces * 4 * sizeof(int));

    return nb::make_tuple(verts_arr, faces_arr);
}


NB_MODULE(_pyquadriflow, m) {
    m.doc() = "Python bindings for QuadriFlow quad-dominant remeshing";

    m.def("quadriflow_remesh", &py_quadriflow_remesh,
        R"doc(
Quad-dominant remeshing using QuadriFlow.

Takes a triangle mesh and produces a quad-dominant mesh with approximately
the requested number of faces.

Parameters
----------
vertices : ndarray, shape (N, 3), dtype float64
    Input triangle mesh vertex positions.
faces : ndarray, shape (M, 3), dtype int32
    Input triangle mesh face indices (0-based).
target_faces : int
    Target number of quad faces in the output.
seed : int
    Random seed for reproducibility.
preserve_sharp : bool
    Preserve sharp features during remeshing.
preserve_boundary : bool
    Preserve mesh boundary edges.
adaptive_scale : bool
    Use adaptive scale for quad sizing.
aggressive_sat : bool
    Use aggressive SAT solver.
minimum_cost_flow : bool
    Use minimum cost flow solver.

Returns
-------
vertices : ndarray, shape (K, 3), dtype float64
    Output quad mesh vertex positions.
faces : ndarray, shape (L, 4), dtype int32
    Output quad mesh face indices (0-based).
)doc",
        nb::arg("vertices"),
        nb::arg("faces"),
        nb::arg("target_faces"),
        nb::arg("seed") = 0,
        nb::arg("preserve_sharp") = false,
        nb::arg("preserve_boundary") = false,
        nb::arg("adaptive_scale") = false,
        nb::arg("aggressive_sat") = false,
        nb::arg("minimum_cost_flow") = false
    );
}
