// Pure C++ wrapper for the QuadriFlow pipeline.
// This header MUST NOT include any Python/nanobind headers to avoid
// potential macro conflicts between QuadriFlow/Eigen and Python internals.

#ifndef PYQUADRIFLOW_PIPELINE_H
#define PYQUADRIFLOW_PIPELINE_H

#include <vector>

struct QuadriFlowResult {
    std::vector<double> vertices;   // flat: [x0,y0,z0, x1,y1,z1, ...]
    std::vector<int> faces;         // flat: [v0,v1,v2,v3, ...] per quad face
    int num_vertices;
    int num_faces;
};

// Run the QuadriFlow quad-dominant remeshing pipeline.
// Input: triangle mesh as flat arrays (vertices Nx3, faces Mx3).
// Output: quad mesh.
QuadriFlowResult run_quadriflow(
    const double* vertices, int num_vertices,
    const int* faces, int num_faces,
    int target_faces,
    int seed,
    bool preserve_sharp,
    bool preserve_boundary,
    bool adaptive_scale,
    bool aggressive_sat,
    bool minimum_cost_flow
);

#endif // PYQUADRIFLOW_PIPELINE_H
