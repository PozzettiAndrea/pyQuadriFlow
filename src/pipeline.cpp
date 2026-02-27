// Pure C++ implementation of the QuadriFlow pipeline.
// NO Python/nanobind headers here — isolates QuadriFlow/Eigen from Python.

#include "pipeline.h"

#include <cstring>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#define _USE_MATH_DEFINES
#include <math.h>
#endif

#include "config.hpp"
#include "field-math.hpp"
#include "optimizer.hpp"
#include "parametrizer.hpp"

using namespace qflow;

// ---------------------------------------------------------------------------
// Extended Parametrizer that can load from raw arrays (no file I/O)
// ---------------------------------------------------------------------------
class Parametrizer2 : public Parametrizer {
public:
    void LoadFromArrays(
        const double* verts, int n_verts,
        const int* face_indices, int n_faces
    ) {
        struct obj_vertex {
            uint32_t p = (uint32_t)-1;
            uint32_t n = (uint32_t)-1;
            uint32_t uv = (uint32_t)-1;

            obj_vertex() {}
            obj_vertex(uint32_t _p) : p(_p) {}

            bool operator==(const obj_vertex& v) const {
                return v.p == p && v.n == n && v.uv == uv;
            }
        };

        struct obj_vertex_hash {
            std::size_t operator()(const obj_vertex& v) const {
                size_t hash = std::hash<uint32_t>()(v.p);
                hash = hash * 37 + std::hash<uint32_t>()(v.uv);
                hash = hash * 37 + std::hash<uint32_t>()(v.n);
                return hash;
            }
        };

        using VertexMap = std::unordered_map<obj_vertex, uint32_t, obj_vertex_hash>;

        std::vector<Vector3d> positions;
        std::vector<uint32_t> indices;
        std::vector<obj_vertex> vertices;
        VertexMap vertexMap;

        positions.reserve(n_verts);
        for (int i = 0; i < n_verts; ++i) {
            positions.emplace_back(verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]);
        }

        for (int i = 0; i < n_faces; ++i) {
            obj_vertex tri[3];
            tri[0] = obj_vertex(face_indices[i * 3]);
            tri[1] = obj_vertex(face_indices[i * 3 + 1]);
            tri[2] = obj_vertex(face_indices[i * 3 + 2]);

            for (int j = 0; j < 3; ++j) {
                const obj_vertex& v = tri[j];
                auto it = vertexMap.find(v);
                if (it == vertexMap.end()) {
                    vertexMap[v] = (uint32_t)vertices.size();
                    indices.push_back((uint32_t)vertices.size());
                    vertices.push_back(v);
                } else {
                    indices.push_back(it->second);
                }
            }
        }

        F.resize(3, indices.size() / 3);
        std::memcpy(F.data(), indices.data(), sizeof(uint32_t) * indices.size());

        V.resize(3, vertices.size());
        for (uint32_t i = 0; i < vertices.size(); ++i) {
            V.col(i) = positions.at(vertices[i].p);
        }

        NormalizeMesh();
    }
};

// ---------------------------------------------------------------------------
// Main pipeline
// ---------------------------------------------------------------------------
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
) {
    if (num_vertices <= 0 || num_faces <= 0) {
        throw std::runtime_error("Input mesh is empty");
    }
    if (target_faces <= 0) {
        throw std::runtime_error("target_faces must be positive");
    }

    Parametrizer2 field;

    // Set flags
    if (preserve_sharp)     field.flag_preserve_sharp = 1;
    if (preserve_boundary)  field.flag_preserve_boundary = 1;
    if (adaptive_scale)     field.flag_adaptive_scale = 1;
    if (aggressive_sat)     field.flag_aggresive_sat = 1;
    if (minimum_cost_flow)  field.flag_minimum_cost_flow = 1;

    field.hierarchy.rng_seed = seed;

    // Load mesh from arrays
    field.LoadFromArrays(vertices, num_vertices, faces, num_faces);
    field.Initialize(target_faces);

    // Handle boundary preservation constraints
    if (field.flag_preserve_boundary) {
        Hierarchy& mRes = field.hierarchy;
        mRes.clearConstraints();
        for (uint32_t i = 0; i < 3 * mRes.mF.cols(); ++i) {
            if (mRes.mE2E[i] == -1) {
                uint32_t i0 = mRes.mF(i % 3, i / 3);
                uint32_t i1 = mRes.mF((i + 1) % 3, i / 3);
                Vector3d p0 = mRes.mV[0].col(i0);
                Vector3d p1 = mRes.mV[0].col(i1);
                Vector3d edge = p1 - p0;
                if (edge.squaredNorm() > 0) {
                    edge.normalize();
                    mRes.mCO[0].col(i0) = p0;
                    mRes.mCO[0].col(i1) = p1;
                    mRes.mCQ[0].col(i0) = edge;
                    mRes.mCQ[0].col(i1) = edge;
                    mRes.mCQw[0][i0] = 1.0;
                    mRes.mCQw[0][i1] = 1.0;
                    mRes.mCOw[0][i0] = 1.0;
                    mRes.mCOw[0][i1] = 1.0;
                }
            }
        }
        mRes.propagateConstraints();
    }

    // Optimization pipeline
    Optimizer::optimize_orientations(field.hierarchy);
    field.ComputeOrientationSingularities();

    if (field.flag_adaptive_scale == 1) {
        field.EstimateSlope();
    }

    Optimizer::optimize_scale(field.hierarchy, field.rho, field.flag_adaptive_scale);
    field.flag_adaptive_scale = 1;
    Optimizer::optimize_positions(field.hierarchy, field.flag_adaptive_scale);
    field.ComputePositionSingularities();

    field.ComputeIndexMap();

    // Extract output mesh
    QuadriFlowResult result;
    result.num_vertices = static_cast<int>(field.O_compact.size());
    result.num_faces = static_cast<int>(field.F_compact.size());

    if (result.num_vertices == 0 || result.num_faces == 0) {
        throw std::runtime_error("QuadriFlow produced an empty mesh");
    }

    result.vertices.resize(result.num_vertices * 3);
    for (int i = 0; i < result.num_vertices; ++i) {
        auto t = field.O_compact[i] * field.normalize_scale + field.normalize_offset;
        result.vertices[i * 3 + 0] = t.x();
        result.vertices[i * 3 + 1] = t.y();
        result.vertices[i * 3 + 2] = t.z();
    }

    result.faces.resize(result.num_faces * 4);
    for (int i = 0; i < result.num_faces; ++i) {
        result.faces[i * 4 + 0] = field.F_compact[i][0];
        result.faces[i * 4 + 1] = field.F_compact[i][1];
        result.faces[i * 4 + 2] = field.F_compact[i][2];
        result.faces[i * 4 + 3] = field.F_compact[i][3];
    }

    return result;
}
