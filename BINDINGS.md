# Binding Coverage

## Mapped

| Function | Description |
|----------|-------------|
| `quadriflow_remesh` | Full pipeline: orientation → singularities → scale → positions → quad extraction |

### Parameters Exposed

| Parameter | Description |
|-----------|-------------|
| `target_faces` | Target number of output quad faces |
| `seed` | Random seed for reproducibility |
| `preserve_sharp` | Preserve sharp features |
| `preserve_boundary` | Preserve mesh boundary edges |
| `adaptive_scale` | Adaptive quad sizing |
| `aggressive_sat` | Aggressive SAT solver |
| `minimum_cost_flow` | Minimum cost flow solver |

## Not Mapped

| Capability | Notes |
|------------|-------|
| Individual optimizer stages | `optimize_orientations`, `optimize_scale`, `optimize_positions` separately |
| Mesh analysis | Valence analysis, sharp edge detection, vertex area computation |
| Mesh repair | Fix holes, flipped faces, valence issues |
| File I/O | Direct OBJ load/save |
| Hierarchy control | Multi-resolution hierarchy management |
| State serialization | Save/load parametrization state |
| CUDA acceleration | GPU-optimized orientation and position optimization |
| Field inspection | Access orientation / position fields directly |
