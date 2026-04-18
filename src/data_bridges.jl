# ============================================================================
# data_bridges.jl — Categorical DB / atlas / TCC bridge semantics
#
# As of v0.3.4, the original 1606-line file has been split into four
# topical submodules under `src/data_bridges/`. Include order matters:
# materializers reference types and helpers defined earlier; the
# cross-cutting compilation-plan / IR / summarization functions in
# `examples.jl` reference both example builders and types.
# ============================================================================

include("data_bridges/types.jl")
include("data_bridges/examples.jl")
include("data_bridges/specs.jl")
include("data_bridges/materializers.jl")
