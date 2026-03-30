# ============================================================================
# FunctorFlowMetalExt — Metal.jl GPU backend for FunctorFlow + Lux
#
# This extension provides GPU device utilities for running FunctorFlow
# neural models on Apple Silicon via Metal.jl.
# ============================================================================

module FunctorFlowMetalExt

using FunctorFlow
using Metal
using Lux
using LuxCore

# Re-export the Lux extension's KETAttentionLayer for Metal GPU use.
# The Lux extension defines the layer; this extension ensures Metal
# compatibility and provides any Metal-specific optimizations.

# Metal.jl integration is mostly automatic via MLDataDevices and NNlib,
# so this extension primarily serves as a dependency gate ensuring
# Metal, Lux, and FunctorFlow are all loaded together.

end # module FunctorFlowMetalExt
