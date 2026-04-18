module FunctorFlowCatlabExt

# ============================================================================
# FunctorFlowCatlabExt — Catlab integration for FunctorFlow.jl
# ============================================================================
#
# Loaded automatically when both `FunctorFlow` and `Catlab` are imported in
# the same Julia session. Provides Catlab-backed methods for FunctorFlow's
# `to_presentation`, `to_symbolic`, and `define_theory` stub functions.
#
# As of FunctorFlow v0.5.0, Catlab is a weak dependency. Without it, the
# above functions raise a `MethodError` directing the user to load Catlab.

using FunctorFlow
using FunctorFlow: Diagram, FFObject, Morphism, Composition,
                   CategoricalModelObject, ModelMorphism, NaturalTransformation
using Catlab
using Catlab.Theories: FreeCategory, Ob, Hom, dom, codom

include("symbolic_catlab.jl")
include("catlab_interop.jl")

end # module
