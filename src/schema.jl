# ============================================================================
# schema.jl — ACSet schema integration stubs (v0.2)
# ============================================================================
#
# As of FunctorFlow v0.2, the ACSet schema lives in the companion package
# CategoricalDiagramSchema.jl (UUID 06663149-d6bb-42b5-8a63-d2553351277c).
# Its schema (`SchCategoricalDiagram`) is shared with CatNet.jl and other
# categorical-diagram packages, so interop at the schema level is the
# identity.
#
# This file declares `to_acset` and `from_acset` as empty generic functions.
# Their methods are provided by the `FunctorFlowSchemaExt` package extension,
# which is loaded automatically when both FunctorFlow and
# CategoricalDiagramSchema are imported:
#
#     using FunctorFlow, CategoricalDiagramSchema
#     acs = to_acset(D)
#     D2  = from_acset(acs; name=:MyDiagram)
#
# If you only `using FunctorFlow`, calling `to_acset(D)` will raise a
# `MethodError` suggesting you also load `CategoricalDiagramSchema`.

"""
    to_acset(D::Diagram) -> CategoricalDiagramACSet

Convert a FunctorFlow Diagram to its shared-schema ACSet representation.

This function is provided by the `FunctorFlowSchemaExt` extension, which
loads automatically when both `FunctorFlow` and `CategoricalDiagramSchema`
are loaded:

    using FunctorFlow, CategoricalDiagramSchema
    acs = to_acset(D)

See also: [`from_acset`](@ref), `CategoricalDiagramSchema`.
"""
function to_acset end

"""
    from_acset(acs; name=:Imported) -> Diagram

Reconstruct a FunctorFlow Diagram from a `CategoricalDiagramACSet`.

Provided by `FunctorFlowSchemaExt`. Load `CategoricalDiagramSchema` to
enable.

See also: [`to_acset`](@ref).
"""
function from_acset end
