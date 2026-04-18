module FunctorFlowSchemaExt

using FunctorFlow
using FunctorFlow: Diagram, FFObject, Morphism, Composition, KanExtension,
                   ObstructionLoss, LEFT, RIGHT,
                   add_object!, add_morphism!, add_left_kan!, add_right_kan!,
                   add_obstruction_loss!, compose!
using CategoricalDiagramSchema
using CategoricalDiagramSchema: CategoricalDiagramACSet, make_diagram,
    DEFAULT_SHAPE, DEFAULT_DTYPE, DEFAULT_METADATA, DEFAULT_WEIGHT
using CategoricalDiagramSchema: add_part!, subpart, nparts, incident

import FunctorFlow: to_acset, from_acset

include("acset_adapter.jl")

end # module
