import Lake
open Lake DSL

package "FunctorFlowProofs"

@[default_target]
lean_lib «FunctorFlowProofs» where
  roots := #[`FunctorFlowProofs]
  globs := #[.andSubmodules `FunctorFlowProofs]
