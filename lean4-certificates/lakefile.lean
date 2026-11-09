import Lake

open System Lake DSL

package VeriTree where
  version := v!"0.1.0"

@[default_target]
lean_lib VeriTree where
  roots := #[`VeriTree]

