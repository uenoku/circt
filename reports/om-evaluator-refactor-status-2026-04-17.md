# OM Evaluator Refactor Status

Date: 2026-04-17

## Scope

This status note captures the current state of the OM evaluator refactor for the `ReferenceValue` handling cleanup and the crash reported in CIRCT issue `#10264`.

User repro:

```sh
./build/bin/domaintool bar.mlir -module Foo
```

User constraint:

- Do not run Docker. All validation below was done on the local host build.

## Current State

Most of the intended internal refactor is implemented. The original `domaintool` crash is fixed, the targeted regression tests are in place, and the focused reproducer passes.

Update after continuing the work on 2026-04-18:

- the remaining `EvaluatorTests.UnknownValuesBasic` regression is fixed
- `evaluateObjectField(...)` now tolerates unknown class-typed placeholders that are not yet materialized as `ObjectValue`
- nested `om.object.field` paths now also convert unknown non-object intermediates into a type-correct unknown result instead of asserting
- the full OM evaluator unit suite now passes locally

## Files Changed

- `include/circt/Dialect/OM/Evaluator/Evaluator.h`
- `lib/Dialect/OM/Evaluator/Evaluator.cpp`
- `unittests/Dialect/OM/Evaluator/EvaluatorTests.cpp`
- `test/Tools/domaintool/reference-bounce.mlir`

## Implemented Changes

### 1. Shared reference resolution helpers

Added internal helpers in `Evaluator.cpp`:

- `ResolutionState`
- `ResolvedValue`
- `ResolvedTypedValue<T>`
- `resolveReferenceValue(...)`
- `resolveValueAs<T>(...)`

These centralize reference traversal and distinguish:

- `Ready`: a final non-reference value was found
- `Pending`: resolution stopped at an unbound reference
- `Failure`: a cycle was detected

### 2. Safer `ReferenceValue::getStrippedValue()`

`ReferenceValue::getStrippedValue()` is now defined in `Evaluator.cpp` and no longer assumes the chain is always valid.

Current behavior:

- returns the final non-reference value when ready
- emits `reference value is not resolved` for pending chains
- emits `reference value contains a cycle` for cyclic chains

### 3. Finalization path updated

`ReferenceValue::finalizeImpl()` now uses the shared resolver path instead of manually peeling references.

Current behavior:

- if the reference chain resolves cleanly, it strips to the final value and finalizes that value
- if the chain is still pending or cyclic, finalization fails instead of dereferencing invalid state

### 4. Parameter handling improved

`createParametersFromOperands(...)` now tries to advance operand values with `evaluateValue(...)` when the initial handle is a `ReferenceValue`.

This allows object arguments to capture already-progressed values instead of preserving extra indirection unnecessarily, while still permitting deferred evaluation when the value is genuinely pending.

### 5. Active object instantiation tracking

Added:

- `activeObjectInstances` to `Evaluator`

This prevents recursive object instantiation from recursing indefinitely when the evaluator is still building a placeholder for the same object instance.

### 6. `evaluateObjectField(...)` partially refactored

This function now:

- evaluates the base operand first
- resolves through bound references using shared helpers
- returns the placeholder when the base is still pending
- propagates unknown field values by creating a type-correct unknown result and wiring the reference to it

This is also the area with the remaining regression described below.

### 7. Shared typed extraction in other evaluators

The ad hoc `ReferenceValue` peeling logic was replaced in:

- property assertions
- integer binary arithmetic
- list concatenation
- string concatenation
- binary equality

These sites now share consistent pending / failure handling.

### 8. New tests added

Unit test:

- `EvaluatorTests.ReferenceValueBounceThroughObject`
- `EvaluatorTests.UnknownValuesNestedObjectFieldPath`

Lit regression:

- `test/Tools/domaintool/reference-bounce.mlir`

These mirror the issue pattern where a field reference is bounced through another object before use.

## Validation Run So Far

### Successful

Built locally:

```sh
ninja -C build CIRCTOMEvaluatorTests domaintool
```

Focused evaluator unit tests:

```sh
./build/unittests/Dialect/OM/Evaluator/CIRCTOMEvaluatorTests \
  --gtest_filter=EvaluatorTests.InstantiateCycle:EvaluatorTests.ReferenceValueBounceThroughObject
```

These passed:

- `EvaluatorTests.InstantiateCycle`
- `EvaluatorTests.ReferenceValueBounceThroughObject`

Targeted lit suites:

```sh
ninja -C build check-circt-dialect-om check-circt-tools-domaintool
```

These passed, including the new `domaintool` regression coverage.

Direct repro:

```sh
./build/bin/domaintool test/Tools/domaintool/reference-bounce.mlir -module Foo
```

This now exits successfully and prints the expected empty JSON structure.

### Additional successful validation after the follow-up fix

Focused regression coverage:

```sh
./build/unittests/Dialect/OM/Evaluator/CIRCTOMEvaluatorTests \
  --gtest_filter=EvaluatorTests.UnknownValuesBasic:EvaluatorTests.UnknownValuesNestedObjectFieldPath
```

Both tests now pass.

Full evaluator unit binary:

```sh
./build/unittests/Dialect/OM/Evaluator/CIRCTOMEvaluatorTests
```

This now passes in full (`39` tests).

## Notes

- `clang-format` has already been run on the C++ files modified so far.
- No public interface changes were introduced in the evaluator API surface.
- The follow-up fix stayed localized to `evaluateObjectField(...)` and did not require backing out the broader refactor.
