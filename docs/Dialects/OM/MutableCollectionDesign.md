# Design proposal: append-only `MutableCollection` properties

**Status:** draft; this document describes the implementation's current choices.

## Summary

Introduce an identity-bearing, append-only property collection:

- FIRRTL: `MutableCollection<T>` / `!firrtl.mutable_collection<T>`.
- OM: `!om.collection<T>`.
- Allocation creates a fresh handle; a resultless push contributes one element.
- Construction programs can transport handles but cannot inspect contents.
- Contributions are unordered and preserve duplicates.
- Object elaboration folds pushes into allocation operands. The evaluator then
  exposes completed collections through the existing list APIs, with no mutable
  runtime value or effect phase.

This generalizes the domain accumulation use case of [Registry PR #10940][registry-pr]
without bottom-up registry output plumbing. The initial Registry type definition
is replaced; this is not a migration of the complete Registry prototype.

## Semantic contract

### Identity, transport, and observation

Each instantiated allocation creates a fresh collection. Equal types, contents,
or constructor arguments do not imply shared identity. Property assignment and
parameter/field transport pass the handle, not a copy of the contents. Existing
single-assignment rules still govern property bindings; push modifies the
referenced collection, not the binding.

Two children receiving the same handle contribute to one collection. Independent
allocations remain distinct. Two pushes of the same value contribute two elements.
Accessing one object through multiple aliases does not replay its constructor.
Unused writer results must not discard contributions to an observable collection.

Construction programs cannot index, enumerate, query length or membership, test
collection equality, or convert a collection into a list. Extracting a handle
from an object field is transport, not a content read. Post-evaluation consumers
can enumerate the completed result as an ordinary list.

OM classes are [graph regions][graph-regions], so operation order is not a
sequencing guarantee. A vector and deterministic traversal may provide stable
output for fixed IR, but clients must not rely on independent pushes' positions.
Tests and collateral consumers should compare multisets or sort explicitly.

### Unknowns and external constructors

Unknown elements and unknown membership are different:

- Pushing an unknown element into a known collection preserves one contribution.
  Other elements and multiplicity remain visible. Unlike `om.list_create`, an
  unknown element does not make the whole collection unknown.
- An unknown collection has unknown membership, not empty membership. A push to
  it cannot identify concrete storage and is erased during object elaboration.
- At the consumer boundary, unknown membership is an unknown list (`Unknown` in
  Python); unknown elements remain individually visible.

Unresolved external constructors are treated as today: elaboration replaces them
with unknown objects and discards their parameters. Their potential contributions
are not represented; collections passed to them retain their known contributions.
Linking external classes before evaluation resolves the common case. Representing
unknown external contributions would require a separate effect-summary design.

### API inputs: seed-value semantics

Each public `instantiate` call is isolated. A list passed where a collection is
expected supplies **initial contents**, not mutable caller storage.

Every collection-typed use of an input list becomes a fresh allocation. This
applies to parameters, object fields, and nested list/collection elements. These
allocations are not memoized: passing the same list twice creates two independent
collections, and aliases between collection-valued fields are not preserved on
re-import. Ordinary object and list import memoization is unchanged.

Within the newly instantiated IR graph, handle transport still preserves aliases.
For example, two children receiving one imported parameter share that parameter's
allocation. The distinction is between importing seed values and transporting
handles after import.

Any-typed inputs remain opaque. OM has no downcast from `!om.any`, so construction
programs cannot push through these opaque references. Existing restrictions on
mutually referential concrete object inputs remain; this proposal does not fix
shared-pointer cycle ownership or introduce persistent state across calls.

## IR and lowering

| Construct | FIRRTL | OM |
| --- | --- | --- |
| Type | `!firrtl.mutable_collection<T>` | `!om.collection<T>` |
| Allocation | `firrtl.prop.collection` | `om.collection` |
| Contribution | `firrtl.prop.push` | `om.collection_push` |

```mlir
%items = firrtl.prop.collection : !firrtl.mutable_collection<string>
firrtl.prop.push %items, %value : !firrtl.mutable_collection<string>, !firrtl.string

%items = om.collection : !om.collection<!om.string>
om.collection_push %items, %value : !om.collection<!om.string>, !om.string

// After object elaboration folds the push into its allocation.
%items = om.collection %value : !om.collection<!om.string>
```

FIRRTL allocations are initially empty. OM allocations accept variadic initial
operands whose types exactly match the collection element type. Push likewise
requires exact element-type agreement; existing `AnyRef`/`Any` casts provide
upcasts. ODS expresses these type constraints and the OM allocation's assembly
format.

Allocation has a result-associated allocation effect and is not pure or
constant-like. Push has a conservative write effect. CSE must not merge distinct
allocations or identical pushes. An unused allocation can be removed after its
contributions have folded into it; pushes must not be removed simply because
they have no results.

FIRRTL operations are restricted to supported module/class bodies, not hardware
`when` regions. Push reads the destination handle, so it requires source or
duplex flow: a local wire or input, not an output port. It is not a property-port
assignment.

[`LowerClasses.cpp`](../../../lib/Dialect/FIRRTL/Transforms/LowerClasses.cpp)
recursively converts the collection element type and lowers the two operations
one-to-one. Existing property discovery scans operands and results, so resultless
pushes participate in class creation without special handling. The parser/emitter
support `MutableCollection<T>()` and `push`; multiply-used allocations need a
binding on emission to preserve shared identity. Textual syntax uses the existing
`missingSpecFIRVersion` gate pending FIRRTL specification agreement.

[`FreezePaths.cpp`](../../../lib/Dialect/OM/Transforms/FreezePaths.cpp) freezes
nested path types in arguments, fields, and container/field-access/unknown result
types. Result types are updated in place, preserving attributes and handles.
Path-producing operations still become frozen path operations with their
per-instance base paths; collections are never flattened into lists in IR.

## Elaboration and evaluation

[`ElaborateObject.cpp`](../../../lib/Dialect/OM/Transforms/ElaborateObject.cpp)
clones constructor bodies, substitutes parameters, folds field accesses, and runs
greedy rewrites. A pass-local push pattern appends one element to a known
allocation and erases the push, without scanning the allocation's users or
rebuilding its operand list for each push. Each instantiated push becomes one operand exactly once.
Resolution never crosses a class boundary. Generic canonicalization preserves
contributions instead of resolving these effects.

After greedy rewriting finishes, each collection's operands are sorted once by
the final IR order of their SSA values: arguments in declaration order, followed
by operation results in block/result order. `isBeforeInBlock` can reuse its order
cache because no more operations are inserted or erased. Initial elements and
contributions participate in this ordering together; multiplicity is preserved.

Allocation and push are not pure, so unknown propagation does not turn a known
collection into unknown membership. The push pattern erases contributions to
unknown destinations. Effects whose destination depends on a class parameter
(including through field access) remain until elaboration exposes an allocation
or unknown destination. firtool and `om-linker` can therefore elaborate public
classes ahead of evaluation without requiring concrete inputs.

The evaluator's pipeline is:

```text
instantiate(class, actual parameters)
  verify parameter types
  create a scratch wrapper class
  import inputs as IR / opaque wrapper parameters
  instantiate the requested class in the wrapper
  elaborate the wrapper, folding every push
  allocate runtime placeholders for all results
  evaluate result-producing operations and collect fields
  return wrapper.root
```

[`Evaluator.cpp`](../../../lib/Dialect/OM/Evaluator/Evaluator.cpp) represents each
collection as a `ListValue`. Only the outer `!om.collection<T>` type becomes
`!om.list<T>` at the runtime boundary; element types retain their IR representation.
Parameter checking and runtime-value factories share that normalization rule.
The importer shares element materialization between lists and collections, but
keeps their operation construction and memoization policies distinct.

Collection evaluation installs references to element placeholders without forcing
deep evaluation, as elaborated objects do for fields. This permits an object to
be registered in a collection referenced by its own fields. `ListValue::setElements`
finishes the value once; it is not an append API. All pushes have already folded,
so runtime evaluation needs neither a mutation phase nor open/sealed states.

Repeated calls do not mutate inputs or previously returned results. Failure does
not publish partial results. Existing evaluator cache and scratch-IR lifetime
management remain unchanged; `ObjectValue` retains a class operation, so the usual
IR/context lifetime requirements still apply.

## Registry comparison

The [reviewed Registry prototype][registry-source] is domain-specific: domain
fields implicitly own initially empty registries and omit them from constructor
inputs. `LowerDomains` gathers local insertions, routes child registry outputs
upward, and concatenates lists. OM receives ordinary lists.

| Design point | Registry prototype | MutableCollection |
| --- | --- | --- |
| Purpose | Domain asset tracking | General property/OM accumulation |
| Ownership | Implicit domain-field ownership | Explicit allocations and shared handles |
| Mutation target | Domain registry field | First-class collection handle |
| Order/multiplicity | Unordered, duplicates preserved | Unordered, duplicates preserved |
| Effects removed | FIRRTL `LowerDomains` | OM object elaboration |
| Generated interface | Additional registry list outputs | Collection-valued fields/parameters |
| Mutable runtime value | None | None |

Registry's restricted ownership permits static bottom-up lowering. A generic
collection can be shared through an object graph unrelated to module hierarchy;
resolving its destination depends on parameters and aliases. Folding in the
elaborated graph avoids separate ownership analysis and hierarchy-specific output
plumbing. The tradeoff is retaining effects longer, rather than [eliminating them
inside FIRRTL][registry-review].

### Domain registry example

[`domain-collections.fir`](../../../test/firtool/domain-collections.fir) declares:

```firrtl
domain ClockDomain:
  name: String
  clockGates: MutableCollection<Path>
```

Child modules push paths through `D.clockGates`. Ordinary domain lowering turns
this into a collection-valued class field; no extra registry outputs or parent
concatenation are needed. After importing a concrete domain, evaluation folds all
contributions into its copied collection. Read the populated domain from
`top.A_out.domainInfo_out`, not the unchanged input domain.

Collection constructor arguments can be supplied through the evaluator API as
typed lists. `domaintool` retains its integer/string CLI arguments and does not
insert collection defaults or accept collection arguments. Its
[Clock Spec JSON regression](../../../test/Tools/domaintool/clock-spec-json.mlir) allocates
the collection inside the OM domain class instead.

Clock Spec JSON reads `clockGates_out` from the returned domain, validates known
paths, and sorts full OM reference-target strings while preserving duplicates.
An empty field emits `"clock_gates": []`; an absent field preserves the existing
JSON format. `source` and `relationship` remain optional string fields.

## Consumer example and validation

Completed collections use existing C++, C, and Python list APIs; no collection
value/type bindings are needed. IR collection types can be parsed with generic
MLIR type APIs when supplying unknown inputs.

[`collection_class.py`](../../../integration_test/Bindings/Python/dialects/collection_class.py)
demonstrates ordinary classes without domains: an assembly registers component
objects in a catalog's collection through unused writer objects.

```python
evaluator = om.Evaluator(module)
catalog = evaluator.instantiate("Catalog", "example")
assembly = evaluator.instantiate("Assembly", catalog)
components = sorted((component.name, component.id)
                    for component in assembly.catalog.components)
assert components == [("alu", 1), ("cache", 2)]
assert len(catalog.components) == 0
```

With [Python bindings enabled](../../PythonBindings.md):

```sh
ninja -C build CIRCTPythonModules
PYTHONPATH="$PWD/build/tools/circt/python_packages/circt_core" \
  python3 integration_test/Bindings/Python/dialects/collection_class.py
```

Tests cover duplicate contributions, distinct allocations under CSE, dead
allocation removal, unused writers, shared handles and field transport, graph
forward/back-references, unknown elements/membership, seed-value import isolation,
cyclic-input rejection, frozen paths (including unknown nested containers),
parser/emitter and bytecode preservation, and end-to-end domain collateral.
Compare membership as a multiset, not a guaranteed order. Lit cases live in the
existing parser, verifier, round-trip, canonicalization, lowering, elaboration,
path-freezing, and Clock Spec JSON files. Runtime behavior is covered in
`unittests/Dialect/OM/Evaluator/EvaluatorTests.cpp`, and Python list/unknown wrapping
is exercised by `collection_class.py` above.

Run `ninja -C build check-circt`; Python integration tests require bindings.
Persistent cross-call mutation, alias-preserving collection import, unknown
external contributions, and migration of the full Registry prototype are separate
semantic changes, not implementation cleanups.

[graph-regions]: https://mlir.llvm.org/docs/LangRef/#graph-regions
[registry-pr]: https://github.com/llvm/circt/pull/10940
[registry-source]: https://github.com/uenoku/circt/tree/c2efe1dc24e15a2d2de050d521f4223fc9f663c5
[registry-review]: https://github.com/llvm/circt/pull/10940#issuecomment-5186428314
