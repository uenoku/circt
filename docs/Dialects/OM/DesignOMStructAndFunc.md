# Separating OM Data Types from Computation

**Status:** Proposal; syntax below is illustrative.

## Motivation

The primary motivation is **linking separately compiled FIRRTL domains**.
`LowerDomains` generates classes describing domain information and associations;
`LowerClasses` lowers these to OM classes and generates module-level classes such
as `Child_Class`. Each compilation unit emits its own domain classes. Linking
then encounters duplicate class symbols: a shared data schema is inseparable
from a constructor definition. For example, using today's OM operations:

```mlir
// Independently emitted in BOTH Unit A and Unit B.
om.class @ClockDomain(%basepath: !om.basepath) {}
om.class @ClockDomain_out(%basepath: !om.basepath,
                          %info: !om.class.type<@ClockDomain>,
                          %paths: !om.list<!om.path>)
    -> (domainInfo_out: !om.class.type<@ClockDomain>,
        associations_out: !om.list<!om.path>) {
  om.class.fields %info, %paths
    : !om.class.type<@ClockDomain>, !om.list<!om.path>
}

// Unit A also imports the child module's generated class.
om.class.extern @Child_Class(%basepath: !om.basepath,
                             %info: !om.class.type<@ClockDomain>,
                             %paths: !om.list<!om.path>)
    -> (clock: !om.class.type<@ClockDomain_out>) {}
```

Unit B provides the `om.class @Child_Class` definition. Even if its definition
satisfies Unit A's external class declaration, linking still encounters two
`@ClockDomain` and two `@ClockDomain_out` definitions. Renaming Unit B's domain
classes avoids symbol collisions but makes its `Child_Class.clock` type differ
from the type expected by Unit A. The desired result is one shared domain schema
and a separately resolved child constructor, as illustrated below.

Separate schemas from computations so compilation units can agree on domain
types while independently defining or importing functions. This also allows
multiple functions to produce the same type, or return ordinary OM values
without defining a class.

## Proposal

Use **`om.struct`** for named data structures and **`om.func`** for computation:

- `om.struct @Name` declares an ordered set of named, typed fields. It has no
  parameters or computation body. `!om.struct<@Name>` identifies the type
  nominally; identical fields do not make differently named types equivalent.
- `om.struct.create` constructs an instance from explicit field values, in
  declaration order. `om.struct.field` reads a named field. Neither operation
  invokes a function.
- `om.func @name` declares typed parameters and results, with a body ending in
  `om.return`. Results may be structs or other OM types; multiple functions may
  return the same struct type. `om.call` invokes a function. A bodyless
  declaration represents an external function.

```mlir
om.struct @Device { address: !om.integer }

om.func @make_device(%base: !om.integer) -> !om.struct<@Device> {
  %offset = om.constant #om.integer<4 : i4> : !om.integer
  %address = om.integer.add %base, %offset : !om.integer
  %device = om.struct.create @Device(%address)
    : (!om.integer) -> !om.struct<@Device>
  om.return %device : !om.struct<@Device>
}

om.func @device_address(%device: !om.struct<@Device>) -> !om.integer {
  %address = om.struct.field %device["address"]
    : (!om.struct<@Device>) -> !om.integer
  om.return %address : !om.integer
}
```

Keep OM's graph-region evaluation model, reference passing, and support for
cyclic object graphs; this is not a switch to imperative execution or structural
value semantics. Struct construction retains object allocation/identity
semantics. Verification checks schema references, unique field names, field
operand types, call signatures, and return types independently.

## FIRRTL Linking Example

`LowerClasses` currently uses `_Class` for classes generated from FIRRTL modules
and external modules (using the external module's `defname`), not for classes
generated directly from domains. Retain that suffix on the generated schema;
use a distinct constructor symbol. Reduced example, with path-building details
omitted:

```mlir
// Shared interface, emitted in both compilation units.
om.struct @ClockDomain {}
om.struct @ClockDomain_out {
  domainInfo_out: !om.struct<@ClockDomain>,
  associations_out: !om.list<!om.path>
}
om.struct @Child_Class { clock: !om.struct<@ClockDomain_out> }

// Unit A: external constructor declaration, with no body.
om.func @make_Child_Class(%basepath: !om.basepath,
                         %info: !om.struct<@ClockDomain>,
                         %paths: !om.list<!om.path>) -> !om.struct<@Child_Class>

om.func @get_child_clock(%basepath: !om.basepath,
                        %info: !om.struct<@ClockDomain>,
                        %paths: !om.list<!om.path>) -> !om.struct<@ClockDomain_out> {
  %child = om.call @make_Child_Class(%basepath, %info, %paths)
    : (!om.basepath, !om.struct<@ClockDomain>, !om.list<!om.path>)
      -> !om.struct<@Child_Class>
  %clock = om.struct.field %child["clock"]
    : (!om.struct<@Child_Class>) -> !om.struct<@ClockDomain_out>
  om.return %clock : !om.struct<@ClockDomain_out>
}

// Unit B: definition satisfying Unit A's external declaration.
om.func @make_Child_Class(%basepath: !om.basepath,
                         %info: !om.struct<@ClockDomain>,
                         %paths: !om.list<!om.path>) -> !om.struct<@Child_Class> {
  %clock = om.struct.create @ClockDomain_out(%info, %paths)
    : (!om.struct<@ClockDomain>, !om.list<!om.path>)
      -> !om.struct<@ClockDomain_out>
  %child = om.struct.create @Child_Class(%clock)
    : (!om.struct<@ClockDomain_out>) -> !om.struct<@Child_Class>
  om.return %child : !om.struct<@Child_Class>
}
```

**Linking contract:** shared schemas use stable, agreed symbol names. The linker
coalesces same-named schemas only when field names, order, and types agree,
resolving referenced schemas consistently; mismatches are errors. This does not
make differently named structs equivalent. Function declarations resolve to a
single definition with a matching signature; conflicting definitions are
errors. The split alone is insufficient without these linker rules.

## Migration

Each existing class becomes a struct plus a constructor function with a distinct
symbol (for example, `@Device` and `@make_device`):

| Existing construct | Replacement |
| --- | --- |
| `om.class` fields / `!om.class.type` | `om.struct` / `!om.struct` |
| `om.class` parameters and body | `om.func` parameters and body |
| `om.class.fields` | `om.struct.create` followed by `om.return` |
| `om.object` | `om.call` to the generated constructor |
| `om.object.field` | `om.struct.field` |
| `om.elaborated_object` | `om.struct.create` |
| `om.class.extern` | Struct declaration plus external function declaration |

Introduce the new constructs alongside the old ones, provide a conversion pass,
and migrate frontends, elaboration, and C++/Python evaluator APIs before retiring
classes. The evaluator entry point becomes function evaluation returning typed
results; field inspection remains schema-based. Preserve field locations and
existing object-graph behavior during conversion.

**Non-goals:** methods, inheritance, generics, mutation, or changes to evaluation
ordering. `om.struct` is preferred over retaining `om.class` because it makes the
schema-only role explicit.
