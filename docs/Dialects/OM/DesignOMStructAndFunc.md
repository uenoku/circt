# Separating OM Data Types from Computation

Proposal; syntax is illustrative.

## Motivation

Separately compiled FIRRTL units each emit domain classes through `LowerDomains`
and `LowerClasses`. Linking produces duplicate definitions because `om.class`
combines a data type and its constructor:

```mlir
// Emitted in both Unit A and Unit B.
om.class @ClockDomain(%basepath: !om.basepath, %name: !om.string)
    -> (name: !om.string) {
  om.class.fields %name : !om.string
}
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

Unit B defines `Child_Class`, but both units also define `ClockDomain` and
`ClockDomain_out`. Renaming those classes changes the type of `Child_Class.clock`
expected by Unit A. We need shared data types and separately linked constructors.

## Proposal

- `om.struct` declares named, typed fields without a constructor.
  `!om.struct<@Name>` is a nominal type.
- `om.struct.create` takes field values in declaration order;
  `om.struct.field` reads a field.
- `om.func` takes typed parameters and returns structs or other OM values through
  `om.return`. `om.call` invokes it. A bodyless declaration is external.

Multiple functions can return the same struct type:

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

Preserve OM's graph-region evaluation, reference passing, object identity, and
cyclic object graphs. Verify field names/types and function signatures.

## FIRRTL Linking Example

Keep `LowerClasses`' `_Class` suffix for module schemas (external modules use
`defname`). Domain schemas have no suffix. Path-building details are omitted:

```mlir
// Shared interface, emitted in both compilation units.
om.struct @ClockDomain { name: !om.string }
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

The linker merges same-named schemas only if field names, order, and types match.
Differently named structs remain distinct. External functions resolve to one
matching definition. Schema/signature mismatches and duplicate function
definitions are errors.

## Migration

Split each class into a struct and a constructor function:

| Existing construct | Replacement |
| --- | --- |
| `om.class` fields / `!om.class.type` | `om.struct` / `!om.struct` |
| `om.class` parameters and body | `om.func` parameters and body |
| `om.class.fields` | `om.struct.create` followed by `om.return` |
| `om.object` | `om.call` to the generated constructor |
| `om.object.field` | `om.struct.field` |
| `om.elaborated_object` | `om.struct.create` |
| `om.class.extern` | Struct declaration plus external function declaration |

Add a conversion pass, then migrate frontends, elaboration, and C++/Python
bindings. The evaluator calls functions and returns typed results. Preserve
field locations and object graphs.

No methods, inheritance, generics, mutation, or evaluation-order changes.
