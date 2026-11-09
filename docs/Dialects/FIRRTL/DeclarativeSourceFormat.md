# Declarative FIRRTL Source Formats

[TOC]

## Overview

FIRRTL operations can declare their `.fir` source syntax with `firrtlFormat`.
This is independent of MLIR's `assemblyFormat`: the latter describes MLIR text,
not FIRRTL source. The `circt-tblgen --firrtl-primitives` backend validates the
format and generates importer token registrations and exporter dispatch and
wrappers. Both directions use the existing parser and emitter helpers.

An empty `firrtlFormat` (the default) does not generate source handling. A
nonempty format opts into both import and export; operation base classes do not
implicitly enable it. The `call` shorthand uses the MLIR mnemonic as the source
spelling.

## Declaring a Format

For an ordinary primitive without attributes, use `call` to specify a call
with the MLIR mnemonic and all operands:

```tablegen
def AddPrimOp : IntBinaryPrimOp<"add", IntType, [Commutative]> {
  let firrtlFormat = "call";
}
```

This describes `add(lhs, rhs)`. It is equivalent to the explicit format
``"`add` `(` operands `)`"``. The shorthand requires a lexer-compatible
mnemonic; use an explicit spelling when the FIRRTL name differs.

For a primitive with integer arguments:

```tablegen
def BitsPrimOp : PrimOp<"bits"> {
  let firrtlFormat = "`bits` `(` operands `,` $hi `,` $lo `)`";
  let arguments = (ins IntType:$input, I32Attr:$hi, I32Attr:$lo);
  // ...
}
```

ODS provides the operand and attribute kinds, counts, and getter names. The
`operands` directive represents all expression operands in ODS order, separated
by commas. Integer attributes are listed explicitly. The format gives the
source spelling and argument order. For example, the FIRRTL
source spelling `integer_add` differs from the MLIR mnemonic `integer.add`
and is declared explicitly. The `bits` MLIR assembly format uses `to`, independently of the
commas in its FIRRTL format.

## Supported Language

The initial format language supports fixed-arity primitive calls:

```text
call-format := "call" | name-literal "`(`" operand-group ("`,`" attribute)* "`)`"
operand-group := "operands" | operand ("`,`" operand)*
operand       := "$" ods-operand-name
attribute     := "$" ods-attribute-name
```

The `call` shorthand requires at least one fixed operand and no attributes or
properties, and uses the operation mnemonic as the source name. In the explicit
form, the name is a backtick-enclosed identifier. Both names must be recognized
by the FIRRTL lexer and usable by the token macros. An operand group is either the `operands` directive
or explicit operand references, not a mixture. References must name every ODS
argument exactly once in ODS order. The directive expands to all operands and
does not include integer attributes; those must be referenced individually.
There must be at least one fixed expression operand, followed by zero or more
named, required `I32Attr` arguments. Participating operations
must have exactly one fixed result and no regions or successors. Properties,
optional or variadic operands, default-valued attributes, reordered or repeated
arguments, optional groups, type directives, and custom directives are not
supported. An unsupported nonempty format is an error, not a request for
handwritten fallback.

Whitespace between format elements does not prescribe source whitespace or line
breaking. The existing lexer recognizes the name and opening parenthesis as a
combined token; the emitter does not put whitespace between them.

The backend rejects duplicate generated spellings. Collisions with handwritten
left-parenthesis keywords are also caught by duplicate token enum definitions at
C++ compile time. The backend cannot validate arbitrary C++ declarations: each
operation must provide the narrow `inferReturnType` overload and builder
signatures used by `parsePrimExp`.

## Version and Feature Metadata

Each `FIRRTLOp` has one version and one diagnostic feature name, shared by the
importer and exporter:

```tablegen
FIRRTLSourceVersion firrtlVersion = FIRRTLBaseVersion;
string firrtlFeature = "Base";
```

For example, `BoolAndOp` declares:

```tablegen
let firrtlFormat = "`bool_and` `(` operands `)`";
let firrtlVersion = FIRRTLReleasedVersion<6, 0, 0>;
let firrtlFeature = "boolean and";
```

A version may be shared by operations through a TableGen `let` block or base
class. Feature names are explicit and need not match the source spelling. The
version records map to existing C++ values:

| TableGen metadata | Generated C++ |
| --- | --- |
| `FIRRTLBaseVersion` | `FIRVersion(0, 0, 0)` |
| `FIRRTLReleasedVersion<M, N, P>` | `FIRVersion(M, N, P)` |
| `FIRRTLUnspecifiedVersion` | `missingSpecFIRVersion` |
| `FIRRTLNextVersion` | `nextFIRVersion` |

Released-version components must fit in `uint16_t`. Unspecified and next
versions are semantically distinct even if their current C++ sentinel values
coincide. See the [FIRRTL dialect rationale](RationaleFIRRTL.md#adding-features)
for the feature-release policy.

The shared version intentionally removes historical import/export asymmetries:
boolean operations now require 6.0.0 in both directions, while `asReset` uses
the base version in both directions. Integer arithmetic uses the 4.0.0 gate in
both directions; integer bitwise operations use the unspecified-version gate.

## Generated Code

The backend emits registrations such as:

```cpp
TOK_LPKEYWORD_PRIM(add, AddPrimOp, 2, 0, FIRVersion(0, 0, 0), "Base")
TOK_LPKEYWORD_PRIM(bits, BitsPrimOp, 1, 2, FIRVersion(0, 0, 0), "Base")
```

The existing token consumers generate lexer entries and expression-parser switch
cases. The parser checks the configured version, parses operands and integer
literals, infers the type, and only then creates the operation. Integer literals
are not narrowed to `I32Attr` before type inference. The existing `parseIntLit`
conversion zero-extends some negative literals (for example, `-1` becomes
`4294967295`); changing that behavior is separate from source generation.

The same definitions produce emitter wrappers and dispatch entries, for
example:

```cpp
void emitExpression(AddPrimOp op) { emitPrimExpr("add", op); }
void emitExpression(BitsPrimOp op) {
  emitPrimExpr("bits", op, {op.getHi(), op.getLo()});
}
```

A non-base version generates a `requireVersion` check. The emitter retains the
existing expression boxes, comma handling, integer printing, and line-breaking
policies. The generated file and its dependencies are wired into the FIRRTL
importer and exporter builds.

## Handwritten and Special Cases

The fixed-call format does not describe all FIRRTL syntax. Variadic `cat` has
version-dependent arity and is exported as a binary tree. Reference operations
have special semantic checks and construction paths. Property and variadic
calls, transparent casts, import-only `tagExtract`, literals, generic intrinsics,
and statements also need specialized handling. Dialect-only operations leave
their format empty.

The existing export-only call and transparent-cast mixins continue to generate
emitter glue for special operations with handwritten importers. They are not
part of the bidirectional `firrtlFormat` language. Extensions such as custom
directives, explicit directionality, or variadic arguments need a construction
and semantic design before they can be supported.
