# Declarative FIRRTL Source Formats

[TOC]

## Overview

FIRRTL operations can declare their `.fir` source syntax with `firrtlFormat`.
This is independent of MLIR's `assemblyFormat`: the latter describes MLIR text,
not FIRRTL source. The `circt-tblgen --firrtl-primitives` backend validates the
format and generates importer token registrations and exporter dispatch and
wrappers. Both directions use the existing parser and emitter helpers.

An empty `firrtlFormat` (the default) does not generate source handling. A
nonempty call format normally opts into both import and export. Operations with
a handwritten importer can explicitly set `firrtlEmitOnly = 1`. A bare operand
format is implicitly emitter-only because it cannot reconstruct the operation.
An operation can set the format directly or inherit it from a shared TableGen
class.

## Declaring a Format

For an ordinary binary primitive, the shared `IntBinaryPrimOp` class builds
the format from its mnemonic using normal TableGen string concatenation:

```tablegen
class IntBinaryPrimOp<string mnemonic, Type resultType,
                      list<Trait> traits = []> : /* ... */ {
  let firrtlFormat = "`" # mnemonic # "` `(` operands `)`";
}

def AddPrimOp : IntBinaryPrimOp<"add", IntType, [Commutative]>;
```

The inherited format describes `add(lhs, rhs)`. An operation can override it
with a different spelling, or with an empty string to opt out. A format can
also be written directly, for example:

```tablegen
let firrtlFormat = "`mux` `(` operands `)`";
```

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
call-format := name-literal "`(`" operand-group ("`,`" attribute)* "`)`"
operand-group := "operands" | operand ("`,`" operand)*
operand       := "$" ods-operand-name
attribute     := "$" ods-attribute-name
```

The name is a backtick-enclosed identifier recognized by the FIRRTL lexer and
usable by the token macros. An operand group is either the `operands` directive
or explicit operand references, not a mixture. References must name every ODS
argument exactly once in ODS order. The directive expands to all operands and
does not include integer attributes; those must be referenced individually.
There must be at least one fixed expression operand, followed by zero or more
named, required `I32Attr` arguments. Participating operations
must have exactly one fixed result and no regions or successors. Properties,
optional operands, default-valued attributes, reordered or repeated arguments,
optional groups, type directives, and custom directives are not supported.
Variadic operands are supported only in emitter-only formats using `operands`. An unsupported nonempty format is an error, not a request for
handwritten fallback.

Whitespace between format elements does not prescribe source whitespace or line
breaking. The existing lexer recognizes the name and opening parenthesis as a
combined token; the emitter does not put whitespace between them.

The backend rejects duplicate generated spellings. Collisions with handwritten
left-parenthesis keywords are also caught by duplicate token enum definitions at
C++ compile time. The backend cannot validate arbitrary C++ declarations:
operations with generated importers must provide the narrow `inferReturnType`
overload and builder signatures used by `parsePrimExp`. Emitter-only formats do
not require those import signatures.

## Emitter-Only Formats

Set `firrtlEmitOnly = 1` on a call format when the source importer needs a
specialized parser. A bare operand format is automatically emitter-only. These
formats generate only an emitter and its dispatch entry, not an importer
registration. For example:

```tablegen
// Transparent cast: there is no source syntax that reconstructs the cast.
let firrtlFormat = "$input";

// The importer handles reference semantics separately.
let firrtlFormat = "`read` `(` $ref `)`";
let firrtlEmitOnly = 1;

// The importer handles the variadic call and its type checks separately.
let firrtlFormat = "`string_concat` `(` operands `)`";
let firrtlEmitOnly = 1;
```

A bare operand format must reference exactly one fixed operand and have no
attributes. It prints that operand directly. An emitter-only call may use a
variadic operand through `operands`; fixed unary calls print directly, while
other calls use the existing primitive emitter. The handwritten importer and
lexer registration, if any, remain responsible for source semantics and
version checks.

## Version Metadata

Each `FIRRTLOp` has one version shared by the importer and exporter:

```tablegen
FIRRTLSourceVersion firrtlVersion = FIRRTLBaseVersion;
```

For example, `BoolAndOp` declares:

```tablegen
let firrtlFormat = "`bool_and` `(` operands `)`";
let firrtlVersion = FIRRTLReleasedVersion<6, 0, 0>;
```

A version may be shared by operations through a TableGen `let` block or base
class. Version diagnostics use the FIRRTL source spelling from the format; for
a bare operand format, they use the operation's MLIR mnemonic. The version
records map to existing C++ values:

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
TOK_LPKEYWORD_PRIM(add, AddPrimOp, 2, 0, FIRVersion(0, 0, 0), "add")
TOK_LPKEYWORD_PRIM(bits, BitsPrimOp, 1, 2, FIRVersion(0, 0, 0), "bits")
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
policies. For an emitter-only format, the version applies to emission; any
handwritten importer supplies its own version check. The generated file and its
dependencies are wired into the FIRRTL importer and exporter builds.

## Handwritten and Special Cases

The format does not describe all FIRRTL syntax. Variadic `cat` has
version-dependent arity and is exported as a binary tree. Reference and
property operations may have special semantic checks and construction paths.
Import-only `tagExtract`, literals, generic intrinsics, and statements also
need specialized handling. Dialect-only operations leave their format empty.

Some of these cases use handwritten parsing and emission; others use an
emitter-only format with a handwritten importer. Extensions such as custom
directives or bidirectional variadic parsing need a construction and semantic
design before they can be supported.
