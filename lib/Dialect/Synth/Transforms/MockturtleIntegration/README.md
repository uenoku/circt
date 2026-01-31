
# Mockturtle Integration

This directory contains utilities to integrate the [mockturtle](https://github.com/mockturtle/mockturtle) library with the Synth dialect.

## Mockturtle

Mockturtle is a C++ logic network library that provides various algorithms for logic synthesis and optimization.
Most of them are provided as header-only libraries, but there are few algorithms that require linking against ABC.

## Exception and RTTI

The mockturtle library uses C++ exceptions and RTTI, which are usually disabled by default in LLVM and MLIR projects.
To avoid build errors we isolate the code that depends on these features. The `Bindings` subdirectory contains the implementation
files that require exceptions and RTTI. Users are expected to define a helper function in `NetworkConversion.h`
and define it under `Bindings` directory. These functions must catch all exceptions if the mockturtle could return them.

## Write a mockturtle-based pass

To write a pass that uses mockturtle, you need to follow these steps:

1. Define a new pass in `SynthMockturtlePasses.td`. You can refer to the existing passes in the same file. Make sure to include the necessary dependent dialects (usually comb, synth, and hw).
2. If the library requires RTTI or exceptions, define a helper function in `NetworkConversion.h` and implement the pass logic in a separate source file under the `Bindings` directory. This file should include the necessary mockturtle headers and handle any exceptions that may be thrown.
3. If the pass does more than just call mockturtle functions (e.g. See Emap.cpp), add a separate implementation file under the `MockturtleIntegration` directory. Otherwise you can implement the pass directly in the `MockturtlePasses.cpp` file.
