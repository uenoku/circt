# Structural choice operation for logic synthesis

This proposes a structural choice operation `synth.choice` for logic synthesis.

## Motivation

In logic synthesis, it is common to have multiple structurally different implementations of the same logical function. For example, multiplier has several implementations, including array multipliers, tree multipliers, and booth multipliers.
Each of these implementations has different characteristics and trade-offs. Usually we can not know which implementation is the best beforehand, and it really depends on the context of the design.
Therefore modern synthesizers are inevitably required to explore multiple implementations of the same logical function during logic synthesis.
The proposed `synth.choice` operation is designed to represent such multiple implementations of the same logical function in a structural way in IR.

## Operation definition

```
    %result = synth.choice %current_choice, %option0, %option1, ... : type
```

### Use case 1: Inserting a choice op during lowering
A first simple use case is to insert choice ops during lowering from high-level operations to low-level operations.
```mlir
%0 = comb.mul %a, %b : i16
```

```
==>
%booth_mul =  .... 

```

### Use case 2: FRAIG
A second use case is to insert choice nodes during FRAIG construction (actually notion of choice nodes originated from this context).
```
%0 = synth.aig.and_inv %a, %b : i1
%1 = synth.aig.and_inv %a, %c : i1

%2 = synth.choice %0, %1 : i1
```

### Use case 3: Technology mapping


### Use case 4: Run different pass pipelines
This is very ultimate use case, but we can clone and run different synthesis pipelines on the same module, and merge the results using `synth.choice`.

```
hw.module @foo(in %a: i16, in %b: i16, out %y: i16) {
  %0 = comb.mul %a, %b : i16
  hw.output %0 : i16
}

==> 
// Run pipeline 1
hw.module @foo_impl0(in %a: i16, in %b: i16, out %y: i16) {
  // Lower comb.mul with pipeline 1 strategy
  hw.output
}
// Run pipeline 2
hw.module @foo_impl1(in %a: i16, in %b: i16, out %y: i16) {
  // Lower comb.mul with pipeline 2 strategy
  hw.output
}
hw.module @foo(in %a: i16, in %b: i16, out %y: i16) {
  %0 = hw.instance "foo_pipeline0" @foo_impl0(%a, %b) : (i16, i16) -> i16
  %1 = hw.instance "foo_pipeline1" @foo_impl1(%a, %b) : (i16, i16) -> i16
  %2 = synth.choice %0, %1 : (i16, i16) -> i16
  hw.output %2 : i16
}
```

## Design points
* Composability with logic depth analysis
    * The reason why `synth.choice` takes the first argument as the "current choice" is to make it composable with logic depth analysis pass. Logic depth analysis pass computes the logic depth of each operation in a bottom-up way, and however if `synth.choice` doesn't have a clear current choice, it becomes difficult to reason about the logic depth of the overall operation (maybe we can take the maximum/minimum depth of the choices but seems too conservative/optimistic).
* Non-region op vs region op
    * `synth.choice` is designed as a non-region op because it is easier to manipulate in most of the use cases. We could isolate entire dataflow into a region and have multiple regions as choices, but it's doesn't composed with Use case 2 and 4 above.

* How to prevent choice explosion
    * Choice ops preserve operations previously eliminated by optimization passes, and therefore the number of choice ops can explode if we are not careful.
    * In practical use cases, we may need to prune non-promising choices based on some heuristics (similar to priority cuts). 

## Egraph

Conceputually, synth.choice is weaker version on equivalence classes represented egraphs. Generic egraph approach might be interesting to explore in the future but using egraph in generic way might be too heavy for practical use cases in logic synthesis (as there are enourmous numeber of isomorphic graphs). Extracting a sub-graph (feasible for egraph but not for exact synthesis) and optimizing it using egraph approach would be an interesting future direction.


## Implementations in other tools 
* Mockturtle
In Mockturtle, choice is implemented as a trait `choice_view`.
* ABC
In abc choice is 

