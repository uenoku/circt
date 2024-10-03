# Yosys Integration

Yosys is de-facto standard as an opensource RTL synthesis and verification tool.
The purpose of the yosys integration is to serve as a baseline for circt-based synthesis flow.

The abstraction level of Yosys IR(RTLIL) is mostly same with CIRCT core dialects while RTLIL is slightly different.


## Build
Yosys integration is disabled by default. To enable Yosys integration, you need to build CIRCT with `-DCIRCT_YOSYS_INTEGRATION_ENABLED=ON`.


## Tranlation between RTLIL

`circt-translate` provides subcommand for translation between CIRCT IR and RTLIL.

* CIRCT IR -> RTLIL

```bash
circt-tranlsate --export-rtlil
```
Currently only subest of core dialects are translted during translation to RTLIL. 

* RTLIL -> CIRCT IR

```bash
circt-tranlsate --import-rtlil
```


## Run Yosys passes on CIRCT IR

To run Yosys passes on CIRCT there are two passes `yosys-optimizer` and `yosys-optimizer-parallel`.

### CIRCT as a parallel Yosys driver

Yosys has a globally context which is a not thread-safe so we cannot parallely run `yosys-optimizer` on each HW module. As a workaround CIRCT provides a `yosys-optimizer-parallel` pass that parallely invokes yosys in child processes. `yosys-optimizer-parallel` cannot be used for transformation that requires module hierarchly (e.g. inlining/flattening etc)


## Testing

Testing Yosys integration is tricky since RTLIL textual format could differ between Yosys versions. Currenly we test the correctness of RTLIL translation by running LEC on the CIRCT IR after import with the original CIRCT IR after export.
