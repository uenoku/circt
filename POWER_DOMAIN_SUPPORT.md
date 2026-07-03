# PowerDomain Support in domaintool

## Summary

Added PowerDomain support to the `domaintool` utility, enabling extraction and JSON generation of power domain information from FIRRTL designs compiled through CIRCT.

## Implementation

### Files Created

1. **tools/domaintool/PowerDomainJSONHandler.cpp**
   - New handler that processes PowerDomain objects from the Object Model (OM)
   - Generates JSON output with power domain names and their associated signals
   - Follows the same pattern as ClockSpecJSONHandler
   - Only emits output when power domains are present (avoids empty output)

2. **test/Tools/domaintool/power-domain-json.mlir**
   - Test case for basic PowerDomain functionality
   - Tests two power domains (PD_A and PD_B) with signal associations
   - Validates JSON output format

3. **test/Tools/domaintool/mixed-domains.mlir**
   - Test case demonstrating ClockDomain and PowerDomain coexistence
   - Shows both handlers can work together in the same design
   - Validates that both JSON outputs are produced correctly

### Files Modified

1. **tools/domaintool/CMakeLists.txt**
   - Added PowerDomainJSONHandler.cpp to the build

2. **tools/domaintool/ClockSpecJSONHandler.cpp**
   - Added check to only emit when there are clock domains to report
   - Prevents empty JSON output when no clocks are present
   - Maintains backward compatibility with existing tests

## PowerDomain Object Model

The implementation expects PowerDomain to be lowered into OM classes following this structure:

```mlir
om.class @PowerDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string
) -> (
  name_out: !om.string
)

om.class @PowerDomain_out(
  %basepath: !om.frozenbasepath,
  %domainInfo_in: !om.class.type<@PowerDomain>,
  %associations_in: !om.list<!om.frozenpath>
) -> (
  domainInfo_out: !om.class.type<@PowerDomain>,
  associations_out: !om.list<!om.frozenpath>
)
```

This matches the pattern used by ClockDomain and is consistent with the FIRRTL domain lowering pass.

## JSON Output Format

The PowerDomainJSON handler produces output in the following format:

```json
{
  "power_domains": [
    {
      "name": "PowerDomainName",
      "associations": [
        "signal_path_1",
        "signal_path_2"
      ]
    }
  ]
}
```

## Usage

```bash
domaintool --module ModuleName \
  --domain PowerDomain,DomainName \
  --assign 0 \
  input.mlir
```

Multiple power domains can be specified:

```bash
domaintool --module ModuleName \
  --domain PowerDomain,PWR_A \
  --domain PowerDomain,PWR_B \
  --assign 0 --assign 1 \
  input.mlir
```

## Handler Architecture

The implementation follows the domaintool handler pattern:

1. **shouldHandle()** - Returns true for OM ClassTypes named "PowerDomain"
2. **handle()** - Extracts domain name and signal associations from OM objects
3. **emit()** - Generates JSON output (only when domains are present)
4. **clear()** - Resets handler state for next invocation

The handler is registered automatically via static initialization, similar to ClockSpecJSONHandler.

## Testing

All tests pass:
- `test/Tools/domaintool/power-domain-json.mlir` - PowerDomain-only test
- `test/Tools/domaintool/mixed-domains.mlir` - Mixed ClockDomain and PowerDomain test
- `test/Tools/domaintool/clock-spec-json.mlir` - Existing ClockDomain tests (unchanged)
- `test/Tools/domaintool/errors.mlir` - Error handling tests (unchanged)

## Future Enhancements

The current implementation provides basic power domain extraction. Future enhancements could include:

1. **Extended PowerDomain attributes** - Support for clamp, retain, voltage crossing, etc. (as shown in the SiFive PowerDomain definition)
2. **Power domain relationships** - Similar to clock domain sync/async relationships
3. **UPF generation** - Direct Unified Power Format output instead of JSON
4. **Hierarchical power domain analysis** - Power domain hierarchy and nesting
5. **Integration with LowerDomains pass** - Ensure proper domain field handling

## References

- Original PowerDomain definition: See user-provided Scala code in request
- ClockDomain implementation: `tools/domaintool/ClockSpecJSONHandler.cpp`
- Domain lowering: `lib/Dialect/FIRRTL/Transforms/LowerDomains.cpp`
- Handler base class: `tools/domaintool/Handler.h`
