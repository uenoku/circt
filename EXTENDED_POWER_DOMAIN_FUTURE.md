# Extended PowerDomain Support - Future Work

## Current Implementation

The current PowerDomain handler supports basic power domain extraction with:
- Domain name
- Signal associations (list of paths)

## SiFive PowerDomain Definition

The user-provided PowerDomain definition includes these features:

```scala
case class PowerDomain(name: String) {
  def containsModule(lm: BaseLazyModule): Unit
  def clamp[T <: Data](wire: T, value: T): Unit
  def clampTo[T <: Data](wire: T, value: T, destPD: PowerDomain): Unit
  def alwaysOn[T <: Data](wire: T): Unit
  def retain[T <: Data](stateBit: T): Unit
  def voltageCrossingPresent(present: Boolean): Unit
}
```

These methods add OM annotations to the design with information about:
1. **Module containment** - Which modules belong to which power domain
2. **Clamp values** - Isolation cell values when domain is powered off
3. **Cross-domain clamps** - Clamp values between different power domains
4. **Always-on signals** - Signals that should never be clamped
5. **Retention** - State bits that should be retained during power-down
6. **Voltage crossings** - Whether voltage level shifters are needed

## Expected OM Structure

For extended PowerDomain support, the FIRRTL domain lowering would need to produce:

```mlir
om.class @PowerDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %voltage_in: !om.integer,
  %alwaysOn_in: !om.bool
) -> (
  name_out: !om.string,
  voltage_out: !om.integer,
  alwaysOn_out: !om.bool,
  clamps_out: !om.list<!om.class.type<@Clamp>>,
  retentions_out: !om.list<!om.class.type<@Retention>>,
  voltageCrossingPresent_out: !om.bool
)

om.class @Clamp(
  %target: !om.frozenpath,
  %width: !om.integer,
  %value: !om.integer,
  %destDomain: !om.string,
  %alwaysOn: !om.bool
) -> (...)

om.class @Retention(
  %target: !om.frozenpath,
  %width: !om.integer
) -> (...)
```

## Enhanced JSON Output

The extended PowerDomain handler would generate:

```json
{
  "power_domains": [
    {
      "name": "CoreDomain",
      "voltage": 800,
      "always_on": false,
      "voltage_crossing_present": true,
      "associations": [
        "Core>datapath",
        "Core>registers"
      ],
      "clamps": [
        {
          "signal": "Core>io_out",
          "width": 32,
          "value": 0,
          "dest_domain": "AON",
          "always_on": false
        }
      ],
      "retentions": [
        {
          "signal": "Core>state_reg",
          "width": 64
        }
      ]
    },
    {
      "name": "AON",
      "voltage": 800,
      "always_on": true,
      "voltage_crossing_present": false,
      "associations": [
        "AON>timer",
        "AON>watchdog"
      ],
      "clamps": [],
      "retentions": []
    }
  ],
  "power_domain_crossings": [
    {
      "from": "CoreDomain",
      "to": "AON",
      "signals": ["Core>io_out"],
      "requires_isolation": true,
      "requires_level_shifter": true
    }
  ]
}
```

## Implementation Steps

To support the extended PowerDomain features:

1. **Update LowerDomains pass**
   - Extract clamp/retain/voltage crossing information from OM annotations
   - Create Clamp and Retention OM classes
   - Add fields to PowerDomain class for these attributes

2. **Enhance PowerDomainJSONHandler**
   - Parse voltage, alwaysOn fields from PowerDomain objects
   - Extract and process Clamp/Retention objects
   - Generate enhanced JSON with all power domain metadata

3. **Add validation**
   - Check clamp values are literals
   - Validate retention signals are registers
   - Verify clamp signals are not inputs (as per PowerDomain definition)

4. **UPF Generation (optional)**
   - Convert JSON to Unified Power Format
   - Generate proper isolation/level-shifter strategies
   - Create power state tables

## Code Pattern for Enhanced Handler

```cpp
LogicalResult handle(const ObjectMap &objectMap) override {
  for (auto &[objectValue, associations] : objectMap) {
    PowerDomainInfo info;
    
    // Extract basic fields
    info.name = getField<StringAttr>(objectValue, "name_out");
    info.voltage = getField<IntegerAttr>(objectValue, "voltage_out");
    info.alwaysOn = getField<BoolAttr>(objectValue, "alwaysOn_out");
    info.voltageCrossingPresent = 
        getField<BoolAttr>(objectValue, "voltageCrossingPresent_out");
    
    // Extract clamps
    auto clampsList = getField<ListValue>(objectValue, "clamps_out");
    for (auto &clamp : clampsList) {
      auto clampObj = cast<ObjectValue>(clamp);
      info.clamps.push_back({
        getField<PathValue>(clampObj, "target"),
        getField<IntegerAttr>(clampObj, "width"),
        getField<IntegerAttr>(clampObj, "value"),
        getField<StringAttr>(clampObj, "destDomain"),
        getField<BoolAttr>(clampObj, "alwaysOn")
      });
    }
    
    // Extract retentions
    auto retentionsList = getField<ListValue>(objectValue, "retentions_out");
    for (auto &retention : retentionsList) {
      auto retObj = cast<ObjectValue>(retention);
      info.retentions.push_back({
        getField<PathValue>(retObj, "target"),
        getField<IntegerAttr>(retObj, "width")
      });
    }
    
    // Store associations
    for (auto &assoc : associations) {
      info.associations.push_back(cast<PathValue>(assoc)->getRef());
    }
    
    powerDomains.push_back(info);
  }
  return success();
}
```

## Testing Strategy

Extended tests would include:
- Power domain with clamp values
- Power domain with retention registers
- Cross-domain isolation cells
- Always-on domains
- Voltage level shifter insertion
- Multi-voltage designs

This framework is ready for these extensions when the FIRRTL domain lowering is enhanced to support the full SiFive PowerDomain API.
