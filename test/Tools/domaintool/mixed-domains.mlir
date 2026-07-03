// Test that both ClockDomain and PowerDomain handlers can work together
// RUN: domaintool --module Mixed --domain ClockDomain,CLK_A,CLK_A,synchronous --domain PowerDomain,PWR_A --assign 0 --assign 1 %s | FileCheck %s

om.class @ClockDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %source_in: !om.string,
  %relationship_in: !om.string
)  -> (
  name_out: !om.string,
  source_out: !om.string,
  relationship_out: !om.string
) {
  om.class.fields %name_in, %source_in, %relationship_in : !om.string, !om.string, !om.string
}

om.class @ClockDomain_out(
  %basepath: !om.frozenbasepath,
  %domainInfo_in: !om.class.type<@ClockDomain>,
  %associations_in: !om.list<!om.frozenpath>
)  -> (
  domainInfo_out: !om.class.type<@ClockDomain>,
  associations_out: !om.list<!om.frozenpath>
) {
  om.class.fields %domainInfo_in, %associations_in : !om.class.type<@ClockDomain>, !om.list<!om.frozenpath>
}

om.class @PowerDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string
)  -> (
  name_out: !om.string
) {
  om.class.fields %name_in : !om.string
}

om.class @PowerDomain_out(
  %basepath: !om.frozenbasepath,
  %domainInfo_in: !om.class.type<@PowerDomain>,
  %associations_in: !om.list<!om.frozenpath>
)  -> (
  domainInfo_out: !om.class.type<@PowerDomain>,
  associations_out: !om.list<!om.frozenpath>
) {
  om.class.fields %domainInfo_in, %associations_in : !om.class.type<@PowerDomain>, !om.list<!om.frozenpath>
}

om.class @Mixed_Class(
  %basepath: !om.frozenbasepath,
  %CLK: !om.class.type<@ClockDomain>,
  %PWR: !om.class.type<@PowerDomain>
)  -> (
  CLK_out: !om.class.type<@ClockDomain_out>,
  PWR_out: !om.class.type<@PowerDomain_out>
) {
  %0 = om.object @ClockDomain_out(%basepath, %CLK, %2) : (
    !om.frozenbasepath,
    !om.class.type<@ClockDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@ClockDomain_out>
  %1 = om.frozenpath_create reference %basepath "Mixed>clk_signal"
  %2 = om.list_create %1 : !om.frozenpath
  
  %3 = om.object @PowerDomain_out(%basepath, %PWR, %5) : (
    !om.frozenbasepath,
    !om.class.type<@PowerDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@PowerDomain_out>
  %4 = om.frozenpath_create reference %basepath "Mixed>pwr_signal"
  %5 = om.list_create %4 : !om.frozenpath
  
  om.class.fields %0, %3 : !om.class.type<@ClockDomain_out>, !om.class.type<@PowerDomain_out>
}

// CHECK:      {
// CHECK-NEXT:   "clocks": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "name_pattern": "CLK_A",
// CHECK-NEXT:       "define_period": "CLK_A_PERIOD",
// CHECK-NEXT:       "clock_relationships": []
// CHECK-NEXT:     }
// CHECK-NEXT:   ],
// CHECK-NEXT:   "static_ports": [],
// CHECK-NEXT:   "asynchronous_ports": [],
// CHECK-NEXT:   "synchronous_ports": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "name_pattern": "CLK_A",
// CHECK-NEXT:       "port_patterns": [
// CHECK-NEXT:         "clk_signal"
// CHECK-NEXT:       ],
// CHECK-NEXT:       "comment": null
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }{
// CHECK-NEXT:   "power_domains": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "name": "PWR_A",
// CHECK-NEXT:       "associations": [
// CHECK-NEXT:         "pwr_signal"
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
