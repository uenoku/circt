// RUN: domaintool --module Foo --domain PowerDomain,PD_A --domain PowerDomain,PD_B --assign 0 --assign 1 %s | FileCheck %s

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

om.class @Foo_Class(
  %basepath: !om.frozenbasepath,
  %PD_A: !om.class.type<@PowerDomain>,
  %PD_B: !om.class.type<@PowerDomain>
)  -> (
  PD_A_out: !om.class.type<@PowerDomain_out>,
  PD_B_out: !om.class.type<@PowerDomain_out>
) {
  %0 = om.object @PowerDomain_out(%basepath, %PD_A, %3) : (
    !om.frozenbasepath,
    !om.class.type<@PowerDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@PowerDomain_out>
  %1 = om.frozenpath_create reference %basepath "Foo>signal_a"
  %2 = om.frozenpath_create reference %basepath "Foo>signal_b"
  %3 = om.list_create %1 : !om.frozenpath
  %4 = om.object @PowerDomain_out(%basepath, %PD_B, %5) : (
    !om.frozenbasepath,
    !om.class.type<@PowerDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@PowerDomain_out>
  %5 = om.list_create %2 : !om.frozenpath
  om.class.fields %0, %4 : !om.class.type<@PowerDomain_out>, !om.class.type<@PowerDomain_out>
}

// CHECK:      {
// CHECK-NEXT:   "power_domains": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "name": "PD_A",
// CHECK-NEXT:       "associations": [
// CHECK-NEXT:         "signal_a"
// CHECK-NEXT:       ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "name": "PD_B",
// CHECK-NEXT:       "associations": [
// CHECK-NEXT:         "signal_b"
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
