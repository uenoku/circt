// Test PowerDomain with extended fields: source, relationship, clampValue, destination
// RUN: domaintool --module Extended --domain PowerDomain,AON,AON,internal,0, --domain PowerDomain,Core,AON,clamped,0, --domain PowerDomain,GPU,AON,clampedtodomain,1,Core --assign 0 --assign 1 --assign 2 %s | FileCheck %s

om.class @PowerDomain(
  %basepath: !om.frozenbasepath,
  %name_in: !om.string,
  %source_in: !om.string,
  %relationship_in: !om.string,
  %clampValue_in: !om.integer,
  %destination_in: !om.string
)  -> (
  name_out: !om.string,
  source_out: !om.string,
  relationship_out: !om.string,
  clampValue_out: !om.integer,
  destination_out: !om.string
) {
  om.class.fields %name_in, %source_in, %relationship_in, %clampValue_in, %destination_in : !om.string, !om.string, !om.string, !om.integer, !om.string
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

om.class @Extended_Class(
  %basepath: !om.frozenbasepath,
  %AON: !om.class.type<@PowerDomain>,
  %Core: !om.class.type<@PowerDomain>,
  %GPU: !om.class.type<@PowerDomain>
)  -> (
  AON_out: !om.class.type<@PowerDomain_out>,
  Core_out: !om.class.type<@PowerDomain_out>,
  GPU_out: !om.class.type<@PowerDomain_out>
) {
  // AON domain - internal, always-on
  %0 = om.object @PowerDomain_out(%basepath, %AON, %4) : (
    !om.frozenbasepath,
    !om.class.type<@PowerDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@PowerDomain_out>
  %1 = om.frozenpath_create reference %basepath "Extended>aon_timer"
  %2 = om.frozenpath_create reference %basepath "Extended>aon_wdog"
  %4 = om.list_create %1, %2 : !om.frozenpath
  
  // Core domain - clamped to 0
  %5 = om.object @PowerDomain_out(%basepath, %Core, %7) : (
    !om.frozenbasepath,
    !om.class.type<@PowerDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@PowerDomain_out>
  %6 = om.frozenpath_create reference %basepath "Extended>core_cpu"
  %7 = om.list_create %6 : !om.frozenpath
  
  // GPU domain - clamped to Core domain with value 1
  %8 = om.object @PowerDomain_out(%basepath, %GPU, %10) : (
    !om.frozenbasepath,
    !om.class.type<@PowerDomain>,
    !om.list<!om.frozenpath>
  ) -> !om.class.type<@PowerDomain_out>
  %9 = om.frozenpath_create reference %basepath "Extended>gpu_compute"
  %10 = om.list_create %9 : !om.frozenpath
  
  om.class.fields %0, %5, %8 : !om.class.type<@PowerDomain_out>, !om.class.type<@PowerDomain_out>, !om.class.type<@PowerDomain_out>
}

// CHECK:      {
// CHECK-NEXT:   "power_domains": [
// CHECK-NEXT:     {
// CHECK-NEXT:       "name": "AON",
// CHECK-NEXT:       "source": "AON",
// CHECK-NEXT:       "relationship": "internal",
// CHECK-NEXT:       "clamp_value": 0,
// CHECK-NEXT:       "destination": "",
// CHECK-NEXT:       "associations": [
// CHECK-NEXT:         "aon_timer",
// CHECK-NEXT:         "aon_wdog"
// CHECK-NEXT:       ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "name": "Core",
// CHECK-NEXT:       "source": "AON",
// CHECK-NEXT:       "relationship": "clamped",
// CHECK-NEXT:       "clamp_value": 0,
// CHECK-NEXT:       "destination": "",
// CHECK-NEXT:       "associations": [
// CHECK-NEXT:         "core_cpu"
// CHECK-NEXT:       ]
// CHECK-NEXT:     },
// CHECK-NEXT:     {
// CHECK-NEXT:       "name": "GPU",
// CHECK-NEXT:       "source": "AON",
// CHECK-NEXT:       "relationship": "clampedtodomain",
// CHECK-NEXT:       "clamp_value": 1,
// CHECK-NEXT:       "destination": "Core",
// CHECK-NEXT:       "associations": [
// CHECK-NEXT:         "gpu_compute"
// CHECK-NEXT:       ]
// CHECK-NEXT:     }
// CHECK-NEXT:   ]
// CHECK-NEXT: }
