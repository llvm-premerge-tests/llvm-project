!RUN: %flang_fc1 -emit-fir -fopenmp %s -o - | FileCheck %s --check-prefixes HOST
!RUN: %flang_fc1 -emit-fir -fopenmp -fopenmp-is-device %s -o - | FileCheck %s --check-prefixes DEVICE

!DEVICE: func.func @_QPread_write_section_omp_outline_0(%[[ARG0:.*]]: !fir.ref<i32>, %[[ARG1:.*]]: !fir.ref<!fir.array<10xi32>>, %[[ARG2:.*]]: !fir.ref<!fir.array<10xi32>>) attributes {omp.declare_target = #omp.declaretarget<device_type = (host), capture_clause = (to)>, omp.outline_parent_name = "_QPread_write_section"} {
!DEVICE:  %c4 = arith.constant 4 : index
!DEVICE:  %c1 = arith.constant 1 : index
!DEVICE:  %c1_0 = arith.constant 1 : index
!DEVICE:  %c1_1 = arith.constant 1 : index
!DEVICE:  %[[BOUNDS0:.*]] = omp.bounds   lower_bound(%c1 : index) upper_bound(%c4 : index) stride(%c1_1 : index) start_idx(%c1_1 : index)
!DEVICE:  %[[MAP0:.*]] = omp.map_entry var_ptr(%[[ARG1]] : !fir.ref<!fir.array<10xi32>>)   map_type_value(35) capture(ByRef) bounds(%[[BOUNDS0]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_read(2:5)"}
!DEVICE:  %c4_2 = arith.constant 4 : index
!DEVICE:  %c1_3 = arith.constant 1 : index
!DEVICE:  %c1_4 = arith.constant 1 : index
!DEVICE:  %c1_5 = arith.constant 1 : index
!DEVICE:  %[[BOUNDS1:.*]] = omp.bounds   lower_bound(%c1_3 : index) upper_bound(%c4_2 : index) stride(%c1_5 : index) start_idx(%c1_5 : index)
!DEVICE:  %[[MAP1:.*]] = omp.map_entry var_ptr(%[[ARG2]] : !fir.ref<!fir.array<10xi32>>)   map_type_value(35) capture(ByRef) bounds(%[[BOUNDS1]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_write(2:5)"}
!DEVICE:  omp.target   map_entries((tofrom -> %[[MAP0]] : !fir.ref<!fir.array<10xi32>>), (tofrom -> %[[MAP1]] : !fir.ref<!fir.array<10xi32>>)) {

!HOST:  func.func @_QPread_write_section() {
!HOST:  %0 = fir.alloca i32 {bindc_name = "i", uniq_name = "_QFread_write_sectionEi"}
!HOST:  %[[READ:.*]] = fir.address_of(@_QFread_write_sectionEsp_read) : !fir.ref<!fir.array<10xi32>>
!HOST:  %[[WRITE:.*]] = fir.address_of(@_QFread_write_sectionEsp_write) : !fir.ref<!fir.array<10xi32>>
!HOST:  %c1 = arith.constant 1 : index
!HOST:  %c1_0 = arith.constant 1 : index
!HOST:  %c4 = arith.constant 4 : index
!HOST:  %[[BOUNDS0:.*]] = omp.bounds   lower_bound(%c1_0 : index) upper_bound(%c4 : index) stride(%c1 : index) start_idx(%c1 : index)
!HOST:  %[[MAP0:.*]] = omp.map_entry var_ptr(%[[READ]] : !fir.ref<!fir.array<10xi32>>)   map_type_value(35) capture(ByRef) bounds(%[[BOUNDS0]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_read(2:5)"}
!HOST:  %c1_1 = arith.constant 1 : index
!HOST:  %c1_2 = arith.constant 1 : index
!HOST:  %c4_3 = arith.constant 4 : index
!HOST:  %[[BOUNDS1:.*]] = omp.bounds   lower_bound(%c1_2 : index) upper_bound(%c4_3 : index) stride(%c1_1 : index) start_idx(%c1_1 : index)
!HOST:  %[[MAP1:.*]] = omp.map_entry var_ptr(%[[WRITE]] : !fir.ref<!fir.array<10xi32>>)   map_type_value(35) capture(ByRef) bounds(%[[BOUNDS1]]) -> !fir.ref<!fir.array<10xi32>> {name = "sp_write(2:5)"}
!HOST:  omp.target   map_entries((tofrom -> %[[MAP0]] : !fir.ref<!fir.array<10xi32>>), (tofrom -> %[[MAP1]] : !fir.ref<!fir.array<10xi32>>)) {

SUBROUTINE READ_WRITE_SECTION()
    INTEGER :: sp_read(10) = (/1,2,3,4,5,6,7,8,9,10/)
    INTEGER :: sp_write(10) = (/0,0,0,0,0,0,0,0,0,0/)

!$omp target map(tofrom:sp_read(2:5)) map(tofrom:sp_write(2:5))
    do i = 2, 5
        sp_write(i) = sp_read(i)
    end do
!$omp end target
END SUBROUTINE READ_WRITE_SECTION
