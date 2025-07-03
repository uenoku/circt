module {
  hw.module @FPU(in %clock : !seq.clock, in %reset : i1, in %io_inst : i32, in %io_fromint_data : i64, in %io_fcsr_rm : i3, out io_fcsr_flags_valid : i1, out io_fcsr_flags_bits : i5, out io_store_data : i64, out io_toint_data : i64, in %io_dmem_resp_val : i1, in %io_dmem_resp_type : i3, in %io_dmem_resp_tag : i5, in %io_dmem_resp_data : i64, in %io_valid : i1, out io_fcsr_rdy : i1, out io_nack_mem : i1, out io_illegal_rm : i1, in %io_killx : i1, in %io_killm : i1, out io_dec_cmd : i5, out io_dec_ldst : i1, out io_dec_wen : i1, out io_dec_ren1 : i1, out io_dec_ren2 : i1, out io_dec_ren3 : i1, out io_dec_swap12 : i1, out io_dec_swap23 : i1, out io_dec_single : i1, out io_dec_fromint : i1, out io_dec_toint : i1, out io_dec_fastpipe : i1, out io_dec_fma : i1, out io_dec_div : i1, out io_dec_sqrt : i1, out io_dec_wflags : i1, out io_sboard_set : i1, out io_sboard_clr : i1, out io_sboard_clra : i5, out io_cp_req_ready : i1, in %io_cp_req_valid : i1, in %io_cp_req_bits_cmd : i5, in %io_cp_req_bits_ldst : i1, in %io_cp_req_bits_wen : i1, in %io_cp_req_bits_ren1 : i1, in %io_cp_req_bits_ren2 : i1, in %io_cp_req_bits_ren3 : i1, in %io_cp_req_bits_swap12 : i1, in %io_cp_req_bits_swap23 : i1, in %io_cp_req_bits_single : i1, in %io_cp_req_bits_fromint : i1, in %io_cp_req_bits_toint : i1, in %io_cp_req_bits_fastpipe : i1, in %io_cp_req_bits_fma : i1, in %io_cp_req_bits_div : i1, in %io_cp_req_bits_sqrt : i1, in %io_cp_req_bits_wflags : i1, in %io_cp_req_bits_rm : i3, in %io_cp_req_bits_typ : i2, in %io_cp_req_bits_in1 : i65, in %io_cp_req_bits_in2 : i65, in %io_cp_req_bits_in3 : i65, in %io_cp_resp_ready : i1, out io_cp_resp_valid : i1, out io_cp_resp_bits_data : i65, out io_cp_resp_bits_exc : i5) {
    %c3_i3 = hw.constant 3 : i3
    %c1879179264_i32 = hw.constant 1879179264 : i32
    %c-1_i9 = hw.constant -1 : i9
    %c256_i10 = hw.constant 256 : i10
    %c0_i2 = hw.constant 0 : i2
    %c32_i7 = hw.constant 32 : i7
    %c-1_i4 = hw.constant -1 : i4
    %c0_i7 = hw.constant 0 : i7
    %c0_i109 = hw.constant 0 : i109
    %c0_i63 = hw.constant 0 : i63
    %c-1_i6 = hw.constant -1 : i6
    %c0_i32 = hw.constant 0 : i32
    %c0_i52 = hw.constant 0 : i52
    %c0_i11 = hw.constant 0 : i11
    %c0_i6 = hw.constant 0 : i6
    %c0_i49 = hw.constant 0 : i49
    %c0_i31 = hw.constant 0 : i31
    %c-1_i5 = hw.constant -1 : i5
    %c0_i16 = hw.constant 0 : i16
    %c0_i9 = hw.constant 0 : i9
    %c0_i23 = hw.constant 0 : i23
    %c0_i8 = hw.constant 0 : i8
    %false = hw.constant false
    %true = hw.constant true
    %c-2_i2 = hw.constant -2 : i2
    %c-1_i2 = hw.constant -1 : i2
    %c-1_i3 = hw.constant -1 : i3
    %c0_i3 = hw.constant 0 : i3
    %c0_i5 = hw.constant 0 : i5
    %c0_i65 = hw.constant 0 : i65
    %c1_i2 = hw.constant 1 : i2
    %c0_i4 = hw.constant 0 : i4
    %ex_reg_valid = seq.firreg %io_valid clock %clock reset sync %reset, %false {firrtl.random_init_start = 0 : ui64} : i1
    %0 = comb.or bin %ex_reg_valid, %io_cp_req_valid {sv.namehint = "req_valid"} : i1
    %ex_reg_inst = seq.firreg %1 clock %clock {firrtl.random_init_start = 1 : ui64} : i32
    %1 = comb.mux bin %io_valid, %io_inst, %ex_reg_inst : i32
    %2 = comb.and bin %459, %io_cp_req_valid {sv.namehint = "ex_cp_valid"} : i1
    %3 = comb.xor bin %io_killx, %true : i1
    %4 = comb.and bin %ex_reg_valid, %3 : i1
    %5 = comb.or bin %4, %2 : i1
    %mem_reg_valid = seq.firreg %5 clock %clock reset sync %reset, %false {firrtl.random_init_start = 33 : ui64} : i1
    %mem_reg_inst = seq.firreg %6 clock %clock {firrtl.random_init_start = 34 : ui64} : i32
    %6 = comb.mux bin %ex_reg_valid, %ex_reg_inst, %mem_reg_inst : i32
    %mem_cp_valid = seq.firreg %2 clock %clock reset sync %reset, %false {firrtl.random_init_start = 66 : ui64} : i1
    %7 = comb.or bin %io_killm, %480 : i1
    %8 = comb.xor bin %mem_cp_valid, %true : i1
    %9 = comb.and bin %7, %8 {sv.namehint = "killm"} : i1
    %10 = comb.xor bin %9, %true : i1
    %11 = comb.or bin %10, %mem_cp_valid : i1
    %12 = comb.and bin %mem_reg_valid, %11 : i1
    %wb_reg_valid = seq.firreg %12 clock %clock reset sync %reset, %false {firrtl.random_init_start = 67 : ui64} : i1
    %wb_cp_valid = seq.firreg %mem_cp_valid clock %clock reset sync %reset, %false {firrtl.random_init_start = 68 : ui64} : i1
    %fp_decoder.io_sigs_cmd, %fp_decoder.io_sigs_ldst, %fp_decoder.io_sigs_wen, %fp_decoder.io_sigs_ren1, %fp_decoder.io_sigs_ren2, %fp_decoder.io_sigs_ren3, %fp_decoder.io_sigs_swap12, %fp_decoder.io_sigs_swap23, %fp_decoder.io_sigs_single, %fp_decoder.io_sigs_fromint, %fp_decoder.io_sigs_toint, %fp_decoder.io_sigs_fastpipe, %fp_decoder.io_sigs_fma, %fp_decoder.io_sigs_div, %fp_decoder.io_sigs_sqrt, %fp_decoder.io_sigs_wflags = hw.instance "fp_decoder" @FPUDecoder(io_inst: %io_inst: i32) -> (io_sigs_cmd: i5, io_sigs_ldst: i1, io_sigs_wen: i1, io_sigs_ren1: i1, io_sigs_ren2: i1, io_sigs_ren3: i1, io_sigs_swap12: i1, io_sigs_swap23: i1, io_sigs_single: i1, io_sigs_fromint: i1, io_sigs_toint: i1, io_sigs_fastpipe: i1, io_sigs_fma: i1, io_sigs_div: i1, io_sigs_sqrt: i1, io_sigs_wflags: i1) {sv.namehint = "fp_decoder.io_sigs_div"}
    %13 = seq.firreg %25 clock %clock {firrtl.random_init_start = 69 : ui64} : i5
    %14 = seq.firreg %26 clock %clock {firrtl.random_init_start = 74 : ui64} : i1
    %15 = seq.firreg %27 clock %clock {firrtl.random_init_start = 78 : ui64} : i1
    %16 = seq.firreg %28 clock %clock {firrtl.random_init_start = 80 : ui64} : i1
    %17 = seq.firreg %29 clock %clock {firrtl.random_init_start = 81 : ui64} : i1
    %18 = seq.firreg %30 clock %clock {firrtl.random_init_start = 82 : ui64} : i1
    %19 = seq.firreg %31 clock %clock {firrtl.random_init_start = 83 : ui64} : i1
    %20 = seq.firreg %32 clock %clock {firrtl.random_init_start = 84 : ui64} : i1
    %21 = seq.firreg %33 clock %clock {firrtl.random_init_start = 85 : ui64} : i1
    %22 = seq.firreg %34 clock %clock {firrtl.random_init_start = 86 : ui64} : i1
    %23 = seq.firreg %35 clock %clock {firrtl.random_init_start = 87 : ui64} : i1
    %24 = seq.firreg %36 clock %clock {firrtl.random_init_start = 88 : ui64} : i1
    %25 = comb.mux bin %io_valid, %fp_decoder.io_sigs_cmd, %13 : i5
    %26 = comb.mux bin %io_valid, %fp_decoder.io_sigs_ldst, %14 : i1
    %27 = comb.mux bin %io_valid, %fp_decoder.io_sigs_ren3, %15 : i1
    %28 = comb.mux bin %io_valid, %fp_decoder.io_sigs_swap23, %16 : i1
    %29 = comb.mux bin %io_valid, %fp_decoder.io_sigs_single, %17 : i1
    %30 = comb.mux bin %io_valid, %fp_decoder.io_sigs_fromint, %18 : i1
    %31 = comb.mux bin %io_valid, %fp_decoder.io_sigs_toint, %19 : i1
    %32 = comb.mux bin %io_valid, %fp_decoder.io_sigs_fastpipe, %20 : i1
    %33 = comb.mux bin %io_valid, %fp_decoder.io_sigs_fma, %21 : i1
    %34 = comb.mux bin %io_valid, %fp_decoder.io_sigs_div, %22 : i1
    %35 = comb.mux bin %io_valid, %fp_decoder.io_sigs_sqrt, %23 : i1
    %36 = comb.mux bin %io_valid, %fp_decoder.io_sigs_wflags, %24 : i1
    %37 = comb.extract %io_cp_req_bits_cmd from 0 : (i5) -> i4
    %38 = comb.extract %13 from 0 : (i5) -> i4
    %39 = comb.mux %2, %37, %38 {sv.namehint = "ex_ctrl_cmd"} : i4
    %40 = comb.mux bin %2, %io_cp_req_bits_single, %17 {sv.namehint = "ex_ctrl_single"} : i1
    %41 = comb.mux bin %2, %io_cp_req_bits_fromint, %18 {sv.namehint = "ex_ctrl_fromint"} : i1
    %42 = comb.mux bin %2, %io_cp_req_bits_toint, %19 {sv.namehint = "ex_ctrl_toint"} : i1
    %43 = comb.mux bin %2, %io_cp_req_bits_fastpipe, %20 {sv.namehint = "ex_ctrl_fastpipe"} : i1
    %44 = comb.mux bin %2, %io_cp_req_bits_fma, %21 {sv.namehint = "ex_ctrl_fma"} : i1
    %45 = comb.mux bin %2, %io_cp_req_bits_div, %22 {sv.namehint = "ex_ctrl_div"} : i1
    %46 = comb.mux bin %2, %io_cp_req_bits_sqrt, %23 {sv.namehint = "ex_ctrl_sqrt"} : i1
    %47 = comb.mux bin %2, %io_cp_req_bits_wflags, %24 {sv.namehint = "ex_ctrl_wflags"} : i1
    %mem_ctrl_single = seq.firreg %48 clock %clock {firrtl.random_init_start = 101 : ui64} : i1
    %mem_ctrl_fromint = seq.firreg %49 clock %clock {firrtl.random_init_start = 102 : ui64} : i1
    %mem_ctrl_toint = seq.firreg %50 clock %clock {firrtl.random_init_start = 103 : ui64} : i1
    %mem_ctrl_fastpipe = seq.firreg %51 clock %clock {firrtl.random_init_start = 104 : ui64} : i1
    %mem_ctrl_fma = seq.firreg %52 clock %clock {firrtl.random_init_start = 105 : ui64} : i1
    %mem_ctrl_div = seq.firreg %53 clock %clock {firrtl.random_init_start = 106 : ui64} : i1
    %mem_ctrl_sqrt = seq.firreg %54 clock %clock {firrtl.random_init_start = 107 : ui64, sv.namehint = "DivSqrtRecF64.io_sqrtOp"} : i1
    %mem_ctrl_wflags = seq.firreg %55 clock %clock {firrtl.random_init_start = 108 : ui64} : i1
    %48 = comb.mux bin %0, %40, %mem_ctrl_single : i1
    %49 = comb.mux bin %0, %41, %mem_ctrl_fromint : i1
    %50 = comb.mux bin %0, %42, %mem_ctrl_toint : i1
    %51 = comb.mux bin %0, %43, %mem_ctrl_fastpipe : i1
    %52 = comb.mux bin %0, %44, %mem_ctrl_fma : i1
    %53 = comb.mux bin %0, %45, %mem_ctrl_div : i1
    %54 = comb.mux bin %0, %46, %mem_ctrl_sqrt : i1
    %55 = comb.mux bin %0, %47, %mem_ctrl_wflags : i1
    %wb_ctrl_toint = seq.firreg %56 clock %clock {firrtl.random_init_start = 123 : ui64} : i1
    %56 = comb.mux bin %mem_reg_valid, %mem_ctrl_toint, %wb_ctrl_toint : i1
    %load_wb = seq.firreg %io_dmem_resp_val clock %clock {firrtl.random_init_start = 129 : ui64} : i1
    %57 = comb.extract %io_dmem_resp_type from 0 : (i3) -> i1
    %58 = comb.xor bin %57, %true : i1
    %load_wb_single = seq.firreg %59 clock %clock {firrtl.random_init_start = 130 : ui64} : i1
    %59 = comb.mux bin %io_dmem_resp_val, %58, %load_wb_single : i1
    %load_wb_data = seq.firreg %60 clock %clock {firrtl.random_init_start = 131 : ui64} : i64
    %60 = comb.mux bin %io_dmem_resp_val, %io_dmem_resp_data, %load_wb_data : i64
    %load_wb_tag = seq.firreg %61 clock %clock {firrtl.random_init_start = 195 : ui64} : i5
    %61 = comb.mux bin %io_dmem_resp_val, %io_dmem_resp_tag, %load_wb_tag : i5
    %62 = comb.extract %load_wb_data from 31 : (i64) -> i1
    %63 = comb.extract %load_wb_data from 23 : (i64) -> i8
    %64 = comb.extract %load_wb_data from 0 : (i64) -> i23
    %65 = comb.icmp bin eq %63, %c0_i8 : i8
    %66 = comb.icmp bin ne %64, %c0_i23 : i23
    %67 = comb.xor bin %66, %true : i1
    %68 = comb.and bin %65, %67 : i1
    %69 = comb.extract %load_wb_data from 7 : (i64) -> i16
    %70 = comb.icmp bin ne %69, %c0_i16 : i16
    %71 = comb.extract %load_wb_data from 15 : (i64) -> i8
    %72 = comb.icmp bin ne %71, %c0_i8 : i8
    %73 = comb.extract %load_wb_data from 19 : (i64) -> i4
    %74 = comb.icmp bin ne %73, %c0_i4 : i4
    %75 = comb.extract %load_wb_data from 22 : (i64) -> i1
    %76 = comb.extract %load_wb_data from 21 : (i64) -> i1
    %77 = comb.extract %load_wb_data from 20 : (i64) -> i1
    %78 = comb.concat %false, %77 : i1, i1
    %79 = comb.mux bin %76, %c-2_i2, %78 : i2
    %80 = comb.mux bin %75, %c-1_i2, %79 : i2
    %81 = comb.extract %load_wb_data from 18 : (i64) -> i1
    %82 = comb.extract %load_wb_data from 17 : (i64) -> i1
    %83 = comb.extract %load_wb_data from 16 : (i64) -> i1
    %84 = comb.concat %false, %83 : i1, i1
    %85 = comb.mux bin %82, %c-2_i2, %84 : i2
    %86 = comb.mux bin %81, %c-1_i2, %85 : i2
    %87 = comb.mux bin %74, %80, %86 : i2
    %88 = comb.concat %74, %87 : i1, i2
    %89 = comb.extract %load_wb_data from 11 : (i64) -> i4
    %90 = comb.icmp bin ne %89, %c0_i4 : i4
    %91 = comb.extract %load_wb_data from 14 : (i64) -> i1
    %92 = comb.extract %load_wb_data from 13 : (i64) -> i1
    %93 = comb.extract %load_wb_data from 12 : (i64) -> i1
    %94 = comb.concat %false, %93 : i1, i1
    %95 = comb.mux bin %92, %c-2_i2, %94 : i2
    %96 = comb.mux bin %91, %c-1_i2, %95 : i2
    %97 = comb.extract %load_wb_data from 10 : (i64) -> i1
    %98 = comb.extract %load_wb_data from 9 : (i64) -> i1
    %99 = comb.extract %load_wb_data from 8 : (i64) -> i1
    %100 = comb.concat %false, %99 : i1, i1
    %101 = comb.mux bin %98, %c-2_i2, %100 : i2
    %102 = comb.mux bin %97, %c-1_i2, %101 : i2
    %103 = comb.mux bin %90, %96, %102 : i2
    %104 = comb.concat %90, %103 : i1, i2
    %105 = comb.mux bin %72, %88, %104 : i3
    %106 = comb.concat %72, %105 : i1, i3
    %107 = comb.extract %load_wb_data from 0 : (i64) -> i7
    %108 = comb.icmp bin ne %107, %c0_i7 : i7
    %109 = comb.extract %load_wb_data from 3 : (i64) -> i4
    %110 = comb.icmp bin ne %109, %c0_i4 : i4
    %111 = comb.extract %load_wb_data from 6 : (i64) -> i1
    %112 = comb.extract %load_wb_data from 5 : (i64) -> i1
    %113 = comb.extract %load_wb_data from 4 : (i64) -> i1
    %114 = comb.concat %false, %113 : i1, i1
    %115 = comb.mux bin %112, %c-2_i2, %114 : i2
    %116 = comb.mux bin %111, %c-1_i2, %115 : i2
    %117 = comb.extract %load_wb_data from 2 : (i64) -> i1
    %118 = comb.extract %load_wb_data from 1 : (i64) -> i1
    %119 = comb.extract %load_wb_data from 0 : (i64) -> i1
    %120 = comb.concat %false, %119 : i1, i1
    %121 = comb.mux bin %118, %c-2_i2, %120 : i2
    %122 = comb.mux bin %117, %c-1_i2, %121 : i2
    %123 = comb.mux bin %110, %116, %122 : i2
    %124 = comb.concat %110, %123 : i1, i2
    %125 = comb.mux bin %108, %124, %c0_i3 : i3
    %126 = comb.concat %108, %125 : i1, i3
    %127 = comb.mux bin %70, %106, %126 : i4
    %128 = comb.concat %70, %127 : i1, i4
    %129 = comb.xor bin %128, %c-1_i5 : i5
    %130 = comb.concat %c0_i31, %64 : i31, i23
    %131 = comb.concat %c0_i49, %129 : i49, i5
    %132 = comb.shl bin %130, %131 : i54
    %133 = comb.extract %132 from 0 : (i54) -> i22
    %134 = comb.concat %133, %false : i22, i1
    %135 = comb.concat %c-1_i4, %70, %127 : i4, i1, i4
    %136 = comb.concat %false, %63 : i1, i8
    %137 = comb.mux bin %65, %135, %136 : i9
    %138 = comb.mux bin %65, %c-2_i2, %c1_i2 : i2
    %139 = comb.concat %c32_i7, %138 : i7, i2
    %140 = comb.add %137, %139 : i9
    %141 = comb.extract %140 from 7 : (i9) -> i2
    %142 = comb.icmp bin eq %141, %c-1_i2 : i2
    %143 = comb.and bin %142, %66 : i1
    %144 = comb.replicate %68 : (i1) -> i3
    %145 = comb.xor %144, %c-1_i3 : i3
    %146 = comb.concat %143, %c0_i6 : i1, i6
    %147 = comb.extract %145 from 1 : (i3) -> i2
    %148 = comb.and bin %141, %147 : i2
    %149 = comb.extract %140 from 0 : (i9) -> i7
    %150 = comb.extract %145 from 0 : (i3) -> i1
    %151 = comb.concat %150, %c-1_i6 : i1, i6
    %152 = comb.and bin %149, %151 : i7
    %153 = comb.or bin %152, %146 : i7
    %154 = comb.mux bin %65, %134, %64 : i23
    %155 = comb.extract %load_wb_data from 63 : (i64) -> i1
    %156 = comb.extract %load_wb_data from 52 : (i64) -> i11
    %157 = comb.extract %load_wb_data from 0 : (i64) -> i52
    %158 = comb.icmp bin eq %156, %c0_i11 : i11
    %159 = comb.icmp bin ne %157, %c0_i52 : i52
    %160 = comb.xor bin %159, %true : i1
    %161 = comb.and bin %158, %160 : i1
    %162 = comb.extract %load_wb_data from 20 : (i64) -> i32
    %163 = comb.icmp bin ne %162, %c0_i32 : i32
    %164 = comb.extract %load_wb_data from 36 : (i64) -> i16
    %165 = comb.icmp bin ne %164, %c0_i16 : i16
    %166 = comb.extract %load_wb_data from 44 : (i64) -> i8
    %167 = comb.icmp bin ne %166, %c0_i8 : i8
    %168 = comb.extract %load_wb_data from 48 : (i64) -> i4
    %169 = comb.icmp bin ne %168, %c0_i4 : i4
    %170 = comb.extract %load_wb_data from 51 : (i64) -> i1
    %171 = comb.extract %load_wb_data from 50 : (i64) -> i1
    %172 = comb.extract %load_wb_data from 49 : (i64) -> i1
    %173 = comb.concat %false, %172 : i1, i1
    %174 = comb.mux bin %171, %c-2_i2, %173 : i2
    %175 = comb.mux bin %170, %c-1_i2, %174 : i2
    %176 = comb.extract %load_wb_data from 47 : (i64) -> i1
    %177 = comb.extract %load_wb_data from 46 : (i64) -> i1
    %178 = comb.extract %load_wb_data from 45 : (i64) -> i1
    %179 = comb.concat %false, %178 : i1, i1
    %180 = comb.mux bin %177, %c-2_i2, %179 : i2
    %181 = comb.mux bin %176, %c-1_i2, %180 : i2
    %182 = comb.mux bin %169, %175, %181 : i2
    %183 = comb.concat %169, %182 : i1, i2
    %184 = comb.extract %load_wb_data from 40 : (i64) -> i4
    %185 = comb.icmp bin ne %184, %c0_i4 : i4
    %186 = comb.extract %load_wb_data from 43 : (i64) -> i1
    %187 = comb.extract %load_wb_data from 42 : (i64) -> i1
    %188 = comb.extract %load_wb_data from 41 : (i64) -> i1
    %189 = comb.concat %false, %188 : i1, i1
    %190 = comb.mux bin %187, %c-2_i2, %189 : i2
    %191 = comb.mux bin %186, %c-1_i2, %190 : i2
    %192 = comb.extract %load_wb_data from 39 : (i64) -> i1
    %193 = comb.extract %load_wb_data from 38 : (i64) -> i1
    %194 = comb.extract %load_wb_data from 37 : (i64) -> i1
    %195 = comb.concat %false, %194 : i1, i1
    %196 = comb.mux bin %193, %c-2_i2, %195 : i2
    %197 = comb.mux bin %192, %c-1_i2, %196 : i2
    %198 = comb.mux bin %185, %191, %197 : i2
    %199 = comb.concat %185, %198 : i1, i2
    %200 = comb.mux bin %167, %183, %199 : i3
    %201 = comb.concat %167, %200 : i1, i3
    %202 = comb.extract %load_wb_data from 28 : (i64) -> i8
    %203 = comb.icmp bin ne %202, %c0_i8 : i8
    %204 = comb.extract %load_wb_data from 32 : (i64) -> i4
    %205 = comb.icmp bin ne %204, %c0_i4 : i4
    %206 = comb.extract %load_wb_data from 35 : (i64) -> i1
    %207 = comb.extract %load_wb_data from 34 : (i64) -> i1
    %208 = comb.extract %load_wb_data from 33 : (i64) -> i1
    %209 = comb.concat %false, %208 : i1, i1
    %210 = comb.mux bin %207, %c-2_i2, %209 : i2
    %211 = comb.mux bin %206, %c-1_i2, %210 : i2
    %212 = comb.extract %load_wb_data from 31 : (i64) -> i1
    %213 = comb.extract %load_wb_data from 30 : (i64) -> i1
    %214 = comb.extract %load_wb_data from 29 : (i64) -> i1
    %215 = comb.concat %false, %214 : i1, i1
    %216 = comb.mux bin %213, %c-2_i2, %215 : i2
    %217 = comb.mux bin %212, %c-1_i2, %216 : i2
    %218 = comb.mux bin %205, %211, %217 : i2
    %219 = comb.concat %205, %218 : i1, i2
    %220 = comb.extract %load_wb_data from 24 : (i64) -> i4
    %221 = comb.icmp bin ne %220, %c0_i4 : i4
    %222 = comb.extract %load_wb_data from 27 : (i64) -> i1
    %223 = comb.extract %load_wb_data from 26 : (i64) -> i1
    %224 = comb.extract %load_wb_data from 25 : (i64) -> i1
    %225 = comb.concat %false, %224 : i1, i1
    %226 = comb.mux bin %223, %c-2_i2, %225 : i2
    %227 = comb.mux bin %222, %c-1_i2, %226 : i2
    %228 = comb.extract %load_wb_data from 23 : (i64) -> i1
    %229 = comb.extract %load_wb_data from 22 : (i64) -> i1
    %230 = comb.extract %load_wb_data from 21 : (i64) -> i1
    %231 = comb.concat %false, %230 : i1, i1
    %232 = comb.mux bin %229, %c-2_i2, %231 : i2
    %233 = comb.mux bin %228, %c-1_i2, %232 : i2
    %234 = comb.mux bin %221, %227, %233 : i2
    %235 = comb.concat %221, %234 : i1, i2
    %236 = comb.mux bin %203, %219, %235 : i3
    %237 = comb.concat %203, %236 : i1, i3
    %238 = comb.mux bin %165, %201, %237 : i4
    %239 = comb.concat %165, %238 : i1, i4
    %240 = comb.extract %load_wb_data from 4 : (i64) -> i16
    %241 = comb.icmp bin ne %240, %c0_i16 : i16
    %242 = comb.extract %load_wb_data from 12 : (i64) -> i8
    %243 = comb.icmp bin ne %242, %c0_i8 : i8
    %244 = comb.extract %load_wb_data from 16 : (i64) -> i4
    %245 = comb.icmp bin ne %244, %c0_i4 : i4
    %246 = comb.extract %load_wb_data from 19 : (i64) -> i1
    %247 = comb.extract %load_wb_data from 18 : (i64) -> i1
    %248 = comb.extract %load_wb_data from 17 : (i64) -> i1
    %249 = comb.concat %false, %248 : i1, i1
    %250 = comb.mux bin %247, %c-2_i2, %249 : i2
    %251 = comb.mux bin %246, %c-1_i2, %250 : i2
    %252 = comb.extract %load_wb_data from 15 : (i64) -> i1
    %253 = comb.extract %load_wb_data from 14 : (i64) -> i1
    %254 = comb.extract %load_wb_data from 13 : (i64) -> i1
    %255 = comb.concat %false, %254 : i1, i1
    %256 = comb.mux bin %253, %c-2_i2, %255 : i2
    %257 = comb.mux bin %252, %c-1_i2, %256 : i2
    %258 = comb.mux bin %245, %251, %257 : i2
    %259 = comb.concat %245, %258 : i1, i2
    %260 = comb.extract %load_wb_data from 8 : (i64) -> i4
    %261 = comb.icmp bin ne %260, %c0_i4 : i4
    %262 = comb.extract %load_wb_data from 11 : (i64) -> i1
    %263 = comb.extract %load_wb_data from 10 : (i64) -> i1
    %264 = comb.extract %load_wb_data from 9 : (i64) -> i1
    %265 = comb.concat %false, %264 : i1, i1
    %266 = comb.mux bin %263, %c-2_i2, %265 : i2
    %267 = comb.mux bin %262, %c-1_i2, %266 : i2
    %268 = comb.extract %load_wb_data from 7 : (i64) -> i1
    %269 = comb.extract %load_wb_data from 6 : (i64) -> i1
    %270 = comb.extract %load_wb_data from 5 : (i64) -> i1
    %271 = comb.concat %false, %270 : i1, i1
    %272 = comb.mux bin %269, %c-2_i2, %271 : i2
    %273 = comb.mux bin %268, %c-1_i2, %272 : i2
    %274 = comb.mux bin %261, %267, %273 : i2
    %275 = comb.concat %261, %274 : i1, i2
    %276 = comb.mux bin %243, %259, %275 : i3
    %277 = comb.concat %243, %276 : i1, i3
    %278 = comb.extract %load_wb_data from 0 : (i64) -> i4
    %279 = comb.icmp bin ne %278, %c0_i4 : i4
    %280 = comb.extract %load_wb_data from 0 : (i64) -> i4
    %281 = comb.icmp bin ne %280, %c0_i4 : i4
    %282 = comb.extract %load_wb_data from 3 : (i64) -> i1
    %283 = comb.extract %load_wb_data from 2 : (i64) -> i1
    %284 = comb.extract %load_wb_data from 1 : (i64) -> i1
    %285 = comb.concat %false, %284 : i1, i1
    %286 = comb.mux bin %283, %c-2_i2, %285 : i2
    %287 = comb.mux bin %282, %c-1_i2, %286 : i2
    %288 = comb.mux bin %281, %287, %c0_i2 : i2
    %289 = comb.concat %281, %288 : i1, i2
    %290 = comb.mux bin %279, %289, %c0_i3 : i3
    %291 = comb.concat %279, %290 : i1, i3
    %292 = comb.mux bin %241, %277, %291 : i4
    %293 = comb.concat %241, %292 : i1, i4
    %294 = comb.mux bin %163, %239, %293 : i5
    %295 = comb.concat %163, %294 : i1, i5
    %296 = comb.xor bin %295, %c-1_i6 : i6
    %297 = comb.concat %c0_i63, %157 : i63, i52
    %298 = comb.concat %c0_i109, %296 : i109, i6
    %299 = comb.shl bin %297, %298 : i115
    %300 = comb.extract %299 from 0 : (i115) -> i51
    %301 = comb.concat %300, %false : i51, i1
    %302 = comb.concat %c-1_i6, %163, %294 : i6, i1, i5
    %303 = comb.concat %false, %156 : i1, i11
    %304 = comb.mux bin %158, %302, %303 : i12
    %305 = comb.mux bin %158, %c-2_i2, %c1_i2 : i2
    %306 = comb.concat %c256_i10, %305 : i10, i2
    %307 = comb.add %304, %306 : i12
    %308 = comb.extract %307 from 10 : (i12) -> i2
    %309 = comb.icmp bin eq %308, %c-1_i2 : i2
    %310 = comb.and bin %309, %159 : i1
    %311 = comb.replicate %161 : (i1) -> i3
    %312 = comb.xor %311, %c-1_i3 : i3
    %313 = comb.concat %310, %c0_i9 : i1, i9
    %314 = comb.extract %312 from 1 : (i3) -> i2
    %315 = comb.and bin %308, %314 : i2
    %316 = comb.extract %307 from 0 : (i12) -> i10
    %317 = comb.extract %312 from 0 : (i3) -> i1
    %318 = comb.concat %317, %c-1_i9 : i1, i9
    %319 = comb.and bin %316, %318 : i10
    %320 = comb.or bin %319, %313 : i10
    %321 = comb.mux bin %158, %301, %157 : i52
    %322 = comb.concat %155, %315, %320, %321 : i1, i2, i10, i52
    %323 = comb.concat %c1879179264_i32, %62, %148, %153, %154 : i32, i1, i2, i7, i23
    %324 = comb.mux bin %load_wb_single, %323, %322 {sv.namehint = "load_wb_data_recoded"} : i65
    %regfile = seq.firmem 0, 1, undefined, port_order {prefix = ""} : <32 x 65>
    seq.firmem.write_port %regfile[%438] = %448, clock %clock enable %455 : <32 x 65>
    seq.firmem.write_port %regfile[%load_wb_tag] = %324, clock %clock enable %load_wb : <32 x 65>
    %325 = seq.firmem.read_port %regfile[%ex_ra1], clock %clock : <32 x 65>
    %326 = seq.firmem.read_port %regfile[%ex_ra2], clock %clock : <32 x 65>
    %327 = seq.firmem.read_port %regfile[%ex_ra3], clock %clock : <32 x 65>
    %ex_ra1 = seq.firreg %337 clock %clock {firrtl.random_init_start = 200 : ui64} : i5
    %ex_ra2 = seq.firreg %343 clock %clock {firrtl.random_init_start = 205 : ui64} : i5
    %ex_ra3 = seq.firreg %346 clock %clock {firrtl.random_init_start = 210 : ui64} : i5
    %328 = comb.xor bin %fp_decoder.io_sigs_swap12, %true : i1
    %329 = comb.extract %io_inst from 15 : (i32) -> i5
    %330 = comb.and bin %fp_decoder.io_sigs_ren1, %328 : i1
    %331 = comb.mux bin %330, %329, %ex_ra1 : i5
    %332 = comb.and bin %fp_decoder.io_sigs_ren1, %fp_decoder.io_sigs_swap12 : i1
    %333 = comb.mux bin %332, %329, %ex_ra2 : i5
    %334 = comb.extract %io_inst from 20 : (i32) -> i5
    %335 = comb.and bin %fp_decoder.io_sigs_ren2, %fp_decoder.io_sigs_swap12 : i1
    %336 = comb.mux bin %335, %334, %331 : i5
    %337 = comb.mux bin %io_valid, %336, %ex_ra1 : i5
    %338 = comb.and bin %fp_decoder.io_sigs_ren2, %fp_decoder.io_sigs_swap23 : i1
    %339 = comb.mux bin %338, %334, %ex_ra3 : i5
    %340 = comb.xor bin %fp_decoder.io_sigs_swap23, %true : i1
    %341 = comb.and bin %fp_decoder.io_sigs_ren2, %328, %340 : i1
    %342 = comb.mux bin %341, %334, %333 : i5
    %343 = comb.mux bin %io_valid, %342, %ex_ra2 : i5
    %344 = comb.extract %io_inst from 27 : (i32) -> i5
    %345 = comb.mux bin %fp_decoder.io_sigs_ren3, %344, %339 : i5
    %346 = comb.mux bin %io_valid, %345, %ex_ra3 : i5
    %347 = comb.extract %ex_reg_inst from 12 : (i32) -> i3
    %348 = comb.icmp bin eq %347, %c-1_i3 : i3
    %349 = comb.mux bin %348, %io_fcsr_rm, %347 {sv.namehint = "ex_rm"} : i3
    %350 = comb.extract %ex_reg_inst from 20 : (i32) -> i2
    %351 = comb.mux bin %2, %io_cp_req_bits_cmd, %13 {sv.namehint = "req_cmd"} : i5
    %352 = comb.mux bin %2, %io_cp_req_bits_ldst, %14 {sv.namehint = "req_ldst"} : i1
    %353 = comb.mux bin %2, %io_cp_req_bits_ren3, %15 {sv.namehint = "req_ren3"} : i1
    %354 = comb.mux bin %2, %io_cp_req_bits_swap23, %16 {sv.namehint = "req_swap23"} : i1
    %355 = comb.mux bin %2, %io_cp_req_bits_single, %17 {sv.namehint = "req_single"} : i1
    %356 = comb.mux bin %2, %io_cp_req_bits_rm, %349 {sv.namehint = "req_rm"} : i3
    %357 = comb.mux bin %2, %io_cp_req_bits_typ, %350 {sv.namehint = "req_typ"} : i2
    %358 = comb.mux bin %2, %io_cp_req_bits_in1, %325 {sv.namehint = "req_in1"} : i65
    %359 = comb.mux bin %io_cp_req_bits_swap23, %io_cp_req_bits_in3, %io_cp_req_bits_in2 : i65
    %360 = comb.mux bin %2, %359, %326 {sv.namehint = "req_in2"} : i65
    %361 = comb.mux bin %io_cp_req_bits_swap23, %io_cp_req_bits_in2, %io_cp_req_bits_in3 : i65
    %362 = comb.mux bin %2, %361, %327 {sv.namehint = "req_in3"} : i65
    %sfma.io_out_bits_data, %sfma.io_out_bits_exc = hw.instance "sfma" @FPUFMAPipe(clock: %clock: !seq.clock, reset: %reset: i1, io_in_valid: %364: i1, io_in_bits_cmd: %351: i5, io_in_bits_ren3: %353: i1, io_in_bits_swap23: %354: i1, io_in_bits_rm: %356: i3, io_in_bits_in1: %358: i65, io_in_bits_in2: %360: i65, io_in_bits_in3: %362: i65) -> (io_out_bits_data: i65, io_out_bits_exc: i5) {sv.namehint = "sfma.io_out_bits_exc"}
    %363 = comb.and bin %0, %44 : i1
    %364 = comb.and bin %363, %40 {sv.namehint = "sfma.io_in_valid"} : i1
    %fpiu.io_as_double_rm, %fpiu.io_as_double_in1, %fpiu.io_as_double_in2, %fpiu.io_out_valid, %fpiu.io_out_bits_lt, %fpiu.io_out_bits_store, %fpiu.io_out_bits_toint, %fpiu.io_out_bits_exc = hw.instance "fpiu" @FPToInt(clock: %clock: !seq.clock, io_in_valid: %370: i1, io_in_bits_cmd: %351: i5, io_in_bits_ldst: %352: i1, io_in_bits_single: %355: i1, io_in_bits_rm: %356: i3, io_in_bits_typ: %357: i2, io_in_bits_in1: %358: i65, io_in_bits_in2: %360: i65) -> (io_as_double_rm: i3, io_as_double_in1: i65, io_as_double_in2: i65, io_out_valid: i1, io_out_bits_lt: i1, io_out_bits_store: i64, io_out_bits_toint: i64, io_out_bits_exc: i5) {sv.namehint = "fpmu.io_lt"}
    %365 = comb.extract %39 from 2 : (i4) -> i2
    %366 = comb.extract %39 from 0 : (i4) -> i1
    %367 = comb.concat %365, %366 : i2, i1
    %368 = comb.icmp bin eq %367, %c3_i3 : i3
    %369 = comb.or bin %42, %45, %46, %368 : i1
    %370 = comb.and bin %0, %369 {sv.namehint = "fpiu.io_in_valid"} : i1
    %371 = comb.and bin %fpiu.io_out_valid, %mem_cp_valid, %mem_ctrl_toint : i1
    %372 = comb.concat %false, %fpiu.io_out_bits_toint : i1, i64
    %373 = comb.mux bin %371, %372, %c0_i65 : i65
    %ifpu.io_out_bits_data, %ifpu.io_out_bits_exc = hw.instance "ifpu" @IntToFP(clock: %clock: !seq.clock, reset: %reset: i1, io_in_valid: %374: i1, io_in_bits_cmd: %351: i5, io_in_bits_single: %355: i1, io_in_bits_rm: %356: i3, io_in_bits_typ: %357: i2, io_in_bits_in1: %376: i65) -> (io_out_bits_data: i65, io_out_bits_exc: i5) {sv.namehint = "ifpu.io_out_bits_exc"}
    %374 = comb.and bin %0, %41 {sv.namehint = "ifpu.io_in_valid"} : i1
    %375 = comb.concat %false, %io_fromint_data : i1, i64
    %376 = comb.mux bin %2, %io_cp_req_bits_in1, %375 {sv.namehint = "ifpu.io_in_bits_in1"} : i65
    %fpmu.io_out_bits_data, %fpmu.io_out_bits_exc = hw.instance "fpmu" @FPToFP(clock: %clock: !seq.clock, reset: %reset: i1, io_in_valid: %377: i1, io_in_bits_cmd: %351: i5, io_in_bits_single: %355: i1, io_in_bits_rm: %356: i3, io_in_bits_in1: %358: i65, io_in_bits_in2: %360: i65, io_lt: %fpiu.io_out_bits_lt: i1) -> (io_out_bits_data: i65, io_out_bits_exc: i5) {sv.namehint = "fpmu.io_out_bits_exc"}
    %377 = comb.and bin %0, %43 {sv.namehint = "fpmu.io_in_valid"} : i1
    %divSqrt_wen = seq.firreg %510 clock %clock {firrtl.random_init_start = 215 : ui64} : i1
    %divSqrt_waddr = seq.firreg %507 clock %clock {firrtl.random_init_start = 216 : ui64} : i5
    %divSqrt_single = seq.firreg %506 clock %clock {firrtl.random_init_start = 221 : ui64} : i1
    %divSqrt_in_flight = seq.firreg %513 clock %clock reset sync %reset, %false {firrtl.random_init_start = 222 : ui64} : i1
    %divSqrt_killed = seq.firreg %505 clock %clock {firrtl.random_init_start = 223 : ui64} : i1
    %FPUFMAPipe.io_out_bits_data, %FPUFMAPipe.io_out_bits_exc = hw.instance "FPUFMAPipe" @FPUFMAPipe_1(clock: %clock: !seq.clock, reset: %reset: i1, io_in_valid: %379: i1, io_in_bits_cmd: %351: i5, io_in_bits_ren3: %353: i1, io_in_bits_swap23: %354: i1, io_in_bits_rm: %356: i3, io_in_bits_in1: %358: i65, io_in_bits_in2: %360: i65, io_in_bits_in3: %362: i65) -> (io_out_bits_data: i65, io_out_bits_exc: i5) {sv.namehint = "FPUFMAPipe.io_out_bits_exc"}
    %378 = comb.xor bin %40, %true : i1
    %379 = comb.and bin %363, %378 {sv.namehint = "FPUFMAPipe.io_in_valid"} : i1
    %380 = comb.and bin %mem_ctrl_fma, %mem_ctrl_single : i1
    %381 = comb.xor bin %mem_ctrl_single, %true : i1
    %382 = comb.and bin %mem_ctrl_fma, %381 : i1
    %383 = comb.or bin %mem_ctrl_fastpipe, %mem_ctrl_fromint : i1
    %wen = seq.firreg %417 clock %clock reset sync %reset, %c0_i3 {firrtl.random_init_start = 224 : ui64} : i3
    %wbInfo_0_rd = seq.firreg %427 clock %clock {firrtl.random_init_start = 227 : ui64} : i5
    %wbInfo_0_single = seq.firreg %421 clock %clock {firrtl.random_init_start = 232 : ui64} : i1
    %wbInfo_0_cp = seq.firreg %420 clock %clock {firrtl.random_init_start = 233 : ui64} : i1
    %wbInfo_0_pipeid = seq.firreg %425 clock %clock {firrtl.random_init_start = 234 : ui64} : i2
    %wbInfo_1_rd = seq.firreg %432 clock %clock {firrtl.random_init_start = 236 : ui64} : i5
    %wbInfo_1_single = seq.firreg %430 clock %clock {firrtl.random_init_start = 241 : ui64} : i1
    %wbInfo_1_cp = seq.firreg %429 clock %clock {firrtl.random_init_start = 242 : ui64} : i1
    %wbInfo_1_pipeid = seq.firreg %431 clock %clock {firrtl.random_init_start = 243 : ui64} : i2
    %wbInfo_2_rd = seq.firreg %437 clock %clock {firrtl.random_init_start = 245 : ui64} : i5
    %wbInfo_2_single = seq.firreg %435 clock %clock {firrtl.random_init_start = 250 : ui64} : i1
    %wbInfo_2_cp = seq.firreg %434 clock %clock {firrtl.random_init_start = 251 : ui64} : i1
    %wbInfo_2_pipeid = seq.firreg %436 clock %clock {firrtl.random_init_start = 252 : ui64} : i2
    %384 = comb.or bin %mem_ctrl_fma, %mem_ctrl_fastpipe, %mem_ctrl_fromint : i1
    %385 = comb.and bin %mem_reg_valid, %384 {sv.namehint = "mem_wen"} : i1
    %386 = comb.and bin %44, %40 : i1
    %387 = comb.or %43, %41 : i1
    %388 = comb.concat %386, %387 : i1, i1
    %389 = comb.concat %382, %380 : i1, i1
    %390 = comb.and %388, %389 : i2
    %391 = comb.icmp bin ne %390, %c0_i2 : i2
    %392 = comb.and bin %385, %391 : i1
    %393 = comb.or %43, %41 : i1
    %394 = comb.extract %wen from 2 : (i3) -> i1
    %395 = comb.and %393, %394 : i1
    %396 = comb.concat %392, %395 : i1, i1
    %397 = comb.icmp bin ne %396, %c0_i2 : i2
    %write_port_busy = seq.firreg %398 clock %clock {firrtl.random_init_start = 254 : ui64} : i1
    %398 = comb.mux bin %0, %397, %write_port_busy : i1
    %399 = comb.extract %wen from 1 : (i3) -> i1
    %400 = comb.mux bin %399, %wbInfo_1_rd, %wbInfo_0_rd : i5
    %401 = comb.mux bin %399, %wbInfo_1_single, %wbInfo_0_single : i1
    %402 = comb.mux bin %399, %wbInfo_1_cp, %wbInfo_0_cp : i1
    %403 = comb.mux bin %399, %wbInfo_1_pipeid, %wbInfo_0_pipeid : i2
    %404 = comb.extract %wen from 2 : (i3) -> i1
    %405 = comb.mux bin %404, %wbInfo_2_rd, %wbInfo_1_rd : i5
    %406 = comb.mux bin %404, %wbInfo_2_single, %wbInfo_1_single : i1
    %407 = comb.mux bin %404, %wbInfo_2_cp, %wbInfo_1_cp : i1
    %408 = comb.mux bin %404, %wbInfo_2_pipeid, %wbInfo_1_pipeid : i2
    %409 = comb.extract %wen from 1 : (i3) -> i2
    %410 = comb.concat %false, %409 : i1, i2
    %411 = comb.extract %wen from 2 : (i3) -> i1
    %412 = comb.or %411, %380 : i1
    %413 = comb.extract %wen from 1 : (i3) -> i1
    %414 = comb.or %413, %383 : i1
    %415 = comb.concat %382, %412, %414 : i1, i1, i1
    %416 = comb.and bin %385, %10 : i1
    %417 = comb.mux bin %416, %415, %410 : i3
    %418 = comb.xor bin %write_port_busy, %true : i1
    %419 = comb.and bin %385, %418, %383 : i1
    %420 = comb.mux bin %419, %mem_cp_valid, %402 : i1
    %421 = comb.mux bin %419, %mem_ctrl_single, %401 : i1
    %422 = comb.replicate %382 : (i1) -> i2
    %423 = comb.concat %380, %mem_ctrl_fromint : i1, i1
    %424 = comb.or bin %423, %422 : i2
    %425 = comb.mux bin %419, %424, %403 : i2
    %426 = comb.extract %mem_reg_inst from 7 : (i32) -> i5
    %427 = comb.mux bin %419, %426, %400 : i5
    %428 = comb.and bin %385, %418, %380 : i1
    %429 = comb.mux bin %428, %mem_cp_valid, %407 : i1
    %430 = comb.mux bin %428, %mem_ctrl_single, %406 : i1
    %431 = comb.mux bin %428, %424, %408 : i2
    %432 = comb.mux bin %428, %426, %405 : i5
    %433 = comb.and bin %385, %418, %382 : i1
    %434 = comb.mux bin %433, %mem_cp_valid, %wbInfo_2_cp : i1
    %435 = comb.mux bin %433, %mem_ctrl_single, %wbInfo_2_single : i1
    %436 = comb.mux bin %433, %424, %wbInfo_2_pipeid : i2
    %437 = comb.mux bin %433, %426, %wbInfo_2_rd : i5
    %438 = comb.mux bin %divSqrt_wen, %divSqrt_waddr, %wbInfo_0_rd {sv.namehint = "waddr"} : i5
    %439 = comb.extract %wbInfo_0_pipeid from 0 : (i2) -> i1
    %440 = comb.extract %wbInfo_0_pipeid from 1 : (i2) -> i1
    %441 = comb.mux bin %439, %FPUFMAPipe.io_out_bits_data, %sfma.io_out_bits_data : i65
    %442 = comb.mux bin %439, %ifpu.io_out_bits_data, %fpmu.io_out_bits_data : i65
    %443 = comb.mux bin %440, %441, %442 : i65
    %444 = comb.mux bin %divSqrt_wen, %516, %443 {sv.namehint = "wdata0"} : i65
    %445 = comb.mux bin %divSqrt_wen, %divSqrt_single, %wbInfo_0_single {sv.namehint = "wsingle"} : i1
    %446 = comb.extract %444 from 0 : (i65) -> i33
    %447 = comb.concat %c1879179264_i32, %446 : i32, i33
    %448 = comb.mux bin %445, %447, %444 {sv.namehint = "wdata"} : i65
    %449 = comb.mux bin %439, %FPUFMAPipe.io_out_bits_exc, %sfma.io_out_bits_exc : i5
    %450 = comb.mux bin %439, %ifpu.io_out_bits_exc, %fpmu.io_out_bits_exc : i5
    %451 = comb.mux bin %440, %449, %450 {sv.namehint = "wexc"} : i5
    %452 = comb.xor bin %wbInfo_0_cp, %true : i1
    %453 = comb.extract %wen from 0 : (i3) -> i1
    %454 = comb.and bin %452, %453 : i1
    %455 = comb.or bin %454, %divSqrt_wen : i1
    %456 = comb.and bin %wbInfo_0_cp, %453 : i1
    %457 = comb.mux bin %456, %448, %373 {sv.namehint = "io_cp_resp_bits_data"} : i65
    %458 = comb.or %456, %371 {sv.namehint = "io_cp_resp_valid"} : i1
    %459 = comb.xor bin %ex_reg_valid, %true {sv.namehint = "io_cp_req_ready"} : i1
    %460 = comb.and bin %wb_reg_valid, %wb_ctrl_toint {sv.namehint = "wb_toint_valid"} : i1
    %wb_toint_exc = seq.firreg %461 clock %clock {firrtl.random_init_start = 255 : ui64} : i5
    %461 = comb.mux bin %mem_ctrl_toint, %fpiu.io_out_bits_exc, %wb_toint_exc : i5
    %462 = comb.or bin %460, %divSqrt_wen, %453 {sv.namehint = "io_fcsr_flags_valid"} : i1
    %463 = comb.mux bin %460, %wb_toint_exc, %c0_i5 : i5
    %464 = comb.mux bin %divSqrt_wen, %518, %c0_i5 : i5
    %465 = comb.mux bin %453, %451, %c0_i5 : i5
    %466 = comb.or bin %463, %464, %465 {sv.namehint = "io_fcsr_flags_bits"} : i5
    %467 = comb.or bin %mem_ctrl_div, %mem_ctrl_sqrt : i1
    %468 = comb.and bin %mem_reg_valid, %467 : i1
    %469 = comb.xor bin %498, %true : i1
    %470 = comb.concat %469, %wen : i1, i3
    %471 = comb.icmp bin ne %470, %c0_i4 : i4
    %472 = comb.and bin %468, %471 {sv.namehint = "units_busy"} : i1
    %473 = comb.and bin %ex_reg_valid, %47 : i1
    %474 = comb.and bin %mem_reg_valid, %mem_ctrl_wflags : i1
    %475 = comb.or bin %473, %474, %460 : i1
    %476 = comb.concat %475, %wen : i1, i3
    %477 = comb.icmp bin ne %476, %c0_i4 : i4
    %478 = comb.or bin %477, %divSqrt_in_flight : i1
    %479 = comb.xor bin %478, %true {sv.namehint = "io_fcsr_rdy"} : i1
    %480 = comb.or bin %472, %write_port_busy, %divSqrt_in_flight {sv.namehint = "io_nack_mem"} : i1
    %481 = comb.xor bin %wb_cp_valid, %true : i1
    %482 = comb.or bin %382, %mem_ctrl_div, %mem_ctrl_sqrt : i1
    %483 = seq.firreg %482 clock %clock {firrtl.random_init_start = 260 : ui64} : i1
    %484 = comb.and bin %wb_reg_valid, %481, %483 {sv.namehint = "io_sboard_set"} : i1
    %485 = comb.icmp bin eq %wbInfo_0_pipeid, %c-1_i2 : i2
    %486 = comb.and bin %453, %485 : i1
    %487 = comb.or bin %divSqrt_wen, %486 : i1
    %488 = comb.and bin %481, %487 {sv.namehint = "io_sboard_clr"} : i1
    %489 = comb.extract %io_inst from 14 : (i32) -> i1
    %490 = comb.extract %io_inst from 12 : (i32) -> i2
    %491 = comb.icmp bin ne %490, %c-1_i2 : i2
    %492 = comb.extract %io_fcsr_rm from 2 : (i3) -> i1
    %493 = comb.or bin %491, %492 : i1
    %494 = comb.and bin %489, %493 {sv.namehint = "io_illegal_rm"} : i1
    %495 = seq.firreg %508 clock %clock {firrtl.random_init_start = 261 : ui64, sv.namehint = "RecFNToRecFN.io_roundingMode"} : i2
    %496 = seq.firreg %514 clock %clock {firrtl.random_init_start = 263 : ui64} : i5
    %497 = seq.firreg %511 clock %clock {firrtl.random_init_start = 268 : ui64, sv.namehint = "RecFNToRecFN.io_in"} : i65
    %DivSqrtRecF64.io_inReady_div, %DivSqrtRecF64.io_inReady_sqrt, %DivSqrtRecF64.io_outValid_div, %DivSqrtRecF64.io_outValid_sqrt, %DivSqrtRecF64.io_out, %DivSqrtRecF64.io_exceptionFlags = hw.instance "DivSqrtRecF64" @DivSqrtRecF64(clock: %clock: !seq.clock, reset: %reset: i1, io_inValid: %501: i1, io_sqrtOp: %mem_ctrl_sqrt: i1, io_a: %fpiu.io_as_double_in1: i65, io_b: %fpiu.io_as_double_in2: i65, io_roundingMode: %502: i2) -> (io_inReady_div: i1, io_inReady_sqrt: i1, io_outValid_div: i1, io_outValid_sqrt: i1, io_out: i65, io_exceptionFlags: i5) {sv.namehint = "DivSqrtRecF64.io_out"}
    %498 = comb.mux bin %mem_ctrl_sqrt, %DivSqrtRecF64.io_inReady_sqrt, %DivSqrtRecF64.io_inReady_div {sv.namehint = "divSqrt_inReady"} : i1
    %499 = comb.or bin %DivSqrtRecF64.io_outValid_div, %DivSqrtRecF64.io_outValid_sqrt : i1
    %500 = comb.xor bin %divSqrt_in_flight, %true : i1
    %501 = comb.and bin %468, %500 {sv.namehint = "DivSqrtRecF64.io_inValid"} : i1
    %502 = comb.extract %fpiu.io_as_double_rm from 0 {sv.namehint = "DivSqrtRecF64.io_roundingMode"} : (i3) -> i2
    %503 = comb.and bin %501, %498 : i1
    %504 = comb.or %503, %divSqrt_in_flight : i1
    %505 = comb.mux bin %503, %9, %divSqrt_killed : i1
    %506 = comb.mux bin %503, %mem_ctrl_single, %divSqrt_single : i1
    %507 = comb.mux bin %503, %426, %divSqrt_waddr : i5
    %508 = comb.mux bin %503, %502, %495 : i2
    %509 = comb.xor bin %divSqrt_killed, %true : i1
    %510 = comb.and %499, %509 : i1
    %511 = comb.mux bin %499, %DivSqrtRecF64.io_out, %497 : i65
    %512 = comb.xor %499, %true : i1
    %513 = comb.and %512, %504 : i1
    %514 = comb.mux bin %499, %DivSqrtRecF64.io_exceptionFlags, %496 : i5
    %RecFNToRecFN.io_out, %RecFNToRecFN.io_exceptionFlags = hw.instance "RecFNToRecFN" @RecFNToRecFN(io_in: %497: i65, io_roundingMode: %495: i2) -> (io_out: i33, io_exceptionFlags: i5) {sv.namehint = "RecFNToRecFN.io_out"}
    %515 = comb.concat %c0_i32, %RecFNToRecFN.io_out : i32, i33
    %516 = comb.mux bin %divSqrt_single, %515, %497 {sv.namehint = "divSqrt_wdata"} : i65
    %517 = comb.mux bin %divSqrt_single, %RecFNToRecFN.io_exceptionFlags, %c0_i5 : i5
    %518 = comb.or bin %496, %517 {sv.namehint = "divSqrt_flags"} : i5
    hw.output %462, %466, %fpiu.io_out_bits_store, %fpiu.io_out_bits_toint, %479, %480, %494, %fp_decoder.io_sigs_cmd, %fp_decoder.io_sigs_ldst, %fp_decoder.io_sigs_wen, %fp_decoder.io_sigs_ren1, %fp_decoder.io_sigs_ren2, %fp_decoder.io_sigs_ren3, %fp_decoder.io_sigs_swap12, %fp_decoder.io_sigs_swap23, %fp_decoder.io_sigs_single, %fp_decoder.io_sigs_fromint, %fp_decoder.io_sigs_toint, %fp_decoder.io_sigs_fastpipe, %fp_decoder.io_sigs_fma, %fp_decoder.io_sigs_div, %fp_decoder.io_sigs_sqrt, %fp_decoder.io_sigs_wflags, %484, %488, %438, %459, %458, %457, %c0_i5 : i1, i5, i64, i64, i1, i1, i1, i5, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i5, i1, i1, i65, i5
  }
  hw.module private @FPUDecoder(in %io_inst : i32, out io_sigs_cmd : i5, out io_sigs_ldst : i1, out io_sigs_wen : i1, out io_sigs_ren1 : i1, out io_sigs_ren2 : i1, out io_sigs_ren3 : i1, out io_sigs_swap12 : i1, out io_sigs_swap23 : i1, out io_sigs_single : i1, out io_sigs_fromint : i1, out io_sigs_toint : i1, out io_sigs_fastpipe : i1, out io_sigs_fma : i1, out io_sigs_div : i1, out io_sigs_sqrt : i1, out io_sigs_wflags : i1) {
    %c-4_i3 = hw.constant -4 : i3
    %c7_i4 = hw.constant 7 : i4
    %c0_i4 = hw.constant 0 : i4
    %c5_i4 = hw.constant 5 : i4
    %c3_i3 = hw.constant 3 : i3
    %c-3_i3 = hw.constant -3 : i3
    %c1_i2 = hw.constant 1 : i2
    %c1_i3 = hw.constant 1 : i3
    %c-1_i3 = hw.constant -1 : i3
    %c-2_i2 = hw.constant -2 : i2
    %c0_i2 = hw.constant 0 : i2
    %c-1_i2 = hw.constant -1 : i2
    %true = hw.constant true
    %0 = comb.extract %io_inst from 2 : (i32) -> i1
    %1 = comb.extract %io_inst from 27 : (i32) -> i1
    %2 = comb.extract %io_inst from 4 : (i32) -> i1
    %3 = comb.concat %1, %2 : i1, i1
    %4 = comb.icmp bin eq %3, %c-1_i2 : i2
    %5 = comb.or bin %0, %4 : i1
    %6 = comb.extract %io_inst from 3 : (i32) -> i1
    %7 = comb.extract %io_inst from 28 : (i32) -> i1
    %8 = comb.extract %io_inst from 4 : (i32) -> i1
    %9 = comb.concat %7, %8 : i1, i1
    %10 = comb.icmp bin eq %9, %c-1_i2 : i2
    %11 = comb.or bin %6, %10 : i1
    %12 = comb.extract %io_inst from 6 : (i32) -> i1
    %13 = comb.xor bin %12, %true {sv.namehint = "decoder_1"} : i1
    %14 = comb.extract %io_inst from 29 : (i32) -> i1
    %15 = comb.or bin %13, %14 : i1
    %16 = comb.extract %io_inst from 30 : (i32) -> i1
    %17 = comb.or bin %13, %16 : i1
    %18 = comb.extract %io_inst from 4 : (i32) -> i1
    %19 = comb.xor bin %18, %true : i1
    %20 = comb.concat %19, %17, %15, %11, %5 {sv.namehint = "decoder_0"} : i1, i1, i1, i1, i1
    %21 = comb.extract %io_inst from 31 : (i32) -> i1
    %22 = comb.extract %io_inst from 5 : (i32) -> i1
    %23 = comb.concat %21, %22 : i1, i1
    %24 = comb.icmp bin eq %23, %c0_i2 : i2
    %25 = comb.extract %io_inst from 4 : (i32) -> i2
    %26 = comb.icmp bin eq %25, %c0_i2 : i2
    %27 = comb.extract %io_inst from 28 : (i32) -> i1
    %28 = comb.extract %io_inst from 5 : (i32) -> i1
    %29 = comb.concat %27, %28 : i1, i1
    %30 = comb.icmp bin eq %29, %c-2_i2 : i2
    %31 = comb.or bin %24, %26, %30 {sv.namehint = "decoder_2"} : i1
    %32 = comb.extract %io_inst from 31 : (i32) -> i1
    %33 = comb.extract %io_inst from 2 : (i32) -> i1
    %34 = comb.concat %32, %33 : i1, i1
    %35 = comb.icmp bin eq %34, %c0_i2 : i2
    %36 = comb.extract %io_inst from 28 : (i32) -> i1
    %37 = comb.extract %io_inst from 2 : (i32) -> i1
    %38 = comb.concat %36, %37 : i1, i1
    %39 = comb.icmp bin eq %38, %c0_i2 : i2
    %40 = comb.extract %io_inst from 6 : (i32) -> i1
    %41 = comb.extract %io_inst from 4 : (i32) -> i1
    %42 = comb.concat %40, %41 : i1, i1
    %43 = comb.icmp bin eq %42, %c-2_i2 {sv.namehint = "decoder_5"} : i2
    %44 = comb.or bin %35, %39, %43 {sv.namehint = "decoder_3"} : i1
    %45 = comb.extract %io_inst from 30 : (i32) -> i1
    %46 = comb.extract %io_inst from 2 : (i32) -> i1
    %47 = comb.concat %45, %46 : i1, i1
    %48 = comb.icmp bin eq %47, %c0_i2 : i2
    %49 = comb.extract %io_inst from 5 : (i32) -> i1
    %50 = comb.or bin %48, %49, %43 {sv.namehint = "decoder_4"} : i1
    %51 = comb.extract %io_inst from 30 : (i32) -> i1
    %52 = comb.extract %io_inst from 28 : (i32) -> i1
    %53 = comb.extract %io_inst from 4 : (i32) -> i1
    %54 = comb.concat %51, %52, %53 : i1, i1, i1
    %55 = comb.icmp bin eq %54, %c-1_i3 : i3
    %56 = comb.or bin %13, %55 {sv.namehint = "decoder_6"} : i1
    %57 = comb.extract %io_inst from 28 : (i32) -> i2
    %58 = comb.extract %io_inst from 4 : (i32) -> i1
    %59 = comb.concat %57, %58 : i2, i1
    %60 = comb.icmp bin eq %59, %c1_i3 {sv.namehint = "decoder_7"} : i3
    %61 = comb.extract %io_inst from 12 : (i32) -> i1
    %62 = comb.extract %io_inst from 6 : (i32) -> i1
    %63 = comb.concat %61, %62 : i1, i1
    %64 = comb.icmp bin eq %63, %c0_i2 : i2
    %65 = comb.extract %io_inst from 25 : (i32) -> i1
    %66 = comb.extract %io_inst from 6 : (i32) -> i1
    %67 = comb.concat %65, %66 : i1, i1
    %68 = comb.icmp bin eq %67, %c1_i2 : i2
    %69 = comb.or bin %64, %68 {sv.namehint = "decoder_8"} : i1
    %70 = comb.extract %io_inst from 31 : (i32) -> i1
    %71 = comb.extract %io_inst from 31 : (i32) -> i1
    %72 = comb.extract %io_inst from 28 : (i32) -> i1
    %73 = comb.extract %io_inst from 4 : (i32) -> i1
    %74 = comb.concat %71, %72, %73 : i1, i1, i1
    %75 = comb.icmp bin eq %74, %c-1_i3 {sv.namehint = "decoder_9"} : i3
    %76 = comb.extract %io_inst from 28 : (i32) -> i1
    %77 = comb.extract %io_inst from 4 : (i32) -> i1
    %78 = comb.concat %70, %76, %77 : i1, i1, i1
    %79 = comb.icmp bin eq %78, %c-3_i3 : i3
    %80 = comb.or bin %49, %79 {sv.namehint = "decoder_10"} : i1
    %81 = comb.extract %io_inst from 31 : (i32) -> i1
    %82 = comb.extract %io_inst from 29 : (i32) -> i1
    %83 = comb.extract %io_inst from 4 : (i32) -> i1
    %84 = comb.concat %81, %82, %83 : i1, i1, i1
    %85 = comb.icmp bin eq %84, %c3_i3 : i3
    %86 = comb.extract %io_inst from 30 : (i32) -> i2
    %87 = comb.extract %io_inst from 30 : (i32) -> i2
    %88 = comb.extract %io_inst from 28 : (i32) -> i1
    %89 = comb.extract %io_inst from 4 : (i32) -> i1
    %90 = comb.concat %87, %88, %89 : i2, i1, i1
    %91 = comb.icmp bin eq %90, %c5_i4 : i4
    %92 = comb.or bin %85, %91 {sv.namehint = "decoder_11"} : i1
    %93 = comb.extract %io_inst from 28 : (i32) -> i3
    %94 = comb.extract %io_inst from 2 : (i32) -> i1
    %95 = comb.concat %93, %94 : i3, i1
    %96 = comb.icmp bin eq %95, %c0_i4 : i4
    %97 = comb.extract %io_inst from 29 : (i32) -> i2
    %98 = comb.extract %io_inst from 27 : (i32) -> i1
    %99 = comb.extract %io_inst from 2 : (i32) -> i1
    %100 = comb.concat %97, %98, %99 : i2, i1, i1
    %101 = comb.icmp bin eq %100, %c0_i4 : i4
    %102 = comb.or bin %96, %101, %43 {sv.namehint = "decoder_12"} : i1
    %103 = comb.extract %io_inst from 30 : (i32) -> i1
    %104 = comb.extract %io_inst from 27 : (i32) -> i2
    %105 = comb.extract %io_inst from 4 : (i32) -> i1
    %106 = comb.concat %103, %104, %105 : i1, i2, i1
    %107 = comb.icmp bin eq %106, %c7_i4 {sv.namehint = "decoder_13"} : i4
    %108 = comb.extract %io_inst from 28 : (i32) -> i1
    %109 = comb.extract %io_inst from 4 : (i32) -> i1
    %110 = comb.concat %86, %108, %109 : i2, i1, i1
    %111 = comb.icmp bin eq %110, %c7_i4 {sv.namehint = "decoder_14"} : i4
    %112 = comb.extract %io_inst from 29 : (i32) -> i1
    %113 = comb.extract %io_inst from 2 : (i32) -> i1
    %114 = comb.concat %112, %113 : i1, i1
    %115 = comb.icmp bin eq %114, %c0_i2 : i2
    %116 = comb.extract %io_inst from 27 : (i32) -> i1
    %117 = comb.extract %io_inst from 13 : (i32) -> i1
    %118 = comb.concat %116, %117 : i1, i1
    %119 = comb.icmp bin eq %118, %c-2_i2 : i2
    %120 = comb.extract %io_inst from 30 : (i32) -> i2
    %121 = comb.extract %io_inst from 2 : (i32) -> i1
    %122 = comb.concat %120, %121 : i2, i1
    %123 = comb.icmp bin eq %122, %c-4_i3 : i3
    %124 = comb.or bin %115, %43, %119, %123 {sv.namehint = "decoder_15"} : i1
    hw.output %20, %13, %31, %44, %50, %43, %56, %60, %69, %75, %80, %92, %102, %107, %111, %124 : i5, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
  }
  hw.module private @FPUFMAPipe(in %clock : !seq.clock, in %reset : i1, in %io_in_valid : i1, in %io_in_bits_cmd : i5, in %io_in_bits_ren3 : i1, in %io_in_bits_swap23 : i1, in %io_in_bits_rm : i3, in %io_in_bits_in1 : i65, in %io_in_bits_in2 : i65, in %io_in_bits_in3 : i65, out io_out_bits_data : i65, out io_out_bits_exc : i5) {
    %c0_i3 = hw.constant 0 : i3
    %c0_i32 = hw.constant 0 : i32
    %false = hw.constant false
    %c2147483648_i65 = hw.constant 2147483648 : i65
    %0 = comb.extract %io_in_bits_in1 from 32 : (i65) -> i1
    %1 = comb.extract %io_in_bits_in2 from 32 : (i65) -> i1
    %2 = comb.xor bin %0, %1 : i1
    %valid = seq.firreg %io_in_valid clock %clock {firrtl.random_init_start = 0 : ui64} : i1
    %in_cmd = seq.firreg %10 clock %clock {firrtl.random_init_start = 1 : ui64} : i5
    %in_rm = seq.firreg %3 clock %clock {firrtl.random_init_start = 21 : ui64} : i3
    %in_in1 = seq.firreg %4 clock %clock {firrtl.random_init_start = 26 : ui64} : i65
    %in_in2 = seq.firreg %12 clock %clock {firrtl.random_init_start = 91 : ui64} : i65
    %in_in3 = seq.firreg %15 clock %clock {firrtl.random_init_start = 156 : ui64} : i65
    %3 = comb.mux bin %io_in_valid, %io_in_bits_rm, %in_rm : i3
    %4 = comb.mux bin %io_in_valid, %io_in_bits_in1, %in_in1 : i65
    %5 = comb.extract %io_in_bits_cmd from 1 : (i5) -> i1
    %6 = comb.or bin %io_in_bits_ren3, %io_in_bits_swap23 : i1
    %7 = comb.and bin %5, %6 : i1
    %8 = comb.extract %io_in_bits_cmd from 0 : (i5) -> i1
    %9 = comb.concat %c0_i3, %7, %8 : i3, i1, i1
    %10 = comb.mux bin %io_in_valid, %9, %in_cmd : i5
    %11 = comb.mux bin %io_in_bits_swap23, %c2147483648_i65, %io_in_bits_in2 : i65
    %12 = comb.mux bin %io_in_valid, %11, %in_in2 : i65
    %13 = comb.concat %c0_i32, %2, %c0_i32 : i32, i1, i32
    %14 = comb.mux bin %6, %io_in_bits_in3, %13 : i65
    %15 = comb.mux bin %io_in_valid, %14, %in_in3 : i65
    %fma.io_out, %fma.io_exceptionFlags = hw.instance "fma" @MulAddRecFN(io_op: %16: i2, io_a: %18: i33, io_b: %19: i33, io_c: %20: i33, io_roundingMode: %17: i2) -> (io_out: i33, io_exceptionFlags: i5) {sv.namehint = "res_exc"}
    %16 = comb.extract %in_cmd from 0 {sv.namehint = "fma.io_op"} : (i5) -> i2
    %17 = comb.extract %in_rm from 0 {sv.namehint = "fma.io_roundingMode"} : (i3) -> i2
    %18 = comb.extract %in_in1 from 0 {sv.namehint = "fma.io_a"} : (i65) -> i33
    %19 = comb.extract %in_in2 from 0 {sv.namehint = "fma.io_b"} : (i65) -> i33
    %20 = comb.extract %in_in3 from 0 {sv.namehint = "fma.io_c"} : (i65) -> i33
    %21 = comb.concat %c0_i32, %fma.io_out {sv.namehint = "res_data"} : i32, i33
    %22 = seq.firreg %valid clock %clock reset sync %reset, %false {firrtl.random_init_start = 221 : ui64} : i1
    %23 = seq.firreg %25 clock %clock {firrtl.random_init_start = 222 : ui64} : i65
    %24 = seq.firreg %26 clock %clock {firrtl.random_init_start = 287 : ui64} : i5
    %25 = comb.mux bin %valid, %21, %23 : i65
    %26 = comb.mux bin %valid, %fma.io_exceptionFlags, %24 : i5
    %27 = seq.firreg %29 clock %clock {firrtl.random_init_start = 293 : ui64, sv.namehint = "io_out_bits_data"} : i65
    %28 = seq.firreg %30 clock %clock {firrtl.random_init_start = 358 : ui64, sv.namehint = "io_out_bits_exc"} : i5
    %29 = comb.mux bin %22, %23, %27 : i65
    %30 = comb.mux bin %22, %24, %28 : i5
    hw.output %27, %28 : i65, i5
  }
  hw.module private @FPToInt(in %clock : !seq.clock, in %io_in_valid : i1, in %io_in_bits_cmd : i5, in %io_in_bits_ldst : i1, in %io_in_bits_single : i1, in %io_in_bits_rm : i3, in %io_in_bits_typ : i2, in %io_in_bits_in1 : i65, in %io_in_bits_in2 : i65, out io_as_double_rm : i3, out io_as_double_in1 : i65, out io_as_double_in2 : i65, out io_out_valid : i1, out io_out_bits_lt : i1, out io_out_bits_store : i64, out io_out_bits_toint : i64, out io_out_bits_exc : i5) {
    %c-1_i3 = hw.constant -1 : i3
    %c1023_i11 = hw.constant 1023 : i11
    %c2_i6 = hw.constant 2 : i6
    %c0_i9 = hw.constant 0 : i9
    %c127_i8 = hw.constant 127 : i8
    %c2_i5 = hw.constant 2 : i5
    %c0_i6 = hw.constant 0 : i6
    %c-3_i3 = hw.constant -3 : i3
    %c1792_i12 = hw.constant 1792 : i12
    %c0_i29 = hw.constant 0 : i29
    %c2_i10 = hw.constant 2 : i10
    %c1_i2 = hw.constant 1 : i2
    %c1_i3 = hw.constant 1 : i3
    %c2_i7 = hw.constant 2 : i7
    %c0_i63 = hw.constant 0 : i63
    %c0_i54 = hw.constant 0 : i54
    %c0_i47 = hw.constant 0 : i47
    %c0_i19 = hw.constant 0 : i19
    %c-1_i2 = hw.constant -1 : i2
    %c0_i2 = hw.constant 0 : i2
    %c-2_i2 = hw.constant -2 : i2
    %true = hw.constant true
    %c0_i3 = hw.constant 0 : i3
    %c0_i23 = hw.constant 0 : i23
    %c0_i52 = hw.constant 0 : i52
    %c0_i5 = hw.constant 0 : i5
    %in_cmd = seq.firreg %0 clock %clock {firrtl.random_init_start = 0 : ui64} : i5
    %in_single = seq.firreg %1 clock %clock {firrtl.random_init_start = 12 : ui64} : i1
    %in_rm = seq.firreg %2 clock %clock {firrtl.random_init_start = 20 : ui64, sv.namehint = "in_rm"} : i3
    %in_typ = seq.firreg %3 clock %clock {firrtl.random_init_start = 23 : ui64} : i2
    %in_in1 = seq.firreg %22 clock %clock {firrtl.random_init_start = 25 : ui64, sv.namehint = "in_in1"} : i65
    %in_in2 = seq.firreg %37 clock %clock {firrtl.random_init_start = 90 : ui64, sv.namehint = "in_in2"} : i65
    %valid = seq.firreg %io_in_valid clock %clock {firrtl.random_init_start = 220 : ui64, sv.namehint = "valid"} : i1
    %0 = comb.mux bin %io_in_valid, %io_in_bits_cmd, %in_cmd : i5
    %1 = comb.mux bin %io_in_valid, %io_in_bits_single, %in_single : i1
    %2 = comb.mux bin %io_in_valid, %io_in_bits_rm, %in_rm : i3
    %3 = comb.mux bin %io_in_valid, %io_in_bits_typ, %in_typ : i2
    %4 = comb.xor bin %io_in_bits_ldst, %true : i1
    %5 = comb.extract %io_in_bits_cmd from 2 : (i5) -> i2
    %6 = comb.icmp bin ne %5, %c-1_i2 : i2
    %7 = comb.and bin %io_in_bits_single, %4, %6 : i1
    %8 = comb.extract %io_in_bits_in1 from 32 : (i65) -> i1
    %9 = comb.extract %io_in_bits_in1 from 0 : (i65) -> i23
    %10 = comb.extract %io_in_bits_in1 from 23 : (i65) -> i9
    %11 = comb.extract %io_in_bits_in1 from 29 : (i65) -> i3
    %12 = comb.concat %c0_i3, %10 : i3, i9
    %13 = comb.add %12, %c1792_i12 : i12
    %14 = comb.icmp bin eq %11, %c0_i3 : i3
    %15 = comb.icmp bin ugt %11, %c-3_i3 : i3
    %16 = comb.or bin %14, %15 : i1
    %17 = comb.extract %13 from 0 : (i12) -> i9
    %18 = comb.concat %11, %17 : i3, i9
    %19 = comb.mux bin %16, %18, %13 : i12
    %20 = comb.concat %8, %19, %9, %c0_i29 : i1, i12, i23, i29
    %21 = comb.mux bin %7, %20, %io_in_bits_in1 : i65
    %22 = comb.mux bin %io_in_valid, %21, %in_in1 : i65
    %23 = comb.extract %io_in_bits_in2 from 32 : (i65) -> i1
    %24 = comb.extract %io_in_bits_in2 from 0 : (i65) -> i23
    %25 = comb.extract %io_in_bits_in2 from 23 : (i65) -> i9
    %26 = comb.extract %io_in_bits_in2 from 29 : (i65) -> i3
    %27 = comb.concat %c0_i3, %25 : i3, i9
    %28 = comb.add %27, %c1792_i12 : i12
    %29 = comb.icmp bin eq %26, %c0_i3 : i3
    %30 = comb.icmp bin ugt %26, %c-3_i3 : i3
    %31 = comb.or bin %29, %30 : i1
    %32 = comb.extract %28 from 0 : (i12) -> i9
    %33 = comb.concat %26, %32 : i3, i9
    %34 = comb.mux bin %31, %33, %28 : i12
    %35 = comb.concat %23, %34, %24, %c0_i29 : i1, i12, i23, i29
    %36 = comb.mux bin %7, %35, %io_in_bits_in2 : i65
    %37 = comb.mux bin %io_in_valid, %36, %in_in2 : i65
    %38 = comb.extract %in_in1 from 32 : (i65) -> i1
    %39 = comb.extract %in_in1 from 0 : (i65) -> i23
    %40 = comb.extract %in_in1 from 23 : (i65) -> i7
    %41 = comb.icmp bin ult %40, %c2_i7 : i7
    %42 = comb.extract %in_in1 from 29 : (i65) -> i3
    %43 = comb.icmp bin eq %42, %c1_i3 : i3
    %44 = comb.extract %in_in1 from 30 : (i65) -> i2
    %45 = comb.icmp bin eq %44, %c1_i2 : i2
    %46 = comb.and bin %45, %41 : i1
    %47 = comb.or bin %43, %46 : i1
    %48 = comb.extract %in_in1 from 24 : (i65) -> i6
    %49 = comb.icmp bin ne %48, %c0_i6 : i6
    %50 = comb.and bin %45, %49 : i1
    %51 = comb.icmp bin eq %44, %c-2_i2 : i2
    %52 = comb.or bin %50, %51 : i1
    %53 = comb.icmp bin eq %44, %c-1_i2 : i2
    %54 = comb.extract %in_in1 from 29 : (i65) -> i1
    %55 = comb.and bin %53, %54 : i1
    %56 = comb.extract %in_in1 from 23 : (i65) -> i5
    %57 = comb.sub %c2_i5, %56 : i5
    %58 = comb.concat %true, %39 : i1, i23
    %59 = comb.concat %c0_i19, %57 : i19, i5
    %60 = comb.shru bin %58, %59 : i24
    %61 = comb.extract %60 from 0 : (i24) -> i23
    %62 = comb.extract %in_in1 from 23 : (i65) -> i8
    %63 = comb.add %62, %c127_i8 : i8
    %64 = comb.replicate %53 : (i1) -> i8
    %65 = comb.mux bin %52, %63, %64 : i8
    %66 = comb.or bin %52, %55 : i1
    %67 = comb.mux bin %47, %61, %c0_i23 : i23
    %68 = comb.mux bin %66, %39, %67 : i23
    %69 = comb.replicate %38 : (i1) -> i33
    %70 = comb.concat %69, %65, %68 {sv.namehint = "unrec_s"} : i33, i8, i23
    %71 = comb.extract %in_in1 from 64 : (i65) -> i1
    %72 = comb.extract %in_in1 from 0 : (i65) -> i52
    %73 = comb.extract %in_in1 from 52 : (i65) -> i10
    %74 = comb.icmp bin ult %73, %c2_i10 : i10
    %75 = comb.extract %in_in1 from 61 : (i65) -> i3
    %76 = comb.icmp bin eq %75, %c1_i3 : i3
    %77 = comb.extract %in_in1 from 62 : (i65) -> i2
    %78 = comb.icmp bin eq %77, %c1_i2 : i2
    %79 = comb.and bin %78, %74 : i1
    %80 = comb.or bin %76, %79 : i1
    %81 = comb.extract %in_in1 from 53 : (i65) -> i9
    %82 = comb.icmp bin ne %81, %c0_i9 : i9
    %83 = comb.and bin %78, %82 : i1
    %84 = comb.icmp bin eq %77, %c-2_i2 : i2
    %85 = comb.or bin %83, %84 : i1
    %86 = comb.icmp bin eq %77, %c-1_i2 : i2
    %87 = comb.extract %in_in1 from 61 : (i65) -> i1
    %88 = comb.and bin %86, %87 : i1
    %89 = comb.extract %in_in1 from 52 : (i65) -> i6
    %90 = comb.sub %c2_i6, %89 : i6
    %91 = comb.concat %true, %72 : i1, i52
    %92 = comb.concat %c0_i47, %90 : i47, i6
    %93 = comb.shru bin %91, %92 : i53
    %94 = comb.extract %93 from 0 : (i53) -> i52
    %95 = comb.extract %in_in1 from 52 : (i65) -> i11
    %96 = comb.add %95, %c1023_i11 : i11
    %97 = comb.replicate %86 : (i1) -> i11
    %98 = comb.mux bin %85, %96, %97 : i11
    %99 = comb.or bin %85, %88 : i1
    %100 = comb.mux bin %80, %94, %c0_i52 : i52
    %101 = comb.mux bin %99, %72, %100 : i52
    %102 = comb.concat %71, %98, %101 : i1, i11, i52
    %103 = comb.mux bin %in_single, %70, %102 {sv.namehint = "unrec_int"} : i64
    %104 = comb.icmp bin eq %42, %c0_i3 : i3
    %105 = comb.xor bin %54, %true : i1
    %106 = comb.and bin %53, %105 : i1
    %107 = comb.icmp eq %42, %c-1_i3 : i3
    %108 = comb.extract %in_in1 from 22 : (i65) -> i1
    %109 = comb.xor bin %108, %true : i1
    %110 = comb.and bin %107, %109 : i1
    %111 = comb.and bin %107, %108 : i1
    %112 = comb.xor bin %38, %true : i1
    %113 = comb.and bin %106, %112 : i1
    %114 = comb.and bin %52, %112 : i1
    %115 = comb.and bin %47, %112 : i1
    %116 = comb.and bin %104, %112 : i1
    %117 = comb.and bin %104, %38 : i1
    %118 = comb.and bin %47, %38 : i1
    %119 = comb.and bin %52, %38 : i1
    %120 = comb.and bin %106, %38 : i1
    %121 = comb.concat %111, %110, %113, %114, %115, %116, %117, %118, %119, %120 {sv.namehint = "classify_s"} : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
    %122 = comb.icmp bin eq %75, %c0_i3 : i3
    %123 = comb.xor bin %87, %true : i1
    %124 = comb.and bin %86, %123 : i1
    %125 = comb.icmp eq %75, %c-1_i3 : i3
    %126 = comb.extract %in_in1 from 51 : (i65) -> i1
    %127 = comb.xor bin %126, %true : i1
    %128 = comb.and bin %125, %127 : i1
    %129 = comb.and bin %125, %126 : i1
    %130 = comb.xor bin %71, %true : i1
    %131 = comb.and bin %124, %130 : i1
    %132 = comb.and bin %85, %130 : i1
    %133 = comb.and bin %80, %130 : i1
    %134 = comb.and bin %122, %130 : i1
    %135 = comb.and bin %122, %71 : i1
    %136 = comb.and bin %80, %71 : i1
    %137 = comb.and bin %85, %71 : i1
    %138 = comb.and bin %124, %71 : i1
    %139 = comb.concat %129, %128, %131, %132, %133, %134, %135, %136, %137, %138 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
    %140 = comb.mux bin %in_single, %121, %139 {sv.namehint = "classify_out"} : i10
    %dcmp.io_lt, %dcmp.io_eq, %dcmp.io_exceptionFlags = hw.instance "dcmp" @CompareRecFN(io_a: %in_in1: i65, io_b: %in_in2: i65, io_signaling: %142: i1) -> (io_lt: i1, io_eq: i1, io_exceptionFlags: i5) {sv.namehint = "dcmp.io_eq"}
    %141 = comb.extract %in_rm from 1 : (i3) -> i1
    %142 = comb.xor bin %141, %true {sv.namehint = "dcmp.io_signaling"} : i1
    %143 = comb.extract %in_rm from 0 : (i3) -> i1
    %144 = comb.concat %c0_i54, %140 : i54, i10
    %145 = comb.mux bin %143, %144, %103 : i64
    %146 = comb.extract %in_cmd from 2 : (i5) -> i2
    %147 = comb.icmp bin eq %146, %c1_i2 : i2
    %148 = comb.extract %in_rm from 0 : (i3) -> i2
    %149 = comb.xor %148, %c-1_i2 : i2
    %150 = comb.concat %dcmp.io_lt, %dcmp.io_eq : i1, i1
    %151 = comb.and bin %149, %150 : i2
    %152 = comb.icmp bin ne %151, %c0_i2 : i2
    %153 = comb.concat %c0_i63, %152 : i63, i1
    %154 = comb.mux bin %147, %153, %145 : i64
    %155 = comb.mux bin %147, %dcmp.io_exceptionFlags, %c0_i5 : i5
    %156 = comb.icmp bin eq %146, %c-2_i2 : i2
    %RecFNToIN.io_out, %RecFNToIN.io_intExceptionFlags = hw.instance "RecFNToIN" @RecFNToIN(io_in: %in_in1: i65, io_roundingMode: %157: i2, io_signedOut: %159: i1) -> (io_out: i32, io_intExceptionFlags: i3) {sv.namehint = "RecFNToIN.io_out"}
    %157 = comb.extract %in_rm from 0 {sv.namehint = "RecFNToIN.io_roundingMode"} : (i3) -> i2
    %158 = comb.extract %in_typ from 0 : (i2) -> i1
    %159 = comb.xor bin %158, %true {sv.namehint = "RecFNToIN.io_signedOut"} : i1
    %160 = comb.extract %in_typ from 1 : (i2) -> i1
    %161 = comb.extract %RecFNToIN.io_out from 31 : (i32) -> i1
    %162 = comb.replicate %161 : (i1) -> i32
    %163 = comb.concat %162, %RecFNToIN.io_out : i32, i32
    %164 = comb.extract %RecFNToIN.io_intExceptionFlags from 1 : (i3) -> i2
    %165 = comb.icmp bin ne %164, %c0_i2 : i2
    %166 = comb.extract %RecFNToIN.io_intExceptionFlags from 0 : (i3) -> i1
    %167 = comb.concat %165, %c0_i3, %166 : i1, i3, i1
    %RecFNToIN_1.io_out, %RecFNToIN_1.io_intExceptionFlags = hw.instance "RecFNToIN_1" @RecFNToIN_1(io_in: %in_in1: i65, io_roundingMode: %157: i2, io_signedOut: %159: i1) -> (io_out: i64, io_intExceptionFlags: i3) {sv.namehint = "RecFNToIN_1.io_out"}
    %168 = comb.mux bin %160, %RecFNToIN_1.io_out, %163 : i64
    %169 = comb.mux bin %156, %168, %154 {sv.namehint = "io_out_bits_toint"} : i64
    %170 = comb.extract %RecFNToIN_1.io_intExceptionFlags from 1 : (i3) -> i2
    %171 = comb.icmp bin ne %170, %c0_i2 : i2
    %172 = comb.extract %RecFNToIN_1.io_intExceptionFlags from 0 : (i3) -> i1
    %173 = comb.concat %171, %c0_i3, %172 : i1, i3, i1
    %174 = comb.mux bin %160, %173, %167 : i5
    %175 = comb.mux bin %156, %174, %155 {sv.namehint = "io_out_bits_exc"} : i5
    hw.output %in_rm, %in_in1, %in_in2, %valid, %dcmp.io_lt, %103, %169, %175 : i3, i65, i65, i1, i1, i64, i64, i5
  }
  hw.module private @IntToFP(in %clock : !seq.clock, in %reset : i1, in %io_in_valid : i1, in %io_in_bits_cmd : i5, in %io_in_bits_single : i1, in %io_in_bits_rm : i3, in %io_in_bits_typ : i2, in %io_in_bits_in1 : i65, out io_out_bits_data : i65, out io_out_bits_exc : i5) {
    %c-1_i9 = hw.constant -1 : i9
    %c256_i10 = hw.constant 256 : i10
    %c0_i2 = hw.constant 0 : i2
    %c-1_i3 = hw.constant -1 : i3
    %c32_i7 = hw.constant 32 : i7
    %c-1_i4 = hw.constant -1 : i4
    %c0_i3 = hw.constant 0 : i3
    %c0_i7 = hw.constant 0 : i7
    %c0_i109 = hw.constant 0 : i109
    %c0_i63 = hw.constant 0 : i63
    %c-1_i6 = hw.constant -1 : i6
    %c0_i52 = hw.constant 0 : i52
    %c0_i11 = hw.constant 0 : i11
    %c0_i32 = hw.constant 0 : i32
    %c0_i6 = hw.constant 0 : i6
    %c0_i49 = hw.constant 0 : i49
    %c0_i31 = hw.constant 0 : i31
    %c-1_i5 = hw.constant -1 : i5
    %c0_i4 = hw.constant 0 : i4
    %c0_i16 = hw.constant 0 : i16
    %c0_i9 = hw.constant 0 : i9
    %c0_i23 = hw.constant 0 : i23
    %true = hw.constant true
    %c0_i8 = hw.constant 0 : i8
    %false = hw.constant false
    %c-2_i2 = hw.constant -2 : i2
    %c-1_i2 = hw.constant -1 : i2
    %c0_i5 = hw.constant 0 : i5
    %c1_i2 = hw.constant 1 : i2
    %0 = seq.firreg %io_in_valid clock %clock reset sync %reset, %false {firrtl.random_init_start = 0 : ui64, sv.namehint = "in_valid"} : i1
    %1 = seq.firreg %6 clock %clock {firrtl.random_init_start = 1 : ui64, sv.namehint = "in_bits_cmd"} : i5
    %2 = seq.firreg %7 clock %clock {firrtl.random_init_start = 13 : ui64, sv.namehint = "in_bits_single"} : i1
    %3 = seq.firreg %8 clock %clock {firrtl.random_init_start = 21 : ui64, sv.namehint = "in_bits_rm"} : i3
    %4 = seq.firreg %9 clock %clock {firrtl.random_init_start = 24 : ui64, sv.namehint = "in_bits_typ"} : i2
    %5 = seq.firreg %10 clock %clock {firrtl.random_init_start = 26 : ui64, sv.namehint = "in_bits_in1"} : i65
    %6 = comb.mux bin %io_in_valid, %io_in_bits_cmd, %1 : i5
    %7 = comb.mux bin %io_in_valid, %io_in_bits_single, %2 : i1
    %8 = comb.mux bin %io_in_valid, %io_in_bits_rm, %3 : i3
    %9 = comb.mux bin %io_in_valid, %io_in_bits_typ, %4 : i2
    %10 = comb.mux bin %io_in_valid, %io_in_bits_in1, %5 : i65
    %11 = comb.extract %5 from 31 : (i65) -> i1
    %12 = comb.extract %5 from 23 : (i65) -> i8
    %13 = comb.extract %5 from 0 : (i65) -> i23
    %14 = comb.icmp bin eq %12, %c0_i8 : i8
    %15 = comb.icmp bin ne %13, %c0_i23 : i23
    %16 = comb.xor bin %15, %true : i1
    %17 = comb.and bin %14, %16 : i1
    %18 = comb.extract %5 from 7 : (i65) -> i16
    %19 = comb.icmp bin ne %18, %c0_i16 : i16
    %20 = comb.extract %5 from 15 : (i65) -> i8
    %21 = comb.icmp bin ne %20, %c0_i8 : i8
    %22 = comb.extract %5 from 19 : (i65) -> i4
    %23 = comb.icmp bin ne %22, %c0_i4 : i4
    %24 = comb.extract %5 from 22 : (i65) -> i1
    %25 = comb.extract %5 from 21 : (i65) -> i1
    %26 = comb.extract %5 from 20 : (i65) -> i1
    %27 = comb.concat %false, %26 : i1, i1
    %28 = comb.mux bin %25, %c-2_i2, %27 : i2
    %29 = comb.mux bin %24, %c-1_i2, %28 : i2
    %30 = comb.extract %5 from 18 : (i65) -> i1
    %31 = comb.extract %5 from 17 : (i65) -> i1
    %32 = comb.extract %5 from 16 : (i65) -> i1
    %33 = comb.concat %false, %32 : i1, i1
    %34 = comb.mux bin %31, %c-2_i2, %33 : i2
    %35 = comb.mux bin %30, %c-1_i2, %34 : i2
    %36 = comb.mux bin %23, %29, %35 : i2
    %37 = comb.concat %23, %36 : i1, i2
    %38 = comb.extract %5 from 11 : (i65) -> i4
    %39 = comb.icmp bin ne %38, %c0_i4 : i4
    %40 = comb.extract %5 from 14 : (i65) -> i1
    %41 = comb.extract %5 from 13 : (i65) -> i1
    %42 = comb.extract %5 from 12 : (i65) -> i1
    %43 = comb.concat %false, %42 : i1, i1
    %44 = comb.mux bin %41, %c-2_i2, %43 : i2
    %45 = comb.mux bin %40, %c-1_i2, %44 : i2
    %46 = comb.extract %5 from 10 : (i65) -> i1
    %47 = comb.extract %5 from 9 : (i65) -> i1
    %48 = comb.extract %5 from 8 : (i65) -> i1
    %49 = comb.concat %false, %48 : i1, i1
    %50 = comb.mux bin %47, %c-2_i2, %49 : i2
    %51 = comb.mux bin %46, %c-1_i2, %50 : i2
    %52 = comb.mux bin %39, %45, %51 : i2
    %53 = comb.concat %39, %52 : i1, i2
    %54 = comb.mux bin %21, %37, %53 : i3
    %55 = comb.concat %21, %54 : i1, i3
    %56 = comb.extract %5 from 0 : (i65) -> i7
    %57 = comb.icmp bin ne %56, %c0_i7 : i7
    %58 = comb.extract %5 from 3 : (i65) -> i4
    %59 = comb.icmp bin ne %58, %c0_i4 : i4
    %60 = comb.extract %5 from 6 : (i65) -> i1
    %61 = comb.extract %5 from 5 : (i65) -> i1
    %62 = comb.extract %5 from 4 : (i65) -> i1
    %63 = comb.concat %false, %62 : i1, i1
    %64 = comb.mux bin %61, %c-2_i2, %63 : i2
    %65 = comb.mux bin %60, %c-1_i2, %64 : i2
    %66 = comb.extract %5 from 2 : (i65) -> i1
    %67 = comb.extract %5 from 1 : (i65) -> i1
    %68 = comb.extract %5 from 0 : (i65) -> i1
    %69 = comb.concat %false, %68 : i1, i1
    %70 = comb.mux bin %67, %c-2_i2, %69 : i2
    %71 = comb.mux bin %66, %c-1_i2, %70 : i2
    %72 = comb.mux bin %59, %65, %71 : i2
    %73 = comb.concat %59, %72 : i1, i2
    %74 = comb.mux bin %57, %73, %c0_i3 : i3
    %75 = comb.concat %57, %74 : i1, i3
    %76 = comb.mux bin %19, %55, %75 : i4
    %77 = comb.concat %19, %76 : i1, i4
    %78 = comb.xor bin %77, %c-1_i5 : i5
    %79 = comb.concat %c0_i31, %13 : i31, i23
    %80 = comb.concat %c0_i49, %78 : i49, i5
    %81 = comb.shl bin %79, %80 : i54
    %82 = comb.extract %81 from 0 : (i54) -> i22
    %83 = comb.concat %82, %false : i22, i1
    %84 = comb.concat %c-1_i4, %19, %76 : i4, i1, i4
    %85 = comb.concat %false, %12 : i1, i8
    %86 = comb.mux bin %14, %84, %85 : i9
    %87 = comb.mux bin %14, %c-2_i2, %c1_i2 : i2
    %88 = comb.concat %c32_i7, %87 : i7, i2
    %89 = comb.add %86, %88 : i9
    %90 = comb.extract %89 from 7 : (i9) -> i2
    %91 = comb.icmp bin eq %90, %c-1_i2 : i2
    %92 = comb.and bin %91, %15 : i1
    %93 = comb.replicate %17 : (i1) -> i3
    %94 = comb.xor %93, %c-1_i3 : i3
    %95 = comb.concat %92, %c0_i6 : i1, i6
    %96 = comb.extract %94 from 1 : (i3) -> i2
    %97 = comb.and bin %90, %96 : i2
    %98 = comb.extract %89 from 0 : (i9) -> i7
    %99 = comb.extract %94 from 0 : (i3) -> i1
    %100 = comb.concat %99, %c-1_i6 : i1, i6
    %101 = comb.and bin %98, %100 : i7
    %102 = comb.or bin %101, %95 : i7
    %103 = comb.mux bin %14, %83, %13 : i23
    %104 = comb.concat %c0_i32, %11, %97, %102, %103 : i32, i1, i2, i7, i23
    %105 = comb.extract %5 from 63 : (i65) -> i1
    %106 = comb.extract %5 from 52 : (i65) -> i11
    %107 = comb.extract %5 from 0 : (i65) -> i52
    %108 = comb.icmp bin eq %106, %c0_i11 : i11
    %109 = comb.icmp bin ne %107, %c0_i52 : i52
    %110 = comb.xor bin %109, %true : i1
    %111 = comb.and bin %108, %110 : i1
    %112 = comb.extract %5 from 20 : (i65) -> i32
    %113 = comb.icmp bin ne %112, %c0_i32 : i32
    %114 = comb.extract %5 from 36 : (i65) -> i16
    %115 = comb.icmp bin ne %114, %c0_i16 : i16
    %116 = comb.extract %5 from 44 : (i65) -> i8
    %117 = comb.icmp bin ne %116, %c0_i8 : i8
    %118 = comb.extract %5 from 48 : (i65) -> i4
    %119 = comb.icmp bin ne %118, %c0_i4 : i4
    %120 = comb.extract %5 from 51 : (i65) -> i1
    %121 = comb.extract %5 from 50 : (i65) -> i1
    %122 = comb.extract %5 from 49 : (i65) -> i1
    %123 = comb.concat %false, %122 : i1, i1
    %124 = comb.mux bin %121, %c-2_i2, %123 : i2
    %125 = comb.mux bin %120, %c-1_i2, %124 : i2
    %126 = comb.extract %5 from 47 : (i65) -> i1
    %127 = comb.extract %5 from 46 : (i65) -> i1
    %128 = comb.extract %5 from 45 : (i65) -> i1
    %129 = comb.concat %false, %128 : i1, i1
    %130 = comb.mux bin %127, %c-2_i2, %129 : i2
    %131 = comb.mux bin %126, %c-1_i2, %130 : i2
    %132 = comb.mux bin %119, %125, %131 : i2
    %133 = comb.concat %119, %132 : i1, i2
    %134 = comb.extract %5 from 40 : (i65) -> i4
    %135 = comb.icmp bin ne %134, %c0_i4 : i4
    %136 = comb.extract %5 from 43 : (i65) -> i1
    %137 = comb.extract %5 from 42 : (i65) -> i1
    %138 = comb.extract %5 from 41 : (i65) -> i1
    %139 = comb.concat %false, %138 : i1, i1
    %140 = comb.mux bin %137, %c-2_i2, %139 : i2
    %141 = comb.mux bin %136, %c-1_i2, %140 : i2
    %142 = comb.extract %5 from 39 : (i65) -> i1
    %143 = comb.extract %5 from 38 : (i65) -> i1
    %144 = comb.extract %5 from 37 : (i65) -> i1
    %145 = comb.concat %false, %144 : i1, i1
    %146 = comb.mux bin %143, %c-2_i2, %145 : i2
    %147 = comb.mux bin %142, %c-1_i2, %146 : i2
    %148 = comb.mux bin %135, %141, %147 : i2
    %149 = comb.concat %135, %148 : i1, i2
    %150 = comb.mux bin %117, %133, %149 : i3
    %151 = comb.concat %117, %150 : i1, i3
    %152 = comb.extract %5 from 28 : (i65) -> i8
    %153 = comb.icmp bin ne %152, %c0_i8 : i8
    %154 = comb.extract %5 from 32 : (i65) -> i4
    %155 = comb.icmp bin ne %154, %c0_i4 : i4
    %156 = comb.extract %5 from 35 : (i65) -> i1
    %157 = comb.extract %5 from 34 : (i65) -> i1
    %158 = comb.extract %5 from 33 : (i65) -> i1
    %159 = comb.concat %false, %158 : i1, i1
    %160 = comb.mux bin %157, %c-2_i2, %159 : i2
    %161 = comb.mux bin %156, %c-1_i2, %160 : i2
    %162 = comb.extract %5 from 31 : (i65) -> i1
    %163 = comb.extract %5 from 30 : (i65) -> i1
    %164 = comb.extract %5 from 29 : (i65) -> i1
    %165 = comb.concat %false, %164 : i1, i1
    %166 = comb.mux bin %163, %c-2_i2, %165 : i2
    %167 = comb.mux bin %162, %c-1_i2, %166 : i2
    %168 = comb.mux bin %155, %161, %167 : i2
    %169 = comb.concat %155, %168 : i1, i2
    %170 = comb.extract %5 from 24 : (i65) -> i4
    %171 = comb.icmp bin ne %170, %c0_i4 : i4
    %172 = comb.extract %5 from 27 : (i65) -> i1
    %173 = comb.extract %5 from 26 : (i65) -> i1
    %174 = comb.extract %5 from 25 : (i65) -> i1
    %175 = comb.concat %false, %174 : i1, i1
    %176 = comb.mux bin %173, %c-2_i2, %175 : i2
    %177 = comb.mux bin %172, %c-1_i2, %176 : i2
    %178 = comb.extract %5 from 23 : (i65) -> i1
    %179 = comb.extract %5 from 22 : (i65) -> i1
    %180 = comb.extract %5 from 21 : (i65) -> i1
    %181 = comb.concat %false, %180 : i1, i1
    %182 = comb.mux bin %179, %c-2_i2, %181 : i2
    %183 = comb.mux bin %178, %c-1_i2, %182 : i2
    %184 = comb.mux bin %171, %177, %183 : i2
    %185 = comb.concat %171, %184 : i1, i2
    %186 = comb.mux bin %153, %169, %185 : i3
    %187 = comb.concat %153, %186 : i1, i3
    %188 = comb.mux bin %115, %151, %187 : i4
    %189 = comb.concat %115, %188 : i1, i4
    %190 = comb.extract %5 from 4 : (i65) -> i16
    %191 = comb.icmp bin ne %190, %c0_i16 : i16
    %192 = comb.extract %5 from 12 : (i65) -> i8
    %193 = comb.icmp bin ne %192, %c0_i8 : i8
    %194 = comb.extract %5 from 16 : (i65) -> i4
    %195 = comb.icmp bin ne %194, %c0_i4 : i4
    %196 = comb.extract %5 from 19 : (i65) -> i1
    %197 = comb.extract %5 from 18 : (i65) -> i1
    %198 = comb.extract %5 from 17 : (i65) -> i1
    %199 = comb.concat %false, %198 : i1, i1
    %200 = comb.mux bin %197, %c-2_i2, %199 : i2
    %201 = comb.mux bin %196, %c-1_i2, %200 : i2
    %202 = comb.extract %5 from 15 : (i65) -> i1
    %203 = comb.extract %5 from 14 : (i65) -> i1
    %204 = comb.extract %5 from 13 : (i65) -> i1
    %205 = comb.concat %false, %204 : i1, i1
    %206 = comb.mux bin %203, %c-2_i2, %205 : i2
    %207 = comb.mux bin %202, %c-1_i2, %206 : i2
    %208 = comb.mux bin %195, %201, %207 : i2
    %209 = comb.concat %195, %208 : i1, i2
    %210 = comb.extract %5 from 8 : (i65) -> i4
    %211 = comb.icmp bin ne %210, %c0_i4 : i4
    %212 = comb.extract %5 from 11 : (i65) -> i1
    %213 = comb.extract %5 from 10 : (i65) -> i1
    %214 = comb.extract %5 from 9 : (i65) -> i1
    %215 = comb.concat %false, %214 : i1, i1
    %216 = comb.mux bin %213, %c-2_i2, %215 : i2
    %217 = comb.mux bin %212, %c-1_i2, %216 : i2
    %218 = comb.extract %5 from 7 : (i65) -> i1
    %219 = comb.extract %5 from 6 : (i65) -> i1
    %220 = comb.extract %5 from 5 : (i65) -> i1
    %221 = comb.concat %false, %220 : i1, i1
    %222 = comb.mux bin %219, %c-2_i2, %221 : i2
    %223 = comb.mux bin %218, %c-1_i2, %222 : i2
    %224 = comb.mux bin %211, %217, %223 : i2
    %225 = comb.concat %211, %224 : i1, i2
    %226 = comb.mux bin %193, %209, %225 : i3
    %227 = comb.concat %193, %226 : i1, i3
    %228 = comb.extract %5 from 0 : (i65) -> i4
    %229 = comb.icmp bin ne %228, %c0_i4 : i4
    %230 = comb.extract %5 from 0 : (i65) -> i4
    %231 = comb.icmp bin ne %230, %c0_i4 : i4
    %232 = comb.extract %5 from 3 : (i65) -> i1
    %233 = comb.extract %5 from 2 : (i65) -> i1
    %234 = comb.extract %5 from 1 : (i65) -> i1
    %235 = comb.concat %false, %234 : i1, i1
    %236 = comb.mux bin %233, %c-2_i2, %235 : i2
    %237 = comb.mux bin %232, %c-1_i2, %236 : i2
    %238 = comb.mux bin %231, %237, %c0_i2 : i2
    %239 = comb.concat %231, %238 : i1, i2
    %240 = comb.mux bin %229, %239, %c0_i3 : i3
    %241 = comb.concat %229, %240 : i1, i3
    %242 = comb.mux bin %191, %227, %241 : i4
    %243 = comb.concat %191, %242 : i1, i4
    %244 = comb.mux bin %113, %189, %243 : i5
    %245 = comb.concat %113, %244 : i1, i5
    %246 = comb.xor bin %245, %c-1_i6 : i6
    %247 = comb.concat %c0_i63, %107 : i63, i52
    %248 = comb.concat %c0_i109, %246 : i109, i6
    %249 = comb.shl bin %247, %248 : i115
    %250 = comb.extract %249 from 0 : (i115) -> i51
    %251 = comb.concat %250, %false : i51, i1
    %252 = comb.concat %c-1_i6, %113, %244 : i6, i1, i5
    %253 = comb.concat %false, %106 : i1, i11
    %254 = comb.mux bin %108, %252, %253 : i12
    %255 = comb.mux bin %108, %c-2_i2, %c1_i2 : i2
    %256 = comb.concat %c256_i10, %255 : i10, i2
    %257 = comb.add %254, %256 : i12
    %258 = comb.extract %257 from 10 : (i12) -> i2
    %259 = comb.icmp bin eq %258, %c-1_i2 : i2
    %260 = comb.and bin %259, %109 : i1
    %261 = comb.replicate %111 : (i1) -> i3
    %262 = comb.xor %261, %c-1_i3 : i3
    %263 = comb.concat %260, %c0_i9 : i1, i9
    %264 = comb.extract %262 from 1 : (i3) -> i2
    %265 = comb.and bin %258, %264 : i2
    %266 = comb.extract %257 from 0 : (i12) -> i10
    %267 = comb.extract %262 from 0 : (i3) -> i1
    %268 = comb.concat %267, %c-1_i9 : i1, i9
    %269 = comb.and bin %266, %268 : i10
    %270 = comb.or bin %269, %263 : i10
    %271 = comb.mux bin %108, %251, %107 : i52
    %272 = comb.concat %105, %265, %270, %271 : i1, i2, i10, i52
    %273 = comb.mux bin %2, %104, %272 : i65
    %274 = comb.extract %5 from 0 : (i65) -> i32
    %275 = comb.extract %4 from 1 : (i2) -> i1
    %276 = comb.extract %4 from 0 : (i2) -> i1
    %277 = comb.extract %5 from 31 : (i65) -> i1
    %278 = comb.xor %276, %true : i1
    %279 = comb.and %278, %277 : i1
    %280 = comb.extract %5 from 0 : (i65) -> i64
    %281 = comb.replicate %279 : (i1) -> i32
    %282 = comb.concat %281, %274 : i32, i32
    %283 = comb.mux %275, %280, %282 {sv.namehint = "INToRecFN.io_in"} : i64
    %284 = comb.extract %1 from 2 : (i5) -> i1
    %INToRecFN.io_out, %INToRecFN.io_exceptionFlags = hw.instance "INToRecFN" @INToRecFN(io_signedIn: %285: i1, io_in: %283: i64, io_roundingMode: %286: i2) -> (io_out: i33, io_exceptionFlags: i5) {sv.namehint = "INToRecFN.io_out"}
    %285 = comb.xor bin %276, %true {sv.namehint = "INToRecFN.io_signedIn"} : i1
    %286 = comb.extract %3 from 0 {sv.namehint = "INToRecFN.io_roundingMode"} : (i3) -> i2
    %INToRecFN_1.io_out, %INToRecFN_1.io_exceptionFlags = hw.instance "INToRecFN_1" @INToRecFN_1(io_signedIn: %285: i1, io_in: %283: i64, io_roundingMode: %286: i2) -> (io_out: i65, io_exceptionFlags: i5) {sv.namehint = "INToRecFN_1.io_out"}
    %287 = comb.extract %INToRecFN_1.io_out from 33 : (i65) -> i32
    %288 = comb.concat %287, %INToRecFN.io_out : i32, i33
    %289 = comb.mux bin %2, %288, %INToRecFN_1.io_out : i65
    %290 = comb.mux bin %284, %273, %289 {sv.namehint = "mux_data"} : i65
    %291 = comb.mux bin %2, %INToRecFN.io_exceptionFlags, %INToRecFN_1.io_exceptionFlags : i5
    %292 = comb.mux bin %284, %c0_i5, %291 {sv.namehint = "mux_exc"} : i5
    %293 = seq.firreg %295 clock %clock {firrtl.random_init_start = 222 : ui64, sv.namehint = "io_out_bits_data"} : i65
    %294 = seq.firreg %296 clock %clock {firrtl.random_init_start = 287 : ui64, sv.namehint = "io_out_bits_exc"} : i5
    %295 = comb.mux bin %0, %290, %293 : i65
    %296 = comb.mux bin %0, %292, %294 : i5
    hw.output %293, %294 : i65, i5
  }
  hw.module private @FPToFP(in %clock : !seq.clock, in %reset : i1, in %io_in_valid : i1, in %io_in_bits_cmd : i5, in %io_in_bits_single : i1, in %io_in_bits_rm : i3, in %io_in_bits_in1 : i65, in %io_in_bits_in2 : i65, out io_out_bits_data : i65, out io_out_bits_exc : i5, in %io_lt : i1) {
    %c-1_i3 = hw.constant -1 : i3
    %c3_i3 = hw.constant 3 : i3
    %c0_i4 = hw.constant 0 : i4
    %true = hw.constant true
    %false = hw.constant false
    %c16143152864309542912_i65 = hw.constant 16143152864309542912 : i65
    %c0_i5 = hw.constant 0 : i5
    %c16143152868071833600_i65 = hw.constant 16143152868071833600 : i65
    %0 = seq.firreg %io_in_valid clock %clock reset sync %reset, %false {firrtl.random_init_start = 0 : ui64, sv.namehint = "in_valid"} : i1
    %1 = seq.firreg %6 clock %clock {firrtl.random_init_start = 1 : ui64, sv.namehint = "in_bits_cmd"} : i5
    %2 = seq.firreg %7 clock %clock {firrtl.random_init_start = 13 : ui64, sv.namehint = "in_bits_single"} : i1
    %3 = seq.firreg %8 clock %clock {firrtl.random_init_start = 21 : ui64, sv.namehint = "in_bits_rm"} : i3
    %4 = seq.firreg %9 clock %clock {firrtl.random_init_start = 26 : ui64, sv.namehint = "RecFNToRecFN.io_in"} : i65
    %5 = seq.firreg %10 clock %clock {firrtl.random_init_start = 91 : ui64, sv.namehint = "in_bits_in2"} : i65
    %6 = comb.mux bin %io_in_valid, %io_in_bits_cmd, %1 : i5
    %7 = comb.mux bin %io_in_valid, %io_in_bits_single, %2 : i1
    %8 = comb.mux bin %io_in_valid, %io_in_bits_rm, %3 : i3
    %9 = comb.mux bin %io_in_valid, %io_in_bits_in1, %4 : i65
    %10 = comb.mux bin %io_in_valid, %io_in_bits_in2, %5 : i65
    %11 = comb.extract %3 from 1 : (i3) -> i1
    %12 = comb.extract %4 from 32 : (i65) -> i33
    %13 = comb.extract %5 from 32 : (i65) -> i33
    %14 = comb.xor %12, %13 : i33
    %15 = comb.extract %3 from 0 : (i3) -> i1
    %16 = comb.replicate %15 : (i1) -> i33
    %17 = comb.extract %5 from 32 : (i65) -> i33
    %18 = comb.xor %16, %17 : i33
    %19 = comb.mux %11, %14, %18 {sv.namehint = "signNum"} : i33
    %20 = comb.extract %19 from 0 : (i33) -> i1
    %21 = comb.extract %4 from 0 : (i65) -> i32
    %22 = comb.extract %4 from 33 : (i65) -> i32
    %23 = comb.concat %22, %20, %21 : i32, i1, i32
    %24 = comb.extract %19 from 32 : (i33) -> i1
    %25 = comb.extract %4 from 0 : (i65) -> i64
    %26 = comb.concat %24, %25 : i1, i64
    %27 = comb.mux bin %2, %23, %26 {sv.namehint = "fsgnj"} : i65
    %28 = comb.extract %1 from 2 : (i5) -> i2
    %29 = comb.extract %1 from 0 : (i5) -> i1
    %30 = comb.concat %28, %29 : i2, i1
    %31 = comb.icmp bin eq %30, %c3_i3 : i3
    %32 = comb.extract %4 from 29 : (i65) -> i3
    %33 = comb.icmp ne %32, %c-1_i3 : i3
    %34 = comb.xor bin %33, %true : i1
    %35 = comb.extract %5 from 29 : (i65) -> i3
    %36 = comb.icmp eq %35, %c-1_i3 : i3
    %37 = comb.extract %4 from 22 : (i65) -> i1
    %38 = comb.xor bin %37, %true : i1
    %39 = comb.and bin %34, %38 : i1
    %40 = comb.extract %5 from 22 : (i65) -> i1
    %41 = comb.xor bin %40, %true : i1
    %42 = comb.and bin %36, %41 : i1
    %43 = comb.or bin %39, %42 : i1
    %44 = comb.and bin %34, %36 : i1
    %45 = comb.or bin %43, %44 : i1
    %46 = comb.icmp bin ne %15, %io_lt : i1
    %47 = comb.and bin %46, %33 : i1
    %48 = comb.or bin %36, %47 : i1
    %49 = comb.extract %4 from 61 : (i65) -> i3
    %50 = comb.icmp ne %49, %c-1_i3 : i3
    %51 = comb.xor bin %50, %true : i1
    %52 = comb.extract %5 from 61 : (i65) -> i3
    %53 = comb.icmp eq %52, %c-1_i3 : i3
    %54 = comb.extract %4 from 51 : (i65) -> i1
    %55 = comb.xor bin %54, %true : i1
    %56 = comb.and bin %51, %55 : i1
    %57 = comb.extract %5 from 51 : (i65) -> i1
    %58 = comb.xor bin %57, %true : i1
    %59 = comb.and bin %53, %58 : i1
    %60 = comb.or bin %56, %59 : i1
    %61 = comb.and bin %51, %53 : i1
    %62 = comb.or bin %60, %61 : i1
    %63 = comb.and bin %46, %50 : i1
    %64 = comb.or bin %53, %63 : i1
    %65 = comb.mux bin %2, %48, %64 : i1
    %66 = comb.mux bin %2, %43, %60 : i1
    %67 = comb.mux bin %2, %45, %62 : i1
    %68 = comb.mux bin %2, %c16143152868071833600_i65, %c16143152864309542912_i65 : i65
    %69 = comb.concat %66, %c0_i4 : i1, i4
    %70 = comb.mux bin %31, %69, %c0_i5 : i5
    %71 = comb.mux bin %65, %4, %5 : i65
    %72 = comb.mux bin %67, %68, %71 : i65
    %73 = comb.mux bin %31, %72, %27 : i65
    %74 = comb.extract %1 from 2 : (i5) -> i1
    %RecFNToRecFN.io_out, %RecFNToRecFN.io_exceptionFlags = hw.instance "RecFNToRecFN" @RecFNToRecFN(io_in: %4: i65, io_roundingMode: %75: i2) -> (io_out: i33, io_exceptionFlags: i5) {sv.namehint = "RecFNToRecFN.io_out"}
    %75 = comb.extract %3 from 0 {sv.namehint = "RecFNToRecFN.io_roundingMode"} : (i3) -> i2
    %RecFNToRecFN_1.io_out, %RecFNToRecFN_1.io_exceptionFlags = hw.instance "RecFNToRecFN_1" @RecFNToRecFN_1(io_in: %76: i33) -> (io_out: i65, io_exceptionFlags: i5) {sv.namehint = "RecFNToRecFN_1.io_out"}
    %76 = comb.extract %4 from 0 {sv.namehint = "RecFNToRecFN_1.io_in"} : (i65) -> i33
    %77 = comb.extract %RecFNToRecFN_1.io_out from 33 : (i65) -> i32
    %78 = comb.concat %77, %RecFNToRecFN.io_out : i32, i33
    %79 = comb.mux bin %2, %78, %RecFNToRecFN_1.io_out : i65
    %80 = comb.mux bin %74, %73, %79 {sv.namehint = "mux_data"} : i65
    %81 = comb.mux bin %2, %RecFNToRecFN.io_exceptionFlags, %RecFNToRecFN_1.io_exceptionFlags : i5
    %82 = comb.mux bin %74, %70, %81 {sv.namehint = "mux_exc"} : i5
    %83 = seq.firreg %85 clock %clock {firrtl.random_init_start = 222 : ui64, sv.namehint = "io_out_bits_data"} : i65
    %84 = seq.firreg %86 clock %clock {firrtl.random_init_start = 287 : ui64, sv.namehint = "io_out_bits_exc"} : i5
    %85 = comb.mux bin %0, %80, %83 : i65
    %86 = comb.mux bin %0, %82, %84 : i5
    hw.output %83, %84 : i65, i5
  }
  hw.module private @FPUFMAPipe_1(in %clock : !seq.clock, in %reset : i1, in %io_in_valid : i1, in %io_in_bits_cmd : i5, in %io_in_bits_ren3 : i1, in %io_in_bits_swap23 : i1, in %io_in_bits_rm : i3, in %io_in_bits_in1 : i65, in %io_in_bits_in2 : i65, in %io_in_bits_in3 : i65, out io_out_bits_data : i65, out io_out_bits_exc : i5) {
    %c0_i3 = hw.constant 0 : i3
    %c0_i64 = hw.constant 0 : i64
    %false = hw.constant false
    %c9223372036854775808_i65 = hw.constant 9223372036854775808 : i65
    %0 = comb.extract %io_in_bits_in1 from 64 : (i65) -> i1
    %1 = comb.extract %io_in_bits_in2 from 64 : (i65) -> i1
    %2 = comb.xor bin %0, %1 : i1
    %3 = comb.concat %2, %c0_i64 {sv.namehint = "zero"} : i1, i64
    %valid = seq.firreg %io_in_valid clock %clock {firrtl.random_init_start = 0 : ui64} : i1
    %in_cmd = seq.firreg %11 clock %clock {firrtl.random_init_start = 1 : ui64} : i5
    %in_rm = seq.firreg %4 clock %clock {firrtl.random_init_start = 21 : ui64} : i3
    %in_in1 = seq.firreg %5 clock %clock {firrtl.random_init_start = 26 : ui64, sv.namehint = "in_in1"} : i65
    %in_in2 = seq.firreg %13 clock %clock {firrtl.random_init_start = 91 : ui64, sv.namehint = "in_in2"} : i65
    %in_in3 = seq.firreg %15 clock %clock {firrtl.random_init_start = 156 : ui64, sv.namehint = "in_in3"} : i65
    %4 = comb.mux bin %io_in_valid, %io_in_bits_rm, %in_rm : i3
    %5 = comb.mux bin %io_in_valid, %io_in_bits_in1, %in_in1 : i65
    %6 = comb.extract %io_in_bits_cmd from 1 : (i5) -> i1
    %7 = comb.or bin %io_in_bits_ren3, %io_in_bits_swap23 : i1
    %8 = comb.and bin %6, %7 : i1
    %9 = comb.extract %io_in_bits_cmd from 0 : (i5) -> i1
    %10 = comb.concat %c0_i3, %8, %9 : i3, i1, i1
    %11 = comb.mux bin %io_in_valid, %10, %in_cmd : i5
    %12 = comb.mux bin %io_in_bits_swap23, %c9223372036854775808_i65, %io_in_bits_in2 : i65
    %13 = comb.mux bin %io_in_valid, %12, %in_in2 : i65
    %14 = comb.mux bin %7, %io_in_bits_in3, %3 : i65
    %15 = comb.mux bin %io_in_valid, %14, %in_in3 : i65
    %fma.io_out, %fma.io_exceptionFlags = hw.instance "fma" @MulAddRecFN_1(io_op: %16: i2, io_a: %in_in1: i65, io_b: %in_in2: i65, io_c: %in_in3: i65, io_roundingMode: %17: i2) -> (io_out: i65, io_exceptionFlags: i5) {sv.namehint = "res_exc"}
    %16 = comb.extract %in_cmd from 0 {sv.namehint = "fma.io_op"} : (i5) -> i2
    %17 = comb.extract %in_rm from 0 {sv.namehint = "fma.io_roundingMode"} : (i3) -> i2
    %18 = seq.firreg %valid clock %clock reset sync %reset, %false {firrtl.random_init_start = 221 : ui64} : i1
    %19 = seq.firreg %21 clock %clock {firrtl.random_init_start = 222 : ui64} : i65
    %20 = seq.firreg %22 clock %clock {firrtl.random_init_start = 287 : ui64} : i5
    %21 = comb.mux bin %valid, %fma.io_out, %19 : i65
    %22 = comb.mux bin %valid, %fma.io_exceptionFlags, %20 : i5
    %23 = seq.firreg %18 clock %clock reset sync %reset, %false {firrtl.random_init_start = 292 : ui64} : i1
    %24 = seq.firreg %26 clock %clock {firrtl.random_init_start = 293 : ui64} : i65
    %25 = seq.firreg %27 clock %clock {firrtl.random_init_start = 358 : ui64} : i5
    %26 = comb.mux bin %18, %19, %24 : i65
    %27 = comb.mux bin %18, %20, %25 : i5
    %28 = seq.firreg %30 clock %clock {firrtl.random_init_start = 364 : ui64, sv.namehint = "io_out_bits_data"} : i65
    %29 = seq.firreg %31 clock %clock {firrtl.random_init_start = 429 : ui64, sv.namehint = "io_out_bits_exc"} : i5
    %30 = comb.mux bin %23, %24, %28 : i65
    %31 = comb.mux bin %23, %25, %29 : i5
    hw.output %28, %29 : i65, i5
  }
  hw.module private @DivSqrtRecF64(in %clock : !seq.clock, in %reset : i1, out io_inReady_div : i1, out io_inReady_sqrt : i1, in %io_inValid : i1, in %io_sqrtOp : i1, in %io_a : i65, in %io_b : i65, in %io_roundingMode : i2, out io_outValid_div : i1, out io_outValid_sqrt : i1, out io_out : i65, out io_exceptionFlags : i5) {
    %ds.io_inReady_div, %ds.io_inReady_sqrt, %ds.io_outValid_div, %ds.io_outValid_sqrt, %ds.io_out, %ds.io_exceptionFlags, %ds.io_usingMulAdd, %ds.io_latchMulAddA_0, %ds.io_mulAddA_0, %ds.io_latchMulAddB_0, %ds.io_mulAddB_0, %ds.io_mulAddC_2 = hw.instance "ds" @DivSqrtRecF64_mulAddZ31(clock: %clock: !seq.clock, reset: %reset: i1, io_inValid: %io_inValid: i1, io_sqrtOp: %io_sqrtOp: i1, io_a: %io_a: i65, io_b: %io_b: i65, io_roundingMode: %io_roundingMode: i2, io_mulAddResult_3: %mul.io_result_s3: i105) -> (io_inReady_div: i1, io_inReady_sqrt: i1, io_outValid_div: i1, io_outValid_sqrt: i1, io_out: i65, io_exceptionFlags: i5, io_usingMulAdd: i4, io_latchMulAddA_0: i1, io_mulAddA_0: i54, io_latchMulAddB_0: i1, io_mulAddB_0: i54, io_mulAddC_2: i105) {sv.namehint = "ds.io_out"}
    %mul.io_result_s3 = hw.instance "mul" @Mul54(clock: %clock: !seq.clock, io_val_s0: %0: i1, io_latch_a_s0: %ds.io_latchMulAddA_0: i1, io_a_s0: %ds.io_mulAddA_0: i54, io_latch_b_s0: %ds.io_latchMulAddB_0: i1, io_b_s0: %ds.io_mulAddB_0: i54, io_c_s2: %ds.io_mulAddC_2: i105) -> (io_result_s3: i105) {sv.namehint = "mul.io_result_s3"}
    %0 = comb.extract %ds.io_usingMulAdd from 0 {sv.namehint = "mul.io_val_s0"} : (i4) -> i1
    hw.output %ds.io_inReady_div, %ds.io_inReady_sqrt, %ds.io_outValid_div, %ds.io_outValid_sqrt, %ds.io_out, %ds.io_exceptionFlags : i1, i1, i1, i1, i65, i5
  }
  hw.module private @MulAddRecFN(in %io_op : i2, in %io_a : i33, in %io_b : i33, in %io_c : i33, in %io_roundingMode : i2, out io_out : i33, out io_exceptionFlags : i5) {
    %false = hw.constant false
    %c0_i24 = hw.constant 0 : i24
    %mulAddRecFN_preMul.io_mulAddA, %mulAddRecFN_preMul.io_mulAddB, %mulAddRecFN_preMul.io_mulAddC, %mulAddRecFN_preMul.io_toPostMul_highExpA, %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNA, %mulAddRecFN_preMul.io_toPostMul_highExpB, %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNB, %mulAddRecFN_preMul.io_toPostMul_signProd, %mulAddRecFN_preMul.io_toPostMul_isZeroProd, %mulAddRecFN_preMul.io_toPostMul_opSignC, %mulAddRecFN_preMul.io_toPostMul_highExpC, %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNC, %mulAddRecFN_preMul.io_toPostMul_isCDominant, %mulAddRecFN_preMul.io_toPostMul_CAlignDist_0, %mulAddRecFN_preMul.io_toPostMul_CAlignDist, %mulAddRecFN_preMul.io_toPostMul_bit0AlignedNegSigC, %mulAddRecFN_preMul.io_toPostMul_highAlignedNegSigC, %mulAddRecFN_preMul.io_toPostMul_sExpSum, %mulAddRecFN_preMul.io_toPostMul_roundingMode = hw.instance "mulAddRecFN_preMul" @MulAddRecFN_preMul(io_op: %io_op: i2, io_a: %io_a: i33, io_b: %io_b: i33, io_c: %io_c: i33, io_roundingMode: %io_roundingMode: i2) -> (io_mulAddA: i24, io_mulAddB: i24, io_mulAddC: i48, io_toPostMul_highExpA: i3, io_toPostMul_isNaN_isQuietNaNA: i1, io_toPostMul_highExpB: i3, io_toPostMul_isNaN_isQuietNaNB: i1, io_toPostMul_signProd: i1, io_toPostMul_isZeroProd: i1, io_toPostMul_opSignC: i1, io_toPostMul_highExpC: i3, io_toPostMul_isNaN_isQuietNaNC: i1, io_toPostMul_isCDominant: i1, io_toPostMul_CAlignDist_0: i1, io_toPostMul_CAlignDist: i7, io_toPostMul_bit0AlignedNegSigC: i1, io_toPostMul_highAlignedNegSigC: i26, io_toPostMul_sExpSum: i11, io_toPostMul_roundingMode: i2) {sv.namehint = "mulAddRecFN_preMul.io_mulAddC"}
    %mulAddRecFN_postMul.io_out, %mulAddRecFN_postMul.io_exceptionFlags = hw.instance "mulAddRecFN_postMul" @MulAddRecFN_postMul(io_fromPreMul_highExpA: %mulAddRecFN_preMul.io_toPostMul_highExpA: i3, io_fromPreMul_isNaN_isQuietNaNA: %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNA: i1, io_fromPreMul_highExpB: %mulAddRecFN_preMul.io_toPostMul_highExpB: i3, io_fromPreMul_isNaN_isQuietNaNB: %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNB: i1, io_fromPreMul_signProd: %mulAddRecFN_preMul.io_toPostMul_signProd: i1, io_fromPreMul_isZeroProd: %mulAddRecFN_preMul.io_toPostMul_isZeroProd: i1, io_fromPreMul_opSignC: %mulAddRecFN_preMul.io_toPostMul_opSignC: i1, io_fromPreMul_highExpC: %mulAddRecFN_preMul.io_toPostMul_highExpC: i3, io_fromPreMul_isNaN_isQuietNaNC: %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNC: i1, io_fromPreMul_isCDominant: %mulAddRecFN_preMul.io_toPostMul_isCDominant: i1, io_fromPreMul_CAlignDist_0: %mulAddRecFN_preMul.io_toPostMul_CAlignDist_0: i1, io_fromPreMul_CAlignDist: %mulAddRecFN_preMul.io_toPostMul_CAlignDist: i7, io_fromPreMul_bit0AlignedNegSigC: %mulAddRecFN_preMul.io_toPostMul_bit0AlignedNegSigC: i1, io_fromPreMul_highAlignedNegSigC: %mulAddRecFN_preMul.io_toPostMul_highAlignedNegSigC: i26, io_fromPreMul_sExpSum: %mulAddRecFN_preMul.io_toPostMul_sExpSum: i11, io_fromPreMul_roundingMode: %mulAddRecFN_preMul.io_toPostMul_roundingMode: i2, io_mulAddResult: %5: i49) -> (io_out: i33, io_exceptionFlags: i5) {sv.namehint = "mulAddRecFN_postMul.io_out"}
    %0 = comb.concat %c0_i24, %mulAddRecFN_preMul.io_mulAddA : i24, i24
    %1 = comb.concat %c0_i24, %mulAddRecFN_preMul.io_mulAddB : i24, i24
    %2 = comb.mul bin %0, %1 : i48
    %3 = comb.concat %false, %2 : i1, i48
    %4 = comb.concat %false, %mulAddRecFN_preMul.io_mulAddC : i1, i48
    %5 = comb.add %3, %4 {sv.namehint = "mulAddRecFN_postMul.io_mulAddResult"} : i49
    hw.output %mulAddRecFN_postMul.io_out, %mulAddRecFN_postMul.io_exceptionFlags : i33, i5
  }
  hw.module private @CompareRecFN(in %io_a : i65, in %io_b : i65, in %io_signaling : i1, out io_lt : i1, out io_eq : i1, out io_exceptionFlags : i5) {
    %c-1_i2 = hw.constant -1 : i2
    %true = hw.constant true
    %c0_i3 = hw.constant 0 : i3
    %false = hw.constant false
    %c0_i4 = hw.constant 0 : i4
    %0 = comb.extract %io_a from 52 : (i65) -> i12
    %1 = comb.extract %io_a from 61 : (i65) -> i3
    %2 = comb.icmp bin ne %1, %c0_i3 : i3
    %3 = comb.xor bin %2, %true {sv.namehint = "rawA_isZero"} : i1
    %4 = comb.extract %io_a from 62 : (i65) -> i2
    %5 = comb.icmp bin eq %4, %c-1_i2 : i2
    %6 = comb.extract %io_a from 64 {sv.namehint = "rawA_sign"} : (i65) -> i1
    %7 = comb.extract %io_a from 61 : (i65) -> i1
    %8 = comb.and bin %5, %7 {sv.namehint = "rawA_isNaN"} : i1
    %9 = comb.xor bin %7, %true : i1
    %10 = comb.concat %false, %0 {sv.namehint = "rawA_sExp"} : i1, i12
    %11 = comb.extract %io_a from 0 : (i65) -> i52
    %12 = comb.extract %io_b from 52 : (i65) -> i12
    %13 = comb.extract %io_b from 61 : (i65) -> i3
    %14 = comb.icmp bin ne %13, %c0_i3 : i3
    %15 = comb.xor bin %14, %true {sv.namehint = "rawB_isZero"} : i1
    %16 = comb.extract %io_b from 62 : (i65) -> i2
    %17 = comb.icmp bin eq %16, %c-1_i2 : i2
    %18 = comb.extract %io_b from 64 {sv.namehint = "rawB_sign"} : (i65) -> i1
    %19 = comb.extract %io_b from 61 : (i65) -> i1
    %20 = comb.and bin %17, %19 {sv.namehint = "rawB_isNaN"} : i1
    %21 = comb.xor bin %19, %true : i1
    %22 = comb.concat %false, %12 {sv.namehint = "rawB_sExp"} : i1, i12
    %23 = comb.extract %io_b from 0 : (i65) -> i52
    %24 = comb.xor bin %8, %true : i1
    %25 = comb.xor bin %20, %true : i1
    %26 = comb.and bin %24, %25 {sv.namehint = "ordered"} : i1
    %27 = comb.and bin %5, %9, %17, %21 {sv.namehint = "bothInfs"} : i1
    %28 = comb.and bin %3, %15 {sv.namehint = "bothZeros"} : i1
    %29 = comb.icmp bin eq %0, %12 {sv.namehint = "eqExps"} : i12
    %30 = comb.icmp bin slt %10, %22 : i13
    %31 = comb.concat %2, %11 : i1, i52
    %32 = comb.concat %14, %23 : i1, i52
    %33 = comb.icmp bin ult %31, %32 : i53
    %34 = comb.and bin %29, %33 : i1
    %35 = comb.or bin %30, %34 {sv.namehint = "common_ltMags"} : i1
    %36 = comb.concat %2, %11 : i1, i52
    %37 = comb.concat %14, %23 : i1, i52
    %38 = comb.icmp bin eq %36, %37 : i53
    %39 = comb.and bin %29, %38 {sv.namehint = "common_eqMags"} : i1
    %40 = comb.xor bin %28, %true : i1
    %41 = comb.xor bin %18, %true : i1
    %42 = comb.and bin %6, %41 : i1
    %43 = comb.xor bin %27, %true : i1
    %44 = comb.xor bin %35, %true : i1
    %45 = comb.xor bin %39, %true : i1
    %46 = comb.and bin %6, %44, %45 : i1
    %47 = comb.and bin %41, %35 : i1
    %48 = comb.or bin %46, %47 : i1
    %49 = comb.and bin %43, %48 : i1
    %50 = comb.or bin %42, %49 : i1
    %51 = comb.icmp bin eq %6, %18 : i1
    %52 = comb.or bin %27, %39 : i1
    %53 = comb.and bin %51, %52 : i1
    %54 = comb.or bin %28, %53 {sv.namehint = "ordered_eq"} : i1
    %55 = comb.extract %io_a from 51 : (i65) -> i1
    %56 = comb.xor bin %55, %true : i1
    %57 = comb.and bin %8, %56 : i1
    %58 = comb.extract %io_b from 51 : (i65) -> i1
    %59 = comb.xor bin %58, %true : i1
    %60 = comb.and bin %20, %59 : i1
    %61 = comb.xor bin %26, %true : i1
    %62 = comb.and bin %io_signaling, %61 : i1
    %63 = comb.or bin %57, %60, %62 {sv.namehint = "invalid"} : i1
    %64 = comb.and bin %26, %40, %50 {sv.namehint = "io_lt"} : i1
    %65 = comb.and bin %26, %54 {sv.namehint = "io_eq"} : i1
    %66 = comb.concat %63, %c0_i4 {sv.namehint = "io_exceptionFlags"} : i1, i4
    hw.output %64, %65, %66 : i1, i1, i5
  }
  hw.module private @RecFNToIN(in %io_in : i65, in %io_roundingMode : i2, in %io_signedOut : i1, out io_out : i32, out io_intExceptionFlags : i3) {
    %c0_i6 = hw.constant 0 : i6
    %c-1_i30 = hw.constant -1 : i30
    %c1_i32 = hw.constant 1 : i32
    %c-1_i11 = hw.constant -1 : i11
    %c30_i11 = hw.constant 30 : i11
    %c31_i11 = hw.constant 31 : i11
    %c0_i2 = hw.constant 0 : i2
    %c0_i51 = hw.constant 0 : i51
    %c0_i79 = hw.constant 0 : i79
    %c-1_i2 = hw.constant -1 : i2
    %c0_i3 = hw.constant 0 : i3
    %c-2_i2 = hw.constant -2 : i2
    %true = hw.constant true
    %c0_i5 = hw.constant 0 : i5
    %c0_i31 = hw.constant 0 : i31
    %c0_i32 = hw.constant 0 : i32
    %0 = comb.extract %io_in from 64 {sv.namehint = "sign"} : (i65) -> i1
    %1 = comb.extract %io_in from 0 {sv.namehint = "fract"} : (i65) -> i52
    %2 = comb.extract %io_in from 61 : (i65) -> i3
    %3 = comb.icmp bin ne %2, %c0_i3 : i3
    %4 = comb.extract %io_in from 62 : (i65) -> i2
    %5 = comb.icmp bin eq %4, %c-1_i2 {sv.namehint = "invalid"} : i2
    %6 = comb.extract %io_in from 61 : (i65) -> i1
    %7 = comb.and bin %5, %6 {sv.namehint = "isNaN"} : i1
    %8 = comb.extract %io_in from 63 {sv.namehint = "notSpecial_magGeOne"} : (i65) -> i1
    %9 = comb.extract %io_in from 52 : (i65) -> i5
    %10 = comb.mux bin %8, %9, %c0_i5 : i5
    %11 = comb.concat %c0_i31, %8, %1 : i31, i1, i52
    %12 = comb.concat %c0_i79, %10 : i79, i5
    %13 = comb.shl bin %11, %12 {sv.namehint = "shiftedSig"} : i84
    %14 = comb.extract %13 from 52 {sv.namehint = "unroundedInt"} : (i84) -> i32
    %15 = comb.extract %13 from 51 : (i84) -> i2
    %16 = comb.extract %13 from 0 : (i84) -> i51
    %17 = comb.icmp bin ne %16, %c0_i51 : i51
    %18 = comb.extract %13 from 51 : (i84) -> i1
    %19 = comb.concat %18, %17 : i1, i1
    %20 = comb.icmp bin ne %19, %c0_i2 : i2
    %21 = comb.mux bin %8, %20, %3 {sv.namehint = "roundInexact"} : i1
    %22 = comb.icmp eq %15, %c-1_i2 : i2
    %23 = comb.icmp eq %19, %c-1_i2 : i2
    %24 = comb.or bin %22, %23 : i1
    %25 = comb.extract %io_in from 52 {sv.namehint = "posExp"} : (i65) -> i11
    %26 = comb.icmp eq %25, %c-1_i11 : i11
    %27 = comb.and %26, %20 : i1
    %28 = comb.mux bin %8, %24, %27 {sv.namehint = "roundIncr_nearestEven"} : i1
    %29 = comb.icmp bin eq %io_roundingMode, %c0_i2 : i2
    %30 = comb.and bin %29, %28 : i1
    %31 = comb.icmp bin eq %io_roundingMode, %c-2_i2 : i2
    %32 = comb.and bin %31, %0, %21 : i1
    %33 = comb.icmp bin eq %io_roundingMode, %c-1_i2 : i2
    %34 = comb.xor bin %0, %true : i1
    %35 = comb.and bin %33, %34, %21 : i1
    %36 = comb.or bin %30, %32, %35 {sv.namehint = "roundIncr"} : i1
    %37 = comb.replicate %0 : (i1) -> i32
    %38 = comb.xor %37, %14 {sv.namehint = "complUnroundedInt"} : i32
    %39 = comb.xor bin %36, %0 : i1
    %40 = comb.add %38, %c1_i32 : i32
    %41 = comb.mux bin %39, %40, %38 {sv.namehint = "roundedInt"} : i32
    %42 = comb.extract %13 from 52 : (i84) -> i30
    %43 = comb.icmp eq %42, %c-1_i30 : i30
    %44 = comb.and bin %43, %36 {sv.namehint = "roundCarryBut2"} : i1
    %45 = comb.extract %io_in from 57 : (i65) -> i6
    %46 = comb.icmp bin ne %45, %c0_i6 : i6
    %47 = comb.icmp bin eq %25, %c31_i11 : i11
    %48 = comb.extract %13 from 52 : (i84) -> i31
    %49 = comb.concat %34, %48 : i1, i31
    %50 = comb.icmp bin ne %49, %c0_i32 : i32
    %51 = comb.or bin %50, %36 : i1
    %52 = comb.and bin %47, %51 : i1
    %53 = comb.icmp bin eq %25, %c30_i11 : i11
    %54 = comb.and bin %34, %53, %44 : i1
    %55 = comb.or bin %46, %52, %54 : i1
    %56 = comb.and %8, %55 {sv.namehint = "overflow_signed"} : i1
    %57 = comb.extract %13 from 82 : (i84) -> i1
    %58 = comb.and bin %47, %57, %44 : i1
    %59 = comb.or bin %0, %46, %58 : i1
    %60 = comb.and bin %0, %36 : i1
    %61 = comb.mux bin %8, %59, %60 {sv.namehint = "overflow_unsigned"} : i1
    %62 = comb.mux bin %io_signedOut, %56, %61 {sv.namehint = "overflow"} : i1
    %63 = comb.xor bin %7, %true : i1
    %64 = comb.and bin %0, %63 {sv.namehint = "excSign"} : i1
    %65 = comb.and bin %io_signedOut, %64 : i1
    %66 = comb.xor bin %64, %true : i1
    %67 = comb.and bin %io_signedOut, %66 : i1
    %68 = comb.replicate %67 : (i1) -> i31
    %69 = comb.concat %65, %68 : i1, i31
    %70 = comb.xor bin %io_signedOut, %true : i1
    %71 = comb.and bin %70, %66 : i1
    %72 = comb.replicate %71 : (i1) -> i32
    %73 = comb.or bin %69, %72 {sv.namehint = "excValue"} : i32
    %74 = comb.xor bin %5, %true : i1
    %75 = comb.xor bin %62, %true : i1
    %76 = comb.and bin %21, %74, %75 {sv.namehint = "inexact"} : i1
    %77 = comb.or bin %5, %62 : i1
    %78 = comb.mux bin %77, %73, %41 {sv.namehint = "io_out"} : i32
    %79 = comb.concat %5, %62, %76 {sv.namehint = "io_intExceptionFlags"} : i1, i1, i1
    hw.output %78, %79 : i32, i3
  }
  hw.module private @RecFNToIN_1(in %io_in : i65, in %io_roundingMode : i2, in %io_signedOut : i1, out io_out : i64, out io_intExceptionFlags : i3) {
    %c0_i5 = hw.constant 0 : i5
    %c-1_i62 = hw.constant -1 : i62
    %c1_i64 = hw.constant 1 : i64
    %c-1_i11 = hw.constant -1 : i11
    %c62_i11 = hw.constant 62 : i11
    %c63_i11 = hw.constant 63 : i11
    %c0_i2 = hw.constant 0 : i2
    %c0_i51 = hw.constant 0 : i51
    %c0_i110 = hw.constant 0 : i110
    %c-1_i2 = hw.constant -1 : i2
    %c0_i3 = hw.constant 0 : i3
    %c-2_i2 = hw.constant -2 : i2
    %true = hw.constant true
    %c0_i6 = hw.constant 0 : i6
    %c0_i63 = hw.constant 0 : i63
    %c0_i64 = hw.constant 0 : i64
    %0 = comb.extract %io_in from 64 {sv.namehint = "sign"} : (i65) -> i1
    %1 = comb.extract %io_in from 0 {sv.namehint = "fract"} : (i65) -> i52
    %2 = comb.extract %io_in from 61 : (i65) -> i3
    %3 = comb.icmp bin ne %2, %c0_i3 : i3
    %4 = comb.extract %io_in from 62 : (i65) -> i2
    %5 = comb.icmp bin eq %4, %c-1_i2 {sv.namehint = "invalid"} : i2
    %6 = comb.extract %io_in from 61 : (i65) -> i1
    %7 = comb.and bin %5, %6 {sv.namehint = "isNaN"} : i1
    %8 = comb.extract %io_in from 63 {sv.namehint = "notSpecial_magGeOne"} : (i65) -> i1
    %9 = comb.extract %io_in from 52 : (i65) -> i6
    %10 = comb.mux bin %8, %9, %c0_i6 : i6
    %11 = comb.concat %c0_i63, %8, %1 : i63, i1, i52
    %12 = comb.concat %c0_i110, %10 : i110, i6
    %13 = comb.shl bin %11, %12 {sv.namehint = "shiftedSig"} : i116
    %14 = comb.extract %13 from 52 {sv.namehint = "unroundedInt"} : (i116) -> i64
    %15 = comb.extract %13 from 51 : (i116) -> i2
    %16 = comb.extract %13 from 0 : (i116) -> i51
    %17 = comb.icmp bin ne %16, %c0_i51 : i51
    %18 = comb.extract %13 from 51 : (i116) -> i1
    %19 = comb.concat %18, %17 : i1, i1
    %20 = comb.icmp bin ne %19, %c0_i2 : i2
    %21 = comb.mux bin %8, %20, %3 {sv.namehint = "roundInexact"} : i1
    %22 = comb.icmp eq %15, %c-1_i2 : i2
    %23 = comb.icmp eq %19, %c-1_i2 : i2
    %24 = comb.or bin %22, %23 : i1
    %25 = comb.extract %io_in from 52 {sv.namehint = "posExp"} : (i65) -> i11
    %26 = comb.icmp eq %25, %c-1_i11 : i11
    %27 = comb.and %26, %20 : i1
    %28 = comb.mux bin %8, %24, %27 {sv.namehint = "roundIncr_nearestEven"} : i1
    %29 = comb.icmp bin eq %io_roundingMode, %c0_i2 : i2
    %30 = comb.and bin %29, %28 : i1
    %31 = comb.icmp bin eq %io_roundingMode, %c-2_i2 : i2
    %32 = comb.and bin %31, %0, %21 : i1
    %33 = comb.icmp bin eq %io_roundingMode, %c-1_i2 : i2
    %34 = comb.xor bin %0, %true : i1
    %35 = comb.and bin %33, %34, %21 : i1
    %36 = comb.or bin %30, %32, %35 {sv.namehint = "roundIncr"} : i1
    %37 = comb.replicate %0 : (i1) -> i64
    %38 = comb.xor %37, %14 {sv.namehint = "complUnroundedInt"} : i64
    %39 = comb.xor bin %36, %0 : i1
    %40 = comb.add %38, %c1_i64 : i64
    %41 = comb.mux bin %39, %40, %38 {sv.namehint = "roundedInt"} : i64
    %42 = comb.extract %13 from 52 : (i116) -> i62
    %43 = comb.icmp eq %42, %c-1_i62 : i62
    %44 = comb.and bin %43, %36 {sv.namehint = "roundCarryBut2"} : i1
    %45 = comb.extract %io_in from 58 : (i65) -> i5
    %46 = comb.icmp bin ne %45, %c0_i5 : i5
    %47 = comb.icmp bin eq %25, %c63_i11 : i11
    %48 = comb.extract %13 from 52 : (i116) -> i63
    %49 = comb.concat %34, %48 : i1, i63
    %50 = comb.icmp bin ne %49, %c0_i64 : i64
    %51 = comb.or bin %50, %36 : i1
    %52 = comb.and bin %47, %51 : i1
    %53 = comb.icmp bin eq %25, %c62_i11 : i11
    %54 = comb.and bin %34, %53, %44 : i1
    %55 = comb.or bin %46, %52, %54 : i1
    %56 = comb.and %8, %55 {sv.namehint = "overflow_signed"} : i1
    %57 = comb.extract %13 from 114 : (i116) -> i1
    %58 = comb.and bin %47, %57, %44 : i1
    %59 = comb.or bin %0, %46, %58 : i1
    %60 = comb.and bin %0, %36 : i1
    %61 = comb.mux bin %8, %59, %60 {sv.namehint = "overflow_unsigned"} : i1
    %62 = comb.mux bin %io_signedOut, %56, %61 {sv.namehint = "overflow"} : i1
    %63 = comb.xor bin %7, %true : i1
    %64 = comb.and bin %0, %63 {sv.namehint = "excSign"} : i1
    %65 = comb.and bin %io_signedOut, %64 : i1
    %66 = comb.xor bin %64, %true : i1
    %67 = comb.and bin %io_signedOut, %66 : i1
    %68 = comb.replicate %67 : (i1) -> i63
    %69 = comb.concat %65, %68 : i1, i63
    %70 = comb.xor bin %io_signedOut, %true : i1
    %71 = comb.and bin %70, %66 : i1
    %72 = comb.replicate %71 : (i1) -> i64
    %73 = comb.or bin %69, %72 {sv.namehint = "excValue"} : i64
    %74 = comb.xor bin %5, %true : i1
    %75 = comb.xor bin %62, %true : i1
    %76 = comb.and bin %21, %74, %75 {sv.namehint = "inexact"} : i1
    %77 = comb.or bin %5, %62 : i1
    %78 = comb.mux bin %77, %73, %41 {sv.namehint = "io_out"} : i64
    %79 = comb.concat %5, %62, %76 {sv.namehint = "io_intExceptionFlags"} : i1, i1, i1
    hw.output %78, %79 : i64, i3
  }
  hw.module private @INToRecFN(in %io_signedIn : i1, in %io_in : i64, in %io_roundingMode : i2, out io_out : i33, out io_exceptionFlags : i5) {
    %c0_i7 = hw.constant 0 : i7
    %c1_i25 = hw.constant 1 : i25
    %c0_i64 = hw.constant 0 : i64
    %c0_i39 = hw.constant 0 : i39
    %c0_i121 = hw.constant 0 : i121
    %c0_i63 = hw.constant 0 : i63
    %c-1_i6 = hw.constant -1 : i6
    %c0_i8 = hw.constant 0 : i8
    %c0_i16 = hw.constant 0 : i16
    %c0_i32 = hw.constant 0 : i32
    %false = hw.constant false
    %c-2_i2 = hw.constant -2 : i2
    %c-1_i2 = hw.constant -1 : i2
    %true = hw.constant true
    %c0_i2 = hw.constant 0 : i2
    %c0_i4 = hw.constant 0 : i4
    %0 = comb.extract %io_in from 63 : (i64) -> i1
    %1 = comb.and bin %io_signedIn, %0 {sv.namehint = "sign"} : i1
    %2 = comb.sub %c0_i64, %io_in : i64
    %3 = comb.mux bin %1, %2, %io_in {sv.namehint = "absIn"} : i64
    %4 = comb.extract %3 from 32 : (i64) -> i32
    %5 = comb.icmp bin ne %4, %c0_i32 : i32
    %6 = comb.extract %3 from 48 : (i64) -> i16
    %7 = comb.icmp bin ne %6, %c0_i16 : i16
    %8 = comb.extract %3 from 56 : (i64) -> i8
    %9 = comb.icmp bin ne %8, %c0_i8 : i8
    %10 = comb.extract %3 from 60 : (i64) -> i4
    %11 = comb.icmp bin ne %10, %c0_i4 : i4
    %12 = comb.extract %3 from 63 : (i64) -> i1
    %13 = comb.extract %3 from 62 : (i64) -> i1
    %14 = comb.extract %3 from 61 : (i64) -> i1
    %15 = comb.concat %false, %14 : i1, i1
    %16 = comb.mux bin %13, %c-2_i2, %15 : i2
    %17 = comb.mux bin %12, %c-1_i2, %16 : i2
    %18 = comb.extract %3 from 59 : (i64) -> i1
    %19 = comb.extract %3 from 58 : (i64) -> i1
    %20 = comb.extract %3 from 57 : (i64) -> i1
    %21 = comb.concat %false, %20 : i1, i1
    %22 = comb.mux bin %19, %c-2_i2, %21 : i2
    %23 = comb.mux bin %18, %c-1_i2, %22 : i2
    %24 = comb.mux bin %11, %17, %23 : i2
    %25 = comb.concat %11, %24 : i1, i2
    %26 = comb.extract %3 from 52 : (i64) -> i4
    %27 = comb.icmp bin ne %26, %c0_i4 : i4
    %28 = comb.extract %3 from 55 : (i64) -> i1
    %29 = comb.extract %3 from 54 : (i64) -> i1
    %30 = comb.extract %3 from 53 : (i64) -> i1
    %31 = comb.concat %false, %30 : i1, i1
    %32 = comb.mux bin %29, %c-2_i2, %31 : i2
    %33 = comb.mux bin %28, %c-1_i2, %32 : i2
    %34 = comb.extract %3 from 51 : (i64) -> i1
    %35 = comb.extract %3 from 50 : (i64) -> i1
    %36 = comb.extract %3 from 49 : (i64) -> i1
    %37 = comb.concat %false, %36 : i1, i1
    %38 = comb.mux bin %35, %c-2_i2, %37 : i2
    %39 = comb.mux bin %34, %c-1_i2, %38 : i2
    %40 = comb.mux bin %27, %33, %39 : i2
    %41 = comb.concat %27, %40 : i1, i2
    %42 = comb.mux bin %9, %25, %41 : i3
    %43 = comb.concat %9, %42 : i1, i3
    %44 = comb.extract %3 from 40 : (i64) -> i8
    %45 = comb.icmp bin ne %44, %c0_i8 : i8
    %46 = comb.extract %3 from 44 : (i64) -> i4
    %47 = comb.icmp bin ne %46, %c0_i4 : i4
    %48 = comb.extract %3 from 47 : (i64) -> i1
    %49 = comb.extract %3 from 46 : (i64) -> i1
    %50 = comb.extract %3 from 45 : (i64) -> i1
    %51 = comb.concat %false, %50 : i1, i1
    %52 = comb.mux bin %49, %c-2_i2, %51 : i2
    %53 = comb.mux bin %48, %c-1_i2, %52 : i2
    %54 = comb.extract %3 from 43 : (i64) -> i1
    %55 = comb.extract %3 from 42 : (i64) -> i1
    %56 = comb.extract %3 from 41 : (i64) -> i1
    %57 = comb.concat %false, %56 : i1, i1
    %58 = comb.mux bin %55, %c-2_i2, %57 : i2
    %59 = comb.mux bin %54, %c-1_i2, %58 : i2
    %60 = comb.mux bin %47, %53, %59 : i2
    %61 = comb.concat %47, %60 : i1, i2
    %62 = comb.extract %3 from 36 : (i64) -> i4
    %63 = comb.icmp bin ne %62, %c0_i4 : i4
    %64 = comb.extract %3 from 39 : (i64) -> i1
    %65 = comb.extract %3 from 38 : (i64) -> i1
    %66 = comb.extract %3 from 37 : (i64) -> i1
    %67 = comb.concat %false, %66 : i1, i1
    %68 = comb.mux bin %65, %c-2_i2, %67 : i2
    %69 = comb.mux bin %64, %c-1_i2, %68 : i2
    %70 = comb.extract %3 from 35 : (i64) -> i1
    %71 = comb.extract %3 from 34 : (i64) -> i1
    %72 = comb.extract %3 from 33 : (i64) -> i1
    %73 = comb.concat %false, %72 : i1, i1
    %74 = comb.mux bin %71, %c-2_i2, %73 : i2
    %75 = comb.mux bin %70, %c-1_i2, %74 : i2
    %76 = comb.mux bin %63, %69, %75 : i2
    %77 = comb.concat %63, %76 : i1, i2
    %78 = comb.mux bin %45, %61, %77 : i3
    %79 = comb.concat %45, %78 : i1, i3
    %80 = comb.mux bin %7, %43, %79 : i4
    %81 = comb.concat %7, %80 : i1, i4
    %82 = comb.extract %3 from 16 : (i64) -> i16
    %83 = comb.icmp bin ne %82, %c0_i16 : i16
    %84 = comb.extract %3 from 24 : (i64) -> i8
    %85 = comb.icmp bin ne %84, %c0_i8 : i8
    %86 = comb.extract %3 from 28 : (i64) -> i4
    %87 = comb.icmp bin ne %86, %c0_i4 : i4
    %88 = comb.extract %3 from 31 : (i64) -> i1
    %89 = comb.extract %3 from 30 : (i64) -> i1
    %90 = comb.extract %3 from 29 : (i64) -> i1
    %91 = comb.concat %false, %90 : i1, i1
    %92 = comb.mux bin %89, %c-2_i2, %91 : i2
    %93 = comb.mux bin %88, %c-1_i2, %92 : i2
    %94 = comb.extract %3 from 27 : (i64) -> i1
    %95 = comb.extract %3 from 26 : (i64) -> i1
    %96 = comb.extract %3 from 25 : (i64) -> i1
    %97 = comb.concat %false, %96 : i1, i1
    %98 = comb.mux bin %95, %c-2_i2, %97 : i2
    %99 = comb.mux bin %94, %c-1_i2, %98 : i2
    %100 = comb.mux bin %87, %93, %99 : i2
    %101 = comb.concat %87, %100 : i1, i2
    %102 = comb.extract %3 from 20 : (i64) -> i4
    %103 = comb.icmp bin ne %102, %c0_i4 : i4
    %104 = comb.extract %3 from 23 : (i64) -> i1
    %105 = comb.extract %3 from 22 : (i64) -> i1
    %106 = comb.extract %3 from 21 : (i64) -> i1
    %107 = comb.concat %false, %106 : i1, i1
    %108 = comb.mux bin %105, %c-2_i2, %107 : i2
    %109 = comb.mux bin %104, %c-1_i2, %108 : i2
    %110 = comb.extract %3 from 19 : (i64) -> i1
    %111 = comb.extract %3 from 18 : (i64) -> i1
    %112 = comb.extract %3 from 17 : (i64) -> i1
    %113 = comb.concat %false, %112 : i1, i1
    %114 = comb.mux bin %111, %c-2_i2, %113 : i2
    %115 = comb.mux bin %110, %c-1_i2, %114 : i2
    %116 = comb.mux bin %103, %109, %115 : i2
    %117 = comb.concat %103, %116 : i1, i2
    %118 = comb.mux bin %85, %101, %117 : i3
    %119 = comb.concat %85, %118 : i1, i3
    %120 = comb.extract %3 from 8 : (i64) -> i8
    %121 = comb.icmp bin ne %120, %c0_i8 : i8
    %122 = comb.extract %3 from 12 : (i64) -> i4
    %123 = comb.icmp bin ne %122, %c0_i4 : i4
    %124 = comb.extract %3 from 15 : (i64) -> i1
    %125 = comb.extract %3 from 14 : (i64) -> i1
    %126 = comb.extract %3 from 13 : (i64) -> i1
    %127 = comb.concat %false, %126 : i1, i1
    %128 = comb.mux bin %125, %c-2_i2, %127 : i2
    %129 = comb.mux bin %124, %c-1_i2, %128 : i2
    %130 = comb.extract %3 from 11 : (i64) -> i1
    %131 = comb.extract %3 from 10 : (i64) -> i1
    %132 = comb.extract %3 from 9 : (i64) -> i1
    %133 = comb.concat %false, %132 : i1, i1
    %134 = comb.mux bin %131, %c-2_i2, %133 : i2
    %135 = comb.mux bin %130, %c-1_i2, %134 : i2
    %136 = comb.mux bin %123, %129, %135 : i2
    %137 = comb.concat %123, %136 : i1, i2
    %138 = comb.extract %3 from 4 : (i64) -> i4
    %139 = comb.icmp bin ne %138, %c0_i4 : i4
    %140 = comb.extract %3 from 7 : (i64) -> i1
    %141 = comb.extract %3 from 6 : (i64) -> i1
    %142 = comb.extract %3 from 5 : (i64) -> i1
    %143 = comb.concat %false, %142 : i1, i1
    %144 = comb.mux bin %141, %c-2_i2, %143 : i2
    %145 = comb.mux bin %140, %c-1_i2, %144 : i2
    %146 = comb.extract %3 from 3 : (i64) -> i1
    %147 = comb.extract %3 from 2 : (i64) -> i1
    %148 = comb.extract %3 from 1 : (i64) -> i1
    %149 = comb.concat %false, %148 : i1, i1
    %150 = comb.mux bin %147, %c-2_i2, %149 : i2
    %151 = comb.mux bin %146, %c-1_i2, %150 : i2
    %152 = comb.mux bin %139, %145, %151 : i2
    %153 = comb.concat %139, %152 : i1, i2
    %154 = comb.mux bin %121, %137, %153 : i3
    %155 = comb.concat %121, %154 : i1, i3
    %156 = comb.mux bin %83, %119, %155 : i4
    %157 = comb.concat %83, %156 : i1, i4
    %158 = comb.mux bin %5, %81, %157 : i5
    %159 = comb.concat %5, %158 : i1, i5
    %160 = comb.xor bin %159, %c-1_i6 {sv.namehint = "normCount"} : i6
    %161 = comb.concat %c0_i63, %3 : i63, i64
    %162 = comb.concat %c0_i121, %160 : i121, i6
    %163 = comb.shl bin %161, %162 : i127
    %164 = comb.extract %163 from 39 : (i127) -> i2
    %165 = comb.extract %163 from 0 : (i127) -> i39
    %166 = comb.icmp bin ne %165, %c0_i39 : i39
    %167 = comb.extract %163 from 39 : (i127) -> i1
    %168 = comb.concat %167, %166 : i1, i1
    %169 = comb.icmp bin ne %168, %c0_i2 {sv.namehint = "inexact"} : i2
    %170 = comb.icmp bin eq %io_roundingMode, %c0_i2 : i2
    %171 = comb.icmp eq %164, %c-1_i2 : i2
    %172 = comb.icmp eq %168, %c-1_i2 : i2
    %173 = comb.or bin %171, %172 : i1
    %174 = comb.and %170, %173 : i1
    %175 = comb.icmp bin eq %io_roundingMode, %c-2_i2 : i2
    %176 = comb.and %175, %1, %169 : i1
    %177 = comb.icmp bin eq %io_roundingMode, %c-1_i2 : i2
    %178 = comb.xor bin %1, %true : i1
    %179 = comb.and %177, %178, %169 : i1
    %180 = comb.or bin %174, %176, %179 {sv.namehint = "round"} : i1
    %181 = comb.extract %163 from 40 : (i127) -> i24
    %182 = comb.concat %false, %181 {sv.namehint = "unroundedNorm"} : i1, i24
    %183 = comb.concat %false, %181 : i1, i24
    %184 = comb.add %183, %c1_i25 : i25
    %185 = comb.mux bin %180, %184, %182 {sv.namehint = "roundedNorm"} : i25
    %186 = comb.extract %185 from 24 : (i25) -> i1
    %187 = comb.concat %c0_i2, %5, %158 : i2, i1, i5
    %188 = comb.concat %c0_i7, %186 : i7, i1
    %189 = comb.add %187, %188 {sv.namehint = "roundedExp"} : i8
    %190 = comb.extract %163 from 63 : (i127) -> i1
    %191 = comb.extract %185 from 0 : (i25) -> i23
    %192 = comb.concat %1, %190, %189, %191 {sv.namehint = "io_out"} : i1, i1, i8, i23
    %193 = comb.concat %c0_i4, %169 {sv.namehint = "io_exceptionFlags"} : i4, i1
    hw.output %192, %193 : i33, i5
  }
  hw.module private @INToRecFN_1(in %io_signedIn : i1, in %io_in : i64, in %io_roundingMode : i2, out io_out : i65, out io_exceptionFlags : i5) {
    %c0_i5 = hw.constant 0 : i5
    %c1_i54 = hw.constant 1 : i54
    %c0_i64 = hw.constant 0 : i64
    %c0_i2 = hw.constant 0 : i2
    %c0_i10 = hw.constant 0 : i10
    %c0_i121 = hw.constant 0 : i121
    %c0_i63 = hw.constant 0 : i63
    %c-1_i6 = hw.constant -1 : i6
    %c0_i8 = hw.constant 0 : i8
    %c0_i16 = hw.constant 0 : i16
    %c0_i32 = hw.constant 0 : i32
    %false = hw.constant false
    %c-2_i2 = hw.constant -2 : i2
    %c-1_i2 = hw.constant -1 : i2
    %true = hw.constant true
    %c0_i4 = hw.constant 0 : i4
    %0 = comb.extract %io_in from 63 : (i64) -> i1
    %1 = comb.and bin %io_signedIn, %0 {sv.namehint = "sign"} : i1
    %2 = comb.sub %c0_i64, %io_in : i64
    %3 = comb.mux bin %1, %2, %io_in {sv.namehint = "absIn"} : i64
    %4 = comb.extract %3 from 32 : (i64) -> i32
    %5 = comb.icmp bin ne %4, %c0_i32 : i32
    %6 = comb.extract %3 from 48 : (i64) -> i16
    %7 = comb.icmp bin ne %6, %c0_i16 : i16
    %8 = comb.extract %3 from 56 : (i64) -> i8
    %9 = comb.icmp bin ne %8, %c0_i8 : i8
    %10 = comb.extract %3 from 60 : (i64) -> i4
    %11 = comb.icmp bin ne %10, %c0_i4 : i4
    %12 = comb.extract %3 from 63 : (i64) -> i1
    %13 = comb.extract %3 from 62 : (i64) -> i1
    %14 = comb.extract %3 from 61 : (i64) -> i1
    %15 = comb.concat %false, %14 : i1, i1
    %16 = comb.mux bin %13, %c-2_i2, %15 : i2
    %17 = comb.mux bin %12, %c-1_i2, %16 : i2
    %18 = comb.extract %3 from 59 : (i64) -> i1
    %19 = comb.extract %3 from 58 : (i64) -> i1
    %20 = comb.extract %3 from 57 : (i64) -> i1
    %21 = comb.concat %false, %20 : i1, i1
    %22 = comb.mux bin %19, %c-2_i2, %21 : i2
    %23 = comb.mux bin %18, %c-1_i2, %22 : i2
    %24 = comb.mux bin %11, %17, %23 : i2
    %25 = comb.concat %11, %24 : i1, i2
    %26 = comb.extract %3 from 52 : (i64) -> i4
    %27 = comb.icmp bin ne %26, %c0_i4 : i4
    %28 = comb.extract %3 from 55 : (i64) -> i1
    %29 = comb.extract %3 from 54 : (i64) -> i1
    %30 = comb.extract %3 from 53 : (i64) -> i1
    %31 = comb.concat %false, %30 : i1, i1
    %32 = comb.mux bin %29, %c-2_i2, %31 : i2
    %33 = comb.mux bin %28, %c-1_i2, %32 : i2
    %34 = comb.extract %3 from 51 : (i64) -> i1
    %35 = comb.extract %3 from 50 : (i64) -> i1
    %36 = comb.extract %3 from 49 : (i64) -> i1
    %37 = comb.concat %false, %36 : i1, i1
    %38 = comb.mux bin %35, %c-2_i2, %37 : i2
    %39 = comb.mux bin %34, %c-1_i2, %38 : i2
    %40 = comb.mux bin %27, %33, %39 : i2
    %41 = comb.concat %27, %40 : i1, i2
    %42 = comb.mux bin %9, %25, %41 : i3
    %43 = comb.concat %9, %42 : i1, i3
    %44 = comb.extract %3 from 40 : (i64) -> i8
    %45 = comb.icmp bin ne %44, %c0_i8 : i8
    %46 = comb.extract %3 from 44 : (i64) -> i4
    %47 = comb.icmp bin ne %46, %c0_i4 : i4
    %48 = comb.extract %3 from 47 : (i64) -> i1
    %49 = comb.extract %3 from 46 : (i64) -> i1
    %50 = comb.extract %3 from 45 : (i64) -> i1
    %51 = comb.concat %false, %50 : i1, i1
    %52 = comb.mux bin %49, %c-2_i2, %51 : i2
    %53 = comb.mux bin %48, %c-1_i2, %52 : i2
    %54 = comb.extract %3 from 43 : (i64) -> i1
    %55 = comb.extract %3 from 42 : (i64) -> i1
    %56 = comb.extract %3 from 41 : (i64) -> i1
    %57 = comb.concat %false, %56 : i1, i1
    %58 = comb.mux bin %55, %c-2_i2, %57 : i2
    %59 = comb.mux bin %54, %c-1_i2, %58 : i2
    %60 = comb.mux bin %47, %53, %59 : i2
    %61 = comb.concat %47, %60 : i1, i2
    %62 = comb.extract %3 from 36 : (i64) -> i4
    %63 = comb.icmp bin ne %62, %c0_i4 : i4
    %64 = comb.extract %3 from 39 : (i64) -> i1
    %65 = comb.extract %3 from 38 : (i64) -> i1
    %66 = comb.extract %3 from 37 : (i64) -> i1
    %67 = comb.concat %false, %66 : i1, i1
    %68 = comb.mux bin %65, %c-2_i2, %67 : i2
    %69 = comb.mux bin %64, %c-1_i2, %68 : i2
    %70 = comb.extract %3 from 35 : (i64) -> i1
    %71 = comb.extract %3 from 34 : (i64) -> i1
    %72 = comb.extract %3 from 33 : (i64) -> i1
    %73 = comb.concat %false, %72 : i1, i1
    %74 = comb.mux bin %71, %c-2_i2, %73 : i2
    %75 = comb.mux bin %70, %c-1_i2, %74 : i2
    %76 = comb.mux bin %63, %69, %75 : i2
    %77 = comb.concat %63, %76 : i1, i2
    %78 = comb.mux bin %45, %61, %77 : i3
    %79 = comb.concat %45, %78 : i1, i3
    %80 = comb.mux bin %7, %43, %79 : i4
    %81 = comb.concat %7, %80 : i1, i4
    %82 = comb.extract %3 from 16 : (i64) -> i16
    %83 = comb.icmp bin ne %82, %c0_i16 : i16
    %84 = comb.extract %3 from 24 : (i64) -> i8
    %85 = comb.icmp bin ne %84, %c0_i8 : i8
    %86 = comb.extract %3 from 28 : (i64) -> i4
    %87 = comb.icmp bin ne %86, %c0_i4 : i4
    %88 = comb.extract %3 from 31 : (i64) -> i1
    %89 = comb.extract %3 from 30 : (i64) -> i1
    %90 = comb.extract %3 from 29 : (i64) -> i1
    %91 = comb.concat %false, %90 : i1, i1
    %92 = comb.mux bin %89, %c-2_i2, %91 : i2
    %93 = comb.mux bin %88, %c-1_i2, %92 : i2
    %94 = comb.extract %3 from 27 : (i64) -> i1
    %95 = comb.extract %3 from 26 : (i64) -> i1
    %96 = comb.extract %3 from 25 : (i64) -> i1
    %97 = comb.concat %false, %96 : i1, i1
    %98 = comb.mux bin %95, %c-2_i2, %97 : i2
    %99 = comb.mux bin %94, %c-1_i2, %98 : i2
    %100 = comb.mux bin %87, %93, %99 : i2
    %101 = comb.concat %87, %100 : i1, i2
    %102 = comb.extract %3 from 20 : (i64) -> i4
    %103 = comb.icmp bin ne %102, %c0_i4 : i4
    %104 = comb.extract %3 from 23 : (i64) -> i1
    %105 = comb.extract %3 from 22 : (i64) -> i1
    %106 = comb.extract %3 from 21 : (i64) -> i1
    %107 = comb.concat %false, %106 : i1, i1
    %108 = comb.mux bin %105, %c-2_i2, %107 : i2
    %109 = comb.mux bin %104, %c-1_i2, %108 : i2
    %110 = comb.extract %3 from 19 : (i64) -> i1
    %111 = comb.extract %3 from 18 : (i64) -> i1
    %112 = comb.extract %3 from 17 : (i64) -> i1
    %113 = comb.concat %false, %112 : i1, i1
    %114 = comb.mux bin %111, %c-2_i2, %113 : i2
    %115 = comb.mux bin %110, %c-1_i2, %114 : i2
    %116 = comb.mux bin %103, %109, %115 : i2
    %117 = comb.concat %103, %116 : i1, i2
    %118 = comb.mux bin %85, %101, %117 : i3
    %119 = comb.concat %85, %118 : i1, i3
    %120 = comb.extract %3 from 8 : (i64) -> i8
    %121 = comb.icmp bin ne %120, %c0_i8 : i8
    %122 = comb.extract %3 from 12 : (i64) -> i4
    %123 = comb.icmp bin ne %122, %c0_i4 : i4
    %124 = comb.extract %3 from 15 : (i64) -> i1
    %125 = comb.extract %3 from 14 : (i64) -> i1
    %126 = comb.extract %3 from 13 : (i64) -> i1
    %127 = comb.concat %false, %126 : i1, i1
    %128 = comb.mux bin %125, %c-2_i2, %127 : i2
    %129 = comb.mux bin %124, %c-1_i2, %128 : i2
    %130 = comb.extract %3 from 11 : (i64) -> i1
    %131 = comb.extract %3 from 10 : (i64) -> i1
    %132 = comb.extract %3 from 9 : (i64) -> i1
    %133 = comb.concat %false, %132 : i1, i1
    %134 = comb.mux bin %131, %c-2_i2, %133 : i2
    %135 = comb.mux bin %130, %c-1_i2, %134 : i2
    %136 = comb.mux bin %123, %129, %135 : i2
    %137 = comb.concat %123, %136 : i1, i2
    %138 = comb.extract %3 from 4 : (i64) -> i4
    %139 = comb.icmp bin ne %138, %c0_i4 : i4
    %140 = comb.extract %3 from 7 : (i64) -> i1
    %141 = comb.extract %3 from 6 : (i64) -> i1
    %142 = comb.extract %3 from 5 : (i64) -> i1
    %143 = comb.concat %false, %142 : i1, i1
    %144 = comb.mux bin %141, %c-2_i2, %143 : i2
    %145 = comb.mux bin %140, %c-1_i2, %144 : i2
    %146 = comb.extract %3 from 3 : (i64) -> i1
    %147 = comb.extract %3 from 2 : (i64) -> i1
    %148 = comb.extract %3 from 1 : (i64) -> i1
    %149 = comb.concat %false, %148 : i1, i1
    %150 = comb.mux bin %147, %c-2_i2, %149 : i2
    %151 = comb.mux bin %146, %c-1_i2, %150 : i2
    %152 = comb.mux bin %139, %145, %151 : i2
    %153 = comb.concat %139, %152 : i1, i2
    %154 = comb.mux bin %121, %137, %153 : i3
    %155 = comb.concat %121, %154 : i1, i3
    %156 = comb.mux bin %83, %119, %155 : i4
    %157 = comb.concat %83, %156 : i1, i4
    %158 = comb.mux bin %5, %81, %157 : i5
    %159 = comb.concat %5, %158 : i1, i5
    %160 = comb.xor bin %159, %c-1_i6 {sv.namehint = "normCount"} : i6
    %161 = comb.concat %c0_i63, %3 : i63, i64
    %162 = comb.concat %c0_i121, %160 : i121, i6
    %163 = comb.shl bin %161, %162 : i127
    %164 = comb.extract %163 from 10 : (i127) -> i2
    %165 = comb.extract %163 from 0 : (i127) -> i10
    %166 = comb.icmp bin ne %165, %c0_i10 : i10
    %167 = comb.extract %163 from 10 : (i127) -> i1
    %168 = comb.concat %167, %166 : i1, i1
    %169 = comb.icmp bin ne %168, %c0_i2 {sv.namehint = "inexact"} : i2
    %170 = comb.icmp bin eq %io_roundingMode, %c0_i2 : i2
    %171 = comb.icmp eq %164, %c-1_i2 : i2
    %172 = comb.icmp eq %168, %c-1_i2 : i2
    %173 = comb.or bin %171, %172 : i1
    %174 = comb.and %170, %173 : i1
    %175 = comb.icmp bin eq %io_roundingMode, %c-2_i2 : i2
    %176 = comb.and %175, %1, %169 : i1
    %177 = comb.icmp bin eq %io_roundingMode, %c-1_i2 : i2
    %178 = comb.xor bin %1, %true : i1
    %179 = comb.and %177, %178, %169 : i1
    %180 = comb.or bin %174, %176, %179 {sv.namehint = "round"} : i1
    %181 = comb.extract %163 from 11 : (i127) -> i53
    %182 = comb.concat %false, %181 {sv.namehint = "unroundedNorm"} : i1, i53
    %183 = comb.concat %false, %181 : i1, i53
    %184 = comb.add %183, %c1_i54 : i54
    %185 = comb.mux bin %180, %184, %182 {sv.namehint = "roundedNorm"} : i54
    %186 = comb.extract %185 from 53 : (i54) -> i1
    %187 = comb.concat %c0_i5, %5, %158 : i5, i1, i5
    %188 = comb.concat %c0_i10, %186 : i10, i1
    %189 = comb.add %187, %188 {sv.namehint = "roundedExp"} : i11
    %190 = comb.extract %163 from 63 : (i127) -> i1
    %191 = comb.extract %185 from 0 : (i54) -> i52
    %192 = comb.concat %1, %190, %189, %191 {sv.namehint = "io_out"} : i1, i1, i11, i52
    %193 = comb.concat %c0_i4, %169 {sv.namehint = "io_exceptionFlags"} : i4, i1
    hw.output %192, %193 : i65, i5
  }
  hw.module private @RecFNToRecFN(in %io_in : i65, in %io_roundingMode : i2, out io_out : i33, out io_exceptionFlags : i5) {
    %c0_i28 = hw.constant 0 : i28
    %c0_i2 = hw.constant 0 : i2
    %c0_i14 = hw.constant 0 : i14
    %c-1792_i14 = hw.constant -1792 : i14
    %c0_i4 = hw.constant 0 : i4
    %c-1_i2 = hw.constant -1 : i2
    %true = hw.constant true
    %c0_i3 = hw.constant 0 : i3
    %false = hw.constant false
    %c-4_i9 = hw.constant -4 : i9
    %0 = comb.extract %io_in from 52 : (i65) -> i12
    %1 = comb.extract %io_in from 61 : (i65) -> i3
    %2 = comb.icmp bin ne %1, %c0_i3 : i3
    %3 = comb.xor bin %2, %true {sv.namehint = "outRawFloat_isZero"} : i1
    %4 = comb.extract %io_in from 62 : (i65) -> i2
    %5 = comb.icmp bin eq %4, %c-1_i2 : i2
    %6 = comb.extract %io_in from 64 {sv.namehint = "outRawFloat_sign"} : (i65) -> i1
    %7 = comb.extract %io_in from 61 : (i65) -> i1
    %8 = comb.and bin %5, %7 {sv.namehint = "outRawFloat_isNaN"} : i1
    %9 = comb.xor bin %7, %true : i1
    %10 = comb.and bin %5, %9 {sv.namehint = "outRawFloat_isInf"} : i1
    %11 = comb.concat %c0_i2, %0 : i2, i12
    %12 = comb.add bin %11, %c-1792_i14 : i14
    %13 = comb.icmp bin slt %12, %c0_i14 : i14
    %14 = comb.extract %12 from 9 : (i14) -> i4
    %15 = comb.icmp bin ne %14, %c0_i4 : i4
    %16 = comb.extract %12 from 0 : (i14) -> i9
    %17 = comb.mux bin %15, %c-4_i9, %16 : i9
    %18 = comb.concat %13, %17 {sv.namehint = "outRawFloat_sExp"} : i1, i9
    %19 = comb.extract %io_in from 28 : (i65) -> i24
    %20 = comb.extract %io_in from 0 : (i65) -> i28
    %21 = comb.icmp bin ne %20, %c0_i28 : i28
    %22 = comb.concat %false, %2, %19, %21 {sv.namehint = "outRawFloat_sig"} : i1, i1, i24, i1
    %23 = comb.extract %io_in from 51 : (i65) -> i1
    %24 = comb.xor bin %23, %true : i1
    %25 = comb.and bin %8, %24 {sv.namehint = "invalidExc"} : i1
    %RoundRawFNToRecFN.io_out, %RoundRawFNToRecFN.io_exceptionFlags = hw.instance "RoundRawFNToRecFN" @RoundRawFNToRecFN(io_invalidExc: %25: i1, io_in_sign: %6: i1, io_in_isNaN: %8: i1, io_in_isInf: %10: i1, io_in_isZero: %3: i1, io_in_sExp: %18: i10, io_in_sig: %22: i27, io_roundingMode: %io_roundingMode: i2) -> (io_out: i33, io_exceptionFlags: i5) {sv.namehint = "RoundRawFNToRecFN.io_out"}
    hw.output %RoundRawFNToRecFN.io_out, %RoundRawFNToRecFN.io_exceptionFlags : i33, i5
  }
  hw.module private @RecFNToRecFN_1(in %io_in : i33, out io_out : i65, out io_exceptionFlags : i5) {
    %c0_i29 = hw.constant 0 : i29
    %c-1_i9 = hw.constant -1 : i9
    %c1792_i12 = hw.constant 1792 : i12
    %c-1_i12 = hw.constant -1 : i12
    %c-1_i2 = hw.constant -1 : i2
    %true = hw.constant true
    %c0_i3 = hw.constant 0 : i3
    %c-1024_i12 = hw.constant -1024 : i12
    %c-512_i12 = hw.constant -512 : i12
    %c0_i4 = hw.constant 0 : i4
    %c0_i12 = hw.constant 0 : i12
    %c-2251799813685248_i52 = hw.constant -2251799813685248 : i52
    %0 = comb.extract %io_in from 23 : (i33) -> i9
    %1 = comb.extract %io_in from 29 : (i33) -> i3
    %2 = comb.icmp bin ne %1, %c0_i3 : i3
    %3 = comb.xor bin %2, %true {sv.namehint = "outRawFloat_isZero"} : i1
    %4 = comb.extract %io_in from 30 : (i33) -> i2
    %5 = comb.icmp bin eq %4, %c-1_i2 : i2
    %6 = comb.extract %io_in from 32 {sv.namehint = "outRawFloat_sign"} : (i33) -> i1
    %7 = comb.extract %io_in from 29 : (i33) -> i1
    %8 = comb.and bin %5, %7 {sv.namehint = "outRawFloat_isNaN"} : i1
    %9 = comb.xor bin %7, %true : i1
    %10 = comb.and bin %5, %9 {sv.namehint = "outRawFloat_isInf"} : i1
    %11 = comb.extract %io_in from 0 : (i33) -> i23
    %12 = comb.concat %c0_i3, %0 : i3, i9
    %13 = comb.add %12, %c1792_i12 {sv.namehint = "outRawFloat_sExp"} : i12
    %14 = comb.extract %io_in from 22 : (i33) -> i1
    %15 = comb.xor bin %14, %true : i1
    %16 = comb.and bin %8, %15 {sv.namehint = "invalidExc"} : i1
    %17 = comb.xor bin %8, %true : i1
    %18 = comb.and bin %6, %17 : i1
    %19 = comb.mux bin %2, %c0_i12, %c-1024_i12 : i12
    %20 = comb.xor bin %19, %c-1_i12 : i12
    %21 = comb.or bin %3, %10 : i1
    %22 = comb.xor %21, %true : i1
    %23 = comb.concat %c-1_i2, %22, %c-1_i9 : i2, i1, i9
    %24 = comb.and bin %13, %20, %23 : i12
    %25 = comb.mux bin %10, %c-1024_i12, %c0_i12 : i12
    %26 = comb.mux bin %8, %c-512_i12, %c0_i12 : i12
    %27 = comb.or bin %24, %25, %26 : i12
    %28 = comb.concat %11, %c0_i29 : i23, i29
    %29 = comb.mux bin %8, %c-2251799813685248_i52, %28 : i52
    %30 = comb.concat %18, %27, %29 {sv.namehint = "io_out"} : i1, i12, i52
    %31 = comb.concat %16, %c0_i4 {sv.namehint = "io_exceptionFlags"} : i1, i4
    hw.output %30, %31 : i65, i5
  }
  hw.module private @MulAddRecFN_1(in %io_op : i2, in %io_a : i65, in %io_b : i65, in %io_c : i65, in %io_roundingMode : i2, out io_out : i65, out io_exceptionFlags : i5) {
    %false = hw.constant false
    %c0_i53 = hw.constant 0 : i53
    %mulAddRecFN_preMul.io_mulAddA, %mulAddRecFN_preMul.io_mulAddB, %mulAddRecFN_preMul.io_mulAddC, %mulAddRecFN_preMul.io_toPostMul_highExpA, %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNA, %mulAddRecFN_preMul.io_toPostMul_highExpB, %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNB, %mulAddRecFN_preMul.io_toPostMul_signProd, %mulAddRecFN_preMul.io_toPostMul_isZeroProd, %mulAddRecFN_preMul.io_toPostMul_opSignC, %mulAddRecFN_preMul.io_toPostMul_highExpC, %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNC, %mulAddRecFN_preMul.io_toPostMul_isCDominant, %mulAddRecFN_preMul.io_toPostMul_CAlignDist_0, %mulAddRecFN_preMul.io_toPostMul_CAlignDist, %mulAddRecFN_preMul.io_toPostMul_bit0AlignedNegSigC, %mulAddRecFN_preMul.io_toPostMul_highAlignedNegSigC, %mulAddRecFN_preMul.io_toPostMul_sExpSum, %mulAddRecFN_preMul.io_toPostMul_roundingMode = hw.instance "mulAddRecFN_preMul" @MulAddRecFN_preMul_1(io_op: %io_op: i2, io_a: %io_a: i65, io_b: %io_b: i65, io_c: %io_c: i65, io_roundingMode: %io_roundingMode: i2) -> (io_mulAddA: i53, io_mulAddB: i53, io_mulAddC: i106, io_toPostMul_highExpA: i3, io_toPostMul_isNaN_isQuietNaNA: i1, io_toPostMul_highExpB: i3, io_toPostMul_isNaN_isQuietNaNB: i1, io_toPostMul_signProd: i1, io_toPostMul_isZeroProd: i1, io_toPostMul_opSignC: i1, io_toPostMul_highExpC: i3, io_toPostMul_isNaN_isQuietNaNC: i1, io_toPostMul_isCDominant: i1, io_toPostMul_CAlignDist_0: i1, io_toPostMul_CAlignDist: i8, io_toPostMul_bit0AlignedNegSigC: i1, io_toPostMul_highAlignedNegSigC: i55, io_toPostMul_sExpSum: i14, io_toPostMul_roundingMode: i2) {sv.namehint = "mulAddRecFN_preMul.io_mulAddC"}
    %mulAddRecFN_postMul.io_out, %mulAddRecFN_postMul.io_exceptionFlags = hw.instance "mulAddRecFN_postMul" @MulAddRecFN_postMul_1(io_fromPreMul_highExpA: %mulAddRecFN_preMul.io_toPostMul_highExpA: i3, io_fromPreMul_isNaN_isQuietNaNA: %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNA: i1, io_fromPreMul_highExpB: %mulAddRecFN_preMul.io_toPostMul_highExpB: i3, io_fromPreMul_isNaN_isQuietNaNB: %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNB: i1, io_fromPreMul_signProd: %mulAddRecFN_preMul.io_toPostMul_signProd: i1, io_fromPreMul_isZeroProd: %mulAddRecFN_preMul.io_toPostMul_isZeroProd: i1, io_fromPreMul_opSignC: %mulAddRecFN_preMul.io_toPostMul_opSignC: i1, io_fromPreMul_highExpC: %mulAddRecFN_preMul.io_toPostMul_highExpC: i3, io_fromPreMul_isNaN_isQuietNaNC: %mulAddRecFN_preMul.io_toPostMul_isNaN_isQuietNaNC: i1, io_fromPreMul_isCDominant: %mulAddRecFN_preMul.io_toPostMul_isCDominant: i1, io_fromPreMul_CAlignDist_0: %mulAddRecFN_preMul.io_toPostMul_CAlignDist_0: i1, io_fromPreMul_CAlignDist: %mulAddRecFN_preMul.io_toPostMul_CAlignDist: i8, io_fromPreMul_bit0AlignedNegSigC: %mulAddRecFN_preMul.io_toPostMul_bit0AlignedNegSigC: i1, io_fromPreMul_highAlignedNegSigC: %mulAddRecFN_preMul.io_toPostMul_highAlignedNegSigC: i55, io_fromPreMul_sExpSum: %mulAddRecFN_preMul.io_toPostMul_sExpSum: i14, io_fromPreMul_roundingMode: %mulAddRecFN_preMul.io_toPostMul_roundingMode: i2, io_mulAddResult: %5: i107) -> (io_out: i65, io_exceptionFlags: i5) {sv.namehint = "mulAddRecFN_postMul.io_out"}
    %0 = comb.concat %c0_i53, %mulAddRecFN_preMul.io_mulAddA : i53, i53
    %1 = comb.concat %c0_i53, %mulAddRecFN_preMul.io_mulAddB : i53, i53
    %2 = comb.mul bin %0, %1 : i106
    %3 = comb.concat %false, %2 : i1, i106
    %4 = comb.concat %false, %mulAddRecFN_preMul.io_mulAddC : i1, i106
    %5 = comb.add %3, %4 {sv.namehint = "mulAddRecFN_postMul.io_mulAddResult"} : i107
    hw.output %mulAddRecFN_postMul.io_out, %mulAddRecFN_postMul.io_exceptionFlags : i65, i5
  }
  hw.module private @DivSqrtRecF64_mulAddZ31(in %clock : !seq.clock, in %reset : i1, out io_inReady_div : i1, out io_inReady_sqrt : i1, in %io_inValid : i1, in %io_sqrtOp : i1, in %io_a : i65, in %io_b : i65, in %io_roundingMode : i2, out io_outValid_div : i1, out io_outValid_sqrt : i1, out io_out : i65, out io_exceptionFlags : i5, out io_usingMulAdd : i4, out io_latchMulAddA_0 : i1, out io_mulAddA_0 : i54, out io_latchMulAddB_0 : i1, out io_mulAddB_0 : i54, out io_mulAddC_2 : i105, in %io_mulAddResult_3 : i105) {
    %c-174763_i19 = hw.constant -174763 : i19
    %c-1_i18 = hw.constant -1 : i18
    %c-1_i9 = hw.constant -1 : i9
    %c-1_i10 = hw.constant -1 : i10
    %c1026_i13 = hw.constant 1026 : i13
    %c1_i54 = hw.constant 1 : i54
    %c-1_i52 = hw.constant -1 : i52
    %c1024_i13 = hw.constant 1024 : i13
    %c1_i2 = hw.constant 1 : i2
    %c1_i7 = hw.constant 1 : i7
    %c1024_i21 = hw.constant 1024 : i21
    %c262144_i20 = hw.constant 262144 : i20
    %c-1_i4 = hw.constant -1 : i4
    %c2_i14 = hw.constant 2 : i14
    %c2_i4 = hw.constant 2 : i4
    %c3_i4 = hw.constant 3 : i4
    %c4_i4 = hw.constant 4 : i4
    %c5_i4 = hw.constant 5 : i4
    %c7_i4 = hw.constant 7 : i4
    %c1_i3 = hw.constant 1 : i3
    %c2_i3 = hw.constant 2 : i3
    %c1_i4 = hw.constant 1 : i4
    %c-1_i12 = hw.constant -1 : i12
    %c-1_i50 = hw.constant -1 : i50
    %c0_i59 = hw.constant 0 : i59
    %c-1_i13 = hw.constant -1 : i13
    %c-1_i53 = hw.constant -1 : i53
    %c-1_i54 = hw.constant -1 : i54
    %c-1_i46 = hw.constant -1 : i46
    %c0_i47 = hw.constant 0 : i47
    %c0_i19 = hw.constant 0 : i19
    %c0_i36 = hw.constant 0 : i36
    %c0_i16 = hw.constant 0 : i16
    %c0_i5 = hw.constant 0 : i5
    %c-1_i11 = hw.constant -1 : i11
    %false = hw.constant false
    %c0_i3 = hw.constant 0 : i3
    %c0_i4 = hw.constant 0 : i4
    %c-1_i2 = hw.constant -1 : i2
    %c-1_i3 = hw.constant -1 : i3
    %true = hw.constant true
    %c-2_i2 = hw.constant -2 : i2
    %c-2_i3 = hw.constant -2 : i3
    %c-3_i3 = hw.constant -3 : i3
    %c-4_i3 = hw.constant -4 : i3
    %c-6_i4 = hw.constant -6 : i4
    %c-7_i4 = hw.constant -7 : i4
    %c-8_i4 = hw.constant -8 : i4
    %c-57_i9 = hw.constant -57 : i9
    %c-148_i9 = hw.constant -148 : i9
    %c-214_i9 = hw.constant -214 : i9
    %c248_i9 = hw.constant 248 : i9
    %c210_i9 = hw.constant 210 : i9
    %c180_i9 = hw.constant 180 : i9
    %c156_i9 = hw.constant 156 : i9
    %c137_i9 = hw.constant 137 : i9
    %c-56_i9 = hw.constant -56 : i9
    %c193_i9 = hw.constant 193 : i9
    %c-189_i9 = hw.constant -189 : i9
    %c0_i10 = hw.constant 0 : i10
    %c3_i3 = hw.constant 3 : i3
    %c974_i13 = hw.constant 974 : i13
    %c974_i12 = hw.constant 974 : i12
    %c-1025_i12 = hw.constant -1025 : i12
    %c-1024_i12 = hw.constant -1024 : i12
    %c-512_i12 = hw.constant -512 : i12
    %c0_i52 = hw.constant 0 : i52
    %c0_i2 = hw.constant 0 : i2
    %c6_i4 = hw.constant 6 : i4
    %c0_i9 = hw.constant 0 : i9
    %c28_i12 = hw.constant 28 : i12
    %c0_i12 = hw.constant 0 : i12
    %c930_i12 = hw.constant 930 : i12
    %c1653_i12 = hw.constant 1653 : i12
    %c-1850_i12 = hw.constant -1850 : i12
    %c-1356_i12 = hw.constant -1356 : i12
    %c-938_i12 = hw.constant -938 : i12
    %c-579_i12 = hw.constant -579 : i12
    %c-268_i12 = hw.constant -268 : i12
    %c47_i10 = hw.constant 47 : i10
    %c479_i10 = hw.constant 479 : i10
    %c333_i10 = hw.constant 333 : i10
    %c-386_i10 = hw.constant -386 : i10
    %c26_i13 = hw.constant 26 : i13
    %c0_i13 = hw.constant 0 : i13
    %c3018_i13 = hw.constant 3018 : i13
    %c-3373_i13 = hw.constant -3373 : i13
    %c-1257_i13 = hw.constant -1257 : i13
    %c0_i20 = hw.constant 0 : i20
    %c0_i21 = hw.constant 0 : i21
    %c0_i25 = hw.constant 0 : i25
    %c0_i24 = hw.constant 0 : i24
    %c0_i15 = hw.constant 0 : i15
    %c0_i14 = hw.constant 0 : i14
    %c0_i53 = hw.constant 0 : i53
    %c0_i46 = hw.constant 0 : i46
    %c0_i51 = hw.constant 0 : i51
    %c0_i30 = hw.constant 0 : i30
    %c0_i33 = hw.constant 0 : i33
    %c0_i105 = hw.constant 0 : i105
    %c0_i104 = hw.constant 0 : i104
    %c0_i54 = hw.constant 0 : i54
    %c0_i56 = hw.constant 0 : i56
    %c-18446744073709551616_i65 = hw.constant -18446744073709551616 : i65
    %c0_i50 = hw.constant 0 : i50
    %c-975_i12 = hw.constant -975 : i12
    %valid_PA = seq.firreg %52 clock %clock reset sync %reset, %false {firrtl.random_init_start = 0 : ui64} : i1
    %sqrtOp_PA = seq.firreg %53 clock %clock {firrtl.random_init_start = 1 : ui64} : i1
    %sign_PA = seq.firreg %54 clock %clock {firrtl.random_init_start = 2 : ui64} : i1
    %specialCodeB_PA = seq.firreg %55 clock %clock {firrtl.random_init_start = 3 : ui64} : i3
    %fractB_51_PA = seq.firreg %57 clock %clock {firrtl.random_init_start = 6 : ui64} : i1
    %roundingMode_PA = seq.firreg %58 clock %clock {firrtl.random_init_start = 7 : ui64} : i2
    %specialCodeA_PA = seq.firreg %60 clock %clock {firrtl.random_init_start = 9 : ui64} : i3
    %fractA_51_PA = seq.firreg %62 clock %clock {firrtl.random_init_start = 12 : ui64} : i1
    %exp_PA = seq.firreg %72 clock %clock {firrtl.random_init_start = 13 : ui64} : i14
    %fractB_other_PA = seq.firreg %74 clock %clock {firrtl.random_init_start = 27 : ui64} : i51
    %fractA_other_PA = seq.firreg %76 clock %clock {firrtl.random_init_start = 78 : ui64} : i51
    %valid_PB = seq.firreg %96 clock %clock reset sync %reset, %false {firrtl.random_init_start = 129 : ui64} : i1
    %sqrtOp_PB = seq.firreg %98 clock %clock {firrtl.random_init_start = 130 : ui64} : i1
    %sign_PB = seq.firreg %100 clock %clock {firrtl.random_init_start = 131 : ui64} : i1
    %specialCodeA_PB = seq.firreg %102 clock %clock {firrtl.random_init_start = 132 : ui64} : i3
    %fractA_51_PB = seq.firreg %104 clock %clock {firrtl.random_init_start = 135 : ui64} : i1
    %specialCodeB_PB = seq.firreg %106 clock %clock {firrtl.random_init_start = 136 : ui64} : i3
    %fractB_51_PB = seq.firreg %108 clock %clock {firrtl.random_init_start = 139 : ui64} : i1
    %roundingMode_PB = seq.firreg %110 clock %clock {firrtl.random_init_start = 140 : ui64} : i2
    %exp_PB = seq.firreg %111 clock %clock {firrtl.random_init_start = 142 : ui64} : i14
    %fractA_0_PB = seq.firreg %113 clock %clock {firrtl.random_init_start = 156 : ui64} : i1
    %fractB_other_PB = seq.firreg %114 clock %clock {firrtl.random_init_start = 157 : ui64} : i51
    %valid_PC = seq.firreg %131 clock %clock reset sync %reset, %false {firrtl.random_init_start = 208 : ui64} : i1
    %sqrtOp_PC = seq.firreg %133 clock %clock {firrtl.random_init_start = 209 : ui64} : i1
    %sign_PC = seq.firreg %135 clock %clock {firrtl.random_init_start = 210 : ui64} : i1
    %specialCodeA_PC = seq.firreg %137 clock %clock {firrtl.random_init_start = 211 : ui64} : i3
    %fractA_51_PC = seq.firreg %139 clock %clock {firrtl.random_init_start = 214 : ui64} : i1
    %specialCodeB_PC = seq.firreg %141 clock %clock {firrtl.random_init_start = 215 : ui64} : i3
    %fractB_51_PC = seq.firreg %143 clock %clock {firrtl.random_init_start = 218 : ui64} : i1
    %roundingMode_PC = seq.firreg %145 clock %clock {firrtl.random_init_start = 219 : ui64} : i2
    %exp_PC = seq.firreg %146 clock %clock {firrtl.random_init_start = 221 : ui64} : i14
    %fractA_0_PC = seq.firreg %147 clock %clock {firrtl.random_init_start = 235 : ui64} : i1
    %fractB_other_PC = seq.firreg %148 clock %clock {firrtl.random_init_start = 236 : ui64} : i51
    %cycleNum_A = seq.firreg %205 clock %clock reset sync %reset, %c0_i3 {firrtl.random_init_start = 287 : ui64} : i3
    %cycleNum_B = seq.firreg %224 clock %clock reset sync %reset, %c0_i4 {firrtl.random_init_start = 290 : ui64} : i4
    %cycleNum_C = seq.firreg %251 clock %clock reset sync %reset, %c0_i3 {firrtl.random_init_start = 294 : ui64} : i3
    %cycleNum_E = seq.firreg %269 clock %clock reset sync %reset, %c0_i3 {firrtl.random_init_start = 297 : ui64} : i3
    %fractR0_A = seq.firreg %435 clock %clock {firrtl.random_init_start = 300 : ui64} : i9
    %hiSqrR0_A_sqrt = seq.firreg %437 clock %clock {firrtl.random_init_start = 309 : ui64} : i10
    %partNegSigma0_A = seq.firreg %443 clock %clock {firrtl.random_init_start = 319 : ui64} : i21
    %nextMulAdd9A_A = seq.firreg %456 clock %clock {firrtl.random_init_start = 340 : ui64} : i9
    %nextMulAdd9B_A = seq.firreg %466 clock %clock {firrtl.random_init_start = 349 : ui64} : i9
    %ER1_B_sqrt = seq.firreg %467 clock %clock {firrtl.random_init_start = 358 : ui64} : i17
    %ESqrR1_B_sqrt = seq.firreg %582 clock %clock {firrtl.random_init_start = 375 : ui64} : i32
    %sigX1_B = seq.firreg %583 clock %clock {firrtl.random_init_start = 407 : ui64} : i58
    %sqrSigma1_C = seq.firreg %584 clock %clock {firrtl.random_init_start = 465 : ui64} : i33
    %sigXN_C = seq.firreg %586 clock %clock {firrtl.random_init_start = 498 : ui64} : i58
    %u_C_sqrt = seq.firreg %588 clock %clock {firrtl.random_init_start = 556 : ui64} : i31
    %E_E_div = seq.firreg %589 clock %clock {firrtl.random_init_start = 587 : ui64} : i1
    %sigT_E = seq.firreg %591 clock %clock {firrtl.random_init_start = 588 : ui64} : i53
    %extraT_E = seq.firreg %593 clock %clock {firrtl.random_init_start = 641 : ui64} : i1
    %isNegRemT_E = seq.firreg %597 clock %clock {firrtl.random_init_start = 642 : ui64} : i1
    %isZeroRemT_E = seq.firreg %604 clock %clock {firrtl.random_init_start = 643 : ui64} : i1
    %0 = comb.xor bin %228, %true : i1
    %1 = comb.xor bin %240, %true : i1
    %2 = comb.xor bin %241, %true : i1
    %3 = comb.xor bin %242, %true : i1
    %4 = comb.xor bin %232, %true : i1
    %5 = comb.xor bin %233, %true : i1
    %6 = comb.xor bin %245, %true : i1
    %7 = comb.xor bin %253, %true : i1
    %8 = comb.xor bin %254, %true : i1
    %9 = comb.and bin %92, %0, %1, %2, %3, %4, %5, %6, %7, %8 {sv.namehint = "io_inReady_div"} : i1
    %10 = comb.xor bin %238, %true : i1
    %11 = comb.and bin %92, %1, %2, %3, %10, %6 {sv.namehint = "io_inReady_sqrt"} : i1
    %12 = comb.xor bin %io_sqrtOp, %true : i1
    %13 = comb.and bin %9, %io_inValid, %12 {sv.namehint = "cyc_S_div"} : i1
    %14 = comb.and bin %11, %io_inValid, %io_sqrtOp {sv.namehint = "cyc_S_sqrt"} : i1
    %15 = comb.or bin %13, %14 {sv.namehint = "cyc_S"} : i1
    %16 = comb.extract %io_a from 64 {sv.namehint = "signA_S"} : (i65) -> i1
    %17 = comb.extract %io_a from 52 {sv.namehint = "expA_S"} : (i65) -> i12
    %18 = comb.extract %io_a from 61 {sv.namehint = "specialCodeA_S"} : (i65) -> i3
    %19 = comb.icmp bin ne %18, %c0_i3 : i3
    %20 = comb.extract %io_a from 62 : (i65) -> i2
    %21 = comb.extract %io_b from 64 {sv.namehint = "signB_S"} : (i65) -> i1
    %22 = comb.extract %io_b from 52 {sv.namehint = "expB_S"} : (i65) -> i12
    %23 = comb.extract %io_b from 61 {sv.namehint = "specialCodeB_S"} : (i65) -> i3
    %24 = comb.icmp bin ne %23, %c0_i3 : i3
    %25 = comb.extract %io_b from 62 : (i65) -> i2
    %26 = comb.xor %io_sqrtOp, %true : i1
    %27 = comb.and %26, %16 : i1
    %28 = comb.xor %27, %21 {sv.namehint = "sign_S"} : i1
    %29 = comb.icmp bin ne %20, %c-1_i2 : i2
    %30 = comb.icmp bin ne %25, %c-1_i2 : i2
    %31 = comb.and bin %29, %30, %19, %24 {sv.namehint = "normalCase_S_div"} : i1
    %32 = comb.xor bin %21, %true : i1
    %33 = comb.and bin %30, %24, %32 {sv.namehint = "normalCase_S_sqrt"} : i1
    %34 = comb.mux bin %io_sqrtOp, %33, %31 {sv.namehint = "normalCase_S"} : i1
    %35 = comb.and bin %13, %31 {sv.namehint = "cyc_A4_div"} : i1
    %36 = comb.and bin %14, %33 {sv.namehint = "cyc_A7_sqrt"} : i1
    %37 = comb.or bin %35, %36 {sv.namehint = "entering_PA_normalCase"} : i1
    %38 = comb.xor bin %127, %true : i1
    %39 = comb.or bin %valid_PA, %38 : i1
    %40 = comb.and bin %15, %39 : i1
    %41 = comb.or bin %37, %40 {sv.namehint = "entering_PA"} : i1
    %42 = comb.xor bin %34, %true : i1
    %43 = comb.xor bin %valid_PA, %true : i1
    %44 = comb.and bin %15, %42, %43 : i1
    %45 = comb.xor bin %valid_PB, %true : i1
    %46 = comb.xor bin %193, %true : i1
    %47 = comb.and bin %45, %46 : i1
    %48 = comb.or bin %126, %47 : i1
    %49 = comb.and bin %44, %48 {sv.namehint = "entering_PB_S"} : i1
    %50 = comb.and bin %44, %45, %193 {sv.namehint = "entering_PC_S"} : i1
    %51 = comb.or bin %41, %91 : i1
    %52 = comb.mux bin %51, %41, %valid_PA : i1
    %53 = comb.mux bin %41, %io_sqrtOp, %sqrtOp_PA : i1
    %54 = comb.mux bin %41, %28, %sign_PA : i1
    %55 = comb.mux bin %41, %23, %specialCodeB_PA : i3
    %56 = comb.extract %io_b from 51 : (i65) -> i1
    %57 = comb.mux bin %41, %56, %fractB_51_PA : i1
    %58 = comb.mux bin %41, %io_roundingMode, %roundingMode_PA : i2
    %59 = comb.and bin %41, %12 : i1
    %60 = comb.mux bin %59, %18, %specialCodeA_PA : i3
    %61 = comb.extract %io_a from 51 : (i65) -> i1
    %62 = comb.mux bin %59, %61, %fractA_51_PA : i1
    %63 = comb.extract %io_b from 63 : (i65) -> i1
    %64 = comb.replicate %63 : (i1) -> i3
    %65 = comb.extract %io_b from 52 : (i65) -> i11
    %66 = comb.xor bin %65, %c-1_i11 : i11
    %67 = comb.concat %c0_i2, %17 : i2, i12
    %68 = comb.concat %64, %66 : i3, i11
    %69 = comb.add %67, %68 : i14
    %70 = comb.concat %c0_i2, %22 : i2, i12
    %71 = comb.mux bin %io_sqrtOp, %70, %69 : i14
    %72 = comb.mux bin %37, %71, %exp_PA : i14
    %73 = comb.extract %io_b from 0 : (i65) -> i51
    %74 = comb.mux bin %37, %73, %fractB_other_PA : i51
    %75 = comb.extract %io_a from 0 : (i65) -> i51
    %76 = comb.mux bin %35, %75, %fractA_other_PA : i51
    %77 = comb.icmp bin ne %specialCodeA_PA, %c0_i3 : i3
    %78 = comb.extract %specialCodeA_PA from 1 : (i3) -> i2
    %79 = comb.concat %true, %fractA_51_PA, %fractA_other_PA {sv.namehint = "sigA_PA"} : i1, i1, i51
    %80 = comb.icmp bin ne %specialCodeB_PA, %c0_i3 : i3
    %81 = comb.extract %specialCodeB_PA from 1 : (i3) -> i2
    %82 = comb.concat %true, %fractB_51_PA, %fractB_other_PA {sv.namehint = "sigB_PA"} : i1, i1, i51
    %83 = comb.icmp bin ne %81, %c-1_i2 : i2
    %84 = comb.xor bin %sign_PA, %true : i1
    %85 = comb.and bin %83, %80, %84 : i1
    %86 = comb.icmp bin ne %78, %c-1_i2 : i2
    %87 = comb.and bin %86, %83, %77, %80 : i1
    %88 = comb.mux bin %sqrtOp_PA, %85, %87 {sv.namehint = "normalCase_PA"} : i1
    %89 = comb.or bin %236, %228 {sv.namehint = "valid_normalCase_leaving_PA"} : i1
    %90 = comb.mux bin %88, %89, %127 {sv.namehint = "valid_leaving_PA"} : i1
    %91 = comb.and bin %valid_PA, %90 {sv.namehint = "leaving_PA"} : i1
    %92 = comb.or bin %43, %90 {sv.namehint = "ready_PA"} : i1
    %93 = comb.and bin %valid_PA, %88, %89 {sv.namehint = "entering_PB_normalCase"} : i1
    %94 = comb.or bin %49, %91 {sv.namehint = "entering_PB"} : i1
    %95 = comb.or bin %94, %126 : i1
    %96 = comb.mux bin %95, %94, %valid_PB : i1
    %97 = comb.mux bin %valid_PA, %sqrtOp_PA, %io_sqrtOp : i1
    %98 = comb.mux bin %94, %97, %sqrtOp_PB : i1
    %99 = comb.mux bin %valid_PA, %sign_PA, %28 : i1
    %100 = comb.mux bin %94, %99, %sign_PB : i1
    %101 = comb.mux bin %valid_PA, %specialCodeA_PA, %18 : i3
    %102 = comb.mux bin %94, %101, %specialCodeA_PB : i3
    %103 = comb.mux bin %valid_PA, %fractA_51_PA, %61 : i1
    %104 = comb.mux bin %94, %103, %fractA_51_PB : i1
    %105 = comb.mux bin %valid_PA, %specialCodeB_PA, %23 : i3
    %106 = comb.mux bin %94, %105, %specialCodeB_PB : i3
    %107 = comb.mux bin %valid_PA, %fractB_51_PA, %56 : i1
    %108 = comb.mux bin %94, %107, %fractB_51_PB : i1
    %109 = comb.mux bin %valid_PA, %roundingMode_PA, %io_roundingMode : i2
    %110 = comb.mux bin %94, %109, %roundingMode_PB : i2
    %111 = comb.mux bin %93, %exp_PA, %exp_PB : i14
    %112 = comb.extract %fractA_other_PA from 0 : (i51) -> i1
    %113 = comb.mux bin %93, %112, %fractA_0_PB : i1
    %114 = comb.mux bin %93, %fractB_other_PA, %fractB_other_PB : i51
    %115 = comb.icmp bin ne %specialCodeA_PB, %c0_i3 : i3
    %116 = comb.extract %specialCodeA_PB from 1 : (i3) -> i2
    %117 = comb.icmp bin ne %specialCodeB_PB, %c0_i3 : i3
    %118 = comb.extract %specialCodeB_PB from 1 : (i3) -> i2
    %119 = comb.icmp bin ne %118, %c-1_i2 : i2
    %120 = comb.xor bin %sign_PB, %true : i1
    %121 = comb.and bin %119, %117, %120 : i1
    %122 = comb.icmp bin ne %116, %c-1_i2 : i2
    %123 = comb.and bin %122, %119, %115, %117 : i1
    %124 = comb.mux bin %sqrtOp_PB, %121, %123 {sv.namehint = "normalCase_PB"} : i1
    %125 = comb.mux bin %124, %255, %193 {sv.namehint = "valid_leaving_PB"} : i1
    %126 = comb.and bin %valid_PB, %125 {sv.namehint = "leaving_PB"} : i1
    %127 = comb.or bin %45, %125 {sv.namehint = "ready_PB"} : i1
    %128 = comb.and bin %valid_PB, %124, %255 {sv.namehint = "entering_PC_normalCase"} : i1
    %129 = comb.or bin %50, %126 {sv.namehint = "entering_PC"} : i1
    %130 = comb.or bin %129, %191 : i1
    %131 = comb.mux bin %130, %129, %valid_PC : i1
    %132 = comb.mux bin %valid_PB, %sqrtOp_PB, %io_sqrtOp : i1
    %133 = comb.mux bin %129, %132, %sqrtOp_PC : i1
    %134 = comb.mux bin %valid_PB, %sign_PB, %28 : i1
    %135 = comb.mux bin %129, %134, %sign_PC : i1
    %136 = comb.mux bin %valid_PB, %specialCodeA_PB, %18 : i3
    %137 = comb.mux bin %129, %136, %specialCodeA_PC : i3
    %138 = comb.mux bin %valid_PB, %fractA_51_PB, %61 : i1
    %139 = comb.mux bin %129, %138, %fractA_51_PC : i1
    %140 = comb.mux bin %valid_PB, %specialCodeB_PB, %23 : i3
    %141 = comb.mux bin %129, %140, %specialCodeB_PC : i3
    %142 = comb.mux bin %valid_PB, %fractB_51_PB, %56 : i1
    %143 = comb.mux bin %129, %142, %fractB_51_PC : i1
    %144 = comb.mux bin %valid_PB, %roundingMode_PB, %io_roundingMode : i2
    %145 = comb.mux bin %129, %144, %roundingMode_PC : i2
    %146 = comb.mux bin %128, %exp_PB, %exp_PC : i14
    %147 = comb.mux bin %128, %fractA_0_PB, %fractA_0_PC : i1
    %148 = comb.mux bin %128, %fractB_other_PB, %fractB_other_PC : i51
    %149 = comb.icmp bin ne %specialCodeA_PC, %c0_i3 : i3
    %150 = comb.xor bin %149, %true {sv.namehint = "isZeroA_PC"} : i1
    %151 = comb.extract %specialCodeA_PC from 1 : (i3) -> i2
    %152 = comb.icmp bin eq %151, %c-1_i2 {sv.namehint = "isSpecialA_PC"} : i2
    %153 = comb.extract %specialCodeA_PC from 0 : (i3) -> i1
    %154 = comb.xor bin %153, %true : i1
    %155 = comb.and bin %152, %154 {sv.namehint = "isInfA_PC"} : i1
    %156 = comb.and bin %152, %153 {sv.namehint = "isNaNA_PC"} : i1
    %157 = comb.xor bin %fractA_51_PC, %true : i1
    %158 = comb.icmp bin ne %specialCodeB_PC, %c0_i3 : i3
    %159 = comb.xor bin %158, %true {sv.namehint = "isZeroB_PC"} : i1
    %160 = comb.extract %specialCodeB_PC from 1 : (i3) -> i2
    %161 = comb.icmp bin eq %160, %c-1_i2 {sv.namehint = "isSpecialB_PC"} : i2
    %162 = comb.extract %specialCodeB_PC from 0 : (i3) -> i1
    %163 = comb.xor bin %162, %true : i1
    %164 = comb.and bin %161, %163 {sv.namehint = "isInfB_PC"} : i1
    %165 = comb.and bin %161, %162 {sv.namehint = "isNaNB_PC"} : i1
    %166 = comb.xor bin %fractB_51_PC, %true : i1
    %167 = comb.and bin %165, %166 {sv.namehint = "isSigNaNB_PC"} : i1
    %168 = comb.concat %true, %fractB_51_PC, %fractB_other_PC {sv.namehint = "sigB_PC"} : i1, i1, i51
    %169 = comb.xor bin %161, %true : i1
    %170 = comb.xor bin %sign_PC, %true : i1
    %171 = comb.and bin %169, %158, %170 : i1
    %172 = comb.xor bin %152, %true : i1
    %173 = comb.and bin %172, %169, %149, %158 : i1
    %174 = comb.mux bin %sqrtOp_PC, %171, %173 {sv.namehint = "normalCase_PC"} : i1
    %175 = comb.add %exp_PC, %c2_i14 {sv.namehint = "expP2_PC"} : i14
    %176 = comb.extract %exp_PC from 0 : (i14) -> i1
    %177 = comb.extract %175 from 1 : (i14) -> i13
    %178 = comb.concat %177, %false : i13, i1
    %179 = comb.extract %exp_PC from 1 : (i14) -> i13
    %180 = comb.concat %179, %true : i13, i1
    %181 = comb.mux bin %176, %178, %180 {sv.namehint = "expP1_PC"} : i14
    %182 = comb.icmp bin ne %roundingMode_PC, %c0_i2 : i2
    %183 = comb.xor bin %182, %true {sv.namehint = "roundingMode_near_even_PC"} : i1
    %184 = comb.icmp bin eq %roundingMode_PC, %c-2_i2 {sv.namehint = "roundingMode_min_PC"} : i2
    %185 = comb.icmp bin eq %roundingMode_PC, %c-1_i2 {sv.namehint = "roundingMode_max_PC"} : i2
    %186 = comb.mux bin %sign_PC, %184, %185 {sv.namehint = "roundMagUp_PC"} : i1
    %187 = comb.or bin %183, %186 {sv.namehint = "overflowY_roundMagUp_PC"} : i1
    %188 = comb.xor bin %186, %true : i1
    %189 = comb.xor bin %174, %true : i1
    %190 = comb.or bin %189, %272 {sv.namehint = "valid_leaving_PC"} : i1
    %191 = comb.and bin %valid_PC, %190 {sv.namehint = "leaving_PC"} : i1
    %192 = comb.xor bin %valid_PC, %true : i1
    %193 = comb.or bin %192, %190 {sv.namehint = "ready_PC"} : i1
    %194 = comb.xor bin %sqrtOp_PC, %true : i1
    %195 = comb.and bin %191, %194 {sv.namehint = "io_outValid_div"} : i1
    %196 = comb.and bin %191, %sqrtOp_PC {sv.namehint = "io_outValid_sqrt"} : i1
    %197 = comb.concat %37, %cycleNum_A : i1, i3
    %198 = comb.icmp bin ne %197, %c0_i4 : i4
    %199 = comb.replicate %35 : (i1) -> i2
    %200 = comb.mux bin %36, %c-2_i3, %c0_i3 : i3
    %201 = comb.concat %false, %199 : i1, i2
    %202 = comb.add %cycleNum_A, %c-1_i3 : i3
    %203 = comb.mux bin %37, %c0_i3, %202 : i3
    %204 = comb.or bin %201, %200, %203 : i3
    %205 = comb.mux bin %198, %204, %cycleNum_A : i3
    %206 = comb.icmp bin eq %cycleNum_A, %c-2_i3 {sv.namehint = "cyc_A6_sqrt"} : i3
    %207 = comb.icmp bin eq %cycleNum_A, %c-3_i3 {sv.namehint = "cyc_A5_sqrt"} : i3
    %208 = comb.icmp bin eq %cycleNum_A, %c-4_i3 {sv.namehint = "cyc_A4_sqrt"} : i3
    %209 = comb.or bin %208, %35 {sv.namehint = "cyc_A4"} : i1
    %210 = comb.icmp bin eq %cycleNum_A, %c3_i3 {sv.namehint = "cyc_A3"} : i3
    %211 = comb.icmp bin eq %cycleNum_A, %c2_i3 {sv.namehint = "cyc_A2"} : i3
    %212 = comb.icmp bin eq %cycleNum_A, %c1_i3 {sv.namehint = "cyc_A1"} : i3
    %213 = comb.xor bin %sqrtOp_PA, %true : i1
    %214 = comb.and bin %210, %213 {sv.namehint = "cyc_A3_div"} : i1
    %215 = comb.and bin %211, %213 {sv.namehint = "cyc_A2_div"} : i1
    %216 = comb.and bin %212, %213 {sv.namehint = "cyc_A1_div"} : i1
    %217 = comb.and bin %210, %sqrtOp_PA {sv.namehint = "cyc_A3_sqrt"} : i1
    %218 = comb.and bin %212, %sqrtOp_PA {sv.namehint = "cyc_A1_sqrt"} : i1
    %219 = comb.concat %212, %cycleNum_B : i1, i4
    %220 = comb.icmp bin ne %219, %c0_i5 : i5
    %221 = comb.mux bin %sqrtOp_PA, %c-6_i4, %c6_i4 : i4
    %222 = comb.add %cycleNum_B, %c-1_i4 : i4
    %223 = comb.mux bin %212, %221, %222 : i4
    %224 = comb.mux bin %220, %223, %cycleNum_B : i4
    %225 = comb.icmp bin eq %cycleNum_B, %c-6_i4 {sv.namehint = "cyc_B10_sqrt"} : i4
    %226 = comb.icmp bin eq %cycleNum_B, %c-7_i4 {sv.namehint = "cyc_B9_sqrt"} : i4
    %227 = comb.icmp bin eq %cycleNum_B, %c-8_i4 {sv.namehint = "cyc_B8_sqrt"} : i4
    %228 = comb.icmp bin eq %cycleNum_B, %c7_i4 {sv.namehint = "cyc_B7_sqrt"} : i4
    %229 = comb.icmp bin eq %cycleNum_B, %c6_i4 {sv.namehint = "cyc_B6"} : i4
    %230 = comb.icmp bin eq %cycleNum_B, %c5_i4 {sv.namehint = "cyc_B5"} : i4
    %231 = comb.icmp bin eq %cycleNum_B, %c4_i4 {sv.namehint = "cyc_B4"} : i4
    %232 = comb.icmp bin eq %cycleNum_B, %c3_i4 {sv.namehint = "cyc_B3"} : i4
    %233 = comb.icmp bin eq %cycleNum_B, %c2_i4 {sv.namehint = "cyc_B2"} : i4
    %234 = comb.icmp bin eq %cycleNum_B, %c1_i4 {sv.namehint = "cyc_B1"} : i4
    %235 = comb.and bin %229, %valid_PA, %213 {sv.namehint = "cyc_B6_div"} : i1
    %236 = comb.and bin %231, %valid_PA, %213 {sv.namehint = "cyc_B4_div"} : i1
    %237 = comb.xor bin %sqrtOp_PB, %true : i1
    %238 = comb.and bin %233, %237 {sv.namehint = "cyc_B2_div"} : i1
    %239 = comb.and bin %234, %237 {sv.namehint = "cyc_B1_div"} : i1
    %240 = comb.and bin %229, %valid_PB, %sqrtOp_PB {sv.namehint = "cyc_B6_sqrt"} : i1
    %241 = comb.and bin %230, %valid_PB, %sqrtOp_PB {sv.namehint = "cyc_B5_sqrt"} : i1
    %242 = comb.and bin %231, %valid_PB, %sqrtOp_PB {sv.namehint = "cyc_B4_sqrt"} : i1
    %243 = comb.and bin %232, %sqrtOp_PB {sv.namehint = "cyc_B3_sqrt"} : i1
    %244 = comb.and bin %233, %sqrtOp_PB {sv.namehint = "cyc_B2_sqrt"} : i1
    %245 = comb.and bin %234, %sqrtOp_PB {sv.namehint = "cyc_B1_sqrt"} : i1
    %246 = comb.concat %234, %cycleNum_C : i1, i3
    %247 = comb.icmp bin ne %246, %c0_i4 : i4
    %248 = comb.mux bin %sqrtOp_PB, %c-2_i3, %c-3_i3 : i3
    %249 = comb.add %cycleNum_C, %c-1_i3 : i3
    %250 = comb.mux bin %234, %248, %249 : i3
    %251 = comb.mux bin %247, %250, %cycleNum_C : i3
    %252 = comb.icmp bin eq %cycleNum_C, %c-2_i3 {sv.namehint = "cyc_C6_sqrt"} : i3
    %253 = comb.icmp bin eq %cycleNum_C, %c-3_i3 {sv.namehint = "cyc_C5"} : i3
    %254 = comb.icmp bin eq %cycleNum_C, %c-4_i3 {sv.namehint = "cyc_C4"} : i3
    %255 = comb.icmp bin eq %cycleNum_C, %c3_i3 {sv.namehint = "cyc_C3"} : i3
    %256 = comb.icmp bin eq %cycleNum_C, %c2_i3 {sv.namehint = "cyc_C2"} : i3
    %257 = comb.icmp bin eq %cycleNum_C, %c1_i3 {sv.namehint = "cyc_C1"} : i3
    %258 = comb.and bin %253, %237 {sv.namehint = "cyc_C5_div"} : i1
    %259 = comb.and bin %254, %237 {sv.namehint = "cyc_C4_div"} : i1
    %260 = comb.and bin %257, %194 {sv.namehint = "cyc_C1_div"} : i1
    %261 = comb.and bin %253, %sqrtOp_PB {sv.namehint = "cyc_C5_sqrt"} : i1
    %262 = comb.and bin %254, %sqrtOp_PB {sv.namehint = "cyc_C4_sqrt"} : i1
    %263 = comb.and bin %255, %sqrtOp_PB {sv.namehint = "cyc_C3_sqrt"} : i1
    %264 = comb.and bin %257, %sqrtOp_PC {sv.namehint = "cyc_C1_sqrt"} : i1
    %265 = comb.concat %257, %cycleNum_E : i1, i3
    %266 = comb.icmp bin ne %265, %c0_i4 : i4
    %267 = comb.add %cycleNum_E, %c-1_i3 : i3
    %268 = comb.mux bin %257, %c-4_i3, %267 : i3
    %269 = comb.mux bin %266, %268, %cycleNum_E : i3
    %270 = comb.icmp bin eq %cycleNum_E, %c3_i3 {sv.namehint = "cyc_E3"} : i3
    %271 = comb.icmp bin eq %cycleNum_E, %c2_i3 {sv.namehint = "cyc_E2"} : i3
    %272 = comb.icmp bin eq %cycleNum_E, %c1_i3 {sv.namehint = "cyc_E1"} : i3
    %273 = comb.and bin %270, %sqrtOp_PC {sv.namehint = "cyc_E3_sqrt"} : i1
    %274 = comb.extract %io_b from 35 : (i65) -> i14
    %275 = comb.mux %35, %274, %c0_i14 {sv.namehint = "zFractB_A4_div"} : i14
    %276 = comb.extract %io_b from 49 : (i65) -> i3
    %277 = comb.icmp bin eq %276, %c0_i3 : i3
    %278 = comb.and bin %35, %277 {sv.namehint = "zLinPiece_0_A4_div"} : i1
    %279 = comb.icmp bin eq %276, %c1_i3 : i3
    %280 = comb.and bin %35, %279 {sv.namehint = "zLinPiece_1_A4_div"} : i1
    %281 = comb.icmp bin eq %276, %c2_i3 : i3
    %282 = comb.and bin %35, %281 {sv.namehint = "zLinPiece_2_A4_div"} : i1
    %283 = comb.icmp bin eq %276, %c3_i3 : i3
    %284 = comb.and bin %35, %283 {sv.namehint = "zLinPiece_3_A4_div"} : i1
    %285 = comb.icmp bin eq %276, %c-4_i3 : i3
    %286 = comb.and bin %35, %285 {sv.namehint = "zLinPiece_4_A4_div"} : i1
    %287 = comb.icmp bin eq %276, %c-3_i3 : i3
    %288 = comb.and bin %35, %287 {sv.namehint = "zLinPiece_5_A4_div"} : i1
    %289 = comb.icmp bin eq %276, %c-2_i3 : i3
    %290 = comb.and bin %35, %289 {sv.namehint = "zLinPiece_6_A4_div"} : i1
    %291 = comb.icmp bin eq %276, %c-1_i3 : i3
    %292 = comb.and bin %35, %291 {sv.namehint = "zLinPiece_7_A4_div"} : i1
    %293 = comb.mux bin %278, %c-57_i9, %c0_i9 : i9
    %294 = comb.mux bin %280, %c-148_i9, %c0_i9 : i9
    %295 = comb.mux bin %282, %c-214_i9, %c0_i9 : i9
    %296 = comb.mux bin %284, %c248_i9, %c0_i9 : i9
    %297 = comb.mux bin %286, %c210_i9, %c0_i9 : i9
    %298 = comb.mux bin %288, %c180_i9, %c0_i9 : i9
    %299 = comb.mux bin %290, %c156_i9, %c0_i9 : i9
    %300 = comb.mux bin %292, %c137_i9, %c0_i9 : i9
    %301 = comb.mux bin %278, %c28_i12, %c0_i12 : i12
    %302 = comb.mux bin %280, %c930_i12, %c0_i12 : i12
    %303 = comb.mux bin %282, %c1653_i12, %c0_i12 : i12
    %304 = comb.mux bin %284, %c-1850_i12, %c0_i12 : i12
    %305 = comb.mux bin %286, %c-1356_i12, %c0_i12 : i12
    %306 = comb.mux bin %288, %c-938_i12, %c0_i12 : i12
    %307 = comb.mux bin %290, %c-579_i12, %c0_i12 : i12
    %308 = comb.mux bin %292, %c-268_i12, %c0_i12 : i12
    %309 = comb.extract %io_b from 42 : (i65) -> i9
    %310 = comb.mux %36, %309, %c0_i9 {sv.namehint = "zFractB_A7_sqrt"} : i9
    %311 = comb.extract %io_b from 52 : (i65) -> i1
    %312 = comb.xor bin %311, %true : i1
    %313 = comb.and bin %36, %312 : i1
    %314 = comb.xor bin %56, %true : i1
    %315 = comb.and bin %313, %314 {sv.namehint = "zQuadPiece_0_A7_sqrt"} : i1
    %316 = comb.and bin %313, %56 {sv.namehint = "zQuadPiece_1_A7_sqrt"} : i1
    %317 = comb.and bin %36, %311 : i1
    %318 = comb.and bin %317, %314 {sv.namehint = "zQuadPiece_2_A7_sqrt"} : i1
    %319 = comb.and bin %317, %56 {sv.namehint = "zQuadPiece_3_A7_sqrt"} : i1
    %320 = comb.mux bin %315, %c-56_i9, %c0_i9 : i9
    %321 = comb.mux bin %316, %c193_i9, %c0_i9 : i9
    %322 = comb.mux bin %318, %c-189_i9, %c0_i9 : i9
    %323 = comb.mux bin %319, %c137_i9, %c0_i9 : i9
    %324 = comb.mux bin %315, %c47_i10, %c0_i10 : i10
    %325 = comb.mux bin %316, %c479_i10, %c0_i10 : i10
    %326 = comb.mux bin %318, %c333_i10, %c0_i10 : i10
    %327 = comb.mux bin %319, %c-386_i10, %c0_i10 : i10
    %328 = comb.or bin %324, %325, %326, %327 {sv.namehint = "zComplK1_A7_sqrt"} : i10
    %329 = comb.extract %exp_PA from 0 : (i14) -> i1
    %330 = comb.xor bin %329, %true : i1
    %331 = comb.and bin %206, %330 : i1
    %332 = comb.xor bin %fractB_51_PA, %true : i1
    %333 = comb.and bin %331, %332 {sv.namehint = "zQuadPiece_0_A6_sqrt"} : i1
    %334 = comb.and bin %331, %fractB_51_PA {sv.namehint = "zQuadPiece_1_A6_sqrt"} : i1
    %335 = comb.and bin %206, %329 : i1
    %336 = comb.and bin %335, %332 {sv.namehint = "zQuadPiece_2_A6_sqrt"} : i1
    %337 = comb.and bin %335, %fractB_51_PA {sv.namehint = "zQuadPiece_3_A6_sqrt"} : i1
    %338 = comb.mux bin %333, %c26_i13, %c0_i13 : i13
    %339 = comb.mux bin %334, %c3018_i13, %c0_i13 : i13
    %340 = comb.mux bin %336, %c-3373_i13, %c0_i13 : i13
    %341 = comb.mux bin %337, %c-1257_i13, %c0_i13 : i13
    %342 = comb.or bin %338, %339, %340, %341 {sv.namehint = "zComplFractK0_A6_sqrt"} : i13
    %343 = comb.extract %275 from 5 : (i14) -> i9
    %344 = comb.mux bin %15, %c0_i9, %nextMulAdd9A_A : i9
    %345 = comb.or bin %343, %320, %321, %322, %323, %344 {sv.namehint = "mulAdd9A_A"} : i9
    %346 = comb.mux bin %15, %c0_i9, %nextMulAdd9B_A : i9
    %347 = comb.or bin %293, %294, %295, %296, %297, %298, %299, %300, %310, %346 {sv.namehint = "mulAdd9B_A"} : i9
    %348 = comb.replicate %36 : (i1) -> i10
    %349 = comb.concat %328, %348 : i10, i10
    %350 = comb.replicate %206 : (i1) -> i6
    %351 = comb.concat %206, %342, %350 : i1, i13, i6
    %352 = comb.or bin %349, %351 : i20
    %353 = comb.replicate %35 : (i1) -> i8
    %354 = comb.extract %352 from 8 : (i20) -> i12
    %355 = comb.or %354, %301, %302, %303, %304, %305, %306, %307, %308 : i12
    %356 = comb.extract %352 from 0 : (i20) -> i8
    %357 = comb.or %356, %353 : i8
    %358 = comb.concat %false, %fractR0_A, %c0_i10 : i1, i9, i10
    %359 = comb.add %358, %c262144_i20 : i20
    %360 = comb.mux bin %207, %359, %c0_i20 : i20
    %361 = comb.concat %355, %357 : i12, i8
    %362 = comb.or bin %361, %360 : i20
    %363 = comb.extract %hiSqrR0_A_sqrt from 9 : (i10) -> i1
    %364 = comb.xor bin %363, %true : i1
    %365 = comb.and bin %208, %364 : i1
    %366 = comb.concat %365, %c0_i10 : i1, i10
    %367 = comb.extract %362 from 11 : (i20) -> i9
    %368 = comb.extract %362 from 0 : (i20) -> i11
    %369 = comb.or bin %368, %366 : i11
    %370 = comb.concat %35, %367, %369 : i1, i9, i11
    %371 = comb.and bin %208, %363 : i1
    %372 = comb.or bin %371, %214 : i1
    %373 = comb.extract %fractB_other_PA from 26 : (i51) -> i21
    %374 = comb.add %373, %c1024_i21 : i21
    %375 = comb.mux bin %372, %374, %c0_i21 : i21
    %376 = comb.or bin %217, %211 : i1
    %377 = comb.mux bin %376, %partNegSigma0_A, %c0_i21 : i21
    %378 = comb.or bin %370, %375, %377 : i21
    %379 = comb.concat %fractR0_A, %c0_i16 : i9, i16
    %380 = comb.mux bin %218, %379, %c0_i25 : i25
    %381 = comb.concat %c0_i4, %378 : i4, i21
    %382 = comb.or bin %381, %380 : i25
    %383 = comb.concat %fractR0_A, %c0_i15 : i9, i15
    %384 = comb.mux bin %216, %383, %c0_i24 : i24
    %385 = comb.extract %382 from 24 : (i25) -> i1
    %386 = comb.extract %382 from 0 : (i25) -> i24
    %387 = comb.or bin %386, %384 : i24
    %388 = comb.concat %c0_i9, %345 : i9, i9
    %389 = comb.concat %c0_i9, %347 : i9, i9
    %390 = comb.mul bin %388, %389 : i18
    %391 = comb.extract %387 from 0 : (i24) -> i18
    %392 = comb.concat %false, %390 : i1, i18
    %393 = comb.concat %false, %391 : i1, i18
    %394 = comb.add %392, %393 : i19
    %395 = comb.extract %394 from 18 : (i19) -> i1
    %396 = comb.extract %387 from 18 : (i24) -> i6
    %397 = comb.concat %385, %396 : i1, i6
    %398 = comb.concat %385, %396 : i1, i6
    %399 = comb.add %398, %c1_i7 : i7
    %400 = comb.mux bin %395, %399, %397 : i7
    %401 = comb.extract %394 from 0 : (i19) -> i18
    %402 = comb.extract %400 from 1 : (i7) -> i1
    %403 = comb.and bin %206, %402 : i1
    %404 = comb.extract %394 from 2 : (i19) -> i16
    %405 = comb.extract %400 from 0 : (i7) -> i2
    %406 = comb.concat %405, %404 : i2, i16
    %407 = comb.xor %406, %c-1_i18 : i18
    %408 = comb.extract %407 from 8 : (i18) -> i9
    %409 = comb.mux %403, %408, %c0_i9 {sv.namehint = "zFractR0_A6_sqrt"} : i9
    %410 = comb.extract %400 from 0 : (i7) -> i1
    %411 = comb.concat %410, %401 : i1, i18
    %412 = comb.extract %394 from 1 : (i19) -> i17
    %413 = comb.extract %400 from 0 : (i7) -> i2
    %414 = comb.concat %413, %412 : i2, i17
    %415 = comb.mux %329, %411, %414 {sv.namehint = "sqrR0_A5_sqrt"} : i19
    %416 = comb.extract %400 from 2 : (i7) -> i1
    %417 = comb.and bin %35, %416 : i1
    %418 = comb.extract %407 from 9 : (i18) -> i9
    %419 = comb.mux %417, %418, %c0_i9 {sv.namehint = "zFractR0_A4_div"} : i9
    %420 = comb.extract %394 from 11 : (i19) -> i1
    %421 = comb.and bin %211, %420 : i1
    %422 = comb.extract %407 from 0 : (i18) -> i9
    %423 = comb.mux %421, %422, %c0_i9 {sv.namehint = "zSigma0_A2"} : i9
    %424 = comb.extract %394 from 10 : (i19) -> i8
    %425 = comb.extract %394 from 9 : (i19) -> i9
    %426 = comb.concat %400, %424 : i7, i8
    %427 = comb.extract %400 from 0 : (i7) -> i6
    %428 = comb.concat %427, %425 : i6, i9
    %429 = comb.mux %sqrtOp_PA, %426, %428 {sv.namehint = "fractR1_A1"} : i15
    %430 = comb.concat %true, %429, %false : i1, i15, i1
    %431 = comb.concat %c1_i2, %429 : i2, i15
    %432 = comb.mux bin %329, %430, %431 {sv.namehint = "ER1_A1_sqrt"} : i17
    %433 = comb.or bin %206, %35 : i1
    %434 = comb.or bin %409, %419 : i9
    %435 = comb.mux bin %433, %434, %fractR0_A : i9
    %436 = comb.extract %415 from 9 : (i19) -> i10
    %437 = comb.mux bin %207, %436, %hiSqrR0_A_sqrt : i10
    %438 = comb.or bin %208, %210 : i1
    %439 = comb.extract %400 from 0 : (i7) -> i3
    %440 = comb.concat %439, %401 : i3, i18
    %441 = comb.concat %c0_i5, %400, %425 : i5, i7, i9
    %442 = comb.mux %208, %440, %441 : i21
    %443 = comb.mux bin %438, %442, %partNegSigma0_A : i21
    %444 = comb.or bin %36, %206, %207, %209 : i1
    %445 = comb.or bin %444, %210, %211 : i1
    %446 = comb.extract %407 from 9 : (i18) -> i9
    %447 = comb.mux %36, %446, %c0_i9 : i9
    %448 = comb.extract %fractB_other_PA from 35 : (i51) -> i9
    %449 = comb.mux bin %208, %448, %c0_i9 : i9
    %450 = comb.extract %275 from 0 : (i14) -> i9
    %451 = comb.or bin %207, %210 : i1
    %452 = comb.extract %fractB_other_PA from 44 : (i51) -> i7
    %453 = comb.concat %true, %fractB_51_PA, %452 : i1, i1, i7
    %454 = comb.mux bin %451, %453, %c0_i9 : i9
    %455 = comb.or bin %447, %409, %449, %450, %454, %423 : i9
    %456 = comb.mux bin %445, %455, %nextMulAdd9A_A : i9
    %457 = comb.or bin %444, %211 : i1
    %458 = comb.extract %415 from 0 : (i19) -> i9
    %459 = comb.mux bin %207, %458, %c0_i9 : i9
    %460 = comb.extract %hiSqrR0_A_sqrt from 0 : (i10) -> i9
    %461 = comb.mux bin %208, %460, %c0_i9 : i9
    %462 = comb.extract %fractR0_A from 1 : (i9) -> i8
    %463 = comb.concat %true, %462 : i1, i8
    %464 = comb.mux bin %211, %463, %c0_i9 : i9
    %465 = comb.or bin %310, %409, %459, %419, %461, %464 : i9
    %466 = comb.mux bin %457, %465, %nextMulAdd9B_A : i9
    %467 = comb.mux bin %218, %432, %ER1_B_sqrt : i17
    %468 = comb.or bin %212, %228 : i1
    %469 = comb.or bin %468, %235, %231, %232, %252, %254, %257 {sv.namehint = "io_latchMulAddA_0"} : i1
    %470 = comb.concat %432, %c0_i36 : i17, i36
    %471 = comb.mux bin %218, %470, %c0_i53 : i53
    %472 = comb.or bin %228, %216 : i1
    %473 = comb.mux bin %472, %82, %c0_i53 : i53
    %474 = comb.mux bin %235, %79, %c0_i53 : i53
    %475 = comb.or bin %471, %473, %474 : i53
    %476 = comb.extract %564 from 12 : (i46) -> i34
    %477 = comb.extract %475 from 0 : (i53) -> i34
    %478 = comb.or bin %477, %476 : i34
    %479 = comb.or bin %232, %252 : i1
    %480 = comb.extract %io_mulAddResult_3 from 59 : (i105) -> i46
    %481 = comb.mux bin %479, %480, %c0_i46 : i46
    %482 = comb.extract %475 from 46 : (i53) -> i7
    %483 = comb.extract %475 from 34 : (i53) -> i12
    %484 = comb.concat %483, %478 : i12, i34
    %485 = comb.extract %sigXN_C from 25 : (i58) -> i33
    %486 = comb.concat %485, %c0_i13 : i33, i13
    %487 = comb.mux bin %259, %486, %c0_i46 : i46
    %488 = comb.concat %u_C_sqrt, %c0_i15 : i31, i15
    %489 = comb.mux bin %262, %488, %c0_i46 : i46
    %490 = comb.or bin %484, %481, %487, %489 : i46
    %491 = comb.concat %482, %490 : i7, i46
    %492 = comb.mux bin %260, %168, %c0_i53 : i53
    %493 = comb.or bin %491, %492 : i53
    %494 = comb.concat %false, %493 : i1, i53
    %495 = comb.or bin %494, %580 {sv.namehint = "io_mulAddA_0"} : i54
    %496 = comb.or bin %468, %240, %231, %252, %254, %257 {sv.namehint = "io_latchMulAddB_0"} : i1
    %497 = comb.concat %true, %429, %c0_i36 : i1, i15, i36
    %498 = comb.mux bin %212, %497, %c0_i52 : i52
    %499 = comb.concat %ESqrR1_B_sqrt, %c0_i19 : i32, i19
    %500 = comb.mux bin %228, %499, %c0_i51 : i51
    %501 = comb.extract %498 from 51 : (i52) -> i1
    %502 = comb.extract %498 from 0 : (i52) -> i51
    %503 = comb.or bin %502, %500 : i51
    %504 = comb.concat %ER1_B_sqrt, %c0_i36 : i17, i36
    %505 = comb.mux bin %240, %504, %c0_i53 : i53
    %506 = comb.concat %false, %501, %503 : i1, i1, i51
    %507 = comb.or bin %506, %505 : i53
    %508 = comb.extract %507 from 46 : (i53) -> i7
    %509 = comb.extract %507 from 0 : (i53) -> i46
    %510 = comb.or bin %509, %564 : i46
    %511 = comb.extract %sqrSigma1_C from 1 : (i33) -> i30
    %512 = comb.mux bin %252, %511, %c0_i30 : i30
    %513 = comb.extract %510 from 0 : (i46) -> i30
    %514 = comb.or bin %513, %512 : i30
    %515 = comb.mux bin %254, %sqrSigma1_C, %c0_i33 : i33
    %516 = comb.extract %510 from 33 : (i46) -> i13
    %517 = comb.extract %510 from 30 : (i46) -> i3
    %518 = comb.concat %517, %514 : i3, i30
    %519 = comb.or bin %518, %515 : i33
    %520 = comb.concat %false, %508, %516, %519 : i1, i7, i13, i33
    %521 = comb.or bin %520, %579 {sv.namehint = "io_mulAddB_0"} : i54
    %522 = comb.or bin %209, %214, %216, %225, %226, %228, %229, %241, %243, %238, %245, %254 : i1
    %523 = comb.or bin %210, %215, %226, %227, %229, %230, %242, %244, %239, %252, %255 : i1
    %524 = comb.or bin %211, %216, %227, %228, %230, %231, %243, %245, %253, %256 : i1
    %525 = comb.or bin %469, %229, %244 : i1
    %526 = comb.concat %522, %523, %524, %525 {sv.namehint = "io_usingMulAdd"} : i1, i1, i1, i1
    %527 = comb.concat %sigX1_B, %c0_i47 : i58, i47
    %528 = comb.mux bin %234, %527, %c0_i105 : i105
    %529 = comb.concat %sigX1_B, %c0_i46 : i58, i46
    %530 = comb.mux bin %252, %529, %c0_i104 : i104
    %531 = comb.extract %528 from 104 : (i105) -> i1
    %532 = comb.extract %528 from 0 : (i105) -> i104
    %533 = comb.or bin %532, %530 : i104
    %534 = comb.concat %531, %533 : i1, i104
    %535 = comb.or bin %262, %256 : i1
    %536 = comb.concat %sigXN_C, %c0_i47 : i58, i47
    %537 = comb.mux bin %535, %536, %c0_i105 : i105
    %538 = comb.or bin %534, %537 : i105
    %539 = comb.xor bin %E_E_div, %true : i1
    %540 = comb.and bin %270, %194, %539 : i1
    %541 = comb.concat %fractA_0_PC, %c0_i53 : i1, i53
    %542 = comb.mux bin %540, %541, %c0_i54 : i54
    %543 = comb.extract %538 from 0 : (i105) -> i54
    %544 = comb.or bin %543, %542 : i54
    %545 = comb.extract %fractB_other_PC from 0 : (i51) -> i1
    %546 = comb.concat %545, %false : i1, i1
    %547 = comb.extract %fractB_other_PC from 1 : (i51) -> i1
    %548 = comb.xor bin %547, %545 : i1
    %549 = comb.concat %548, %545 : i1, i1
    %550 = comb.mux bin %176, %546, %549 : i2
    %551 = comb.xor bin %extraT_E, %true : i1
    %552 = comb.concat %551, %false : i1, i1
    %553 = comb.xor bin %550, %552 : i2
    %554 = comb.concat %553, %c0_i54 : i2, i54
    %555 = comb.mux bin %273, %554, %c0_i56 : i56
    %556 = comb.extract %538 from 56 : (i105) -> i49
    %557 = comb.extract %538 from 54 : (i105) -> i2
    %558 = comb.concat %557, %544 : i2, i54
    %559 = comb.or bin %558, %555 : i56
    %560 = comb.concat %556, %559 {sv.namehint = "io_mulAddC_2"} : i49, i56
    %561 = comb.extract %io_mulAddResult_3 from 72 {sv.namehint = "ESqrR1_B8_sqrt"} : (i105) -> i32
    %562 = comb.extract %io_mulAddResult_3 from 45 : (i105) -> i46
    %563 = comb.xor bin %562, %c-1_i46 : i46
    %564 = comb.mux bin %231, %563, %c0_i46 {sv.namehint = "zSigma1_B4"} : i46
    %565 = comb.extract %io_mulAddResult_3 from 47 {sv.namehint = "sqrSigma1_B1"} : (i105) -> i33
    %566 = comb.extract %io_mulAddResult_3 from 47 {sv.namehint = "sigXNU_B3_CX"} : (i105) -> i58
    %567 = comb.extract %io_mulAddResult_3 from 104 : (i105) -> i1
    %568 = comb.xor bin %567, %true {sv.namehint = "E_C1_div"} : i1
    %569 = comb.and bin %260, %567 : i1
    %570 = comb.or bin %569, %264 : i1
    %571 = comb.extract %io_mulAddResult_3 from 51 : (i105) -> i54
    %572 = comb.xor bin %571, %c-1_i54 : i54
    %573 = comb.mux bin %570, %572, %c0_i54 : i54
    %574 = comb.and bin %260, %568 : i1
    %575 = comb.extract %io_mulAddResult_3 from 50 : (i105) -> i53
    %576 = comb.xor bin %575, %c-1_i53 : i53
    %577 = comb.concat %false, %576 : i1, i53
    %578 = comb.mux bin %574, %577, %c0_i54 : i54
    %579 = comb.or bin %573, %578 {sv.namehint = "zComplSigT_C1"} : i54
    %580 = comb.mux bin %264, %572, %c0_i54 {sv.namehint = "zComplSigT_C1_sqrt"} : i54
    %581 = comb.xor bin %579, %c-1_i54 {sv.namehint = "sigT_C1"} : i54
    %582 = comb.mux bin %227, %561, %ESqrR1_B_sqrt : i32
    %583 = comb.mux bin %232, %566, %sigX1_B : i58
    %584 = comb.mux bin %234, %565, %sqrSigma1_C : i33
    %585 = comb.or bin %252, %258, %263 : i1
    %586 = comb.mux bin %585, %566, %sigXN_C : i58
    %587 = comb.extract %io_mulAddResult_3 from 73 : (i105) -> i31
    %588 = comb.mux bin %261, %587, %u_C_sqrt : i31
    %589 = comb.mux bin %257, %568, %E_E_div : i1
    %590 = comb.extract %581 from 1 : (i54) -> i53
    %591 = comb.mux bin %257, %590, %sigT_E : i53
    %592 = comb.extract %581 from 0 : (i54) -> i1
    %593 = comb.mux bin %257, %592, %extraT_E : i1
    %594 = comb.extract %io_mulAddResult_3 from 55 : (i105) -> i1
    %595 = comb.extract %io_mulAddResult_3 from 53 : (i105) -> i1
    %596 = comb.mux bin %sqrtOp_PC, %594, %595 : i1
    %597 = comb.mux bin %271, %596, %isNegRemT_E : i1
    %598 = comb.extract %io_mulAddResult_3 from 0 : (i105) -> i54
    %599 = comb.icmp bin eq %598, %c0_i54 : i54
    %600 = comb.extract %io_mulAddResult_3 from 54 : (i105) -> i2
    %601 = comb.icmp bin eq %600, %c0_i2 : i2
    %602 = comb.or bin %194, %601 : i1
    %603 = comb.and bin %599, %602 : i1
    %604 = comb.mux bin %271, %603, %isZeroRemT_E : i1
    %605 = comb.and bin %194, %E_E_div : i1
    %606 = comb.mux bin %605, %exp_PC, %c0_i14 : i14
    %607 = comb.or %sqrtOp_PC, %E_E_div : i1
    %608 = comb.mux bin %607, %c0_i14, %181 : i14
    %609 = comb.or bin %606, %608 : i14
    %610 = comb.add %179, %c1024_i13 : i13
    %611 = comb.mux bin %sqrtOp_PC, %610, %c0_i13 : i13
    %612 = comb.extract %609 from 13 : (i14) -> i1
    %613 = comb.extract %609 from 0 : (i14) -> i13
    %614 = comb.or bin %613, %611 {sv.namehint = "posExpX_E"} : i13
    %615 = comb.concat %612, %614 {sv.namehint = "sExpX_E"} : i1, i13
    %616 = comb.xor bin %614, %c-1_i13 : i13
    %617 = comb.extract %616 from 12 : (i13) -> i1
    %618 = comb.extract %616 from 11 : (i13) -> i1
    %619 = comb.extract %616 from 10 : (i13) -> i1
    %620 = comb.extract %616 from 9 : (i13) -> i1
    %621 = comb.extract %616 from 8 : (i13) -> i1
    %622 = comb.extract %616 from 7 : (i13) -> i1
    %623 = comb.extract %616 from 6 : (i13) -> i1
    %624 = comb.extract %616 from 0 : (i13) -> i6
    %625 = comb.concat %c0_i59, %624 : i59, i6
    %626 = comb.shrs bin %c-18446744073709551616_i65, %625 : i65
    %627 = comb.extract %626 from 18 : (i65) -> i2
    %628 = comb.extract %626 from 22 : (i65) -> i2
    %629 = comb.extract %626 from 26 : (i65) -> i2
    %630 = comb.extract %626 from 30 : (i65) -> i2
    %631 = comb.extract %626 from 34 : (i65) -> i2
    %632 = comb.extract %626 from 38 : (i65) -> i2
    %633 = comb.extract %626 from 20 : (i65) -> i2
    %634 = comb.concat %633, %628 : i2, i2
    %635 = comb.extract %626 from 24 : (i65) -> i2
    %636 = comb.extract %626 from 28 : (i65) -> i2
    %637 = comb.concat %636, %630 : i2, i2
    %638 = comb.extract %626 from 32 : (i65) -> i2
    %639 = comb.extract %626 from 36 : (i65) -> i2
    %640 = comb.concat %639, %632 : i2, i2
    %641 = comb.extract %626 from 45 : (i65) -> i1
    %642 = comb.extract %626 from 37 : (i65) -> i1
    %643 = comb.concat %627, %633, %628, %635, %629, %636, %630, %638, %631, %642 : i2, i2, i2, i2, i2, i2, i2, i2, i2, i1
    %644 = comb.and %643, %c-174763_i19 : i19
    %645 = comb.extract %626 from 14 : (i65) -> i1
    %646 = comb.extract %626 from 16 : (i65) -> i1
    %647 = comb.extract %626 from 18 : (i65) -> i1
    %648 = comb.and %634, %c5_i4 : i4
    %649 = comb.extract %626 from 24 : (i65) -> i1
    %650 = comb.extract %626 from 26 : (i65) -> i1
    %651 = comb.and %637, %c5_i4 : i4
    %652 = comb.extract %626 from 32 : (i65) -> i1
    %653 = comb.extract %626 from 34 : (i65) -> i1
    %654 = comb.and %640, %c5_i4 : i4
    %655 = comb.extract %626 from 40 : (i65) -> i1
    %656 = comb.extract %626 from 42 : (i65) -> i1
    %657 = comb.extract %626 from 44 : (i65) -> i1
    %658 = comb.extract %626 from 15 : (i65) -> i1
    %659 = comb.extract %626 from 17 : (i65) -> i1
    %660 = comb.extract %644 from 15 : (i19) -> i4
    %661 = comb.or %660, %648 : i4
    %662 = comb.extract %626 from 23 : (i65) -> i1
    %663 = comb.extract %644 from 13 : (i19) -> i1
    %664 = comb.or %663, %649 : i1
    %665 = comb.extract %626 from 25 : (i65) -> i1
    %666 = comb.extract %644 from 7 : (i19) -> i4
    %667 = comb.or %666, %651 : i4
    %668 = comb.extract %626 from 31 : (i65) -> i1
    %669 = comb.extract %644 from 5 : (i19) -> i1
    %670 = comb.or %669, %652 : i1
    %671 = comb.extract %626 from 33 : (i65) -> i1
    %672 = comb.extract %644 from 0 : (i19) -> i3
    %673 = comb.concat %672, %false : i3, i1
    %674 = comb.or %673, %654 : i4
    %675 = comb.extract %626 from 39 : (i65) -> i1
    %676 = comb.extract %626 from 41 : (i65) -> i1
    %677 = comb.extract %626 from 43 : (i65) -> i1
    %678 = comb.extract %626 from 61 : (i65) -> i1
    %679 = comb.extract %626 from 46 : (i65) -> i1
    %680 = comb.extract %626 from 48 : (i65) -> i1
    %681 = comb.extract %626 from 50 : (i65) -> i1
    %682 = comb.extract %626 from 52 : (i65) -> i1
    %683 = comb.extract %626 from 54 : (i65) -> i1
    %684 = comb.extract %626 from 56 : (i65) -> i1
    %685 = comb.extract %626 from 58 : (i65) -> i1
    %686 = comb.extract %626 from 60 : (i65) -> i1
    %687 = comb.extract %626 from 47 : (i65) -> i1
    %688 = comb.extract %626 from 49 : (i65) -> i1
    %689 = comb.extract %626 from 51 : (i65) -> i1
    %690 = comb.extract %626 from 53 : (i65) -> i1
    %691 = comb.extract %626 from 55 : (i65) -> i1
    %692 = comb.extract %626 from 57 : (i65) -> i1
    %693 = comb.extract %626 from 59 : (i65) -> i1
    %694 = comb.extract %626 from 62 : (i65) -> i1
    %695 = comb.extract %626 from 63 : (i65) -> i1
    %696 = comb.concat %645, %658, %646, %659, %647, %661, %662, %664, %665, %650, %667, %668, %670, %671, %653, %674, %675, %655, %676, %656, %677, %657, %641, %679, %687, %680, %688, %681, %689, %682, %690, %683, %691, %684, %692, %685, %693, %686, %678, %694, %695 : i1, i1, i1, i1, i1, i4, i1, i1, i1, i1, i4, i1, i1, i1, i1, i4, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
    %697 = comb.xor bin %696, %c-1_i50 : i50
    %698 = comb.or bin %620, %621, %622, %623 : i1
    %699 = comb.mux bin %698, %c0_i50, %697 : i50
    %700 = comb.xor bin %699, %c-1_i50 : i50
    %701 = comb.concat %700, %c-1_i3 : i50, i3
    %702 = comb.extract %626 from 0 : (i65) -> i1
    %703 = comb.extract %626 from 1 : (i65) -> i1
    %704 = comb.extract %626 from 2 : (i65) -> i1
    %705 = comb.concat %702, %703, %704 : i1, i1, i1
    %706 = comb.and bin %620, %621, %622, %623 : i1
    %707 = comb.mux bin %706, %705, %c0_i3 : i3
    %708 = comb.concat %c0_i50, %707 : i50, i3
    %709 = comb.mux bin %619, %701, %708 : i53
    %710 = comb.and bin %617, %618 : i1
    %711 = comb.mux bin %710, %709, %c0_i53 {sv.namehint = "roundMask_E"} : i53
    %712 = comb.concat %false, %711 : i1, i53
    %713 = comb.xor %711, %c-1_i53 : i53
    %714 = comb.concat %true, %713 : i1, i53
    %715 = comb.concat %711, %true : i53, i1
    %716 = comb.and bin %714, %715 {sv.namehint = "incrPosMask_E"} : i54
    %717 = comb.extract %713 from 1 : (i53) -> i52
    %718 = comb.concat %true, %717 : i1, i52
    %719 = comb.and bin %sigT_E, %718, %711 : i53
    %720 = comb.icmp bin ne %719, %c0_i53 {sv.namehint = "hiRoundPosBitT_E"} : i53
    %721 = comb.extract %711 from 1 : (i53) -> i52
    %722 = comb.extract %sigT_E from 0 : (i53) -> i52
    %723 = comb.xor %722, %c-1_i52 : i52
    %724 = comb.and bin %723, %721 : i52
    %725 = comb.icmp bin eq %724, %c0_i52 {sv.namehint = "all1sHiRoundExtraT_E"} : i52
    %726 = comb.extract %711 from 0 : (i53) -> i1
    %727 = comb.xor bin %726, %true : i1
    %728 = comb.concat %727, %719 : i1, i53
    %729 = comb.icmp bin ne %728, %c0_i54 : i54
    %730 = comb.and bin %729, %725 {sv.namehint = "all1sHiRoundT_E"} : i1
    %731 = comb.concat %false, %sigT_E : i1, i53
    %732 = comb.concat %c0_i53, %186 : i53, i1
    %733 = comb.add %731, %732 {sv.namehint = "sigAdjT_E"} : i54
    %734 = comb.xor bin %711, %c-1_i53 : i53
    %735 = comb.concat %true, %734 : i1, i53
    %736 = comb.and bin %733, %735 {sv.namehint = "sigY0_E"} : i54
    %737 = comb.or bin %733, %712 : i54
    %738 = comb.add %737, %c1_i54 {sv.namehint = "sigY1_E"} : i54
    %739 = comb.xor bin %isNegRemT_E, %true : i1
    %740 = comb.xor bin %isZeroRemT_E, %true : i1
    %741 = comb.and bin %739, %740 : i1
    %742 = comb.mux bin %sqrtOp_PC, %741, %isNegRemT_E {sv.namehint = "trueLtX_E1"} : i1
    %743 = comb.xor bin %742, %true : i1
    %744 = comb.and bin %726, %743, %725, %extraT_E : i1
    %745 = comb.xor bin %720, %744 {sv.namehint = "hiRoundPosBit_E1"} : i1
    %746 = comb.or bin %740, %551 : i1
    %747 = comb.concat %746, %724 : i1, i52
    %748 = comb.icmp bin eq %747, %c0_i53 : i53
    %749 = comb.and bin %183, %745, %748 : i1
    %750 = comb.mux bin %749, %716, %c0_i54 {sv.namehint = "roundEvenMask_E1"} : i54
    %751 = comb.and bin %188, %182, %extraT_E, %743, %730 : i1
    %752 = comb.and bin %extraT_E, %743 : i1
    %753 = comb.and bin %752, %740 : i1
    %754 = comb.xor bin %730, %true : i1
    %755 = comb.or bin %753, %754 : i1
    %756 = comb.and bin %186, %755 : i1
    %757 = comb.or bin %extraT_E, %743 : i1
    %758 = comb.and bin %757, %727 : i1
    %759 = comb.and bin %752, %725 : i1
    %760 = comb.or bin %720, %758, %759 : i1
    %761 = comb.and bin %183, %760 : i1
    %762 = comb.or bin %751, %756, %761 : i1
    %763 = comb.mux bin %762, %738, %736 : i54
    %764 = comb.xor bin %750, %c-1_i54 : i54
    %765 = comb.extract %763 from 0 : (i54) -> i52
    %766 = comb.extract %764 from 0 : (i54) -> i52
    %767 = comb.and bin %765, %766 {sv.namehint = "fractY_E1"} : i52
    %768 = comb.concat %745, %746, %724 : i1, i1, i52
    %769 = comb.icmp bin ne %768, %c0_i54 {sv.namehint = "inexactY_E1"} : i54
    %770 = comb.extract %763 from 53 : (i54) -> i1
    %771 = comb.extract %764 from 53 : (i54) -> i1
    %772 = comb.and bin %770, %771 : i1
    %773 = comb.mux bin %772, %c0_i14, %615 : i14
    %774 = comb.and bin %772, %194 : i1
    %775 = comb.and bin %774, %E_E_div : i1
    %776 = comb.mux bin %775, %181, %c0_i14 : i14
    %777 = comb.and bin %774, %539 : i1
    %778 = comb.mux bin %777, %175, %c0_i14 : i14
    %779 = comb.or bin %773, %776, %778 : i14
    %780 = comb.and bin %772, %sqrtOp_PC : i1
    %781 = comb.add %177, %c1024_i13 : i13
    %782 = comb.mux bin %780, %781, %c0_i13 : i13
    %783 = comb.extract %779 from 13 : (i14) -> i1
    %784 = comb.extract %779 from 0 : (i14) -> i13
    %785 = comb.or bin %784, %782 : i13
    %786 = comb.extract %785 from 0 {sv.namehint = "expY_E1"} : (i13) -> i12
    %787 = comb.xor bin %783, %true : i1
    %788 = comb.extract %785 from 10 : (i13) -> i3
    %789 = comb.icmp bin ugt %788, %c2_i3 : i3
    %790 = comb.icmp bin ult %785, %c974_i13 : i13
    %791 = comb.or bin %783, %790 {sv.namehint = "totalUnderflowY_E1"} : i1
    %792 = comb.icmp bin ult %614, %c1026_i13 : i13
    %793 = comb.and bin %792, %769 : i1
    %794 = comb.or bin %791, %793 {sv.namehint = "underflowY_E1"} : i1
    %795 = comb.xor bin %165, %true : i1
    %796 = comb.and bin %795, %158, %sign_PC : i1
    %797 = comb.and bin %150, %159 : i1
    %798 = comb.and bin %155, %164 : i1
    %799 = comb.or bin %797, %798 : i1
    %800 = comb.mux bin %sqrtOp_PC, %796, %799 {sv.namehint = "notSigNaN_invalid_PC"} : i1
    %801 = comb.and bin %194, %156, %157 : i1
    %802 = comb.or bin %801, %167, %800 {sv.namehint = "invalid_PC"} : i1
    %803 = comb.and bin %194, %172, %149, %159 {sv.namehint = "infinity_PC"} : i1
    %804 = comb.and bin %174, %787, %789 {sv.namehint = "overflow_E1"} : i1
    %805 = comb.and bin %174, %794 {sv.namehint = "underflow_E1"} : i1
    %806 = comb.and bin %174, %769 : i1
    %807 = comb.or bin %804, %805, %806 {sv.namehint = "inexact_E1"} : i1
    %808 = comb.and bin %791, %188 : i1
    %809 = comb.or bin %150, %164, %808 : i1
    %810 = comb.mux bin %sqrtOp_PC, %159, %809 {sv.namehint = "notSpecial_isZeroOut_E1"} : i1
    %811 = comb.and bin %174, %791, %186 {sv.namehint = "pegMinFiniteMagOut_E1"} : i1
    %812 = comb.xor bin %187, %true : i1
    %813 = comb.and bin %804, %812 {sv.namehint = "pegMaxFiniteMagOut_E1"} : i1
    %814 = comb.and bin %804, %187 : i1
    %815 = comb.or bin %155, %159, %814 : i1
    %816 = comb.mux bin %sqrtOp_PC, %164, %815 {sv.namehint = "notNaN_isInfOut_E1"} : i1
    %817 = comb.and bin %194, %156 : i1
    %818 = comb.or bin %817, %165, %800 {sv.namehint = "isNaNOut_PC"} : i1
    %819 = comb.xor bin %818, %true : i1
    %820 = comb.xor %sqrtOp_PC, %true : i1
    %821 = comb.or %820, %159 : i1
    %822 = comb.and bin %819, %821, %sign_PC {sv.namehint = "signOut_PC"} : i1
    %823 = comb.mux bin %810, %c-512_i12, %c0_i12 : i12
    %824 = comb.xor bin %823, %c-1_i12 : i12
    %825 = comb.mux bin %811, %c-975_i12, %c0_i12 : i12
    %826 = comb.xor bin %825, %c-1_i12 : i12
    %827 = comb.xor %813, %true : i1
    %828 = comb.concat %true, %827, %c-1_i10 : i1, i1, i10
    %829 = comb.xor %816, %true : i1
    %830 = comb.concat %c-1_i2, %829, %c-1_i9 : i2, i1, i9
    %831 = comb.and bin %786, %824, %826, %828, %830 : i12
    %832 = comb.mux bin %811, %c974_i12, %c0_i12 : i12
    %833 = comb.mux bin %813, %c-1025_i12, %c0_i12 : i12
    %834 = comb.mux bin %816, %c-1024_i12, %c0_i12 : i12
    %835 = comb.mux bin %818, %c-512_i12, %c0_i12 : i12
    %836 = comb.or bin %831, %832, %833, %834, %835 {sv.namehint = "expOut_E1"} : i12
    %837 = comb.or bin %810, %791, %818 : i1
    %838 = comb.concat %818, %c0_i51 : i1, i51
    %839 = comb.mux bin %837, %838, %767 : i52
    %840 = comb.replicate %813 : (i1) -> i52
    %841 = comb.or bin %839, %840 {sv.namehint = "fractOut_E1"} : i52
    %842 = comb.concat %822, %836, %841 {sv.namehint = "io_out"} : i1, i12, i52
    %843 = comb.concat %802, %803, %804, %805, %807 {sv.namehint = "io_exceptionFlags"} : i1, i1, i1, i1, i1
    hw.output %9, %11, %195, %196, %842, %843, %526, %469, %495, %496, %521, %560 : i1, i1, i1, i1, i65, i5, i4, i1, i54, i1, i54, i105
  }
  hw.module private @Mul54(in %clock : !seq.clock, in %io_val_s0 : i1, in %io_latch_a_s0 : i1, in %io_a_s0 : i54, in %io_latch_b_s0 : i1, in %io_b_s0 : i54, in %io_c_s2 : i105, out io_result_s3 : i105) {
    %c0_i51 = hw.constant 0 : i51
    %val_s1 = seq.firreg %io_val_s0 clock %clock {firrtl.random_init_start = 0 : ui64} : i1
    %val_s2 = seq.firreg %val_s1 clock %clock {firrtl.random_init_start = 1 : ui64} : i1
    %reg_a_s1 = seq.firreg %1 clock %clock {firrtl.random_init_start = 2 : ui64} : i54
    %reg_b_s1 = seq.firreg %3 clock %clock {firrtl.random_init_start = 56 : ui64} : i54
    %reg_a_s2 = seq.firreg %4 clock %clock {firrtl.random_init_start = 110 : ui64} : i54
    %reg_b_s2 = seq.firreg %5 clock %clock {firrtl.random_init_start = 164 : ui64} : i54
    %reg_result_s3 = seq.firreg %10 clock %clock {firrtl.random_init_start = 218 : ui64, sv.namehint = "io_result_s3"} : i105
    %0 = comb.and bin %io_val_s0, %io_latch_a_s0 : i1
    %1 = comb.mux bin %0, %io_a_s0, %reg_a_s1 : i54
    %2 = comb.and bin %io_val_s0, %io_latch_b_s0 : i1
    %3 = comb.mux bin %2, %io_b_s0, %reg_b_s1 : i54
    %4 = comb.mux bin %val_s1, %reg_a_s1, %reg_a_s2 : i54
    %5 = comb.mux bin %val_s1, %reg_b_s1, %reg_b_s2 : i54
    %6 = comb.concat %c0_i51, %reg_a_s2 : i51, i54
    %7 = comb.concat %c0_i51, %reg_b_s2 : i51, i54
    %8 = comb.mul %6, %7 : i105
    %9 = comb.add %8, %io_c_s2 : i105
    %10 = comb.mux bin %val_s2, %9, %reg_result_s3 : i105
    hw.output %reg_result_s3 : i105
  }
  hw.module private @MulAddRecFN_preMul(in %io_op : i2, in %io_a : i33, in %io_b : i33, in %io_c : i33, in %io_roundingMode : i2, out io_mulAddA : i24, out io_mulAddB : i24, out io_mulAddC : i48, out io_toPostMul_highExpA : i3, out io_toPostMul_isNaN_isQuietNaNA : i1, out io_toPostMul_highExpB : i3, out io_toPostMul_isNaN_isQuietNaNB : i1, out io_toPostMul_signProd : i1, out io_toPostMul_isZeroProd : i1, out io_toPostMul_opSignC : i1, out io_toPostMul_highExpC : i3, out io_toPostMul_isNaN_isQuietNaNC : i1, out io_toPostMul_isCDominant : i1, out io_toPostMul_CAlignDist_0 : i1, out io_toPostMul_CAlignDist : i7, out io_toPostMul_bit0AlignedNegSigC : i1, out io_toPostMul_highAlignedNegSigC : i26, out io_toPostMul_sExpSum : i11, out io_toPostMul_roundingMode : i2) {
    %c27_i11 = hw.constant 27 : i11
    %c74_i10 = hw.constant 74 : i10
    %c25_i10 = hw.constant 25 : i10
    %c0_i24 = hw.constant 0 : i24
    %c0_i68 = hw.constant 0 : i68
    %c0_i59 = hw.constant 0 : i59
    %c0_i2 = hw.constant 0 : i2
    %c0_i10 = hw.constant 0 : i10
    %true = hw.constant true
    %c0_i3 = hw.constant 0 : i3
    %c-54_i7 = hw.constant -54 : i7
    %c-1_i14 = hw.constant -1 : i14
    %c0_i7 = hw.constant 0 : i7
    %c-18446744073709551616_i65 = hw.constant -18446744073709551616 : i65
    %0 = comb.extract %io_a from 32 {sv.namehint = "signA"} : (i33) -> i1
    %1 = comb.extract %io_a from 23 {sv.namehint = "expA"} : (i33) -> i9
    %2 = comb.extract %io_a from 0 {sv.namehint = "fractA"} : (i33) -> i23
    %3 = comb.extract %io_a from 29 {sv.namehint = "io_toPostMul_highExpA"} : (i33) -> i3
    %4 = comb.icmp bin ne %3, %c0_i3 : i3
    %5 = comb.xor bin %4, %true {sv.namehint = "isZeroA"} : i1
    %6 = comb.concat %4, %2 {sv.namehint = "sigA"} : i1, i23
    %7 = comb.extract %io_b from 32 {sv.namehint = "signB"} : (i33) -> i1
    %8 = comb.extract %io_b from 0 {sv.namehint = "fractB"} : (i33) -> i23
    %9 = comb.extract %io_b from 29 {sv.namehint = "io_toPostMul_highExpB"} : (i33) -> i3
    %10 = comb.icmp bin ne %9, %c0_i3 : i3
    %11 = comb.xor bin %10, %true {sv.namehint = "isZeroB"} : i1
    %12 = comb.concat %10, %8 {sv.namehint = "sigB"} : i1, i23
    %13 = comb.extract %io_c from 32 : (i33) -> i1
    %14 = comb.extract %io_op from 0 : (i2) -> i1
    %15 = comb.xor bin %13, %14 {sv.namehint = "opSignC"} : i1
    %16 = comb.extract %io_c from 23 {sv.namehint = "expC"} : (i33) -> i9
    %17 = comb.extract %io_c from 0 {sv.namehint = "fractC"} : (i33) -> i23
    %18 = comb.extract %io_c from 29 {sv.namehint = "io_toPostMul_highExpC"} : (i33) -> i3
    %19 = comb.icmp bin ne %18, %c0_i3 : i3
    %20 = comb.concat %19, %17 {sv.namehint = "sigC"} : i1, i23
    %21 = comb.extract %io_op from 1 : (i2) -> i1
    %22 = comb.xor bin %0, %7, %21 {sv.namehint = "signProd"} : i1
    %23 = comb.or bin %5, %11 {sv.namehint = "isZeroProd"} : i1
    %24 = comb.extract %io_b from 31 : (i33) -> i1
    %25 = comb.xor bin %24, %true : i1
    %26 = comb.replicate %25 : (i1) -> i3
    %27 = comb.extract %io_b from 23 : (i33) -> i8
    %28 = comb.concat %c0_i2, %1 : i2, i9
    %29 = comb.concat %26, %27 : i3, i8
    %30 = comb.add %28, %29, %c27_i11 {sv.namehint = "sExpAlignedProd"} : i11
    %31 = comb.xor bin %22, %15 {sv.namehint = "doSubMags"} : i1
    %32 = comb.concat %c0_i2, %16 : i2, i9
    %33 = comb.sub %30, %32 : i11
    %34 = comb.extract %33 from 10 : (i11) -> i1
    %35 = comb.or bin %23, %34 {sv.namehint = "CAlignDist_floor"} : i1
    %36 = comb.extract %33 from 0 : (i11) -> i10
    %37 = comb.icmp bin eq %36, %c0_i10 : i10
    %38 = comb.or bin %35, %37 {sv.namehint = "CAlignDist_0"} : i1
    %39 = comb.icmp bin ult %36, %c25_i10 : i10
    %40 = comb.or bin %35, %39 : i1
    %41 = comb.and bin %19, %40 {sv.namehint = "isCDominant"} : i1
    %42 = comb.icmp bin ult %36, %c74_i10 : i10
    %43 = comb.extract %33 from 0 : (i11) -> i7
    %44 = comb.mux bin %42, %43, %c-54_i7 : i7
    %45 = comb.mux bin %35, %c0_i7, %44 {sv.namehint = "CAlignDist"} : i7
    %46 = comb.concat %c0_i2, %16 : i2, i9
    %47 = comb.mux bin %35, %46, %30 {sv.namehint = "sExpSum"} : i11
    %48 = comb.extract %45 from 6 : (i7) -> i1
    %49 = comb.extract %45 from 0 : (i7) -> i6
    %50 = comb.concat %c0_i59, %49 : i59, i6
    %51 = comb.shrs bin %c-18446744073709551616_i65, %50 : i65
    %52 = comb.extract %51 from 61 : (i65) -> i1
    %53 = comb.extract %51 from 54 : (i65) -> i1
    %54 = comb.extract %51 from 56 : (i65) -> i1
    %55 = comb.extract %51 from 58 : (i65) -> i1
    %56 = comb.extract %51 from 60 : (i65) -> i1
    %57 = comb.extract %51 from 55 : (i65) -> i1
    %58 = comb.extract %51 from 57 : (i65) -> i1
    %59 = comb.extract %51 from 59 : (i65) -> i1
    %60 = comb.extract %51 from 62 : (i65) -> i1
    %61 = comb.extract %51 from 63 : (i65) -> i1
    %62 = comb.concat %53, %57, %54, %58, %55, %59, %56, %52, %60, %61, %c-1_i14 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i14
    %63 = comb.extract %51 from 7 : (i65) -> i1
    %64 = comb.extract %51 from 0 : (i65) -> i1
    %65 = comb.extract %51 from 2 : (i65) -> i1
    %66 = comb.extract %51 from 4 : (i65) -> i1
    %67 = comb.extract %51 from 6 : (i65) -> i1
    %68 = comb.extract %51 from 1 : (i65) -> i1
    %69 = comb.extract %51 from 3 : (i65) -> i1
    %70 = comb.extract %51 from 5 : (i65) -> i1
    %71 = comb.extract %51 from 8 : (i65) -> i1
    %72 = comb.extract %51 from 9 : (i65) -> i1
    %73 = comb.extract %51 from 10 : (i65) -> i1
    %74 = comb.extract %51 from 11 : (i65) -> i1
    %75 = comb.extract %51 from 12 : (i65) -> i1
    %76 = comb.extract %51 from 13 : (i65) -> i1
    %77 = comb.concat %c0_i10, %64, %68, %65, %69, %66, %70, %67, %63, %71, %72, %73, %74, %75, %76 : i10, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
    %78 = comb.mux bin %48, %62, %77 {sv.namehint = "CExtraMask"} : i24
    %79 = comb.replicate %31 : (i1) -> i24
    %80 = comb.xor %79, %20 {sv.namehint = "negSigC"} : i24
    %81 = comb.replicate %31 : (i1) -> i50
    %82 = comb.concat %31, %80, %81 : i1, i24, i50
    %83 = comb.concat %c0_i68, %45 : i68, i7
    %84 = comb.shrs bin %82, %83 : i75
    %85 = comb.and bin %20, %78 : i24
    %86 = comb.icmp bin ne %85, %c0_i24 : i24
    %87 = comb.xor bin %86, %31 {sv.namehint = "io_toPostMul_bit0AlignedNegSigC"} : i1
    %88 = comb.extract %84 from 0 {sv.namehint = "io_mulAddC"} : (i75) -> i48
    %89 = comb.extract %io_a from 22 {sv.namehint = "io_toPostMul_isNaN_isQuietNaNA"} : (i33) -> i1
    %90 = comb.extract %io_b from 22 {sv.namehint = "io_toPostMul_isNaN_isQuietNaNB"} : (i33) -> i1
    %91 = comb.extract %io_c from 22 {sv.namehint = "io_toPostMul_isNaN_isQuietNaNC"} : (i33) -> i1
    %92 = comb.extract %84 from 48 {sv.namehint = "io_toPostMul_highAlignedNegSigC"} : (i75) -> i26
    hw.output %6, %12, %88, %3, %89, %9, %90, %22, %23, %15, %18, %91, %41, %38, %45, %87, %92, %47, %io_roundingMode : i24, i24, i48, i3, i1, i3, i1, i1, i1, i1, i3, i1, i1, i1, i7, i1, i26, i11, i2
  }
  hw.module private @MulAddRecFN_postMul(in %io_fromPreMul_highExpA : i3, in %io_fromPreMul_isNaN_isQuietNaNA : i1, in %io_fromPreMul_highExpB : i3, in %io_fromPreMul_isNaN_isQuietNaNB : i1, in %io_fromPreMul_signProd : i1, in %io_fromPreMul_isZeroProd : i1, in %io_fromPreMul_opSignC : i1, in %io_fromPreMul_highExpC : i3, in %io_fromPreMul_isNaN_isQuietNaNC : i1, in %io_fromPreMul_isCDominant : i1, in %io_fromPreMul_CAlignDist_0 : i1, in %io_fromPreMul_CAlignDist : i7, in %io_fromPreMul_bit0AlignedNegSigC : i1, in %io_fromPreMul_highAlignedNegSigC : i26, in %io_fromPreMul_sExpSum : i11, in %io_fromPreMul_roundingMode : i2, in %io_mulAddResult : i49, out io_out : i33, out io_exceptionFlags : i5) {
    %c0_i22 = hw.constant 0 : i22
    %c-1_i6 = hw.constant -1 : i6
    %c-1_i7 = hw.constant -1 : i7
    %c1_i10 = hw.constant 1 : i10
    %c-1_i25 = hw.constant -1 : i25
    %c-1_i5 = hw.constant -1 : i5
    %c-55_i7 = hw.constant -55 : i7
    %c1_i26 = hw.constant 1 : i26
    %c3_i3 = hw.constant 3 : i3
    %c-1_i9 = hw.constant -1 : i9
    %c0_i52 = hw.constant 0 : i52
    %c-1_i26 = hw.constant -1 : i26
    %c-1_i21 = hw.constant -1 : i21
    %c0_i59 = hw.constant 0 : i59
    %c-1_i10 = hw.constant -1 : i10
    %c0_i38 = hw.constant 0 : i38
    %c0_i13 = hw.constant 0 : i13
    %c0_i10 = hw.constant 0 : i10
    %c0_i34 = hw.constant 0 : i34
    %c-1_i75 = hw.constant -1 : i75
    %c0_i8 = hw.constant 0 : i8
    %c0_i18 = hw.constant 0 : i18
    %c0_i2 = hw.constant 0 : i2
    %c0_i3 = hw.constant 0 : i3
    %false = hw.constant false
    %c-1_i2 = hw.constant -1 : i2
    %c-2_i2 = hw.constant -2 : i2
    %true = hw.constant true
    %c-1_i16 = hw.constant -1 : i16
    %c0_i16 = hw.constant 0 : i16
    %c0_i32 = hw.constant 0 : i32
    %c-1_i4 = hw.constant -1 : i4
    %c-126_i8 = hw.constant -126 : i8
    %c-127_i8 = hw.constant -127 : i8
    %c-64_i9 = hw.constant -64 : i9
    %c0_i9 = hw.constant 0 : i9
    %c-129_i9 = hw.constant -129 : i9
    %c-128_i9 = hw.constant -128 : i9
    %c0_i42 = hw.constant 0 : i42
    %c-65536_i17 = hw.constant -65536 : i17
    %c-18446744073709551616_i65 = hw.constant -18446744073709551616 : i65
    %c0_i21 = hw.constant 0 : i21
    %c0_i4 = hw.constant 0 : i4
    %c0_i25 = hw.constant 0 : i25
    %c0_i26 = hw.constant 0 : i26
    %c-108_i9 = hw.constant -108 : i9
    %c107_i9 = hw.constant 107 : i9
    %0 = comb.icmp bin eq %io_fromPreMul_highExpA, %c0_i3 {sv.namehint = "isZeroA"} : i3
    %1 = comb.extract %io_fromPreMul_highExpA from 1 : (i3) -> i2
    %2 = comb.icmp bin eq %1, %c-1_i2 {sv.namehint = "isSpecialA"} : i2
    %3 = comb.extract %io_fromPreMul_highExpA from 0 : (i3) -> i1
    %4 = comb.xor bin %3, %true : i1
    %5 = comb.and bin %2, %4 {sv.namehint = "isInfA"} : i1
    %6 = comb.and bin %2, %3 {sv.namehint = "isNaNA"} : i1
    %7 = comb.xor bin %io_fromPreMul_isNaN_isQuietNaNA, %true : i1
    %8 = comb.and bin %6, %7 {sv.namehint = "isSigNaNA"} : i1
    %9 = comb.icmp bin eq %io_fromPreMul_highExpB, %c0_i3 {sv.namehint = "isZeroB"} : i3
    %10 = comb.extract %io_fromPreMul_highExpB from 1 : (i3) -> i2
    %11 = comb.icmp bin eq %10, %c-1_i2 {sv.namehint = "isSpecialB"} : i2
    %12 = comb.extract %io_fromPreMul_highExpB from 0 : (i3) -> i1
    %13 = comb.xor bin %12, %true : i1
    %14 = comb.and bin %11, %13 {sv.namehint = "isInfB"} : i1
    %15 = comb.and bin %11, %12 {sv.namehint = "isNaNB"} : i1
    %16 = comb.xor bin %io_fromPreMul_isNaN_isQuietNaNB, %true : i1
    %17 = comb.and bin %15, %16 {sv.namehint = "isSigNaNB"} : i1
    %18 = comb.icmp bin ne %io_fromPreMul_highExpC, %c0_i3 : i3
    %19 = comb.xor bin %18, %true {sv.namehint = "isZeroC"} : i1
    %20 = comb.extract %io_fromPreMul_highExpC from 1 : (i3) -> i2
    %21 = comb.icmp bin eq %20, %c-1_i2 {sv.namehint = "isSpecialC"} : i2
    %22 = comb.extract %io_fromPreMul_highExpC from 0 : (i3) -> i1
    %23 = comb.xor bin %22, %true : i1
    %24 = comb.and bin %21, %23 {sv.namehint = "isInfC"} : i1
    %25 = comb.and bin %21, %22 {sv.namehint = "isNaNC"} : i1
    %26 = comb.xor bin %io_fromPreMul_isNaN_isQuietNaNC, %true : i1
    %27 = comb.and bin %25, %26 {sv.namehint = "isSigNaNC"} : i1
    %28 = comb.icmp bin eq %io_fromPreMul_roundingMode, %c0_i2 {sv.namehint = "roundingMode_nearest_even"} : i2
    %29 = comb.icmp bin eq %io_fromPreMul_roundingMode, %c-2_i2 {sv.namehint = "signZeroNotEqOpSigns"} : i2
    %30 = comb.icmp bin eq %io_fromPreMul_roundingMode, %c-1_i2 {sv.namehint = "roundingMode_max"} : i2
    %31 = comb.xor bin %io_fromPreMul_signProd, %io_fromPreMul_opSignC {sv.namehint = "doSubMags"} : i1
    %32 = comb.extract %io_mulAddResult from 48 : (i49) -> i1
    %33 = comb.add %io_fromPreMul_highAlignedNegSigC, %c1_i26 : i26
    %34 = comb.mux bin %32, %33, %io_fromPreMul_highAlignedNegSigC : i26
    %35 = comb.extract %io_mulAddResult from 0 : (i49) -> i48
    %36 = comb.concat %34, %35, %io_fromPreMul_bit0AlignedNegSigC {sv.namehint = "sigSum"} : i26, i48, i1
    %37 = comb.extract %34 from 0 : (i26) -> i2
    %38 = comb.extract %io_mulAddResult from 1 : (i49) -> i47
    %39 = comb.concat %37, %38 : i2, i47
    %40 = comb.extract %34 from 0 : (i26) -> i1
    %41 = comb.concat %40, %35 : i1, i48
    %42 = comb.xor %39, %41 : i49
    %43 = comb.extract %42 from 31 : (i49) -> i18
    %44 = comb.icmp bin ne %43, %c0_i18 : i18
    %45 = comb.extract %42 from 47 : (i49) -> i2
    %46 = comb.icmp bin ne %45, %c0_i2 : i2
    %47 = comb.extract %42 from 48 : (i49) -> i1
    %48 = comb.extract %42 from 39 : (i49) -> i8
    %49 = comb.icmp bin ne %48, %c0_i8 : i8
    %50 = comb.extract %42 from 43 : (i49) -> i4
    %51 = comb.icmp bin ne %50, %c0_i4 : i4
    %52 = comb.extract %42 from 46 : (i49) -> i1
    %53 = comb.extract %42 from 45 : (i49) -> i1
    %54 = comb.extract %42 from 44 : (i49) -> i1
    %55 = comb.concat %false, %54 : i1, i1
    %56 = comb.mux bin %53, %c-2_i2, %55 : i2
    %57 = comb.mux bin %52, %c-1_i2, %56 : i2
    %58 = comb.extract %42 from 42 : (i49) -> i1
    %59 = comb.extract %42 from 41 : (i49) -> i1
    %60 = comb.extract %42 from 40 : (i49) -> i1
    %61 = comb.concat %false, %60 : i1, i1
    %62 = comb.mux bin %59, %c-2_i2, %61 : i2
    %63 = comb.mux bin %58, %c-1_i2, %62 : i2
    %64 = comb.mux bin %51, %57, %63 : i2
    %65 = comb.concat %51, %64 : i1, i2
    %66 = comb.extract %42 from 35 : (i49) -> i4
    %67 = comb.icmp bin ne %66, %c0_i4 : i4
    %68 = comb.extract %42 from 38 : (i49) -> i1
    %69 = comb.extract %42 from 37 : (i49) -> i1
    %70 = comb.extract %42 from 36 : (i49) -> i1
    %71 = comb.concat %false, %70 : i1, i1
    %72 = comb.mux bin %69, %c-2_i2, %71 : i2
    %73 = comb.mux bin %68, %c-1_i2, %72 : i2
    %74 = comb.extract %42 from 34 : (i49) -> i1
    %75 = comb.extract %42 from 33 : (i49) -> i1
    %76 = comb.extract %42 from 32 : (i49) -> i1
    %77 = comb.concat %false, %76 : i1, i1
    %78 = comb.mux bin %75, %c-2_i2, %77 : i2
    %79 = comb.mux bin %74, %c-1_i2, %78 : i2
    %80 = comb.mux bin %67, %73, %79 : i2
    %81 = comb.concat %67, %80 : i1, i2
    %82 = comb.mux bin %49, %65, %81 : i3
    %83 = comb.concat %49, %82 : i1, i3
    %84 = comb.concat %c0_i3, %47 : i3, i1
    %85 = comb.mux bin %46, %84, %83 : i4
    %86 = comb.concat %46, %85 : i1, i4
    %87 = comb.extract %42 from 15 : (i49) -> i16
    %88 = comb.icmp bin ne %87, %c0_i16 : i16
    %89 = comb.extract %42 from 23 : (i49) -> i8
    %90 = comb.icmp bin ne %89, %c0_i8 : i8
    %91 = comb.extract %42 from 27 : (i49) -> i4
    %92 = comb.icmp bin ne %91, %c0_i4 : i4
    %93 = comb.extract %42 from 30 : (i49) -> i1
    %94 = comb.extract %42 from 29 : (i49) -> i1
    %95 = comb.extract %42 from 28 : (i49) -> i1
    %96 = comb.concat %false, %95 : i1, i1
    %97 = comb.mux bin %94, %c-2_i2, %96 : i2
    %98 = comb.mux bin %93, %c-1_i2, %97 : i2
    %99 = comb.extract %42 from 26 : (i49) -> i1
    %100 = comb.extract %42 from 25 : (i49) -> i1
    %101 = comb.extract %42 from 24 : (i49) -> i1
    %102 = comb.concat %false, %101 : i1, i1
    %103 = comb.mux bin %100, %c-2_i2, %102 : i2
    %104 = comb.mux bin %99, %c-1_i2, %103 : i2
    %105 = comb.mux bin %92, %98, %104 : i2
    %106 = comb.concat %92, %105 : i1, i2
    %107 = comb.extract %42 from 19 : (i49) -> i4
    %108 = comb.icmp bin ne %107, %c0_i4 : i4
    %109 = comb.extract %42 from 22 : (i49) -> i1
    %110 = comb.extract %42 from 21 : (i49) -> i1
    %111 = comb.extract %42 from 20 : (i49) -> i1
    %112 = comb.concat %false, %111 : i1, i1
    %113 = comb.mux bin %110, %c-2_i2, %112 : i2
    %114 = comb.mux bin %109, %c-1_i2, %113 : i2
    %115 = comb.extract %42 from 18 : (i49) -> i1
    %116 = comb.extract %42 from 17 : (i49) -> i1
    %117 = comb.extract %42 from 16 : (i49) -> i1
    %118 = comb.concat %false, %117 : i1, i1
    %119 = comb.mux bin %116, %c-2_i2, %118 : i2
    %120 = comb.mux bin %115, %c-1_i2, %119 : i2
    %121 = comb.mux bin %108, %114, %120 : i2
    %122 = comb.concat %108, %121 : i1, i2
    %123 = comb.mux bin %90, %106, %122 : i3
    %124 = comb.concat %90, %123 : i1, i3
    %125 = comb.extract %42 from 7 : (i49) -> i8
    %126 = comb.icmp bin ne %125, %c0_i8 : i8
    %127 = comb.extract %42 from 11 : (i49) -> i4
    %128 = comb.icmp bin ne %127, %c0_i4 : i4
    %129 = comb.extract %42 from 14 : (i49) -> i1
    %130 = comb.extract %42 from 13 : (i49) -> i1
    %131 = comb.extract %42 from 12 : (i49) -> i1
    %132 = comb.concat %false, %131 : i1, i1
    %133 = comb.mux bin %130, %c-2_i2, %132 : i2
    %134 = comb.mux bin %129, %c-1_i2, %133 : i2
    %135 = comb.extract %42 from 10 : (i49) -> i1
    %136 = comb.extract %42 from 9 : (i49) -> i1
    %137 = comb.extract %42 from 8 : (i49) -> i1
    %138 = comb.concat %false, %137 : i1, i1
    %139 = comb.mux bin %136, %c-2_i2, %138 : i2
    %140 = comb.mux bin %135, %c-1_i2, %139 : i2
    %141 = comb.mux bin %128, %134, %140 : i2
    %142 = comb.concat %128, %141 : i1, i2
    %143 = comb.extract %42 from 3 : (i49) -> i4
    %144 = comb.icmp bin ne %143, %c0_i4 : i4
    %145 = comb.extract %42 from 6 : (i49) -> i1
    %146 = comb.extract %42 from 5 : (i49) -> i1
    %147 = comb.extract %42 from 4 : (i49) -> i1
    %148 = comb.concat %false, %147 : i1, i1
    %149 = comb.mux bin %146, %c-2_i2, %148 : i2
    %150 = comb.mux bin %145, %c-1_i2, %149 : i2
    %151 = comb.extract %42 from 2 : (i49) -> i1
    %152 = comb.extract %42 from 1 : (i49) -> i1
    %153 = comb.extract %42 from 0 : (i49) -> i1
    %154 = comb.concat %false, %153 : i1, i1
    %155 = comb.mux bin %152, %c-2_i2, %154 : i2
    %156 = comb.mux bin %151, %c-1_i2, %155 : i2
    %157 = comb.mux bin %144, %150, %156 : i2
    %158 = comb.concat %144, %157 : i1, i2
    %159 = comb.mux bin %126, %142, %158 : i3
    %160 = comb.concat %126, %159 : i1, i3
    %161 = comb.mux bin %88, %124, %160 : i4
    %162 = comb.concat %88, %161 : i1, i4
    %163 = comb.mux bin %44, %86, %162 : i5
    %164 = comb.concat %false, %44, %163 : i1, i1, i5
    %165 = comb.sub %c-55_i7, %164 {sv.namehint = "estNormNeg_dist"} : i7
    %166 = comb.extract %io_mulAddResult from 0 : (i49) -> i17
    %167 = comb.concat %166, %io_fromPreMul_bit0AlignedNegSigC : i17, i1
    %168 = comb.icmp bin ne %167, %c0_i18 : i18
    %169 = comb.xor bin %36, %c-1_i75 {sv.namehint = "complSigSum"} : i75
    %170 = comb.extract %169 from 0 : (i75) -> i18
    %171 = comb.icmp bin ne %170, %c0_i18 : i18
    %172 = comb.or bin %io_fromPreMul_CAlignDist_0, %31 : i1
    %173 = comb.extract %io_fromPreMul_CAlignDist from 0 : (i7) -> i5
    %174 = comb.add %173, %c-1_i5 : i5
    %175 = comb.concat %c0_i2, %174 : i2, i5
    %176 = comb.mux bin %172, %io_fromPreMul_CAlignDist, %175 {sv.namehint = "CDom_estNormDist"} : i7
    %177 = comb.xor bin %31, %true : i1
    %178 = comb.extract %176 from 4 : (i7) -> i1
    %179 = comb.xor bin %178, %true : i1
    %180 = comb.extract %io_mulAddResult from 33 : (i49) -> i15
    %181 = comb.extract %io_mulAddResult from 0 : (i49) -> i33
    %182 = comb.concat %181, %io_fromPreMul_bit0AlignedNegSigC : i33, i1
    %183 = comb.icmp bin ne %182, %c0_i34 : i34
    %184 = comb.concat %34, %180, %183 : i26, i15, i1
    %185 = comb.or %31, %178 : i1
    %186 = comb.mux bin %185, %c0_i42, %184 : i42
    %187 = comb.and bin %177, %178 : i1
    %188 = comb.extract %io_mulAddResult from 17 : (i49) -> i31
    %189 = comb.extract %34 from 0 : (i26) -> i10
    %190 = comb.concat %189, %188, %168 : i10, i31, i1
    %191 = comb.mux bin %187, %190, %c0_i42 : i42
    %192 = comb.and bin %31, %179 : i1
    %193 = comb.extract %169 from 34 : (i75) -> i41
    %194 = comb.extract %169 from 0 : (i75) -> i34
    %195 = comb.icmp bin ne %194, %c0_i34 : i34
    %196 = comb.concat %193, %195 : i41, i1
    %197 = comb.mux bin %192, %196, %c0_i42 : i42
    %198 = comb.and bin %31, %178 : i1
    %199 = comb.extract %169 from 18 : (i75) -> i41
    %200 = comb.concat %199, %171 : i41, i1
    %201 = comb.mux bin %198, %200, %c0_i42 : i42
    %202 = comb.or bin %186, %191, %197, %201 {sv.namehint = "CDom_firstNormAbsSigSum"} : i42
    %203 = comb.extract %io_mulAddResult from 17 : (i49) -> i31
    %204 = comb.extract %34 from 0 : (i26) -> i2
    %205 = comb.xor bin %171, %true : i1
    %206 = comb.mux bin %31, %205, %168 : i1
    %207 = comb.extract %io_mulAddResult from 0 : (i49) -> i42
    %208 = comb.extract %165 from 5 : (i7) -> i1
    %209 = comb.extract %165 from 4 : (i7) -> i1
    %210 = comb.extract %io_mulAddResult from 0 : (i49) -> i26
    %211 = comb.replicate %31 : (i1) -> i16
    %212 = comb.concat %210, %211 : i26, i16
    %213 = comb.mux bin %209, %212, %207 : i42
    %214 = comb.extract %io_mulAddResult from 0 : (i49) -> i10
    %215 = comb.replicate %31 : (i1) -> i32
    %216 = comb.concat %214, %215 : i10, i32
    %217 = comb.concat %c0_i8, %204, %203, %206 : i8, i2, i31, i1
    %218 = comb.mux bin %209, %217, %216 : i42
    %219 = comb.mux bin %208, %213, %218 {sv.namehint = "notCDom_pos_firstNormAbsSigSum"} : i42
    %220 = comb.extract %169 from 18 : (i75) -> i32
    %221 = comb.extract %169 from 1 : (i75) -> i42
    %222 = comb.extract %169 from 1 : (i75) -> i27
    %223 = comb.concat %222, %c0_i16 : i27, i16
    %224 = comb.concat %false, %221 : i1, i42
    %225 = comb.mux bin %209, %223, %224 : i43
    %226 = comb.extract %169 from 1 : (i75) -> i11
    %227 = comb.concat %226, %c0_i32 : i11, i32
    %228 = comb.concat %c0_i10, %220, %171 : i10, i32, i1
    %229 = comb.mux bin %209, %228, %227 : i43
    %230 = comb.mux bin %208, %225, %229 {sv.namehint = "notCDom_neg_cFirstNormAbsSigSum"} : i43
    %231 = comb.extract %34 from 2 {sv.namehint = "notCDom_signSigSum"} : (i26) -> i1
    %232 = comb.and bin %31, %18 : i1
    %233 = comb.mux bin %io_fromPreMul_isCDominant, %232, %231 {sv.namehint = "doNegSignSum"} : i1
    %234 = comb.mux bin %io_fromPreMul_isCDominant, %176, %165 {sv.namehint = "estNormDist"} : i7
    %235 = comb.concat %false, %202 : i1, i42
    %236 = comb.mux bin %io_fromPreMul_isCDominant, %235, %230 : i43
    %237 = comb.mux bin %io_fromPreMul_isCDominant, %202, %219 : i42
    %238 = comb.concat %false, %237 : i1, i42
    %239 = comb.mux bin %231, %236, %238 {sv.namehint = "cFirstNormAbsSigSum"} : i43
    %240 = comb.xor bin %io_fromPreMul_isCDominant, %true : i1
    %241 = comb.xor bin %231, %true : i1
    %242 = comb.and bin %240, %241, %31 {sv.namehint = "doIncrSig"} : i1
    %243 = comb.extract %234 from 0 {sv.namehint = "estNormDist_5"} : (i7) -> i4
    %244 = comb.xor bin %243, %c-1_i4 {sv.namehint = "normTo2ShiftDist"} : i4
    %245 = comb.concat %c0_i13, %244 : i13, i4
    %246 = comb.shrs bin %c-65536_i17, %245 : i17
    %247 = comb.extract %246 from 8 : (i17) -> i1
    %248 = comb.extract %246 from 1 : (i17) -> i1
    %249 = comb.extract %246 from 3 : (i17) -> i1
    %250 = comb.extract %246 from 5 : (i17) -> i1
    %251 = comb.extract %246 from 7 : (i17) -> i1
    %252 = comb.extract %246 from 2 : (i17) -> i1
    %253 = comb.extract %246 from 4 : (i17) -> i1
    %254 = comb.extract %246 from 6 : (i17) -> i1
    %255 = comb.extract %246 from 9 : (i17) -> i1
    %256 = comb.extract %246 from 10 : (i17) -> i1
    %257 = comb.extract %246 from 11 : (i17) -> i1
    %258 = comb.extract %246 from 12 : (i17) -> i1
    %259 = comb.extract %246 from 13 : (i17) -> i1
    %260 = comb.extract %246 from 14 : (i17) -> i1
    %261 = comb.extract %246 from 15 : (i17) -> i1
    %262 = comb.concat %248, %252, %249, %253, %250, %254, %251, %247, %255, %256, %257, %258, %259, %260, %261, %true {sv.namehint = "absSigSumExtraMask"} : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
    %263 = comb.extract %239 from 1 : (i43) -> i42
    %264 = comb.concat %c0_i38, %244 : i38, i4
    %265 = comb.shru bin %263, %264 : i42
    %266 = comb.extract %239 from 0 : (i43) -> i16
    %267 = comb.xor bin %266, %c-1_i16 : i16
    %268 = comb.and bin %267, %262 : i16
    %269 = comb.icmp bin eq %268, %c0_i16 : i16
    %270 = comb.and bin %266, %262 : i16
    %271 = comb.icmp bin ne %270, %c0_i16 : i16
    %272 = comb.mux bin %242, %269, %271 : i1
    %273 = comb.extract %265 from 25 : (i42) -> i2
    %274 = comb.icmp bin eq %273, %c0_i2 {sv.namehint = "sigX3Shift1"} : i2
    %275 = comb.concat %c0_i4, %234 : i4, i7
    %276 = comb.sub %io_fromPreMul_sExpSum, %275 {sv.namehint = "sExpX3"} : i11
    %277 = comb.extract %265 from 24 : (i42) -> i3
    %278 = comb.icmp bin ne %277, %c0_i3 : i3
    %279 = comb.xor bin %278, %true {sv.namehint = "isZeroY"} : i1
    %280 = comb.xor bin %io_fromPreMul_signProd, %233 : i1
    %281 = comb.mux bin %278, %280, %29 {sv.namehint = "signY"} : i1
    %282 = comb.extract %276 from 0 {sv.namehint = "sExpX3_13"} : (i11) -> i10
    %283 = comb.extract %276 from 10 : (i11) -> i1
    %284 = comb.replicate %283 : (i1) -> i27
    %285 = comb.xor bin %282, %c-1_i10 : i10
    %286 = comb.extract %285 from 9 : (i10) -> i1
    %287 = comb.extract %285 from 8 : (i10) -> i1
    %288 = comb.extract %285 from 7 : (i10) -> i1
    %289 = comb.extract %285 from 6 : (i10) -> i1
    %290 = comb.extract %285 from 0 : (i10) -> i6
    %291 = comb.concat %c0_i59, %290 : i59, i6
    %292 = comb.shrs bin %c-18446744073709551616_i65, %291 : i65
    %293 = comb.extract %292 from 58 : (i65) -> i1
    %294 = comb.extract %292 from 43 : (i65) -> i1
    %295 = comb.extract %292 from 45 : (i65) -> i1
    %296 = comb.extract %292 from 47 : (i65) -> i1
    %297 = comb.extract %292 from 49 : (i65) -> i1
    %298 = comb.extract %292 from 51 : (i65) -> i1
    %299 = comb.extract %292 from 53 : (i65) -> i1
    %300 = comb.extract %292 from 55 : (i65) -> i1
    %301 = comb.extract %292 from 57 : (i65) -> i1
    %302 = comb.extract %292 from 44 : (i65) -> i1
    %303 = comb.extract %292 from 46 : (i65) -> i1
    %304 = comb.extract %292 from 48 : (i65) -> i1
    %305 = comb.extract %292 from 50 : (i65) -> i1
    %306 = comb.extract %292 from 52 : (i65) -> i1
    %307 = comb.extract %292 from 54 : (i65) -> i1
    %308 = comb.extract %292 from 56 : (i65) -> i1
    %309 = comb.extract %292 from 59 : (i65) -> i1
    %310 = comb.extract %292 from 60 : (i65) -> i1
    %311 = comb.extract %292 from 61 : (i65) -> i1
    %312 = comb.extract %292 from 62 : (i65) -> i1
    %313 = comb.extract %292 from 63 : (i65) -> i1
    %314 = comb.concat %294, %302, %295, %303, %296, %304, %297, %305, %298, %306, %299, %307, %300, %308, %301, %293, %309, %310, %311, %312, %313 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
    %315 = comb.xor bin %314, %c-1_i21 : i21
    %316 = comb.mux bin %289, %c0_i21, %315 : i21
    %317 = comb.xor bin %316, %c-1_i21 : i21
    %318 = comb.concat %317, %c-1_i4 : i21, i4
    %319 = comb.extract %292 from 0 : (i65) -> i1
    %320 = comb.extract %292 from 1 : (i65) -> i1
    %321 = comb.extract %292 from 2 : (i65) -> i1
    %322 = comb.extract %292 from 3 : (i65) -> i1
    %323 = comb.concat %319, %320, %321, %322 : i1, i1, i1, i1
    %324 = comb.mux bin %289, %323, %c0_i4 : i4
    %325 = comb.concat %c0_i21, %324 : i21, i4
    %326 = comb.mux bin %288, %318, %325 : i25
    %327 = comb.and bin %286, %287 : i1
    %328 = comb.mux bin %327, %326, %c0_i25 : i25
    %329 = comb.extract %265 from 25 : (i42) -> i1
    %330 = comb.extract %328 from 1 : (i25) -> i24
    %331 = comb.extract %328 from 0 : (i25) -> i1
    %332 = comb.or bin %331, %329 : i1
    %333 = comb.concat %330, %332, %c-1_i2 : i24, i1, i2
    %334 = comb.or bin %284, %333 {sv.namehint = "roundMask"} : i27
    %335 = comb.extract %334 from 1 : (i27) -> i26
    %336 = comb.xor bin %335, %c-1_i26 : i26
    %337 = comb.extract %334 from 0 : (i27) -> i26
    %338 = comb.extract %265 from 0 : (i42) -> i25
    %339 = comb.concat %338, %272 : i25, i1
    %340 = comb.and bin %339, %337, %336 : i26
    %341 = comb.icmp bin ne %340, %c0_i26 {sv.namehint = "roundPosBit"} : i26
    %342 = comb.and bin %339, %335 : i26
    %343 = comb.icmp bin ne %342, %c0_i26 {sv.namehint = "anyRoundExtra"} : i26
    %344 = comb.extract %265 from 0 : (i42) -> i25
    %345 = comb.concat %344, %272 : i25, i1
    %346 = comb.xor %345, %c-1_i26 : i26
    %347 = comb.and bin %346, %335 : i26
    %348 = comb.icmp bin eq %347, %c0_i26 {sv.namehint = "allRoundExtra"} : i26
    %349 = comb.concat %340, %342 : i26, i26
    %350 = comb.icmp bin ne %349, %c0_i52 {sv.namehint = "anyRound"} : i52
    %351 = comb.and bin %341, %348 {sv.namehint = "allRound"} : i1
    %352 = comb.mux bin %281, %29, %30 {sv.namehint = "roundDirectUp"} : i1
    %353 = comb.xor bin %242, %true : i1
    %354 = comb.and bin %353, %28, %341, %343 : i1
    %355 = comb.and bin %353, %352, %350 : i1
    %356 = comb.and bin %242, %351 : i1
    %357 = comb.and bin %242, %28, %341 : i1
    %358 = comb.and bin %242, %352 : i1
    %359 = comb.or bin %354, %355, %356, %357, %358 {sv.namehint = "roundUp"} : i1
    %360 = comb.xor bin %341, %true : i1
    %361 = comb.and bin %28, %360, %348 : i1
    %362 = comb.xor bin %343, %true : i1
    %363 = comb.and bin %28, %341, %362 : i1
    %364 = comb.mux bin %242, %361, %363 {sv.namehint = "roundEven"} : i1
    %365 = comb.xor bin %351, %true : i1
    %366 = comb.mux bin %242, %365, %350 {sv.namehint = "inexactY"} : i1
    %367 = comb.extract %265 from 26 : (i42) -> i1
    %368 = comb.extract %265 from 1 : (i42) -> i25
    %369 = comb.extract %334 from 2 : (i27) -> i25
    %370 = comb.or %368, %369 : i25
    %371 = comb.concat %367, %370 : i1, i25
    %372 = comb.add %371, %c1_i26 {sv.namehint = "roundUp_sigY3"} : i26
    %373 = comb.extract %334 from 2 : (i27) -> i25
    %374 = comb.xor %373, %c-1_i25 : i25
    %375 = comb.extract %265 from 1 : (i42) -> i25
    %376 = comb.and %375, %374 : i25
    %377 = comb.concat %false, %376 : i1, i25
    %378 = comb.or %359, %364 : i1
    %379 = comb.mux bin %378, %c0_i26, %377 : i26
    %380 = comb.mux bin %359, %372, %c0_i26 : i26
    %381 = comb.and bin %372, %336 : i26
    %382 = comb.mux bin %364, %381, %c0_i26 : i26
    %383 = comb.or bin %379, %380, %382 {sv.namehint = "sigY3"} : i26
    %384 = comb.extract %383 from 25 : (i26) -> i1
    %385 = comb.extract %276 from 0 : (i11) -> i10
    %386 = comb.add %385, %c1_i10 : i10
    %387 = comb.mux %384, %386, %c0_i10 : i10
    %388 = comb.extract %383 from 24 : (i26) -> i1
    %389 = comb.extract %276 from 0 : (i11) -> i10
    %390 = comb.mux %388, %389, %c0_i10 : i10
    %391 = comb.extract %383 from 24 : (i26) -> i2
    %392 = comb.icmp bin eq %391, %c0_i2 : i2
    %393 = comb.extract %276 from 0 : (i11) -> i10
    %394 = comb.add %393, %c-1_i10 : i10
    %395 = comb.mux %392, %394, %c0_i10 : i10
    %396 = comb.or %387, %390, %395 {sv.namehint = "sExpY"} : i10
    %397 = comb.extract %396 from 0 {sv.namehint = "expY"} : (i10) -> i9
    %398 = comb.extract %383 from 0 : (i26) -> i23
    %399 = comb.extract %383 from 1 : (i26) -> i23
    %400 = comb.mux bin %274, %398, %399 {sv.namehint = "fractY"} : i23
    %401 = comb.extract %396 from 7 : (i10) -> i3
    %402 = comb.icmp bin eq %401, %c3_i3 {sv.namehint = "overflowY"} : i3
    %403 = comb.extract %396 from 9 : (i10) -> i1
    %404 = comb.icmp bin ult %397, %c107_i9 : i9
    %405 = comb.or bin %403, %404 : i1
    %406 = comb.and bin %278, %405 {sv.namehint = "totalUnderflowY"} : i1
    %407 = comb.mux bin %274, %c-126_i8, %c-127_i8 : i8
    %408 = comb.concat %c0_i2, %407 : i2, i8
    %409 = comb.icmp bin ule %282, %408 : i10
    %410 = comb.or bin %283, %409 : i1
    %411 = comb.and bin %29, %281 : i1
    %412 = comb.xor bin %281, %true : i1
    %413 = comb.and bin %30, %412 : i1
    %414 = comb.or bin %411, %413 {sv.namehint = "roundMagUp"} : i1
    %415 = comb.or bin %28, %414 {sv.namehint = "overflowY_roundMagUp"} : i1
    %416 = comb.or bin %2, %11 {sv.namehint = "mulSpecial"} : i1
    %417 = comb.or bin %416, %21 {sv.namehint = "addSpecial"} : i1
    %418 = comb.and bin %io_fromPreMul_isZeroProd, %19 {sv.namehint = "notSpecial_addZeros"} : i1
    %419 = comb.xor bin %417, %true : i1
    %420 = comb.xor bin %418, %true : i1
    %421 = comb.and bin %419, %420 {sv.namehint = "commonCase"} : i1
    %422 = comb.and bin %5, %9 : i1
    %423 = comb.and bin %0, %14 : i1
    %424 = comb.xor bin %6, %true : i1
    %425 = comb.xor bin %15, %true : i1
    %426 = comb.or bin %5, %14 : i1
    %427 = comb.and bin %424, %425, %426, %24, %31 : i1
    %428 = comb.or bin %422, %423, %427 {sv.namehint = "notSigNaN_invalid"} : i1
    %429 = comb.or bin %8, %17, %27, %428 {sv.namehint = "invalid"} : i1
    %430 = comb.and bin %421, %402 {sv.namehint = "overflow"} : i1
    %431 = comb.and bin %421, %366, %410 {sv.namehint = "underflow"} : i1
    %432 = comb.and bin %421, %366 : i1
    %433 = comb.or bin %430, %432 {sv.namehint = "inexact"} : i1
    %434 = comb.or bin %418, %279, %406 {sv.namehint = "notSpecial_isZeroOut"} : i1
    %435 = comb.and bin %421, %406, %414 {sv.namehint = "pegMinFiniteMagOut"} : i1
    %436 = comb.xor bin %415, %true : i1
    %437 = comb.and bin %430, %436 {sv.namehint = "pegMaxFiniteMagOut"} : i1
    %438 = comb.and bin %430, %415 : i1
    %439 = comb.or bin %426, %24, %438 {sv.namehint = "notNaN_isInfOut"} : i1
    %440 = comb.or bin %6, %15, %25, %428 {sv.namehint = "isNaNOut"} : i1
    %441 = comb.and bin %177, %io_fromPreMul_opSignC : i1
    %442 = comb.xor bin %21, %true : i1
    %443 = comb.and bin %416, %442, %io_fromPreMul_signProd : i1
    %444 = comb.xor bin %416, %true : i1
    %445 = comb.and bin %444, %21, %io_fromPreMul_opSignC : i1
    %446 = comb.and bin %444, %418, %31, %29 : i1
    %447 = comb.or bin %441, %443, %445, %446 {sv.namehint = "uncommonCaseSignOut"} : i1
    %448 = comb.xor bin %440, %true : i1
    %449 = comb.and bin %448, %447 : i1
    %450 = comb.and bin %421, %281 : i1
    %451 = comb.or bin %449, %450 {sv.namehint = "signOut"} : i1
    %452 = comb.mux bin %434, %c-64_i9, %c0_i9 : i9
    %453 = comb.xor bin %452, %c-1_i9 : i9
    %454 = comb.mux bin %435, %c-108_i9, %c0_i9 : i9
    %455 = comb.xor bin %454, %c-1_i9 : i9
    %456 = comb.xor %437, %true : i1
    %457 = comb.concat %true, %456, %c-1_i7 : i1, i1, i7
    %458 = comb.xor %439, %true : i1
    %459 = comb.concat %c-1_i2, %458, %c-1_i6 : i2, i1, i6
    %460 = comb.and bin %397, %453, %455, %457, %459 : i9
    %461 = comb.mux bin %435, %c107_i9, %c0_i9 : i9
    %462 = comb.mux bin %437, %c-129_i9, %c0_i9 : i9
    %463 = comb.mux bin %439, %c-128_i9, %c0_i9 : i9
    %464 = comb.mux bin %440, %c-64_i9, %c0_i9 : i9
    %465 = comb.or bin %460, %461, %462, %463, %464 {sv.namehint = "expOut"} : i9
    %466 = comb.and bin %406, %414 : i1
    %467 = comb.or bin %466, %440 : i1
    %468 = comb.concat %440, %c0_i22 : i1, i22
    %469 = comb.mux bin %467, %468, %400 : i23
    %470 = comb.replicate %437 : (i1) -> i23
    %471 = comb.or bin %469, %470 {sv.namehint = "fractOut"} : i23
    %472 = comb.concat %451, %465, %471 {sv.namehint = "io_out"} : i1, i9, i23
    %473 = comb.concat %429, %false, %430, %431, %433 {sv.namehint = "io_exceptionFlags"} : i1, i1, i1, i1, i1
    hw.output %472, %473 : i33, i5
  }
  hw.module private @RoundRawFNToRecFN(in %io_invalidExc : i1, in %io_in_sign : i1, in %io_in_isNaN : i1, in %io_in_isInf : i1, in %io_in_isZero : i1, in %io_in_sExp : i10, in %io_in_sig : i27, in %io_roundingMode : i2, out io_out : i33, out io_exceptionFlags : i5) {
    %c-1_i6 = hw.constant -1 : i6
    %c-1_i7 = hw.constant -1 : i7
    %c2_i4 = hw.constant 2 : i4
    %c0_i53 = hw.constant 0 : i53
    %c-1_i24 = hw.constant -1 : i24
    %c107_i11 = hw.constant 107 : i11
    %c1_i26 = hw.constant 1 : i26
    %c0_i10 = hw.constant 0 : i10
    %c-1_i26 = hw.constant -1 : i26
    %c0_i27 = hw.constant 0 : i27
    %c-1_i22 = hw.constant -1 : i22
    %c0_i59 = hw.constant 0 : i59
    %c-1_i9 = hw.constant -1 : i9
    %c0_i2 = hw.constant 0 : i2
    %c-2_i2 = hw.constant -2 : i2
    %c-1_i2 = hw.constant -1 : i2
    %c0_i25 = hw.constant 0 : i25
    %c-1_i3 = hw.constant -1 : i3
    %true = hw.constant true
    %c0_i26 = hw.constant 0 : i26
    %c-64_i9 = hw.constant -64 : i9
    %c107_i9 = hw.constant 107 : i9
    %c-129_i9 = hw.constant -129 : i9
    %c-128_i9 = hw.constant -128 : i9
    %c-18446744073709551616_i65 = hw.constant -18446744073709551616 : i65
    %c0_i22 = hw.constant 0 : i22
    %c0_i3 = hw.constant 0 : i3
    %c129_i9 = hw.constant 129 : i9
    %c130_i9 = hw.constant 130 : i9
    %c0_i9 = hw.constant 0 : i9
    %c-108_i9 = hw.constant -108 : i9
    %false = hw.constant false
    %0 = comb.icmp bin eq %io_roundingMode, %c0_i2 {sv.namehint = "roundingMode_nearest_even"} : i2
    %1 = comb.icmp bin eq %io_roundingMode, %c-2_i2 {sv.namehint = "roundingMode_min"} : i2
    %2 = comb.icmp bin eq %io_roundingMode, %c-1_i2 {sv.namehint = "roundingMode_max"} : i2
    %3 = comb.and bin %1, %io_in_sign : i1
    %4 = comb.xor bin %io_in_sign, %true : i1
    %5 = comb.and bin %2, %4 : i1
    %6 = comb.or bin %3, %5 {sv.namehint = "roundMagUp"} : i1
    %7 = comb.extract %io_in_sig from 26 {sv.namehint = "doShiftSigDown1"} : (i27) -> i1
    %8 = comb.icmp bin slt %io_in_sExp, %c0_i10 {sv.namehint = "isNegExp"} : i10
    %9 = comb.replicate %8 : (i1) -> i25
    %10 = comb.extract %io_in_sExp from 0 : (i10) -> i9
    %11 = comb.xor bin %10, %c-1_i9 : i9
    %12 = comb.extract %11 from 8 : (i9) -> i1
    %13 = comb.extract %11 from 7 : (i9) -> i1
    %14 = comb.extract %11 from 6 : (i9) -> i1
    %15 = comb.extract %11 from 0 : (i9) -> i6
    %16 = comb.concat %c0_i59, %15 : i59, i6
    %17 = comb.shrs bin %c-18446744073709551616_i65, %16 : i65
    %18 = comb.extract %17 from 57 : (i65) -> i1
    %19 = comb.extract %17 from 42 : (i65) -> i1
    %20 = comb.extract %17 from 44 : (i65) -> i1
    %21 = comb.extract %17 from 46 : (i65) -> i1
    %22 = comb.extract %17 from 48 : (i65) -> i1
    %23 = comb.extract %17 from 50 : (i65) -> i1
    %24 = comb.extract %17 from 52 : (i65) -> i1
    %25 = comb.extract %17 from 54 : (i65) -> i1
    %26 = comb.extract %17 from 56 : (i65) -> i1
    %27 = comb.extract %17 from 43 : (i65) -> i1
    %28 = comb.extract %17 from 45 : (i65) -> i1
    %29 = comb.extract %17 from 47 : (i65) -> i1
    %30 = comb.extract %17 from 49 : (i65) -> i1
    %31 = comb.extract %17 from 51 : (i65) -> i1
    %32 = comb.extract %17 from 53 : (i65) -> i1
    %33 = comb.extract %17 from 55 : (i65) -> i1
    %34 = comb.extract %17 from 58 : (i65) -> i1
    %35 = comb.extract %17 from 59 : (i65) -> i1
    %36 = comb.extract %17 from 60 : (i65) -> i1
    %37 = comb.extract %17 from 61 : (i65) -> i1
    %38 = comb.extract %17 from 62 : (i65) -> i1
    %39 = comb.extract %17 from 63 : (i65) -> i1
    %40 = comb.concat %19, %27, %20, %28, %21, %29, %22, %30, %23, %31, %24, %32, %25, %33, %26, %18, %34, %35, %36, %37, %38, %39 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
    %41 = comb.xor bin %40, %c-1_i22 : i22
    %42 = comb.mux bin %14, %c0_i22, %41 : i22
    %43 = comb.xor bin %42, %c-1_i22 : i22
    %44 = comb.concat %43, %c-1_i3 : i22, i3
    %45 = comb.extract %17 from 0 : (i65) -> i1
    %46 = comb.extract %17 from 1 : (i65) -> i1
    %47 = comb.extract %17 from 2 : (i65) -> i1
    %48 = comb.concat %45, %46, %47 : i1, i1, i1
    %49 = comb.mux bin %14, %48, %c0_i3 : i3
    %50 = comb.concat %c0_i22, %49 : i22, i3
    %51 = comb.mux bin %13, %44, %50 : i25
    %52 = comb.mux bin %12, %51, %c0_i25 : i25
    %53 = comb.or bin %9, %52 : i25
    %54 = comb.extract %53 from 1 : (i25) -> i24
    %55 = comb.extract %53 from 0 : (i25) -> i1
    %56 = comb.or bin %55, %7 : i1
    %57 = comb.concat %8, %54, %56, %true {sv.namehint = "shiftedRoundMask"} : i1, i24, i1, i1
    %58 = comb.xor %8, %true : i1
    %59 = comb.xor %54, %c-1_i24 : i24
    %60 = comb.xor %56, %true : i1
    %61 = comb.extract %io_in_sig from 1 : (i27) -> i26
    %62 = comb.concat %58, %59, %60 : i1, i24, i1
    %63 = comb.concat %54, %56, %true : i24, i1, i1
    %64 = comb.and %61, %62, %63 : i26
    %65 = comb.icmp bin ne %64, %c0_i26 {sv.namehint = "roundPosBit"} : i26
    %66 = comb.and bin %io_in_sig, %57 : i27
    %67 = comb.concat %64, %66 : i26, i27
    %68 = comb.icmp bin ne %67, %c0_i53 {sv.namehint = "anyRound"} : i53
    %69 = comb.and bin %0, %65 : i1
    %70 = comb.and bin %6, %68 : i1
    %71 = comb.or bin %69, %70 : i1
    %72 = comb.extract %io_in_sig from 2 : (i27) -> i25
    %73 = comb.concat %54, %56 : i24, i1
    %74 = comb.or %72, %73 : i25
    %75 = comb.concat %false, %74 : i1, i25
    %76 = comb.add bin %75, %c1_i26 : i26
    %77 = comb.icmp bin eq %66, %c0_i27 : i27
    %78 = comb.and bin %69, %77 : i1
    %79 = comb.concat %54, %56, %true : i24, i1, i1
    %80 = comb.mux bin %78, %79, %c0_i26 : i26
    %81 = comb.xor bin %80, %c-1_i26 : i26
    %82 = comb.and bin %76, %81 : i26
    %83 = comb.xor %54, %c-1_i24 : i24
    %84 = comb.xor %56, %true : i1
    %85 = comb.extract %io_in_sig from 2 : (i27) -> i25
    %86 = comb.concat %83, %84 : i24, i1
    %87 = comb.and bin %85, %86 : i25
    %88 = comb.concat %false, %87 : i1, i25
    %89 = comb.mux bin %71, %82, %88 {sv.namehint = "roundedSig"} : i26
    %90 = comb.extract %89 from 24 : (i26) -> i2
    %91 = comb.extract %io_in_sExp from 9 : (i10) -> i1
    %92 = comb.concat %91, %io_in_sExp : i1, i10
    %93 = comb.concat %c0_i9, %90 : i9, i2
    %94 = comb.add bin %92, %93 {sv.namehint = "sRoundedExp"} : i11
    %95 = comb.extract %94 from 0 {sv.namehint = "common_expOut"} : (i11) -> i9
    %96 = comb.extract %89 from 1 : (i26) -> i23
    %97 = comb.extract %89 from 0 : (i26) -> i23
    %98 = comb.mux bin %7, %96, %97 {sv.namehint = "common_fractOut"} : i23
    %99 = comb.extract %94 from 7 : (i11) -> i4
    %100 = comb.icmp bin sgt %99, %c2_i4 {sv.namehint = "common_overflow"} : i4
    %101 = comb.icmp bin slt %94, %c107_i11 {sv.namehint = "common_totalUnderflow"} : i11
    %102 = comb.mux bin %7, %c129_i9, %c130_i9 : i9
    %103 = comb.extract %102 from 8 : (i9) -> i1
    %104 = comb.concat %103, %102 : i1, i9
    %105 = comb.icmp bin slt %io_in_sExp, %104 : i10
    %106 = comb.or bin %io_invalidExc, %io_in_isNaN {sv.namehint = "isNaNOut"} : i1
    %107 = comb.xor bin %106, %true : i1
    %108 = comb.xor bin %io_in_isInf, %true : i1
    %109 = comb.xor bin %io_in_isZero, %true : i1
    %110 = comb.and bin %107, %108, %109 {sv.namehint = "commonCase"} : i1
    %111 = comb.and bin %110, %100 {sv.namehint = "overflow"} : i1
    %112 = comb.and bin %110, %68, %105 {sv.namehint = "underflow"} : i1
    %113 = comb.and bin %110, %68 : i1
    %114 = comb.or bin %111, %113 {sv.namehint = "inexact"} : i1
    %115 = comb.or bin %0, %6 {sv.namehint = "overflow_roundMagUp"} : i1
    %116 = comb.and bin %110, %101, %6 {sv.namehint = "pegMinNonzeroMagOut"} : i1
    %117 = comb.xor bin %115, %true : i1
    %118 = comb.and bin %111, %117 {sv.namehint = "pegMaxFiniteMagOut"} : i1
    %119 = comb.and bin %111, %115 : i1
    %120 = comb.or bin %io_in_isInf, %119 {sv.namehint = "notNaN_isInfOut"} : i1
    %121 = comb.xor %106, %true : i1
    %122 = comb.and %121, %io_in_sign {sv.namehint = "signOut"} : i1
    %123 = comb.or bin %io_in_isZero, %101 : i1
    %124 = comb.mux bin %123, %c-64_i9, %c0_i9 : i9
    %125 = comb.xor bin %124, %c-1_i9 : i9
    %126 = comb.mux bin %116, %c-108_i9, %c0_i9 : i9
    %127 = comb.xor bin %126, %c-1_i9 : i9
    %128 = comb.xor %118, %true : i1
    %129 = comb.concat %true, %128, %c-1_i7 : i1, i1, i7
    %130 = comb.xor %120, %true : i1
    %131 = comb.concat %c-1_i2, %130, %c-1_i6 : i2, i1, i6
    %132 = comb.and bin %95, %125, %127, %129, %131 : i9
    %133 = comb.mux bin %116, %c107_i9, %c0_i9 : i9
    %134 = comb.mux bin %118, %c-129_i9, %c0_i9 : i9
    %135 = comb.mux bin %120, %c-128_i9, %c0_i9 : i9
    %136 = comb.mux bin %106, %c-64_i9, %c0_i9 : i9
    %137 = comb.or bin %132, %133, %134, %135, %136 {sv.namehint = "expOut"} : i9
    %138 = comb.or bin %101, %106 : i1
    %139 = comb.concat %106, %c0_i22 : i1, i22
    %140 = comb.mux bin %138, %139, %98 : i23
    %141 = comb.replicate %118 : (i1) -> i23
    %142 = comb.or bin %140, %141 {sv.namehint = "fractOut"} : i23
    %143 = comb.concat %122, %137, %142 {sv.namehint = "io_out"} : i1, i9, i23
    %144 = comb.concat %io_invalidExc, %false, %111, %112, %114 {sv.namehint = "io_exceptionFlags"} : i1, i1, i1, i1, i1
    hw.output %143, %144 : i33, i5
  }
  hw.module private @MulAddRecFN_preMul_1(in %io_op : i2, in %io_a : i65, in %io_b : i65, in %io_c : i65, in %io_roundingMode : i2, out io_mulAddA : i53, out io_mulAddB : i53, out io_mulAddC : i106, out io_toPostMul_highExpA : i3, out io_toPostMul_isNaN_isQuietNaNA : i1, out io_toPostMul_highExpB : i3, out io_toPostMul_isNaN_isQuietNaNB : i1, out io_toPostMul_signProd : i1, out io_toPostMul_isZeroProd : i1, out io_toPostMul_opSignC : i1, out io_toPostMul_highExpC : i3, out io_toPostMul_isNaN_isQuietNaNC : i1, out io_toPostMul_isCDominant : i1, out io_toPostMul_CAlignDist_0 : i1, out io_toPostMul_CAlignDist : i8, out io_toPostMul_bit0AlignedNegSigC : i1, out io_toPostMul_highAlignedNegSigC : i55, out io_toPostMul_sExpSum : i14, out io_toPostMul_roundingMode : i2) {
    %false = hw.constant false
    %c-174763_i19 = hw.constant -174763 : i19
    %c5_i4 = hw.constant 5 : i4
    %c56_i14 = hw.constant 56 : i14
    %c161_i13 = hw.constant 161 : i13
    %c54_i13 = hw.constant 54 : i13
    %c0_i53 = hw.constant 0 : i53
    %c0_i154 = hw.constant 0 : i154
    %c-1_i33 = hw.constant -1 : i33
    %c0_i59 = hw.constant 0 : i59
    %c0_i2 = hw.constant 0 : i2
    %c0_i13 = hw.constant 0 : i13
    %true = hw.constant true
    %c0_i3 = hw.constant 0 : i3
    %c-95_i8 = hw.constant -95 : i8
    %c-1_i20 = hw.constant -1 : i20
    %c0_i8 = hw.constant 0 : i8
    %c-18446744073709551616_i65 = hw.constant -18446744073709551616 : i65
    %c0_i33 = hw.constant 0 : i33
    %c0_i20 = hw.constant 0 : i20
    %0 = comb.extract %io_a from 64 {sv.namehint = "signA"} : (i65) -> i1
    %1 = comb.extract %io_a from 52 {sv.namehint = "expA"} : (i65) -> i12
    %2 = comb.extract %io_a from 0 {sv.namehint = "fractA"} : (i65) -> i52
    %3 = comb.extract %io_a from 61 {sv.namehint = "io_toPostMul_highExpA"} : (i65) -> i3
    %4 = comb.icmp bin ne %3, %c0_i3 : i3
    %5 = comb.xor bin %4, %true {sv.namehint = "isZeroA"} : i1
    %6 = comb.concat %4, %2 {sv.namehint = "sigA"} : i1, i52
    %7 = comb.extract %io_b from 64 {sv.namehint = "signB"} : (i65) -> i1
    %8 = comb.extract %io_b from 0 {sv.namehint = "fractB"} : (i65) -> i52
    %9 = comb.extract %io_b from 61 {sv.namehint = "io_toPostMul_highExpB"} : (i65) -> i3
    %10 = comb.icmp bin ne %9, %c0_i3 : i3
    %11 = comb.xor bin %10, %true {sv.namehint = "isZeroB"} : i1
    %12 = comb.concat %10, %8 {sv.namehint = "sigB"} : i1, i52
    %13 = comb.extract %io_c from 64 : (i65) -> i1
    %14 = comb.extract %io_op from 0 : (i2) -> i1
    %15 = comb.xor bin %13, %14 {sv.namehint = "opSignC"} : i1
    %16 = comb.extract %io_c from 52 {sv.namehint = "expC"} : (i65) -> i12
    %17 = comb.extract %io_c from 0 {sv.namehint = "fractC"} : (i65) -> i52
    %18 = comb.extract %io_c from 61 {sv.namehint = "io_toPostMul_highExpC"} : (i65) -> i3
    %19 = comb.icmp bin ne %18, %c0_i3 : i3
    %20 = comb.concat %19, %17 {sv.namehint = "sigC"} : i1, i52
    %21 = comb.extract %io_op from 1 : (i2) -> i1
    %22 = comb.xor bin %0, %7, %21 {sv.namehint = "signProd"} : i1
    %23 = comb.or bin %5, %11 {sv.namehint = "isZeroProd"} : i1
    %24 = comb.extract %io_b from 63 : (i65) -> i1
    %25 = comb.xor bin %24, %true : i1
    %26 = comb.replicate %25 : (i1) -> i3
    %27 = comb.extract %io_b from 52 : (i65) -> i11
    %28 = comb.concat %c0_i2, %1 : i2, i12
    %29 = comb.concat %26, %27 : i3, i11
    %30 = comb.add %28, %29, %c56_i14 {sv.namehint = "sExpAlignedProd"} : i14
    %31 = comb.xor bin %22, %15 {sv.namehint = "doSubMags"} : i1
    %32 = comb.concat %c0_i2, %16 : i2, i12
    %33 = comb.sub %30, %32 : i14
    %34 = comb.extract %33 from 13 : (i14) -> i1
    %35 = comb.or bin %23, %34 {sv.namehint = "CAlignDist_floor"} : i1
    %36 = comb.extract %33 from 0 : (i14) -> i13
    %37 = comb.icmp bin eq %36, %c0_i13 : i13
    %38 = comb.or bin %35, %37 {sv.namehint = "CAlignDist_0"} : i1
    %39 = comb.icmp bin ult %36, %c54_i13 : i13
    %40 = comb.or bin %35, %39 : i1
    %41 = comb.and bin %19, %40 {sv.namehint = "isCDominant"} : i1
    %42 = comb.icmp bin ult %36, %c161_i13 : i13
    %43 = comb.extract %33 from 0 : (i14) -> i8
    %44 = comb.mux bin %42, %43, %c-95_i8 : i8
    %45 = comb.mux bin %35, %c0_i8, %44 {sv.namehint = "CAlignDist"} : i8
    %46 = comb.concat %c0_i2, %16 : i2, i12
    %47 = comb.mux bin %35, %46, %30 {sv.namehint = "sExpSum"} : i14
    %48 = comb.extract %45 from 7 : (i8) -> i1
    %49 = comb.extract %45 from 6 : (i8) -> i1
    %50 = comb.extract %45 from 0 : (i8) -> i6
    %51 = comb.concat %c0_i59, %50 : i59, i6
    %52 = comb.shrs bin %c-18446744073709551616_i65, %51 : i65
    %53 = comb.extract %52 from 35 : (i65) -> i2
    %54 = comb.extract %52 from 39 : (i65) -> i2
    %55 = comb.extract %52 from 43 : (i65) -> i2
    %56 = comb.extract %52 from 47 : (i65) -> i2
    %57 = comb.extract %52 from 51 : (i65) -> i2
    %58 = comb.extract %52 from 55 : (i65) -> i2
    %59 = comb.extract %52 from 37 : (i65) -> i2
    %60 = comb.concat %59, %54 : i2, i2
    %61 = comb.extract %52 from 41 : (i65) -> i2
    %62 = comb.extract %52 from 45 : (i65) -> i2
    %63 = comb.concat %62, %56 : i2, i2
    %64 = comb.extract %52 from 49 : (i65) -> i2
    %65 = comb.extract %52 from 53 : (i65) -> i2
    %66 = comb.concat %65, %58 : i2, i2
    %67 = comb.extract %52 from 62 : (i65) -> i1
    %68 = comb.extract %52 from 54 : (i65) -> i1
    %69 = comb.concat %53, %59, %54, %61, %55, %62, %56, %64, %57, %68 : i2, i2, i2, i2, i2, i2, i2, i2, i2, i1
    %70 = comb.and %69, %c-174763_i19 : i19
    %71 = comb.extract %52 from 31 : (i65) -> i1
    %72 = comb.extract %52 from 33 : (i65) -> i1
    %73 = comb.extract %52 from 35 : (i65) -> i1
    %74 = comb.and %60, %c5_i4 : i4
    %75 = comb.extract %52 from 41 : (i65) -> i1
    %76 = comb.extract %52 from 43 : (i65) -> i1
    %77 = comb.and %63, %c5_i4 : i4
    %78 = comb.extract %52 from 49 : (i65) -> i1
    %79 = comb.extract %52 from 51 : (i65) -> i1
    %80 = comb.and %66, %c5_i4 : i4
    %81 = comb.extract %52 from 57 : (i65) -> i1
    %82 = comb.extract %52 from 59 : (i65) -> i1
    %83 = comb.extract %52 from 61 : (i65) -> i1
    %84 = comb.extract %52 from 32 : (i65) -> i1
    %85 = comb.extract %52 from 34 : (i65) -> i1
    %86 = comb.extract %70 from 15 : (i19) -> i4
    %87 = comb.or %86, %74 : i4
    %88 = comb.extract %52 from 40 : (i65) -> i1
    %89 = comb.extract %70 from 13 : (i19) -> i1
    %90 = comb.or %89, %75 : i1
    %91 = comb.extract %52 from 42 : (i65) -> i1
    %92 = comb.extract %70 from 7 : (i19) -> i4
    %93 = comb.or %92, %77 : i4
    %94 = comb.extract %52 from 48 : (i65) -> i1
    %95 = comb.extract %70 from 5 : (i19) -> i1
    %96 = comb.or %95, %78 : i1
    %97 = comb.extract %52 from 50 : (i65) -> i1
    %98 = comb.extract %70 from 0 : (i19) -> i3
    %99 = comb.concat %98, %false : i3, i1
    %100 = comb.or %99, %80 : i4
    %101 = comb.extract %52 from 56 : (i65) -> i1
    %102 = comb.extract %52 from 58 : (i65) -> i1
    %103 = comb.extract %52 from 60 : (i65) -> i1
    %104 = comb.extract %52 from 63 : (i65) -> i1
    %105 = comb.concat %71, %84, %72, %85, %73, %87, %88, %90, %91, %76, %93, %94, %96, %97, %79, %100, %101, %81, %102, %82, %103, %83, %67, %104 : i1, i1, i1, i1, i1, i4, i1, i1, i1, i1, i4, i1, i1, i1, i1, i4, i1, i1, i1, i1, i1, i1, i1, i1
    %106 = comb.xor bin %105, %c-1_i33 : i33
    %107 = comb.mux bin %49, %c0_i33, %106 : i33
    %108 = comb.xor bin %107, %c-1_i33 : i33
    %109 = comb.concat %108, %c-1_i20 : i33, i20
    %110 = comb.extract %52 from 15 : (i65) -> i1
    %111 = comb.extract %52 from 0 : (i65) -> i1
    %112 = comb.extract %52 from 2 : (i65) -> i1
    %113 = comb.extract %52 from 4 : (i65) -> i1
    %114 = comb.extract %52 from 6 : (i65) -> i1
    %115 = comb.extract %52 from 8 : (i65) -> i1
    %116 = comb.extract %52 from 10 : (i65) -> i1
    %117 = comb.extract %52 from 12 : (i65) -> i1
    %118 = comb.extract %52 from 14 : (i65) -> i1
    %119 = comb.extract %52 from 1 : (i65) -> i1
    %120 = comb.extract %52 from 3 : (i65) -> i1
    %121 = comb.extract %52 from 5 : (i65) -> i1
    %122 = comb.extract %52 from 7 : (i65) -> i1
    %123 = comb.extract %52 from 9 : (i65) -> i1
    %124 = comb.extract %52 from 11 : (i65) -> i1
    %125 = comb.extract %52 from 13 : (i65) -> i1
    %126 = comb.extract %52 from 16 : (i65) -> i1
    %127 = comb.extract %52 from 17 : (i65) -> i1
    %128 = comb.extract %52 from 18 : (i65) -> i1
    %129 = comb.extract %52 from 19 : (i65) -> i1
    %130 = comb.concat %111, %119, %112, %120, %113, %121, %114, %122, %115, %123, %116, %124, %117, %125, %118, %110, %126, %127, %128, %129 : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
    %131 = comb.mux bin %49, %130, %c0_i20 : i20
    %132 = comb.concat %c0_i33, %131 : i33, i20
    %133 = comb.mux bin %48, %109, %132 {sv.namehint = "CExtraMask"} : i53
    %134 = comb.replicate %31 : (i1) -> i53
    %135 = comb.xor %134, %20 {sv.namehint = "negSigC"} : i53
    %136 = comb.replicate %31 : (i1) -> i108
    %137 = comb.concat %31, %135, %136 : i1, i53, i108
    %138 = comb.concat %c0_i154, %45 : i154, i8
    %139 = comb.shrs bin %137, %138 : i162
    %140 = comb.and bin %20, %133 : i53
    %141 = comb.icmp bin ne %140, %c0_i53 : i53
    %142 = comb.xor bin %141, %31 {sv.namehint = "io_toPostMul_bit0AlignedNegSigC"} : i1
    %143 = comb.extract %139 from 0 {sv.namehint = "io_mulAddC"} : (i162) -> i106
    %144 = comb.extract %io_a from 51 {sv.namehint = "io_toPostMul_isNaN_isQuietNaNA"} : (i65) -> i1
    %145 = comb.extract %io_b from 51 {sv.namehint = "io_toPostMul_isNaN_isQuietNaNB"} : (i65) -> i1
    %146 = comb.extract %io_c from 51 {sv.namehint = "io_toPostMul_isNaN_isQuietNaNC"} : (i65) -> i1
    %147 = comb.extract %139 from 106 {sv.namehint = "io_toPostMul_highAlignedNegSigC"} : (i162) -> i55
    hw.output %6, %12, %143, %3, %144, %9, %145, %22, %23, %15, %18, %146, %41, %38, %45, %142, %147, %47, %io_roundingMode : i53, i53, i106, i3, i1, i3, i1, i1, i1, i1, i3, i1, i1, i1, i8, i1, i55, i14, i2
  }
  hw.module private @MulAddRecFN_postMul_1(in %io_fromPreMul_highExpA : i3, in %io_fromPreMul_isNaN_isQuietNaNA : i1, in %io_fromPreMul_highExpB : i3, in %io_fromPreMul_isNaN_isQuietNaNB : i1, in %io_fromPreMul_signProd : i1, in %io_fromPreMul_isZeroProd : i1, in %io_fromPreMul_opSignC : i1, in %io_fromPreMul_highExpC : i3, in %io_fromPreMul_isNaN_isQuietNaNC : i1, in %io_fromPreMul_isCDominant : i1, in %io_fromPreMul_CAlignDist_0 : i1, in %io_fromPreMul_CAlignDist : i8, in %io_fromPreMul_bit0AlignedNegSigC : i1, in %io_fromPreMul_highAlignedNegSigC : i55, in %io_fromPreMul_sExpSum : i14, in %io_fromPreMul_roundingMode : i2, in %io_mulAddResult : i107, out io_out : i65, out io_exceptionFlags : i5) {
    %c-174763_i19 = hw.constant -174763 : i19
    %c0_i51 = hw.constant 0 : i51
    %c-1_i9 = hw.constant -1 : i9
    %c-1_i10 = hw.constant -1 : i10
    %c0_i13 = hw.constant 0 : i13
    %c1_i13 = hw.constant 1 : i13
    %c-1_i54 = hw.constant -1 : i54
    %c5_i4 = hw.constant 5 : i4
    %c0_i6 = hw.constant 0 : i6
    %c-1_i6 = hw.constant -1 : i6
    %c-96_i8 = hw.constant -96 : i8
    %c1_i55 = hw.constant 1 : i55
    %c3_i3 = hw.constant 3 : i3
    %c-1_i12 = hw.constant -1 : i12
    %c0_i110 = hw.constant 0 : i110
    %c-1_i55 = hw.constant -1 : i55
    %c-1_i50 = hw.constant -1 : i50
    %c0_i59 = hw.constant 0 : i59
    %c-1_i13 = hw.constant -1 : i13
    %c-1_i32 = hw.constant -1 : i32
    %c0_i82 = hw.constant 0 : i82
    %c0_i28 = hw.constant 0 : i28
    %c-1_i5 = hw.constant -1 : i5
    %c0_i23 = hw.constant 0 : i23
    %c0_i11 = hw.constant 0 : i11
    %c0_i21 = hw.constant 0 : i21
    %c0_i76 = hw.constant 0 : i76
    %c-1_i162 = hw.constant -1 : i162
    %c0_i32 = hw.constant 0 : i32
    %c0_i8 = hw.constant 0 : i8
    %c0_i16 = hw.constant 0 : i16
    %c0_i44 = hw.constant 0 : i44
    %c0_i2 = hw.constant 0 : i2
    %c0_i3 = hw.constant 0 : i3
    %false = hw.constant false
    %c-1_i2 = hw.constant -1 : i2
    %c-2_i2 = hw.constant -2 : i2
    %true = hw.constant true
    %c0_i86 = hw.constant 0 : i86
    %c0_i22 = hw.constant 0 : i22
    %c0_i54 = hw.constant 0 : i54
    %c-1_i4 = hw.constant -1 : i4
    %c-1022_i11 = hw.constant -1022 : i11
    %c-1023_i11 = hw.constant -1023 : i11
    %c-512_i12 = hw.constant -512 : i12
    %c0_i12 = hw.constant 0 : i12
    %c-1025_i12 = hw.constant -1025 : i12
    %c-1024_i12 = hw.constant -1024 : i12
    %c0_i87 = hw.constant 0 : i87
    %c-4294967296_i33 = hw.constant -4294967296 : i33
    %c-18446744073709551616_i65 = hw.constant -18446744073709551616 : i65
    %c0_i50 = hw.constant 0 : i50
    %c0_i4 = hw.constant 0 : i4
    %c0_i55 = hw.constant 0 : i55
    %c-975_i12 = hw.constant -975 : i12
    %c974_i12 = hw.constant 974 : i12
    %0 = comb.icmp bin eq %io_fromPreMul_highExpA, %c0_i3 {sv.namehint = "isZeroA"} : i3
    %1 = comb.extract %io_fromPreMul_highExpA from 1 : (i3) -> i2
    %2 = comb.icmp bin eq %1, %c-1_i2 {sv.namehint = "isSpecialA"} : i2
    %3 = comb.extract %io_fromPreMul_highExpA from 0 : (i3) -> i1
    %4 = comb.xor bin %3, %true : i1
    %5 = comb.and bin %2, %4 {sv.namehint = "isInfA"} : i1
    %6 = comb.and bin %2, %3 {sv.namehint = "isNaNA"} : i1
    %7 = comb.xor bin %io_fromPreMul_isNaN_isQuietNaNA, %true : i1
    %8 = comb.and bin %6, %7 {sv.namehint = "isSigNaNA"} : i1
    %9 = comb.icmp bin eq %io_fromPreMul_highExpB, %c0_i3 {sv.namehint = "isZeroB"} : i3
    %10 = comb.extract %io_fromPreMul_highExpB from 1 : (i3) -> i2
    %11 = comb.icmp bin eq %10, %c-1_i2 {sv.namehint = "isSpecialB"} : i2
    %12 = comb.extract %io_fromPreMul_highExpB from 0 : (i3) -> i1
    %13 = comb.xor bin %12, %true : i1
    %14 = comb.and bin %11, %13 {sv.namehint = "isInfB"} : i1
    %15 = comb.and bin %11, %12 {sv.namehint = "isNaNB"} : i1
    %16 = comb.xor bin %io_fromPreMul_isNaN_isQuietNaNB, %true : i1
    %17 = comb.and bin %15, %16 {sv.namehint = "isSigNaNB"} : i1
    %18 = comb.icmp bin ne %io_fromPreMul_highExpC, %c0_i3 : i3
    %19 = comb.xor bin %18, %true {sv.namehint = "isZeroC"} : i1
    %20 = comb.extract %io_fromPreMul_highExpC from 1 : (i3) -> i2
    %21 = comb.icmp bin eq %20, %c-1_i2 {sv.namehint = "isSpecialC"} : i2
    %22 = comb.extract %io_fromPreMul_highExpC from 0 : (i3) -> i1
    %23 = comb.xor bin %22, %true : i1
    %24 = comb.and bin %21, %23 {sv.namehint = "isInfC"} : i1
    %25 = comb.and bin %21, %22 {sv.namehint = "isNaNC"} : i1
    %26 = comb.xor bin %io_fromPreMul_isNaN_isQuietNaNC, %true : i1
    %27 = comb.and bin %25, %26 {sv.namehint = "isSigNaNC"} : i1
    %28 = comb.icmp bin eq %io_fromPreMul_roundingMode, %c0_i2 {sv.namehint = "roundingMode_nearest_even"} : i2
    %29 = comb.icmp bin eq %io_fromPreMul_roundingMode, %c-2_i2 {sv.namehint = "signZeroNotEqOpSigns"} : i2
    %30 = comb.icmp bin eq %io_fromPreMul_roundingMode, %c-1_i2 {sv.namehint = "roundingMode_max"} : i2
    %31 = comb.xor bin %io_fromPreMul_signProd, %io_fromPreMul_opSignC {sv.namehint = "doSubMags"} : i1
    %32 = comb.extract %io_mulAddResult from 106 : (i107) -> i1
    %33 = comb.add %io_fromPreMul_highAlignedNegSigC, %c1_i55 : i55
    %34 = comb.mux bin %32, %33, %io_fromPreMul_highAlignedNegSigC : i55
    %35 = comb.extract %io_mulAddResult from 0 : (i107) -> i106
    %36 = comb.concat %34, %35, %io_fromPreMul_bit0AlignedNegSigC {sv.namehint = "sigSum"} : i55, i106, i1
    %37 = comb.extract %34 from 0 : (i55) -> i2
    %38 = comb.extract %io_mulAddResult from 1 : (i107) -> i105
    %39 = comb.concat %37, %38 : i2, i105
    %40 = comb.extract %34 from 0 : (i55) -> i1
    %41 = comb.concat %40, %35 : i1, i106
    %42 = comb.xor %39, %41 : i107
    %43 = comb.extract %42 from 63 : (i107) -> i44
    %44 = comb.icmp bin ne %43, %c0_i44 : i44
    %45 = comb.extract %42 from 95 : (i107) -> i12
    %46 = comb.icmp bin ne %45, %c0_i12 : i12
    %47 = comb.extract %42 from 103 : (i107) -> i4
    %48 = comb.icmp bin ne %47, %c0_i4 : i4
    %49 = comb.extract %42 from 106 : (i107) -> i1
    %50 = comb.extract %42 from 105 : (i107) -> i1
    %51 = comb.extract %42 from 104 : (i107) -> i1
    %52 = comb.concat %false, %51 : i1, i1
    %53 = comb.mux bin %50, %c-2_i2, %52 : i2
    %54 = comb.mux bin %49, %c-1_i2, %53 : i2
    %55 = comb.extract %42 from 99 : (i107) -> i4
    %56 = comb.icmp bin ne %55, %c0_i4 : i4
    %57 = comb.extract %42 from 102 : (i107) -> i1
    %58 = comb.extract %42 from 101 : (i107) -> i1
    %59 = comb.extract %42 from 100 : (i107) -> i1
    %60 = comb.concat %false, %59 : i1, i1
    %61 = comb.mux bin %58, %c-2_i2, %60 : i2
    %62 = comb.mux bin %57, %c-1_i2, %61 : i2
    %63 = comb.extract %42 from 98 : (i107) -> i1
    %64 = comb.extract %42 from 97 : (i107) -> i1
    %65 = comb.extract %42 from 96 : (i107) -> i1
    %66 = comb.concat %false, %65 : i1, i1
    %67 = comb.mux bin %64, %c-2_i2, %66 : i2
    %68 = comb.mux bin %63, %c-1_i2, %67 : i2
    %69 = comb.mux bin %56, %62, %68 : i2
    %70 = comb.concat %56, %69 : i1, i2
    %71 = comb.concat %false, %54 : i1, i2
    %72 = comb.mux bin %48, %71, %70 : i3
    %73 = comb.extract %42 from 79 : (i107) -> i16
    %74 = comb.icmp bin ne %73, %c0_i16 : i16
    %75 = comb.extract %42 from 87 : (i107) -> i8
    %76 = comb.icmp bin ne %75, %c0_i8 : i8
    %77 = comb.extract %42 from 91 : (i107) -> i4
    %78 = comb.icmp bin ne %77, %c0_i4 : i4
    %79 = comb.extract %42 from 94 : (i107) -> i1
    %80 = comb.extract %42 from 93 : (i107) -> i1
    %81 = comb.extract %42 from 92 : (i107) -> i1
    %82 = comb.concat %false, %81 : i1, i1
    %83 = comb.mux bin %80, %c-2_i2, %82 : i2
    %84 = comb.mux bin %79, %c-1_i2, %83 : i2
    %85 = comb.extract %42 from 90 : (i107) -> i1
    %86 = comb.extract %42 from 89 : (i107) -> i1
    %87 = comb.extract %42 from 88 : (i107) -> i1
    %88 = comb.concat %false, %87 : i1, i1
    %89 = comb.mux bin %86, %c-2_i2, %88 : i2
    %90 = comb.mux bin %85, %c-1_i2, %89 : i2
    %91 = comb.mux bin %78, %84, %90 : i2
    %92 = comb.concat %78, %91 : i1, i2
    %93 = comb.extract %42 from 83 : (i107) -> i4
    %94 = comb.icmp bin ne %93, %c0_i4 : i4
    %95 = comb.extract %42 from 86 : (i107) -> i1
    %96 = comb.extract %42 from 85 : (i107) -> i1
    %97 = comb.extract %42 from 84 : (i107) -> i1
    %98 = comb.concat %false, %97 : i1, i1
    %99 = comb.mux bin %96, %c-2_i2, %98 : i2
    %100 = comb.mux bin %95, %c-1_i2, %99 : i2
    %101 = comb.extract %42 from 82 : (i107) -> i1
    %102 = comb.extract %42 from 81 : (i107) -> i1
    %103 = comb.extract %42 from 80 : (i107) -> i1
    %104 = comb.concat %false, %103 : i1, i1
    %105 = comb.mux bin %102, %c-2_i2, %104 : i2
    %106 = comb.mux bin %101, %c-1_i2, %105 : i2
    %107 = comb.mux bin %94, %100, %106 : i2
    %108 = comb.concat %94, %107 : i1, i2
    %109 = comb.mux bin %76, %92, %108 : i3
    %110 = comb.concat %76, %109 : i1, i3
    %111 = comb.extract %42 from 71 : (i107) -> i8
    %112 = comb.icmp bin ne %111, %c0_i8 : i8
    %113 = comb.extract %42 from 75 : (i107) -> i4
    %114 = comb.icmp bin ne %113, %c0_i4 : i4
    %115 = comb.extract %42 from 78 : (i107) -> i1
    %116 = comb.extract %42 from 77 : (i107) -> i1
    %117 = comb.extract %42 from 76 : (i107) -> i1
    %118 = comb.concat %false, %117 : i1, i1
    %119 = comb.mux bin %116, %c-2_i2, %118 : i2
    %120 = comb.mux bin %115, %c-1_i2, %119 : i2
    %121 = comb.extract %42 from 74 : (i107) -> i1
    %122 = comb.extract %42 from 73 : (i107) -> i1
    %123 = comb.extract %42 from 72 : (i107) -> i1
    %124 = comb.concat %false, %123 : i1, i1
    %125 = comb.mux bin %122, %c-2_i2, %124 : i2
    %126 = comb.mux bin %121, %c-1_i2, %125 : i2
    %127 = comb.mux bin %114, %120, %126 : i2
    %128 = comb.concat %114, %127 : i1, i2
    %129 = comb.extract %42 from 67 : (i107) -> i4
    %130 = comb.icmp bin ne %129, %c0_i4 : i4
    %131 = comb.extract %42 from 70 : (i107) -> i1
    %132 = comb.extract %42 from 69 : (i107) -> i1
    %133 = comb.extract %42 from 68 : (i107) -> i1
    %134 = comb.concat %false, %133 : i1, i1
    %135 = comb.mux bin %132, %c-2_i2, %134 : i2
    %136 = comb.mux bin %131, %c-1_i2, %135 : i2
    %137 = comb.extract %42 from 66 : (i107) -> i1
    %138 = comb.extract %42 from 65 : (i107) -> i1
    %139 = comb.extract %42 from 64 : (i107) -> i1
    %140 = comb.concat %false, %139 : i1, i1
    %141 = comb.mux bin %138, %c-2_i2, %140 : i2
    %142 = comb.mux bin %137, %c-1_i2, %141 : i2
    %143 = comb.mux bin %130, %136, %142 : i2
    %144 = comb.concat %130, %143 : i1, i2
    %145 = comb.mux bin %112, %128, %144 : i3
    %146 = comb.concat %112, %145 : i1, i3
    %147 = comb.mux bin %74, %110, %146 : i4
    %148 = comb.concat %74, %147 : i1, i4
    %149 = comb.concat %false, %48, %72 : i1, i1, i3
    %150 = comb.mux bin %46, %149, %148 : i5
    %151 = comb.concat %46, %150 : i1, i5
    %152 = comb.extract %42 from 31 : (i107) -> i32
    %153 = comb.icmp bin ne %152, %c0_i32 : i32
    %154 = comb.extract %42 from 47 : (i107) -> i16
    %155 = comb.icmp bin ne %154, %c0_i16 : i16
    %156 = comb.extract %42 from 55 : (i107) -> i8
    %157 = comb.icmp bin ne %156, %c0_i8 : i8
    %158 = comb.extract %42 from 59 : (i107) -> i4
    %159 = comb.icmp bin ne %158, %c0_i4 : i4
    %160 = comb.extract %42 from 62 : (i107) -> i1
    %161 = comb.extract %42 from 61 : (i107) -> i1
    %162 = comb.extract %42 from 60 : (i107) -> i1
    %163 = comb.concat %false, %162 : i1, i1
    %164 = comb.mux bin %161, %c-2_i2, %163 : i2
    %165 = comb.mux bin %160, %c-1_i2, %164 : i2
    %166 = comb.extract %42 from 58 : (i107) -> i1
    %167 = comb.extract %42 from 57 : (i107) -> i1
    %168 = comb.extract %42 from 56 : (i107) -> i1
    %169 = comb.concat %false, %168 : i1, i1
    %170 = comb.mux bin %167, %c-2_i2, %169 : i2
    %171 = comb.mux bin %166, %c-1_i2, %170 : i2
    %172 = comb.mux bin %159, %165, %171 : i2
    %173 = comb.concat %159, %172 : i1, i2
    %174 = comb.extract %42 from 51 : (i107) -> i4
    %175 = comb.icmp bin ne %174, %c0_i4 : i4
    %176 = comb.extract %42 from 54 : (i107) -> i1
    %177 = comb.extract %42 from 53 : (i107) -> i1
    %178 = comb.extract %42 from 52 : (i107) -> i1
    %179 = comb.concat %false, %178 : i1, i1
    %180 = comb.mux bin %177, %c-2_i2, %179 : i2
    %181 = comb.mux bin %176, %c-1_i2, %180 : i2
    %182 = comb.extract %42 from 50 : (i107) -> i1
    %183 = comb.extract %42 from 49 : (i107) -> i1
    %184 = comb.extract %42 from 48 : (i107) -> i1
    %185 = comb.concat %false, %184 : i1, i1
    %186 = comb.mux bin %183, %c-2_i2, %185 : i2
    %187 = comb.mux bin %182, %c-1_i2, %186 : i2
    %188 = comb.mux bin %175, %181, %187 : i2
    %189 = comb.concat %175, %188 : i1, i2
    %190 = comb.mux bin %157, %173, %189 : i3
    %191 = comb.concat %157, %190 : i1, i3
    %192 = comb.extract %42 from 39 : (i107) -> i8
    %193 = comb.icmp bin ne %192, %c0_i8 : i8
    %194 = comb.extract %42 from 43 : (i107) -> i4
    %195 = comb.icmp bin ne %194, %c0_i4 : i4
    %196 = comb.extract %42 from 46 : (i107) -> i1
    %197 = comb.extract %42 from 45 : (i107) -> i1
    %198 = comb.extract %42 from 44 : (i107) -> i1
    %199 = comb.concat %false, %198 : i1, i1
    %200 = comb.mux bin %197, %c-2_i2, %199 : i2
    %201 = comb.mux bin %196, %c-1_i2, %200 : i2
    %202 = comb.extract %42 from 42 : (i107) -> i1
    %203 = comb.extract %42 from 41 : (i107) -> i1
    %204 = comb.extract %42 from 40 : (i107) -> i1
    %205 = comb.concat %false, %204 : i1, i1
    %206 = comb.mux bin %203, %c-2_i2, %205 : i2
    %207 = comb.mux bin %202, %c-1_i2, %206 : i2
    %208 = comb.mux bin %195, %201, %207 : i2
    %209 = comb.concat %195, %208 : i1, i2
    %210 = comb.extract %42 from 35 : (i107) -> i4
    %211 = comb.icmp bin ne %210, %c0_i4 : i4
    %212 = comb.extract %42 from 38 : (i107) -> i1
    %213 = comb.extract %42 from 37 : (i107) -> i1
    %214 = comb.extract %42 from 36 : (i107) -> i1
    %215 = comb.concat %false, %214 : i1, i1
    %216 = comb.mux bin %213, %c-2_i2, %215 : i2
    %217 = comb.mux bin %212, %c-1_i2, %216 : i2
    %218 = comb.extract %42 from 34 : (i107) -> i1
    %219 = comb.extract %42 from 33 : (i107) -> i1
    %220 = comb.extract %42 from 32 : (i107) -> i1
    %221 = comb.concat %false, %220 : i1, i1
    %222 = comb.mux bin %219, %c-2_i2, %221 : i2
    %223 = comb.mux bin %218, %c-1_i2, %222 : i2
    %224 = comb.mux bin %211, %217, %223 : i2
    %225 = comb.concat %211, %224 : i1, i2
    %226 = comb.mux bin %193, %209, %225 : i3
    %227 = comb.concat %193, %226 : i1, i3
    %228 = comb.mux bin %155, %191, %227 : i4
    %229 = comb.concat %155, %228 : i1, i4
    %230 = comb.extract %42 from 15 : (i107) -> i16
    %231 = comb.icmp bin ne %230, %c0_i16 : i16
    %232 = comb.extract %42 from 23 : (i107) -> i8
    %233 = comb.icmp bin ne %232, %c0_i8 : i8
    %234 = comb.extract %42 from 27 : (i107) -> i4
    %235 = comb.icmp bin ne %234, %c0_i4 : i4
    %236 = comb.extract %42 from 30 : (i107) -> i1
    %237 = comb.extract %42 from 29 : (i107) -> i1
    %238 = comb.extract %42 from 28 : (i107) -> i1
    %239 = comb.concat %false, %238 : i1, i1
    %240 = comb.mux bin %237, %c-2_i2, %239 : i2
    %241 = comb.mux bin %236, %c-1_i2, %240 : i2
    %242 = comb.extract %42 from 26 : (i107) -> i1
    %243 = comb.extract %42 from 25 : (i107) -> i1
    %244 = comb.extract %42 from 24 : (i107) -> i1
    %245 = comb.concat %false, %244 : i1, i1
    %246 = comb.mux bin %243, %c-2_i2, %245 : i2
    %247 = comb.mux bin %242, %c-1_i2, %246 : i2
    %248 = comb.mux bin %235, %241, %247 : i2
    %249 = comb.concat %235, %248 : i1, i2
    %250 = comb.extract %42 from 19 : (i107) -> i4
    %251 = comb.icmp bin ne %250, %c0_i4 : i4
    %252 = comb.extract %42 from 22 : (i107) -> i1
    %253 = comb.extract %42 from 21 : (i107) -> i1
    %254 = comb.extract %42 from 20 : (i107) -> i1
    %255 = comb.concat %false, %254 : i1, i1
    %256 = comb.mux bin %253, %c-2_i2, %255 : i2
    %257 = comb.mux bin %252, %c-1_i2, %256 : i2
    %258 = comb.extract %42 from 18 : (i107) -> i1
    %259 = comb.extract %42 from 17 : (i107) -> i1
    %260 = comb.extract %42 from 16 : (i107) -> i1
    %261 = comb.concat %false, %260 : i1, i1
    %262 = comb.mux bin %259, %c-2_i2, %261 : i2
    %263 = comb.mux bin %258, %c-1_i2, %262 : i2
    %264 = comb.mux bin %251, %257, %263 : i2
    %265 = comb.concat %251, %264 : i1, i2
    %266 = comb.mux bin %233, %249, %265 : i3
    %267 = comb.concat %233, %266 : i1, i3
    %268 = comb.extract %42 from 7 : (i107) -> i8
    %269 = comb.icmp bin ne %268, %c0_i8 : i8
    %270 = comb.extract %42 from 11 : (i107) -> i4
    %271 = comb.icmp bin ne %270, %c0_i4 : i4
    %272 = comb.extract %42 from 14 : (i107) -> i1
    %273 = comb.extract %42 from 13 : (i107) -> i1
    %274 = comb.extract %42 from 12 : (i107) -> i1
    %275 = comb.concat %false, %274 : i1, i1
    %276 = comb.mux bin %273, %c-2_i2, %275 : i2
    %277 = comb.mux bin %272, %c-1_i2, %276 : i2
    %278 = comb.extract %42 from 10 : (i107) -> i1
    %279 = comb.extract %42 from 9 : (i107) -> i1
    %280 = comb.extract %42 from 8 : (i107) -> i1
    %281 = comb.concat %false, %280 : i1, i1
    %282 = comb.mux bin %279, %c-2_i2, %281 : i2
    %283 = comb.mux bin %278, %c-1_i2, %282 : i2
    %284 = comb.mux bin %271, %277, %283 : i2
    %285 = comb.concat %271, %284 : i1, i2
    %286 = comb.extract %42 from 3 : (i107) -> i4
    %287 = comb.icmp bin ne %286, %c0_i4 : i4
    %288 = comb.extract %42 from 6 : (i107) -> i1
    %289 = comb.extract %42 from 5 : (i107) -> i1
    %290 = comb.extract %42 from 4 : (i107) -> i1
    %291 = comb.concat %false, %290 : i1, i1
    %292 = comb.mux bin %289, %c-2_i2, %291 : i2
    %293 = comb.mux bin %288, %c-1_i2, %292 : i2
    %294 = comb.extract %42 from 2 : (i107) -> i1
    %295 = comb.extract %42 from 1 : (i107) -> i1
    %296 = comb.extract %42 from 0 : (i107) -> i1
    %297 = comb.concat %false, %296 : i1, i1
    %298 = comb.mux bin %295, %c-2_i2, %297 : i2
    %299 = comb.mux bin %294, %c-1_i2, %298 : i2
    %300 = comb.mux bin %287, %293, %299 : i2
    %301 = comb.concat %287, %300 : i1, i2
    %302 = comb.mux bin %269, %285, %301 : i3
    %303 = comb.concat %269, %302 : i1, i3
    %304 = comb.mux bin %231, %267, %303 : i4
    %305 = comb.concat %231, %304 : i1, i4
    %306 = comb.mux bin %153, %229, %305 : i5
    %307 = comb.concat %153, %306 : i1, i5
    %308 = comb.mux bin %44, %151, %307 : i6
    %309 = comb.concat %false, %44, %308 : i1, i1, i6
    %310 = comb.sub %c-96_i8, %309 {sv.namehint = "estNormNeg_dist"} : i8
    %311 = comb.extract %io_mulAddResult from 0 : (i107) -> i43
    %312 = comb.concat %311, %io_fromPreMul_bit0AlignedNegSigC : i43, i1
    %313 = comb.icmp bin ne %312, %c0_i44 : i44
    %314 = comb.xor bin %36, %c-1_i162 {sv.namehint = "complSigSum"} : i162
    %315 = comb.extract %314 from 0 : (i162) -> i44
    %316 = comb.icmp bin ne %315, %c0_i44 : i44
    %317 = comb.or bin %io_fromPreMul_CAlignDist_0, %31 : i1
    %318 = comb.extract %io_fromPreMul_CAlignDist from 0 : (i8) -> i6
    %319 = comb.add %318, %c-1_i6 : i6
    %320 = comb.concat %c0_i2, %319 : i2, i6
    %321 = comb.mux bin %317, %io_fromPreMul_CAlignDist, %320 {sv.namehint = "CDom_estNormDist"} : i8
    %322 = comb.xor bin %31, %true : i1
    %323 = comb.extract %321 from 5 : (i8) -> i1
    %324 = comb.xor bin %323, %true : i1
    %325 = comb.extract %io_mulAddResult from 75 : (i107) -> i31
    %326 = comb.extract %io_mulAddResult from 0 : (i107) -> i75
    %327 = comb.concat %326, %io_fromPreMul_bit0AlignedNegSigC : i75, i1
    %328 = comb.icmp bin ne %327, %c0_i76 : i76
    %329 = comb.concat %34, %325, %328 : i55, i31, i1
    %330 = comb.or %31, %323 : i1
    %331 = comb.mux bin %330, %c0_i87, %329 : i87
    %332 = comb.and bin %322, %323 : i1
    %333 = comb.extract %io_mulAddResult from 43 : (i107) -> i63
    %334 = comb.extract %34 from 0 : (i55) -> i23
    %335 = comb.concat %334, %333, %313 : i23, i63, i1
    %336 = comb.mux bin %332, %335, %c0_i87 : i87
    %337 = comb.and bin %31, %324 : i1
    %338 = comb.extract %314 from 76 : (i162) -> i86
    %339 = comb.extract %314 from 0 : (i162) -> i76
    %340 = comb.icmp bin ne %339, %c0_i76 : i76
    %341 = comb.concat %338, %340 : i86, i1
    %342 = comb.mux bin %337, %341, %c0_i87 : i87
    %343 = comb.and bin %31, %323 : i1
    %344 = comb.extract %314 from 44 : (i162) -> i86
    %345 = comb.concat %344, %316 : i86, i1
    %346 = comb.mux bin %343, %345, %c0_i87 : i87
    %347 = comb.or bin %331, %336, %342, %346 {sv.namehint = "CDom_firstNormAbsSigSum"} : i87
    %348 = comb.extract %io_mulAddResult from 43 : (i107) -> i63
    %349 = comb.extract %34 from 0 : (i55) -> i2
    %350 = comb.xor bin %316, %true : i1
    %351 = comb.mux bin %31, %350, %313 : i1
    %352 = comb.extract %310 from 4 : (i8) -> i1
    %353 = comb.extract %io_mulAddResult from 0 : (i107) -> i1
    %354 = comb.replicate %31 : (i1) -> i86
    %355 = comb.concat %353, %354 : i1, i86
    %356 = comb.concat %c0_i21, %349, %348, %351 : i21, i2, i63, i1
    %357 = comb.mux bin %352, %356, %355 : i87
    %358 = comb.extract %io_mulAddResult from 11 : (i107) -> i86
    %359 = comb.extract %314 from 1 : (i162) -> i11
    %360 = comb.icmp bin ne %359, %c0_i11 : i11
    %361 = comb.xor bin %360, %true : i1
    %362 = comb.extract %io_mulAddResult from 0 : (i107) -> i11
    %363 = comb.icmp bin ne %362, %c0_i11 : i11
    %364 = comb.mux bin %31, %361, %363 : i1
    %365 = comb.concat %358, %364 : i86, i1
    %366 = comb.extract %310 from 6 : (i8) -> i1
    %367 = comb.extract %310 from 5 : (i8) -> i1
    %368 = comb.extract %io_mulAddResult from 0 : (i107) -> i65
    %369 = comb.replicate %31 : (i1) -> i22
    %370 = comb.concat %368, %369 : i65, i22
    %371 = comb.mux bin %367, %370, %365 : i87
    %372 = comb.extract %io_mulAddResult from 0 : (i107) -> i33
    %373 = comb.replicate %31 : (i1) -> i54
    %374 = comb.concat %372, %373 : i33, i54
    %375 = comb.mux bin %367, %357, %374 : i87
    %376 = comb.mux bin %366, %371, %375 {sv.namehint = "notCDom_pos_firstNormAbsSigSum"} : i87
    %377 = comb.extract %314 from 44 : (i162) -> i64
    %378 = comb.extract %314 from 1 : (i162) -> i2
    %379 = comb.concat %378, %c0_i86 : i2, i86
    %380 = comb.concat %c0_i23, %377, %316 : i23, i64, i1
    %381 = comb.mux bin %352, %380, %379 : i88
    %382 = comb.extract %314 from 12 : (i162) -> i87
    %383 = comb.concat %382, %360 : i87, i1
    %384 = comb.extract %314 from 1 : (i162) -> i66
    %385 = comb.concat %384, %c0_i22 : i66, i22
    %386 = comb.mux bin %367, %385, %383 : i88
    %387 = comb.extract %314 from 1 : (i162) -> i34
    %388 = comb.concat %387, %c0_i54 : i34, i54
    %389 = comb.mux bin %367, %381, %388 : i88
    %390 = comb.mux bin %366, %386, %389 {sv.namehint = "notCDom_neg_cFirstNormAbsSigSum"} : i88
    %391 = comb.extract %34 from 2 {sv.namehint = "notCDom_signSigSum"} : (i55) -> i1
    %392 = comb.and bin %31, %18 : i1
    %393 = comb.mux bin %io_fromPreMul_isCDominant, %392, %391 {sv.namehint = "doNegSignSum"} : i1
    %394 = comb.mux bin %io_fromPreMul_isCDominant, %321, %310 {sv.namehint = "estNormDist"} : i8
    %395 = comb.concat %false, %347 : i1, i87
    %396 = comb.mux bin %io_fromPreMul_isCDominant, %395, %390 : i88
    %397 = comb.mux bin %io_fromPreMul_isCDominant, %347, %376 : i87
    %398 = comb.concat %false, %397 : i1, i87
    %399 = comb.mux bin %391, %396, %398 {sv.namehint = "cFirstNormAbsSigSum"} : i88
    %400 = comb.xor bin %io_fromPreMul_isCDominant, %true : i1
    %401 = comb.xor bin %391, %true : i1
    %402 = comb.and bin %400, %401, %31 {sv.namehint = "doIncrSig"} : i1
    %403 = comb.extract %394 from 0 {sv.namehint = "estNormDist_5"} : (i8) -> i5
    %404 = comb.xor bin %403, %c-1_i5 {sv.namehint = "normTo2ShiftDist"} : i5
    %405 = comb.concat %c0_i28, %404 : i28, i5
    %406 = comb.shrs bin %c-4294967296_i33, %405 : i33
    %407 = comb.extract %406 from 16 : (i33) -> i1
    %408 = comb.extract %406 from 1 : (i33) -> i1
    %409 = comb.extract %406 from 3 : (i33) -> i1
    %410 = comb.extract %406 from 5 : (i33) -> i1
    %411 = comb.extract %406 from 7 : (i33) -> i1
    %412 = comb.extract %406 from 9 : (i33) -> i1
    %413 = comb.extract %406 from 11 : (i33) -> i1
    %414 = comb.extract %406 from 13 : (i33) -> i1
    %415 = comb.extract %406 from 15 : (i33) -> i1
    %416 = comb.extract %406 from 2 : (i33) -> i1
    %417 = comb.extract %406 from 4 : (i33) -> i1
    %418 = comb.extract %406 from 6 : (i33) -> i1
    %419 = comb.extract %406 from 8 : (i33) -> i1
    %420 = comb.extract %406 from 10 : (i33) -> i1
    %421 = comb.extract %406 from 12 : (i33) -> i1
    %422 = comb.extract %406 from 14 : (i33) -> i1
    %423 = comb.extract %406 from 24 : (i33) -> i1
    %424 = comb.extract %406 from 17 : (i33) -> i1
    %425 = comb.extract %406 from 19 : (i33) -> i1
    %426 = comb.extract %406 from 21 : (i33) -> i1
    %427 = comb.extract %406 from 23 : (i33) -> i1
    %428 = comb.extract %406 from 18 : (i33) -> i1
    %429 = comb.extract %406 from 20 : (i33) -> i1
    %430 = comb.extract %406 from 22 : (i33) -> i1
    %431 = comb.extract %406 from 25 : (i33) -> i1
    %432 = comb.extract %406 from 26 : (i33) -> i1
    %433 = comb.extract %406 from 27 : (i33) -> i1
    %434 = comb.extract %406 from 28 : (i33) -> i1
    %435 = comb.extract %406 from 29 : (i33) -> i1
    %436 = comb.extract %406 from 30 : (i33) -> i1
    %437 = comb.extract %406 from 31 : (i33) -> i1
    %438 = comb.concat %408, %416, %409, %417, %410, %418, %411, %419, %412, %420, %413, %421, %414, %422, %415, %407, %424, %428, %425, %429, %426, %430, %427, %423, %431, %432, %433, %434, %435, %436, %437, %true {sv.namehint = "absSigSumExtraMask"} : i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
    %439 = comb.extract %399 from 1 : (i88) -> i87
    %440 = comb.concat %c0_i82, %404 : i82, i5
    %441 = comb.shru bin %439, %440 : i87
    %442 = comb.extract %399 from 0 : (i88) -> i32
    %443 = comb.xor bin %442, %c-1_i32 : i32
    %444 = comb.and bin %443, %438 : i32
    %445 = comb.icmp bin eq %444, %c0_i32 : i32
    %446 = comb.and bin %442, %438 : i32
    %447 = comb.icmp bin ne %446, %c0_i32 : i32
    %448 = comb.mux bin %402, %445, %447 : i1
    %449 = comb.extract %441 from 54 : (i87) -> i2
    %450 = comb.icmp bin eq %449, %c0_i2 {sv.namehint = "sigX3Shift1"} : i2
    %451 = comb.concat %c0_i6, %394 : i6, i8
    %452 = comb.sub %io_fromPreMul_sExpSum, %451 {sv.namehint = "sExpX3"} : i14
    %453 = comb.extract %441 from 53 : (i87) -> i3
    %454 = comb.icmp bin ne %453, %c0_i3 : i3
    %455 = comb.xor bin %454, %true {sv.namehint = "isZeroY"} : i1
    %456 = comb.xor bin %io_fromPreMul_signProd, %393 : i1
    %457 = comb.mux bin %454, %456, %29 {sv.namehint = "signY"} : i1
    %458 = comb.extract %452 from 0 {sv.namehint = "sExpX3_13"} : (i14) -> i13
    %459 = comb.extract %452 from 13 : (i14) -> i1
    %460 = comb.replicate %459 : (i1) -> i56
    %461 = comb.xor bin %458, %c-1_i13 : i13
    %462 = comb.extract %461 from 12 : (i13) -> i1
    %463 = comb.extract %461 from 11 : (i13) -> i1
    %464 = comb.extract %461 from 10 : (i13) -> i1
    %465 = comb.extract %461 from 9 : (i13) -> i1
    %466 = comb.extract %461 from 8 : (i13) -> i1
    %467 = comb.extract %461 from 7 : (i13) -> i1
    %468 = comb.extract %461 from 6 : (i13) -> i1
    %469 = comb.extract %461 from 0 : (i13) -> i6
    %470 = comb.concat %c0_i59, %469 : i59, i6
    %471 = comb.shrs bin %c-18446744073709551616_i65, %470 : i65
    %472 = comb.extract %471 from 18 : (i65) -> i2
    %473 = comb.extract %471 from 22 : (i65) -> i2
    %474 = comb.extract %471 from 26 : (i65) -> i2
    %475 = comb.extract %471 from 30 : (i65) -> i2
    %476 = comb.extract %471 from 34 : (i65) -> i2
    %477 = comb.extract %471 from 38 : (i65) -> i2
    %478 = comb.extract %471 from 20 : (i65) -> i2
    %479 = comb.concat %478, %473 : i2, i2
    %480 = comb.extract %471 from 24 : (i65) -> i2
    %481 = comb.extract %471 from 28 : (i65) -> i2
    %482 = comb.concat %481, %475 : i2, i2
    %483 = comb.extract %471 from 32 : (i65) -> i2
    %484 = comb.extract %471 from 36 : (i65) -> i2
    %485 = comb.concat %484, %477 : i2, i2
    %486 = comb.extract %471 from 45 : (i65) -> i1
    %487 = comb.extract %471 from 37 : (i65) -> i1
    %488 = comb.concat %472, %478, %473, %480, %474, %481, %475, %483, %476, %487 : i2, i2, i2, i2, i2, i2, i2, i2, i2, i1
    %489 = comb.and %488, %c-174763_i19 : i19
    %490 = comb.extract %471 from 14 : (i65) -> i1
    %491 = comb.extract %471 from 16 : (i65) -> i1
    %492 = comb.extract %471 from 18 : (i65) -> i1
    %493 = comb.and %479, %c5_i4 : i4
    %494 = comb.extract %471 from 24 : (i65) -> i1
    %495 = comb.extract %471 from 26 : (i65) -> i1
    %496 = comb.and %482, %c5_i4 : i4
    %497 = comb.extract %471 from 32 : (i65) -> i1
    %498 = comb.extract %471 from 34 : (i65) -> i1
    %499 = comb.and %485, %c5_i4 : i4
    %500 = comb.extract %471 from 40 : (i65) -> i1
    %501 = comb.extract %471 from 42 : (i65) -> i1
    %502 = comb.extract %471 from 44 : (i65) -> i1
    %503 = comb.extract %471 from 15 : (i65) -> i1
    %504 = comb.extract %471 from 17 : (i65) -> i1
    %505 = comb.extract %489 from 15 : (i19) -> i4
    %506 = comb.or %505, %493 : i4
    %507 = comb.extract %471 from 23 : (i65) -> i1
    %508 = comb.extract %489 from 13 : (i19) -> i1
    %509 = comb.or %508, %494 : i1
    %510 = comb.extract %471 from 25 : (i65) -> i1
    %511 = comb.extract %489 from 7 : (i19) -> i4
    %512 = comb.or %511, %496 : i4
    %513 = comb.extract %471 from 31 : (i65) -> i1
    %514 = comb.extract %489 from 5 : (i19) -> i1
    %515 = comb.or %514, %497 : i1
    %516 = comb.extract %471 from 33 : (i65) -> i1
    %517 = comb.extract %489 from 0 : (i19) -> i3
    %518 = comb.concat %517, %false : i3, i1
    %519 = comb.or %518, %499 : i4
    %520 = comb.extract %471 from 39 : (i65) -> i1
    %521 = comb.extract %471 from 41 : (i65) -> i1
    %522 = comb.extract %471 from 43 : (i65) -> i1
    %523 = comb.extract %471 from 61 : (i65) -> i1
    %524 = comb.extract %471 from 46 : (i65) -> i1
    %525 = comb.extract %471 from 48 : (i65) -> i1
    %526 = comb.extract %471 from 50 : (i65) -> i1
    %527 = comb.extract %471 from 52 : (i65) -> i1
    %528 = comb.extract %471 from 54 : (i65) -> i1
    %529 = comb.extract %471 from 56 : (i65) -> i1
    %530 = comb.extract %471 from 58 : (i65) -> i1
    %531 = comb.extract %471 from 60 : (i65) -> i1
    %532 = comb.extract %471 from 47 : (i65) -> i1
    %533 = comb.extract %471 from 49 : (i65) -> i1
    %534 = comb.extract %471 from 51 : (i65) -> i1
    %535 = comb.extract %471 from 53 : (i65) -> i1
    %536 = comb.extract %471 from 55 : (i65) -> i1
    %537 = comb.extract %471 from 57 : (i65) -> i1
    %538 = comb.extract %471 from 59 : (i65) -> i1
    %539 = comb.extract %471 from 62 : (i65) -> i1
    %540 = comb.extract %471 from 63 : (i65) -> i1
    %541 = comb.concat %490, %503, %491, %504, %492, %506, %507, %509, %510, %495, %512, %513, %515, %516, %498, %519, %520, %500, %521, %501, %522, %502, %486, %524, %532, %525, %533, %526, %534, %527, %535, %528, %536, %529, %537, %530, %538, %531, %523, %539, %540 : i1, i1, i1, i1, i1, i4, i1, i1, i1, i1, i4, i1, i1, i1, i1, i4, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1, i1
    %542 = comb.xor bin %541, %c-1_i50 : i50
    %543 = comb.or bin %465, %466, %467, %468 : i1
    %544 = comb.mux bin %543, %c0_i50, %542 : i50
    %545 = comb.xor bin %544, %c-1_i50 : i50
    %546 = comb.concat %545, %c-1_i4 : i50, i4
    %547 = comb.extract %471 from 0 : (i65) -> i1
    %548 = comb.extract %471 from 1 : (i65) -> i1
    %549 = comb.extract %471 from 2 : (i65) -> i1
    %550 = comb.extract %471 from 3 : (i65) -> i1
    %551 = comb.concat %547, %548, %549, %550 : i1, i1, i1, i1
    %552 = comb.and bin %465, %466, %467, %468 : i1
    %553 = comb.mux bin %552, %551, %c0_i4 : i4
    %554 = comb.concat %c0_i50, %553 : i50, i4
    %555 = comb.mux bin %464, %546, %554 : i54
    %556 = comb.and bin %462, %463 : i1
    %557 = comb.mux bin %556, %555, %c0_i54 : i54
    %558 = comb.extract %441 from 54 : (i87) -> i1
    %559 = comb.extract %557 from 1 : (i54) -> i53
    %560 = comb.extract %557 from 0 : (i54) -> i1
    %561 = comb.or bin %560, %558 : i1
    %562 = comb.concat %559, %561, %c-1_i2 : i53, i1, i2
    %563 = comb.or bin %460, %562 {sv.namehint = "roundMask"} : i56
    %564 = comb.extract %563 from 1 : (i56) -> i55
    %565 = comb.xor bin %564, %c-1_i55 : i55
    %566 = comb.extract %563 from 0 : (i56) -> i55
    %567 = comb.extract %441 from 0 : (i87) -> i54
    %568 = comb.concat %567, %448 : i54, i1
    %569 = comb.and bin %568, %566, %565 : i55
    %570 = comb.icmp bin ne %569, %c0_i55 {sv.namehint = "roundPosBit"} : i55
    %571 = comb.and bin %568, %564 : i55
    %572 = comb.icmp bin ne %571, %c0_i55 {sv.namehint = "anyRoundExtra"} : i55
    %573 = comb.extract %441 from 0 : (i87) -> i54
    %574 = comb.concat %573, %448 : i54, i1
    %575 = comb.xor %574, %c-1_i55 : i55
    %576 = comb.and bin %575, %564 : i55
    %577 = comb.icmp bin eq %576, %c0_i55 {sv.namehint = "allRoundExtra"} : i55
    %578 = comb.concat %569, %571 : i55, i55
    %579 = comb.icmp bin ne %578, %c0_i110 {sv.namehint = "anyRound"} : i110
    %580 = comb.and bin %570, %577 {sv.namehint = "allRound"} : i1
    %581 = comb.mux bin %457, %29, %30 {sv.namehint = "roundDirectUp"} : i1
    %582 = comb.xor bin %402, %true : i1
    %583 = comb.and bin %582, %28, %570, %572 : i1
    %584 = comb.and bin %582, %581, %579 : i1
    %585 = comb.and bin %402, %580 : i1
    %586 = comb.and bin %402, %28, %570 : i1
    %587 = comb.and bin %402, %581 : i1
    %588 = comb.or bin %583, %584, %585, %586, %587 {sv.namehint = "roundUp"} : i1
    %589 = comb.xor bin %570, %true : i1
    %590 = comb.and bin %28, %589, %577 : i1
    %591 = comb.xor bin %572, %true : i1
    %592 = comb.and bin %28, %570, %591 : i1
    %593 = comb.mux bin %402, %590, %592 {sv.namehint = "roundEven"} : i1
    %594 = comb.xor bin %580, %true : i1
    %595 = comb.mux bin %402, %594, %579 {sv.namehint = "inexactY"} : i1
    %596 = comb.extract %441 from 55 : (i87) -> i1
    %597 = comb.extract %441 from 1 : (i87) -> i54
    %598 = comb.extract %563 from 2 : (i56) -> i54
    %599 = comb.or %597, %598 : i54
    %600 = comb.concat %596, %599 : i1, i54
    %601 = comb.add %600, %c1_i55 {sv.namehint = "roundUp_sigY3"} : i55
    %602 = comb.extract %563 from 2 : (i56) -> i54
    %603 = comb.xor %602, %c-1_i54 : i54
    %604 = comb.extract %441 from 1 : (i87) -> i54
    %605 = comb.and %604, %603 : i54
    %606 = comb.concat %false, %605 : i1, i54
    %607 = comb.or %588, %593 : i1
    %608 = comb.mux bin %607, %c0_i55, %606 : i55
    %609 = comb.mux bin %588, %601, %c0_i55 : i55
    %610 = comb.and bin %601, %565 : i55
    %611 = comb.mux bin %593, %610, %c0_i55 : i55
    %612 = comb.or bin %608, %609, %611 {sv.namehint = "sigY3"} : i55
    %613 = comb.extract %612 from 54 : (i55) -> i1
    %614 = comb.extract %452 from 0 : (i14) -> i13
    %615 = comb.add %614, %c1_i13 : i13
    %616 = comb.mux %613, %615, %c0_i13 : i13
    %617 = comb.extract %612 from 53 : (i55) -> i1
    %618 = comb.extract %452 from 0 : (i14) -> i13
    %619 = comb.mux %617, %618, %c0_i13 : i13
    %620 = comb.extract %612 from 53 : (i55) -> i2
    %621 = comb.icmp bin eq %620, %c0_i2 : i2
    %622 = comb.extract %452 from 0 : (i14) -> i13
    %623 = comb.add %622, %c-1_i13 : i13
    %624 = comb.mux %621, %623, %c0_i13 : i13
    %625 = comb.or %616, %619, %624 {sv.namehint = "sExpY"} : i13
    %626 = comb.extract %625 from 0 {sv.namehint = "expY"} : (i13) -> i12
    %627 = comb.extract %612 from 0 : (i55) -> i52
    %628 = comb.extract %612 from 1 : (i55) -> i52
    %629 = comb.mux bin %450, %627, %628 {sv.namehint = "fractY"} : i52
    %630 = comb.extract %625 from 10 : (i13) -> i3
    %631 = comb.icmp bin eq %630, %c3_i3 {sv.namehint = "overflowY"} : i3
    %632 = comb.extract %625 from 12 : (i13) -> i1
    %633 = comb.icmp bin ult %626, %c974_i12 : i12
    %634 = comb.or bin %632, %633 : i1
    %635 = comb.and bin %454, %634 {sv.namehint = "totalUnderflowY"} : i1
    %636 = comb.mux bin %450, %c-1022_i11, %c-1023_i11 : i11
    %637 = comb.concat %c0_i2, %636 : i2, i11
    %638 = comb.icmp bin ule %458, %637 : i13
    %639 = comb.or bin %459, %638 : i1
    %640 = comb.and bin %29, %457 : i1
    %641 = comb.xor bin %457, %true : i1
    %642 = comb.and bin %30, %641 : i1
    %643 = comb.or bin %640, %642 {sv.namehint = "roundMagUp"} : i1
    %644 = comb.or bin %28, %643 {sv.namehint = "overflowY_roundMagUp"} : i1
    %645 = comb.or bin %2, %11 {sv.namehint = "mulSpecial"} : i1
    %646 = comb.or bin %645, %21 {sv.namehint = "addSpecial"} : i1
    %647 = comb.and bin %io_fromPreMul_isZeroProd, %19 {sv.namehint = "notSpecial_addZeros"} : i1
    %648 = comb.xor bin %646, %true : i1
    %649 = comb.xor bin %647, %true : i1
    %650 = comb.and bin %648, %649 {sv.namehint = "commonCase"} : i1
    %651 = comb.and bin %5, %9 : i1
    %652 = comb.and bin %0, %14 : i1
    %653 = comb.xor bin %6, %true : i1
    %654 = comb.xor bin %15, %true : i1
    %655 = comb.or bin %5, %14 : i1
    %656 = comb.and bin %653, %654, %655, %24, %31 : i1
    %657 = comb.or bin %651, %652, %656 {sv.namehint = "notSigNaN_invalid"} : i1
    %658 = comb.or bin %8, %17, %27, %657 {sv.namehint = "invalid"} : i1
    %659 = comb.and bin %650, %631 {sv.namehint = "overflow"} : i1
    %660 = comb.and bin %650, %595, %639 {sv.namehint = "underflow"} : i1
    %661 = comb.and bin %650, %595 : i1
    %662 = comb.or bin %659, %661 {sv.namehint = "inexact"} : i1
    %663 = comb.or bin %647, %455, %635 {sv.namehint = "notSpecial_isZeroOut"} : i1
    %664 = comb.and bin %650, %635, %643 {sv.namehint = "pegMinFiniteMagOut"} : i1
    %665 = comb.xor bin %644, %true : i1
    %666 = comb.and bin %659, %665 {sv.namehint = "pegMaxFiniteMagOut"} : i1
    %667 = comb.and bin %659, %644 : i1
    %668 = comb.or bin %655, %24, %667 {sv.namehint = "notNaN_isInfOut"} : i1
    %669 = comb.or bin %6, %15, %25, %657 {sv.namehint = "isNaNOut"} : i1
    %670 = comb.and bin %322, %io_fromPreMul_opSignC : i1
    %671 = comb.xor bin %21, %true : i1
    %672 = comb.and bin %645, %671, %io_fromPreMul_signProd : i1
    %673 = comb.xor bin %645, %true : i1
    %674 = comb.and bin %673, %21, %io_fromPreMul_opSignC : i1
    %675 = comb.and bin %673, %647, %31, %29 : i1
    %676 = comb.or bin %670, %672, %674, %675 {sv.namehint = "uncommonCaseSignOut"} : i1
    %677 = comb.xor bin %669, %true : i1
    %678 = comb.and bin %677, %676 : i1
    %679 = comb.and bin %650, %457 : i1
    %680 = comb.or bin %678, %679 {sv.namehint = "signOut"} : i1
    %681 = comb.mux bin %663, %c-512_i12, %c0_i12 : i12
    %682 = comb.xor bin %681, %c-1_i12 : i12
    %683 = comb.mux bin %664, %c-975_i12, %c0_i12 : i12
    %684 = comb.xor bin %683, %c-1_i12 : i12
    %685 = comb.xor %666, %true : i1
    %686 = comb.concat %true, %685, %c-1_i10 : i1, i1, i10
    %687 = comb.xor %668, %true : i1
    %688 = comb.concat %c-1_i2, %687, %c-1_i9 : i2, i1, i9
    %689 = comb.and bin %626, %682, %684, %686, %688 : i12
    %690 = comb.mux bin %664, %c974_i12, %c0_i12 : i12
    %691 = comb.mux bin %666, %c-1025_i12, %c0_i12 : i12
    %692 = comb.mux bin %668, %c-1024_i12, %c0_i12 : i12
    %693 = comb.mux bin %669, %c-512_i12, %c0_i12 : i12
    %694 = comb.or bin %689, %690, %691, %692, %693 {sv.namehint = "expOut"} : i12
    %695 = comb.and bin %635, %643 : i1
    %696 = comb.or bin %695, %669 : i1
    %697 = comb.concat %669, %c0_i51 : i1, i51
    %698 = comb.mux bin %696, %697, %629 : i52
    %699 = comb.replicate %666 : (i1) -> i52
    %700 = comb.or bin %698, %699 {sv.namehint = "fractOut"} : i52
    %701 = comb.concat %680, %694, %700 {sv.namehint = "io_out"} : i1, i12, i52
    %702 = comb.concat %658, %false, %659, %660, %662 {sv.namehint = "io_exceptionFlags"} : i1, i1, i1, i1, i1
    hw.output %701, %702 : i65, i5
  }
  om.class @FPU_Class(%basepath: !om.basepath) {
  }
}
