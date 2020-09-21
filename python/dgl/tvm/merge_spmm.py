import tvm

def merge_spmm(num_rows, num_cols, nnz, indice_type, feat_type, feat_len):
    irb = tvm.tir.ir_builder.create()
    def build_ir(indptr, indices, ufeat, out):
        indptr_ptr = irb.buffer_ptr(indptr)
        indices_ptr = irb.buffer_ptr(indices)
        ufeat_ptr = irb.buffer_ptr(ufeat)
        # efeat_ptr = irb.buffer_ptr(efeat)
        out_ptr = irb.buffer_ptr(out)
        row_outer = tvm.te.thread_axis("blockIdx.x", "row_outer")
        row_inner = tvm.te.thread_axis("threadIdx.y", "row_inner")
        row_factor = 4 if feat_len < 64 else 8
        irb.scope_attr(row_outer, "thread_extent", (num_rows + row_factor - 1) // row_factor)
        irb.scope_attr(row_inner, "thread_extent", row_factor)
        elem_inner = tvm.te.thread_axis('threadIdx.x', 'elem_inner')
        irb.scope_attr(elem_inner, "thread_extent", 32)
        feat_outer = tvm.te.thread_axis('blockIdx.y', 'feat_outer')
        CF = 1 if feat_len < 64 else 2
        irb.scope_attr(feat_outer, "thread_extent", (feat_len + CF*32 - 1)// (32 * CF))
        sm_k = irb.allocate(indice_type, (32*row_factor,), name='sm_k', scope='shared')
        # sm_v = irb.allocate(feat_type, (32,), name='sm_v', scope='shared')
        result = irb.allocate(feat_type, (CF,), name='result', scope='local')
        with irb.for_range(0, CF, name="cf", for_type="unroll") as cf:
            result[cf] = 0.0
        # with irb.if_scope(row_outer * row_factor + row_inner < num_rows):
        row_start = indptr_ptr[row_outer * row_factor + row_inner]
        row_end = indptr_ptr[row_outer * row_factor + row_inner + 1]
        with irb.for_range(0, (row_end-row_start+31) // 32, name='elem_outer') as elem_outer:
            with irb.if_scope(row_start + elem_outer * 32 + elem_inner < row_end):
                sm_k[elem_inner] = indices_ptr[row_start + elem_outer * 32 + elem_inner]
                # sm_v[elem_inner] = efeat_ptr[row_start + elem_outer * 32 + elem_inner]
            with irb.for_range(0, 32, name='kk') as kk:
                with irb.if_scope(row_start + elem_outer * 32 + kk < row_end):
                    # result[0] += sm_v[kk] * ufeat_ptr[sm_k[kk] * feat_len + feat_outer + elem_inner]
                    with irb.for_range(0, CF, name="cf", for_type="unroll") as cf:
                        result[cf] += ufeat_ptr[sm_k[kk] * feat_len + feat_outer * CF * 32  + cf * 32 + elem_inner]
            with irb.for_range(0, CF, name="cf", for_type="unroll") as cf:
                out_ptr[(row_outer * row_factor + row_inner) * feat_len + feat_outer * CF * 32 + cf * 32 + elem_inner] = result[cf]
        return irb.get()
    indptr = tvm.te.placeholder((num_rows+1,), indice_type, 'indptr')
    indices = tvm.te.placeholder((nnz,), indice_type, name='indices')
    ufeat = tvm.te.placeholder((num_cols, feat_len), feat_type, name='ufeat')
    # efeat = tvm.te.placeholder((nnz,1), feat_type, name='efeat')
    out = tvm.te.extern((num_rows, feat_len), [indptr, indices, ufeat],
        lambda ins, outs: build_ir(ins[0], ins[1], ins[2], outs[0]), dtype=feat_type, name='out')
    sched = tvm.te.create_schedule(out.op)
    # print(tvm.lower(sched, [indptr, indices, ufeat, efeat, out]))
    f = tvm.build(sched, [indptr, indices, ufeat, out], target='cuda')
    print(f.imported_modules[0].get_source())
    return f

# num_rows = num_cols = 100
# nnz = 1000
# indice_type = 'int32'
# feat_type = 'float32'
# feat_len = 32
# merge_spmm(num_rows, num_cols, nnz, indice_type, feat_type, feat_len)