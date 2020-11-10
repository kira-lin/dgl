/*!
 * Copyright (c) 2020 by Contributors
 * \file array/cuda/ge_spmm.cuh
 * \brief GE-SpMM CUDA kernel function header.
 */
#ifndef DGL_ARRAY_CUDA_GE_SPMM_CUH_
#define DGL_ARRAY_CUDA_GE_SPMM_CUH_

#include "macro.cuh"
#include "atomic.cuh"
#include "../../runtime/cuda/cuda_common.h"
#include "./utils.h"

namespace dgl {

using namespace cuda;

namespace aten {
namespace cuda {

/*! 
 * \brief: TODO(zihao)
 */
template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
__global__ void GESpMMSumKernel(
    const DType* __restrict__ ufeat,
    const DType* __restrict__ efeat,
    DType* __restrict__ out,
    Idx* __restrict__ arg_u,
    Idx* __restrict__ arg_e,
    const Idx* __restrict__ indptr,
    const Idx* __restrict__ indices,
    const int64_t num_rows, const int64_t num_cols,
    const int64_t feat_len) {
  extern __shared__ char smem[];
  // Idx* col = nullptr;
  DType* val = (DType *) smem;
  // if (BinaryOp::use_rhs) {  // use edge feature.
  //   if (sizeof(Idx) >= sizeof(DType)) {
  //     // handle alignment issue: https://forums.developer.nvidia.com/t/dynamic-shared-memory-allocation/21671/3
  //     col = (Idx*) smem;
  //     val = (DType*) &col[blockDim.y * blockDim.x];
  //   } else {
  //     val = (DType*) smem;
  //     col = (Idx*) &val[blockDim.y * blockDim.x];
  //   }
  // } else {
  //   col = (Idx*) smem;
  // }

  Idx ty = blockIdx.x * blockDim.z + threadIdx.z;  // iterate over destination nodes
  const Idx tx = threadIdx.x;
  const Idx head = threadIdx.y;
  const Idx num_heads = blockDim.y;
  const Idx sm_base = 32 * num_heads * threadIdx.z;
  const Idx sm_offset = sm_base + 32 * head;
  const int D = feat_len / num_heads;
  const Idx rid = ty;
  if (rid < num_rows) {
    const Idx low = indptr[rid], high = indptr[rid + 1];
    // Idx fid = (blockIdx.y * 64) + tx;  // iterate over feature dimension
    Idx fid = (blockIdx.y * 32) + tx;  // iterate over feature dimension

    // DType accum_0 = ReduceOp::zero,
    //       accum_1 = ReduceOp::zero;
    // Idx argu_0 = 0, arge_0 = 0,
    //     argu_1 = 0, arge_1 = 0;
    DType accum = ReduceOp::zero;
    Idx argu = 0, arge = 0;

    // if (blockIdx.y != gridDim.y - 1) {
      for (Idx left = low; left < high; left += 32) {
        __syncthreads();
        if (left + tx < high) {
          // col[sm_offset + tx] = indices[left + tx]; 
          if (BinaryOp::use_rhs) {
            // original layout
            // val[sm_offset + tx] = efeat[left + head * 32 + tx];
            // new layout
            val[sm_base + tx * num_heads + head] = efeat[left + head * 32 + tx];
          }
        }
        __syncthreads();
        for (Idx i = 0; i < 32 && left + i < high; ++i) {
          const Idx eid = left + i; 
          const Idx cid = indices[left + i];
          const Idx offset = feat_len * cid + head * D + fid;
          if (BinaryOp::use_rhs) {
            // original layout
            // const DType weight = val[sm_base + i * num_heads + head];
            // new layout
            const DType weight = val[sm_offset + i];
            // ReduceOp::Call(&accum_0, &argu_0, &arge_0,
            //   BinaryOp::Call(ufeat + offset, &weight), cid, eid);
            // ReduceOp::Call(&accum_1, &argu_1, &arge_1,
            //   BinaryOp::Call(ufeat + offset + 32, &weight), cid, eid);
            ReduceOp::Call(&accum, &argu, &arge,
              BinaryOp::Call(ufeat + offset, &weight), cid, eid);
          } else {
            // ReduceOp::Call(&accum_0, &argu_0, &arge_0,
            //   ufeat[offset], fid, eid);
            // ReduceOp::Call(&accum_1, &argu_1, &arge_1,
            //   ufeat[offset + 32], cid, eid);
            ReduceOp::Call(&accum, &argu, &arge,
              ufeat[offset + 32], cid, eid);
          }
        }

        // out[feat_len * rid + head * D +fid] = accum_0;
        // if (ReduceOp::require_arg && BinaryOp::use_lhs)
        //   arg_u[feat_len * rid + fid] = argu_0;
        // if (ReduceOp::require_arg && BinaryOp::use_rhs)
        //   arg_e[feat_len * rid + fid] = arge_0;

        // out[feat_len * rid + head * D + fid + 32] = accum_1;
        // if (ReduceOp::require_arg && BinaryOp::use_rhs)
        //   arg_u[feat_len * rid + fid + 32] = argu_1;
        // if (ReduceOp::require_arg && BinaryOp::use_rhs)
        //   arg_e[feat_len * rid + fid + 32] = arge_1; 

        out[feat_len * rid + head * D +fid] = accum;
        if (ReduceOp::require_arg && BinaryOp::use_lhs)
          arg_u[feat_len * rid + fid] = argu;
        if (ReduceOp::require_arg && BinaryOp::use_rhs)
          arg_e[feat_len * rid + fid] = arge;
      }
    // } else {
    //   bool left_inbound = fid < D,
    //        right_inbound = fid + 32 < D;
    //   for (int left = low; left < high; left += 32) {
    //     if (left + tx < high) {
    //       col[sm_offset + tx] = indices[left + tx]; 
    //       if (BinaryOp::use_rhs)
    //         val[sm_offset + tx] = efeat[(left + tx) * num_heads + head];
    //     }

    //     for (int i = 0; i < 32 && left + i < high; ++i) {
    //       const Idx eid = left + i; 
    //       const Idx cid = col[sm_offset + i];
    //       const Idx offset = feat_len * cid + head * D + fid;
    //       if (BinaryOp::use_rhs) {
    //         const DType weight = val[sm_offset + i];
    //         if (left_inbound)
    //           ReduceOp::Call(&accum_0, &argu_0, &arge_0,
    //             BinaryOp::Call(ufeat + offset, &weight), cid, eid);
    //         if (right_inbound)
    //           ReduceOp::Call(&accum_1, &argu_1, &arge_1,
    //             BinaryOp::Call(ufeat + offset + 32, &weight), cid, eid);
    //       } else {
    //         if (left_inbound)
    //           ReduceOp::Call(&accum_0, &argu_0, &arge_0,
    //             ufeat[offset], fid, eid);
    //         if (right_inbound)
    //           ReduceOp::Call(&accum_1, &argu_1, &arge_1,
    //             ufeat[offset + 32], cid, eid);
    //       }
    //     }

    //     if (left_inbound) {
    //       out[feat_len * rid + head * D + fid] = accum_0;
    //       if (ReduceOp::require_arg && BinaryOp::use_lhs)
    //         arg_u[feat_len * rid + fid] = argu_0;
    //       if (ReduceOp::require_arg && BinaryOp::use_rhs)
    //         arg_e[feat_len * rid + fid] = arge_0;
    //     }

    //     if (right_inbound) {
    //       out[feat_len * rid + head * D + fid + 32] = accum_1;
    //       if (ReduceOp::require_arg && BinaryOp::use_rhs)
    //         arg_u[feat_len * rid + fid + 32] = argu_1;
    //       if (ReduceOp::require_arg && BinaryOp::use_rhs)
    //         arg_e[feat_len * rid + fid + 32] = arge_1; 
    //     }
    //   }
    // }
  }
}

template <typename Idx, typename DType,
          typename BinaryOp, typename ReduceOp>
void GESpMMCsr(
    const CSRMatrix& csr,
    NDArray ufeat, NDArray efeat,
    NDArray out, NDArray argu, NDArray arge,
    int64_t feat_len, int num_heads) {
  const Idx *indptr = csr.indptr.Ptr<Idx>();
  const Idx *indices = csr.indices.Ptr<Idx>();
  const DType *ufeat_data = ufeat.Ptr<DType>();
  const DType *efeat_data = efeat.Ptr<DType>();
  DType *out_data = out.Ptr<DType>();
  Idx *argu_data = argu.Ptr<Idx>();
  Idx *arge_data = arge.Ptr<Idx>();

  auto* thr_entry = runtime::CUDAThreadEntry::ThreadLocal();
  
  const int ntx = 32;
  const int nty = num_heads;
  const int ntz = 4;
  const int nby = (feat_len / num_heads + ntx - 1) / ntx;
  const int nbx = FindNumBlocks<'x'>((csr.num_rows + ntz - 1) / ntz);
  const dim3 nblks(nbx, nby);
  const dim3 nthrs(ntx, nty, ntz);
  const int sh_mem_size = BinaryOp::use_rhs ? 32 * ntz * num_heads * sizeof(DType) : 0;

  CUDA_KERNEL_CALL((GESpMMSumKernel<Idx, DType, BinaryOp, ReduceOp>),
      nblks, nthrs, sh_mem_size, thr_entry->stream,
      ufeat_data, efeat_data, out_data, argu_data, arge_data,
      indptr, indices,
      csr.num_rows, csr.num_cols,
      feat_len);
}

}  // namespace cuda
}  // namespace aten
}  // namespace dgl

#endif