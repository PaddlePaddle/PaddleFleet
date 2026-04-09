# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2025, Ted Zadouri, Markus Hoehnerbach, Jay Shah, Tri Dao.
import math
from typing import Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, const_expr
from cutlass.utils import LayoutEnum
from cutlass.cute.nvgpu import cpasync, tcgen05
import cutlass.utils.blackwell_helpers as sm100_utils_basic
from cutlass.pipeline import PipelineAsync, PipelineConsumer

from flash_mask.cute import utils
from flash_mask.cute import copy_utils
from flash_mask.cute import pipeline
from flash_mask.cute.blackwell_helpers import gemm_w_idx, gemm_ptx_w_idx  # noqa
from flash_mask.cute.mask import AttentionMask
from flash_mask.cute.seqlen_info import SeqlenInfoQK
from flash_mask.cute.block_info import BlockInfo
from flash_mask.cute.tile_scheduler import (
    TileSchedulerArguments,
    SingleTileScheduler,
    SingleTileLPTBwdScheduler,  # noqa
    ParamsBase,
)

from flash_mask.cute import barrier
from flash_mask.cute.named_barrier import NamedBarrierBwdSm100
from flash_mask.cute.flashmask_utils import FlashMaskInfo


class FlashAttentionBackwardSm100:
    arch = 100

    def __init__(
        self,
        head_dim: int,
        head_dim_v: Optional[int] = None,
        is_causal: bool = False,
        is_local: bool = False,
        qhead_per_kvhead: cutlass.Constexpr[int] = 1,
        tile_m: int = 128,
        tile_n: int = 128,
        is_persistent: bool = False,
        deterministic: bool = False,
        cluster_size: int = 1,
    ):
        # padding head_dim to a multiple of 64 to match head_dim_rounded in interface
        hdim_multiple_of = 64
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        assert head_dim == head_dim_v, "head_dim and head_dim_v must be the same for now"
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        assert self.tile_hdim == self.tile_hdimv, (
            "tile_hdim and tile_hdimv must be the same for now"
        )
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv

        self.tile_m = tile_m
        self.tile_n = tile_n
        self.debug_print = False

        # CTA tiler
        self.cta_tiler = (tile_m, tile_n, self.tile_hdim)
        # S = K @ Q.T
        self.mma_tiler_kq = (tile_n, tile_m, self.tile_hdim)
        # dP = V @ dO.T
        self.mma_tiler_vdo = (tile_n, tile_m, self.tile_hdimv)
        # dV = P.T @ dO
        self.mma_tiler_pdo = (tile_n, self.tile_hdimv, tile_m)
        # dK = dS.T @ Q (N, M) (M, D)
        self.mma_tiler_dsq = (tile_n, self.tile_hdimv, tile_m)
        # dQ = dS @ K
        self.mma_tiler_dsk = (tile_m, self.tile_hdimv, tile_n)

        self.acc_dtype = Float32
        self.startend_row_indices_dtype = Int32

        assert cluster_size in (1, 2), "Only cluster_size=1 or 2 is supported"
        self.cluster_shape_mn = (cluster_size, 1)
        self.is_persistent = is_persistent
        self.is_causal = is_causal
        self.is_local = False
        self.qhead_per_kvhead = qhead_per_kvhead
        self.pack_gqa = False
        self.use_tma_store = True
        self.deterministic = deterministic

        # Speed optimizations, does not affect correctness
        self.shuffle_LSE = False
        self.shuffle_dPsum = False
        self.use_smem_dS_for_mma_dK = self.deterministic and self.is_causal

        self.reduce_warp_ids = (0, 1, 2, 3)
        self.compute_warp_ids = (4, 5, 6, 7, 8, 9, 10, 11)
        self.mma_warp_id = 12
        self.load_warp_id = 13
        self.epi_warp_id = 14
        self.empty_warp_id = 15

        # 16 warps -> 512 threads
        self.threads_per_cta = cute.arch.WARP_SIZE * len(
            (
                *self.reduce_warp_ids,
                *self.compute_warp_ids,
                self.mma_warp_id,
                self.load_warp_id,
                self.epi_warp_id,
                self.empty_warp_id,
            )
        )

        # NamedBarrier
        self.compute_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.Compute),
            num_threads=len(self.compute_warp_ids) * cute.arch.WARP_SIZE,
        )
        # self.epilogue_sync_barrier = pipeline.NamedBarrier(
        #     barrier_id=2,
        #     num_threads=self.num_compute_warps * self.threads_per_warp,
        # )
        self.reduce_sync_barrier = cutlass.pipeline.NamedBarrier(
            barrier_id=int(NamedBarrierBwdSm100.dQaccReduce),
            num_threads=len(self.reduce_warp_ids) * cute.arch.WARP_SIZE,
        )

        # TMEM setup
        SM100_TMEM_CAPACITY_COLUMNS = 512
        self.tmem_alloc_cols = SM100_TMEM_CAPACITY_COLUMNS

        # self.tmem_dK_offset = 0
        # self.tmem_dV_offset = self.tmem_dK_offset + self.tile_hdim
        # self.tmem_dQ_offset = self.tmem_dV_offset + self.tile_hdimv
        # self.tmem_dP_offset = self.tmem_dQ_offset  # overlap with dQ
        # self.tmem_S_offset = self.tmem_dQ_offset + max(self.tile_m, self.tile_hdim)
        # self.tmem_P_offset = self.tmem_S_offset  # overlap with S
        # self.tmem_total = self.tmem_S_offset + self.tile_n
        # assert self.tmem_total <= self.tmem_alloc_cols

        self.tmem_S_offset = 0
        self.tmem_P_offset = 0  # overlap with S
        self.tmem_dV_offset = self.tmem_S_offset + self.tile_n
        self.tmem_dP_offset = self.tmem_dV_offset + self.tile_hdimv
        self.tmem_dQ_offset = self.tmem_dP_offset  # overlap with dP
        self.tmem_dK_offset = self.tmem_dP_offset + self.tile_m
        self.tmem_dS_offset = self.tmem_dP_offset  # overlap with dP

        if (not is_causal and not is_local) or deterministic:
            self.num_regs_reduce = 152
            self.num_regs_compute = 136
        else:
            self.num_regs_reduce = 136
            self.num_regs_compute = 144
        self.num_regs_other = 96 - 8
        self.num_regs_empty = 24
        assert self.num_regs_reduce + self.num_regs_compute * 2 + self.num_regs_other <= 512

        self.buffer_align_bytes = 1024

    def _setup_attributes(self):
        self.Q_stage = 2
        self.dO_stage = 1
        # LSE_stage = Q_stage and dPsum_stage = dO_stage
        # self.sdKVaccum_stage = 2
        # number of tma reduce adds per dQacc mma
        self.dQ_reduce_ncol = 32 if self.tile_hdim % 32 == 0 else 16
        self.sdQaccum_stage = 64 // self.dQ_reduce_ncol
        assert self.tile_hdim % self.dQ_reduce_ncol == 0
        self.dQaccum_reduce_stage = self.tile_hdim // self.dQ_reduce_ncol
        self.cluster_reduce_dQ = False and cute.size(self.cluster_shape_mn) > 1
        # number of tma reduce adds for dKacc and dVacc epilogue
        self.dK_reduce_ncol = 32

    def _get_tiled_mma(self):
        cta_group = tcgen05.CtaGroup.ONE
        # S = K @ Q.T
        tiled_mma_S = sm100_utils_basic.make_trivial_tiled_mma(
            self.q_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_kq[:2],
        )
        # dP = V @ dO.T
        tiled_mma_dP = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,
            self.acc_dtype,
            cta_group,
            self.mma_tiler_vdo[:2],
        )
        # dV += P @ dO --> (K, MN) major
        tiled_mma_dV = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # P_major_mode
            tcgen05.OperandMajorMode.MN,  # dO_major_mode
            self.acc_dtype,
            cta_group,
            self.mma_tiler_pdo[:2],
            a_source=tcgen05.OperandSource.TMEM,
        )
        # dK += dS.T @ Q
        if const_expr(self.use_smem_dS_for_mma_dK):
            mma_dK_a_src = tcgen05.OperandSource.SMEM
        else:
            mma_dK_a_src = tcgen05.OperandSource.TMEM
        tiled_mma_dK = sm100_utils_basic.make_trivial_tiled_mma(
            self.do_dtype,
            tcgen05.OperandMajorMode.K,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Q_major_mode
            self.acc_dtype,
            cta_group,
            self.mma_tiler_dsq[:2],
            a_source=mma_dK_a_src,
        )
        # dQ = dS @ K
        tiled_mma_dQ = sm100_utils_basic.make_trivial_tiled_mma(
            self.k_dtype,
            tcgen05.OperandMajorMode.MN,  # dS_major_mode
            tcgen05.OperandMajorMode.MN,  # Kt_major_mode
            self.acc_dtype,
            cta_group,
            self.mma_tiler_dsk[:2],
        )
        return tiled_mma_S, tiled_mma_dP, tiled_mma_dK, tiled_mma_dV, tiled_mma_dQ

    def _setup_smem_layout(self):
        # S = K @ Q.T
        sK_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.k_dtype,
            1,
        )
        self.sK_layout = cute.slice_(sK_layout, (None, None, None, 0))
        self.sQ_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_S,
            self.mma_tiler_kq,
            self.q_dtype,
            self.Q_stage,
        )
        # dP = V @ dO.T
        sV_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.v_dtype,
            1,
        )
        self.sV_layout = cute.slice_(sV_layout, (None, None, None, 0))
        self.sdOt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dP,
            self.mma_tiler_vdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dV += P @ dO
        tP_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            1,
        )
        self.tP_layout = cute.slice_(tP_layout, (None, None, None, 0))
        self.sdO_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dV,
            self.mma_tiler_pdo,
            self.do_dtype,
            self.dO_stage,
        )
        # dK += dS.T @ Q
        sdSt_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.sdSt_layout = cute.slice_(sdSt_layout, (None, None, None, 0))
        tdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.ds_dtype,
            1,
        )
        self.tdS_layout = cute.slice_(tdS_layout, (None, None, None, 0))
        self.sQt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dK,
            self.mma_tiler_dsq,
            self.q_dtype,
            self.Q_stage,
        )
        # dQ = dS @ K
        sdS_layout = sm100_utils_basic.make_smem_layout_a(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.ds_dtype,
            1,
        )
        self.sdS_layout = cute.slice_(sdS_layout, (None, None, None, 0))
        sKt_layout = sm100_utils_basic.make_smem_layout_b(
            self.tiled_mma_dQ,
            self.mma_tiler_dsk,
            self.k_dtype,
            1,
        )
        self.sKt_layout = cute.slice_(sKt_layout, (None, None, None, 0))
        self.sdQaccum_layout = cute.make_layout(
            (self.tile_m * self.dQ_reduce_ncol, self.sdQaccum_stage)
        )
        self.sLSE_layout = cute.make_layout(
            shape=(self.tile_m, self.Q_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdPsum_layout = cute.make_layout(
            shape=(self.tile_m, self.dO_stage),
            stride=(1, cute.round_up(self.tile_m, 64)),
        )
        self.sdKV_epi_tile = (
            self.tile_n,
            min(128 // (self.dk_dtype.width // 8),  (self.tile_hdim // 2)) # 64 or 32
        )  # subtiles mma_tiler_dsq[:2] = mma_tiler_pdo[:2]
        # headdim_64 gets 1 stage
        self.num_epi_stages = max(1, (self.tile_hdim // 2) // self.sdKV_epi_tile[1])

        self.sdKV_flat_epi_tile = self.tile_n * (self.tile_hdim // 2) // self.num_epi_stages

        # TODO: dK and dV could have different shapes
        if const_expr(self.qhead_per_kvhead == 1):
            self.sdKV_layout = sm100_utils_basic.make_smem_layout_epi(
                self.dk_dtype,
                LayoutEnum.ROW_MAJOR,
                self.sdKV_epi_tile,
                2,  # num compute wgs
            )
        else:
            self.sdKV_layout = cute.make_layout((self.tile_n * self.dK_reduce_ncol, 2))

        # TODO(GuoxiaWang): 2 means only support flashmask startend_row_indices.shape[-1] <= 2
        self.sStartEndRowIndices_layout = cute.make_layout(
            shape=(self.tile_n, 2),
            stride=(1, self.tile_n),
        )

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mdO: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdK: cute.Tensor,
        mdV: cute.Tensor,
        softmax_scale: Float32,
        stream: cuda.CUstream,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        softcap: Float32 | float | None = None,
        window_size_left: Int32 | int | None = None,
        window_size_right: Int32 | int | None = None,
        mdQ_semaphore: Optional[cute.Tensor] = None,
        mdK_semaphore: Optional[cute.Tensor] = None,
        mdV_semaphore: Optional[cute.Tensor] = None,
        flashmask_info: Optional[FlashMaskInfo] = None,
    ):
        assert all(x is None for x in (mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK)), (
            "Variable sequence length is not supported yet in FlashAttentionBackwardSm100"
        )
        self.q_dtype = mQ.element_type
        self.k_dtype = mK.element_type
        self.v_dtype = mV.element_type
        self.do_dtype = mdO.element_type
        self.lse_dtype = mLSE.element_type
        self.dpsum_dtype = mdPsum.element_type
        self.dqaccum_dtype = mdQaccum.element_type
        self.dk_dtype = mdK.element_type
        self.dv_dtype = mdV.element_type
        self.ds_dtype = self.q_dtype

        self.enable_flashmask = cutlass.const_expr(flashmask_info is not None)

        if const_expr(self.qhead_per_kvhead > 1):
            assert self.dk_dtype.width == 32, "Must accumulate dK in float precision for GQA"
            assert self.dv_dtype.width == 32, "Must accumulate dV in float precision for GQA"

        # Assume all strides are divisible by 128 bits except the last stride
        new_stride = lambda t: (
            *(cute.assume(s, divby=128 // t.element_type.width) for s in t.stride[:-1]),
            t.stride[-1],
        )
        (mdQaccum,) = [
            cute.make_tensor(t.iterator, cute.make_layout(t.shape, stride=new_stride(t)))
            if t is not None
            else None
            for t in (mdQaccum,)
        ]

        layout_transpose = [1, 3, 2, 0]  # (b, s, n, h) --> (s, h, n, b)
        mQ, mK, mV, mdO = [utils.select(t, mode=layout_transpose) for t in (mQ, mK, mV, mdO)]
        LSE_dPsum_dQaccum_transpose = [2, 1, 0]  # (b, n, s) --> (s, n, b)
        mLSE, mdPsum, mdQaccum = [
            utils.select(t, mode=LSE_dPsum_dQaccum_transpose) for t in (mLSE, mdPsum, mdQaccum)
        ]
        if const_expr(self.qhead_per_kvhead == 1):
            layout_dKV_transpose = layout_transpose
        else:
            layout_dKV_transpose = LSE_dPsum_dQaccum_transpose
        mdK, mdV = [utils.select(t, mode=layout_dKV_transpose) for t in (mdK, mdV)]
        dO_transpose = [1, 0, 2, 3]  # (s, h, n, b) --> (h, s, n, b)
        mdO = utils.select(mdO, mode=dO_transpose)

        semaphore_transpose = [2, 3, 1, 0]  # (b, n, block, stage) -> (block, stage, n, b)
        if const_expr(self.deterministic):
            assert mdQ_semaphore is not None
            mdQ_semaphore = utils.select(mdQ_semaphore, mode=semaphore_transpose)

        if const_expr(self.deterministic and self.qhead_per_kvhead > 1):
            assert mdK_semaphore is not None
            assert mdV_semaphore is not None
            mdK_semaphore, mdV_semaphore = [
                utils.select(t, mode=semaphore_transpose) for t in (mdK_semaphore, mdV_semaphore)
            ]
        else:
            mdK_semaphore = None
            mdV_semaphore = None

        self._setup_attributes()
        (
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dK,
            self.tiled_mma_dV,
            self.tiled_mma_dQ,
        ) = self._get_tiled_mma()
        self._setup_smem_layout()

        cta_group = tcgen05.CtaGroup.ONE

        self.cluster_shape_mnk = (*self.cluster_shape_mn, 1)
        self.cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (self.tiled_mma_S.thr_id.shape,),
        )
        self.num_mcast_ctas_b = cute.size(self.cluster_layout_vmnk.shape[1])
        self.is_q_do_mcast = self.num_mcast_ctas_b > 1

        if const_expr(self.qhead_per_kvhead == 1):
            self.mdK_layout_enum = LayoutEnum.from_tensor(mdK)
            self.mdV_layout_enum = LayoutEnum.from_tensor(mdV)
            dK_major_mode = self.mdK_layout_enum.mma_major_mode()
            dV_major_mode = self.mdV_layout_enum.mma_major_mode()
            if const_expr(dK_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdK is wrong")
            if const_expr(dV_major_mode != tcgen05.OperandMajorMode.K):
                raise RuntimeError("The layout of mdV is wrong")

        if const_expr(self.use_tma_store and self.qhead_per_kvhead == 1):
            tma_copy_op_dKV = cpasync.CopyBulkTensorTileS2GOp()
            tma_atom_dK, mdK_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdK,
                cute.select(self.sdKV_layout, mode=[0, 1]),
                self.sdKV_epi_tile,
                1,  # no mcast
            )
            tma_atom_dV, mdV_tma_tensor = cpasync.make_tiled_tma_atom(
                tma_copy_op_dKV,
                mdV,
                cute.select(self.sdKV_layout, mode=[0, 1]),
                self.sdKV_epi_tile,
                1,  # no mcast
            )
        else:
            mdV_tma_tensor = mdV
            mdK_tma_tensor = mdK
            tma_atom_dV = None
            tma_atom_dK = None

        if const_expr(self.qhead_per_kvhead == 1):
            thr_layout_r2s_dKV = cute.make_ordered_layout((128, 1), order=(1, 0))  # 128 threads
            val_layout_r2s_dKV = cute.make_ordered_layout(
                (1, 128 // self.dk_dtype.width), order=(1, 0)
            )  # 4 or 8 vals for 16 byte store
            copy_atom_r2s_dKV = cute.make_copy_atom(
                cute.nvgpu.CopyUniversalOp(),
                self.dk_dtype,
                num_bits_per_copy=128,
            )
            tiled_copy_r2s_dKV = cute.make_tiled_copy_tv(
                copy_atom_r2s_dKV, thr_layout_r2s_dKV, val_layout_r2s_dKV
            )
        else:
            tiled_copy_r2s_dKV = copy_utils.tiled_copy_1d(
                Float32, 128, num_copy_elems=128 // Float32.width
            )

        tma_load_op = cpasync.CopyBulkTensorTileG2SOp(cta_group)
        tma_load_op_multicast = cpasync.CopyBulkTensorTileG2SMulticastOp(cta_group)

        # S.T = K @ Q.T
        tma_atom_K, tma_tensor_K = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mK,
            cute.select(self.sK_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        Q_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_S.thr_id
        )
        tma_atom_Q, tma_tensor_Q = cute.nvgpu.make_tiled_tma_atom_B(
            # tma_load_op if const_expr(self.cluster_shape_mnk[0] == 1) else tma_load_op_multicast,
            Q_tma_op,
            mQ,
            cute.select(self.sQ_layout, mode=[0, 1, 2]),
            self.mma_tiler_kq,
            self.tiled_mma_S,
            self.cluster_layout_vmnk.shape,
        )
        # dP.T = V @ dO.T
        tma_atom_V, tma_tensor_V = cute.nvgpu.make_tiled_tma_atom_A(
            tma_load_op,
            mV,
            cute.select(self.sV_layout, mode=[0, 1, 2]),
            self.mma_tiler_vdo,
            self.tiled_mma_dP,
            self.cluster_layout_vmnk.shape,
        )
        dO_tma_op = sm100_utils_basic.cluster_shape_to_tma_atom_B(
            self.cluster_shape_mnk, self.tiled_mma_dV.thr_id
        )
        tma_atom_dO, tma_tensor_dO = cute.nvgpu.make_tiled_tma_atom_B(
            # tma_load_op if const_expr(self.cluster_shape_mnk[0] == 1) else tma_load_op_multicast,
            dO_tma_op,
            mdO,
            cute.select(self.sdO_layout, mode=[0, 1, 2]),
            self.mma_tiler_pdo,
            self.tiled_mma_dV,
            self.cluster_layout_vmnk.shape,
        )

        self.tma_copy_bytes = {
            name: cute.size_in_bytes(mX.element_type, cute.select(layout, mode=[0, 1, 2]))
            for name, mX, layout in [
                ("Q", mQ, self.sQ_layout),
                ("K", mK, self.sK_layout),
                ("V", mV, self.sV_layout),
                ("dO", mdO, self.sdO_layout),
            ]
        }
        self.tma_copy_bytes["LSE"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dPsum"] = self.tile_m * Float32.width // 8
        self.tma_copy_bytes["dQ"] = self.tile_m * self.dQ_reduce_ncol * Float32.width // 8
        self.tma_copy_bytes["dKacc"] = self.tile_n * self.dK_reduce_ncol * Float32.width // 8

        # TileScheduler = SingleTileScheduler
        if const_expr(self.deterministic):
            TileScheduler = SingleTileLPTBwdScheduler
        else:
            TileScheduler = SingleTileScheduler
        self.spt = self.is_causal and self.deterministic
        tile_sched_args = TileSchedulerArguments(
            cute.ceil_div(cute.size(mK.shape[0]), self.cta_tiler[0]),
            cute.size(mQ.shape[2]),  # num_heads = num_query_heads
            cute.size(mK.shape[3]),
            1,  # num_splits
            cute.size(mQ.shape[0]),  # pass seqlen_q for seqlen_k
            mQ.shape[1],
            mV.shape[1],
            total_q=cute.size(mQ.shape[0]),
            tile_shape_mn=self.cta_tiler[:2],
            cluster_shape_mn=self.cluster_shape_mnk[:2],
            mCuSeqlensQ=None,
            mSeqUsedQ=None,
            qhead_per_kvhead_packgqa=1,
            element_size=self.k_dtype.width // 8,
            is_persistent=self.is_persistent,
            lpt=self.spt,
        )

        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        self.tile_scheduler_cls = TileScheduler
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        # cute.printf("grid_dim = {}", grid_dim)

        # Compute allocation sizes for shared buffers that are reused
        # sQ is reused for sdK, sdO is reused for sdV
        sQ_alloc_bytes = max(
            cute.size_in_bytes(self.q_dtype, self.sQ_layout),
            cute.size_in_bytes(self.dk_dtype, self.sdKV_layout),
        )
        sdO_alloc_bytes = max(
            cute.size_in_bytes(self.dv_dtype, self.sdKV_layout),
            cute.size_in_bytes(self.do_dtype, self.sdO_layout),
        )
        # Sanity check that layouts fit in allocation
        sdV_bytes = cute.size_in_bytes(self.dv_dtype, self.sdKV_layout)
        sdK_bytes = cute.size_in_bytes(self.dk_dtype, self.sdKV_layout)
        assert sdV_bytes <= sdO_alloc_bytes, "sdV doesn't fit in sdO storage allocation"
        assert sdK_bytes <= sQ_alloc_bytes, "sdK doesn't fit in sQ storage allocation"

        @cute.struct
        class SharedStorage:
            Q_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dO_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            LSE_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.Q_stage]
            dPsum_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * self.dO_stage]
            S_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dP_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dS_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 1]
            dKV_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2 * 2]
            dQ_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 2]
            dQ_cluster_full_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.dQaccum_reduce_stage // 2
            ]
            dQ_cluster_empty_mbar_ptr: cute.struct.MemRange[
                cutlass.Int64, self.dQaccum_reduce_stage // 2
            ]
            tmem_holding_buf: Int32
            tmem_dealloc_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            flashmask_loaded_mbar_ptr: cute.struct.MemRange[cutlass.Int64, 1]
            sFM_max_min_ptr: cute.struct.MemRange[cutlass.Int32, 8]
            # 240
            sdPsum: cute.struct.Align[
                cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
                128,
            ]

            # Smem tensors
            # sQ is reused for sdK which in the non-MHA case needs float32
            sQ: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sQ_alloc_bytes],
                self.buffer_align_bytes,
            ]
            # self.sQ_layout S<3,4,3> o 0 o ((128,16),1,(4,2),2):((64,1),0,(16,8192),16384)
            # 128 * 16 * 4 * 2 * 2 = 32768 * 2 = 65536
            # 66560
            sK: cute.struct.Align[
                cute.struct.MemRange[self.k_dtype, cute.cosize(self.sK_layout)],
                self.buffer_align_bytes,
            ]
            # self.sK_layout S<3,4,3> o 0 o ((128,16),1,(4,2)):((64,1),0,(16,8192))
            # 128 * 16 * 4 * 2 = 16384 * 2 = 32768
            # 99328
            sV: cute.struct.Align[
                cute.struct.MemRange[self.v_dtype, cute.cosize(self.sV_layout)],
                self.buffer_align_bytes,
            ]
            # self.sV_layout S<3,4,3> o 0 o ((128,16),1,(4,2)):((64,1),0,(16,8192))
            # 128 * 16 * 4 * 2 = 16384 * 2 = 32768
            # 132096
            # sdO is reused for sdV which in the non-MHA case needs float32
            sdO: cute.struct.Align[
                cute.struct.MemRange[cute.Uint8, sdO_alloc_bytes],
                self.buffer_align_bytes,
            ]
            # self.sdO_layout S<3,4,3> o 0 o (((64,2),16),1,8,1):(((1,8192),64),0,1024,0)
            # 64 * 2 * 16 * 8 = 16384 * 2 = 32768
            # 164864
            sdQaccum: cute.struct.Align[
                cute.struct.MemRange[self.dqaccum_dtype, cute.cosize(self.sdQaccum_layout)],
                self.buffer_align_bytes,
            ]
            # self.sdQaccum_layout (4096,2):(1,4096)
            # 4096 * 2 = 8192 * 4 = 32768
            # 197632
            sdS: cute.struct.Align[
                cute.struct.MemRange[self.ds_dtype, cute.cosize(self.sdSt_layout)],
                128,
            ]
            # self.sdSt_layout S<3,4,3> o 0 o ((128,16),1,(4,2)):((64,1),0,(16,8192))
            # 128 * 16 * 4 * 2 = 16384 * 2 = 32768
            # 230400
            sLSE: cute.struct.Align[
                cute.struct.MemRange[self.lse_dtype, cute.cosize(self.sLSE_layout)],
                128,
            ]
            # self.sLSE_layout (128,2):(1,128)
            # 128 * 2 = 256 * 4 = 1024
            # 231424
            #sdPsum: cute.struct.Align[
            #    cute.struct.MemRange[self.dpsum_dtype, cute.cosize(self.sdPsum_layout)],
            #    128,
            #]
            # self.sdPsum_layout (128,1):(1,128)
            # 128 * 1 = 128 * 4 = 512
            # 232448
            sStartEndRowIndices: cute.struct.Align[
                cute.struct.MemRange[self.startend_row_indices_dtype, cute.cosize(self.sStartEndRowIndices_layout)],
                64,
            ]
            # sStartEndRowIndices_layout (128,4):(1,128)
            # 128 * 4 = 512 * 4 = 2048
            # 234496

        self.shared_storage = SharedStorage
        #print("self.shared_storage.size_in_bytes()", self.shared_storage.size_in_bytes())

        LOG2_E = math.log2(math.e)
        softmax_scale_log2 = softmax_scale * LOG2_E
        self.kernel(
            tma_tensor_Q,
            tma_tensor_K,
            tma_tensor_V,
            mLSE,
            mdPsum,
            tma_tensor_dO,
            mdV,
            mdK,
            mdQaccum,
            mdV_tma_tensor,
            mdK_tma_tensor,
            mdQ_semaphore,
            mdK_semaphore,
            mdV_semaphore,
            tma_atom_Q,
            tma_atom_K,
            tma_atom_V,
            tma_atom_dO,
            tma_atom_dV,
            tma_atom_dK,
            flashmask_info,
            self.sQ_layout,
            self.sQt_layout,
            self.sK_layout,
            self.sV_layout,
            self.sLSE_layout,
            self.sdPsum_layout,
            self.sdO_layout,
            self.sdOt_layout,
            self.sdSt_layout,
            self.sdS_layout,
            self.sKt_layout,
            self.sdQaccum_layout,
            self.sdKV_layout,
            self.tP_layout,
            self.tdS_layout,
            self.sStartEndRowIndices_layout,
            self.tiled_mma_S,
            self.tiled_mma_dP,
            self.tiled_mma_dV,
            self.tiled_mma_dK,
            self.tiled_mma_dQ,
            tiled_copy_r2s_dKV,
            softmax_scale,
            softmax_scale_log2,
            tile_sched_params,
        ).launch(
            grid=grid_dim,
            block=[self.threads_per_cta, 1, 1],
            cluster=self.cluster_shape_mnk if cute.size(self.cluster_shape_mnk) > 1 else None,
            smem=self.shared_storage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=1,
        )

    @cute.kernel
    def kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        mdQaccum: cute.Tensor,
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        mdQ_semaphore: Optional[cute.Tensor],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        flashmask_info: Optional[FlashMaskInfo],
        sQ_layout: cute.ComposedLayout,
        sQt_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sLSE_layout: cute.Layout,
        sdPsum_layout: cute.Layout,
        sdO_layout: cute.ComposedLayout,
        sdOt_layout: cute.ComposedLayout,
        sdSt_layout: cute.ComposedLayout,
        sdS_layout: cute.ComposedLayout,
        sKt_layout: cute.ComposedLayout,
        sdQaccum_layout: cute.Layout,
        sdKV_layout: cute.ComposedLayout | cute.Layout,
        tP_layout: cute.ComposedLayout,
        tdS_layout: cute.ComposedLayout,
        sStartEndRowIndices_layout: cute.Layout,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        tiled_copy_r2s_dKV: cute.TiledCopy,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        tile_sched_params: ParamsBase,
    ):
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())

        # Prefetch tma descriptor
        if warp_idx == self.load_warp_id:
            with cute.arch.elect_one():
                cpasync.prefetch_descriptor(tma_atom_Q)
                cpasync.prefetch_descriptor(tma_atom_K)
                cpasync.prefetch_descriptor(tma_atom_V)
                cpasync.prefetch_descriptor(tma_atom_dO)
                if const_expr(tma_atom_dV is not None):
                    cpasync.prefetch_descriptor(tma_atom_dV)
                if const_expr(tma_atom_dK is not None):
                    cpasync.prefetch_descriptor(tma_atom_dK)

        cluster_layout_vmnk = cute.tiled_divide(
            cute.make_layout(self.cluster_shape_mnk),
            (tiled_mma_S.thr_id.shape,),
        )

        # Alloc
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(self.shared_storage)

        tmem_dealloc_mbar_ptr = storage.tmem_dealloc_mbar_ptr.data_ptr()
        flashmask_loaded_mbar_ptr = storage.flashmask_loaded_mbar_ptr.data_ptr()
        dQ_cluster_full_mbar_ptr = storage.dQ_cluster_full_mbar_ptr.data_ptr()
        dQ_cluster_empty_mbar_ptr = storage.dQ_cluster_empty_mbar_ptr.data_ptr()

        if warp_idx == 1:
            cute.arch.mbarrier_init(
                tmem_dealloc_mbar_ptr, cute.arch.WARP_SIZE * len(self.compute_warp_ids)
            )
            cute.arch.mbarrier_init(
                flashmask_loaded_mbar_ptr, cute.arch.WARP_SIZE
            )
        if const_expr(self.cluster_reduce_dQ):
            if warp_idx == 4:
                for i in range(self.dQaccum_reduce_stage // 2):
                    cute.arch.mbarrier_init(dQ_cluster_full_mbar_ptr + i, 1)
                    cute.arch.mbarrier_init(dQ_cluster_empty_mbar_ptr + i, 1)

        # UMMA producers and AsyncThread consumers
        pipeline_producer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )
        # Only 1 thread per warp will signal
        pipeline_consumer_group_MMA_AsyncThread = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)
        )
        pipeline_S_P = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.S_mbar_ptr.data_ptr(),
        )
        pipeline_dP = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dP_mbar_ptr.data_ptr(),
        )
        pipeline_dKV = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=2,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread,
            barrier_storage=storage.dKV_mbar_ptr.data_ptr(),
        )
        pipeline_consumer_group_MMA_AsyncThread_dQ = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread,
            len(self.reduce_warp_ids),
        )  # Compute
        pipeline_dQ = cutlass.pipeline.PipelineUmmaAsync.create(
            num_stages=1,
            producer_group=pipeline_producer_group_MMA_AsyncThread,
            consumer_group=pipeline_consumer_group_MMA_AsyncThread_dQ,
            barrier_storage=storage.dQ_mbar_ptr.data_ptr(),
        )

        # AsyncThread producers and UMMA consumers
        # Only 1 thread per warp will signal
        pipeline_PdS_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids)
        )  # Compute
        pipeline_PdS_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id])
        )  # MMA
        pipeline_dS = cutlass.pipeline.PipelineAsyncUmma.create(
            num_stages=1,
            producer_group=pipeline_PdS_producer_group,
            consumer_group=pipeline_PdS_consumer_group,
            barrier_storage=storage.dS_mbar_ptr.data_ptr(),
        )

        # TMA producer and UMMA consumers
        pipeline_producer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.load_warp_id])
        )
        # The arrive count is the number of mcast size
        pipeline_consumer_group = cutlass.pipeline.CooperativeGroup(
            cutlass.pipeline.Agent.Thread, len([self.mma_warp_id]) * self.num_mcast_ctas_b
        )
        pipeline_consumer_group_compute = cutlass.pipeline.CooperativeGroup(
            # cutlass.pipeline.Agent.Thread, len(self.compute_warp_ids) * self.num_mcast_ctas_b
            cutlass.pipeline.Agent.Thread,
            len(self.compute_warp_ids) * 1,
        )
        pipeline_LSE = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.LSE_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["LSE"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            # init_wait=False,
        )
        pipeline_dPsum = cutlass.pipeline.PipelineTmaAsync.create(
            barrier_storage=storage.dPsum_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group_compute,
            tx_count=self.tma_copy_bytes["dPsum"],
            # cta_layout_vmnk=cluster_layout_vmnk,
            # init_wait=False,
        )
        pipeline_Q = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.Q_mbar_ptr.data_ptr(),
            num_stages=self.Q_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["Q"],
            cta_layout_vmnk=cluster_layout_vmnk,
            init_wait=False,
        )
        pipeline_dO = pipeline.PipelineTmaUmma.create(
            barrier_storage=storage.dO_mbar_ptr.data_ptr(),
            num_stages=self.dO_stage,
            producer_group=pipeline_producer_group,
            consumer_group=pipeline_consumer_group,
            tx_count=self.tma_copy_bytes["dO"],
            cta_layout_vmnk=cluster_layout_vmnk,
            init_wait=True,
        )

        sQ = storage.sQ.get_tensor(sQ_layout.outer, swizzle=sQ_layout.inner, dtype=self.q_dtype)
        sQt = cute.make_tensor(cute.recast_ptr(sQ.iterator, sQt_layout.inner, dtype=self.q_dtype), sQt_layout.outer)
        sK = storage.sK.get_tensor(sK_layout.outer, swizzle=sK_layout.inner)
        sKt = cute.make_tensor(cute.recast_ptr(sK.iterator, sKt_layout.inner), sKt_layout.outer)
        sV = storage.sV.get_tensor(sV_layout.outer, swizzle=sV_layout.inner)
        sdSt = storage.sdS.get_tensor(sdSt_layout.outer, swizzle=sdSt_layout.inner)
        sdS = cute.make_tensor(cute.recast_ptr(sdSt.iterator, sdS_layout.inner), sdS_layout.outer)
        sdO = storage.sdO.get_tensor(sdO_layout.outer, swizzle=sdO_layout.inner, dtype=self.do_dtype)
        sdOt = cute.make_tensor(cute.recast_ptr(sdO.iterator, sdOt_layout.inner, dtype=self.do_dtype), sdOt_layout.outer)
        sLSE = storage.sLSE.get_tensor(sLSE_layout)
        sdPsum = storage.sdPsum.get_tensor(sdPsum_layout)
        if const_expr(self.qhead_per_kvhead == 1):
            sdV = storage.sdO.get_tensor(
                sdKV_layout.outer, swizzle=sdKV_layout.inner, dtype=self.dv_dtype
            )
            sdK = storage.sQ.get_tensor(
                sdKV_layout.outer, swizzle=sdKV_layout.inner, dtype=self.dk_dtype
            )
        else:
            sdV = storage.sdO.get_tensor(sdKV_layout, dtype=self.dv_dtype)
            sdK = storage.sQ.get_tensor(sdKV_layout, dtype=self.dk_dtype)

        # Buffer sizing is guaranteed by max(...) in SharedStorage declarations
        # for both sQ (reused as sdK) and sdO (reused as sdV)
        sdQaccum = storage.sdQaccum.get_tensor(sdQaccum_layout)
        sStartEndRowIndices = storage.sStartEndRowIndices.get_tensor(sStartEndRowIndices_layout)
        sFM_max_min = cute.make_tensor(storage.sFM_max_min_ptr.data_ptr(), cute.make_layout((cutlass.Int32(8)), stride=(cutlass.Int32(1))))

        # TMEM
        # This is a fake tensor, by right need to retrieve tmem_ptr. But we know that we always
        # request 512 columns of tmem, so we know that it starts at 0.
        tmem_ptr = cute.make_ptr(Float32, 0, mem_space=cute.AddressSpace.tmem, assumed_align=16)
        # S
        thr_mma_S = tiled_mma_S.get_slice(0)
        Sacc_shape = thr_mma_S.partition_shape_C(self.mma_tiler_kq[:2])  # (M, N)
        tStS = thr_mma_S.make_fragment_C(Sacc_shape)
        # (MMA, MMA_M, MMA_N)
        tStS = cute.make_tensor(tmem_ptr + self.tmem_S_offset, tStS.layout)
        # dP
        thr_mma_dP = tiled_mma_dP.get_slice(0)
        dPacc_shape = thr_mma_dP.partition_shape_C(self.mma_tiler_vdo[:2])
        tdPtdP = thr_mma_dP.make_fragment_C(dPacc_shape)
        tdPtdP = cute.make_tensor(tmem_ptr + self.tmem_dP_offset, tdPtdP.layout)
        # dV
        thr_mma_dV = tiled_mma_dV.get_slice(0)
        dvacc_shape = thr_mma_dV.partition_shape_C(self.mma_tiler_pdo[:2])
        tdVtdV = thr_mma_dV.make_fragment_C(dvacc_shape)
        tdVtdV = cute.make_tensor(tmem_ptr + self.tmem_dV_offset, tdVtdV.layout)
        tP = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_P_offset, dtype=self.do_dtype), tP_layout.outer
        )
        # dK
        thr_mma_dK = tiled_mma_dK.get_slice(0)
        dkacc_shape = thr_mma_dK.partition_shape_C(self.mma_tiler_dsq[:2])
        tdKtdK = thr_mma_dK.make_fragment_C(dkacc_shape)
        tdKtdK = cute.make_tensor(tmem_ptr + self.tmem_dK_offset, tdKtdK.layout)
        tdS = cute.make_tensor(
            cute.recast_ptr(tmem_ptr + self.tmem_dS_offset, dtype=self.ds_dtype), tdS_layout.outer
        )
        # dQ
        thr_mma_dQ = tiled_mma_dQ.get_slice(0)
        dQacc_shape = thr_mma_dQ.partition_shape_C(self.mma_tiler_dsk[:2])
        tdQtdQ = thr_mma_dQ.make_fragment_C(dQacc_shape)
        tdQtdQ = cute.make_tensor(tmem_ptr + self.tmem_dQ_offset, tdQtdQ.layout)

        block_info = BlockInfo(
            self.tile_m,
            # self.tile_n,
            self.tile_n * self.cluster_shape_mnk[0],  # careful, this case is not very well-tested
            self.is_causal,
            self.is_local,
            False,  # is_split_kv
            None,
            None,
            qhead_per_kvhead_packgqa=1,
        )
        SeqlenInfoCls = partial(
            SeqlenInfoQK.create,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=None,
            mCuSeqlensK=None,
            mSeqUsedQ=None,
            mSeqUsedK=None,
        )
        TileSchedulerCls = partial(self.tile_scheduler_cls.create, tile_sched_params)

        # TODO: support local
        AttentionMaskCls = partial(
            AttentionMask,
            self.tile_m,
            self.tile_n,
            swap_AB=True,
        )

        #  EMPTY
        # (15)
        if warp_idx == self.empty_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        #  EPI
        # (14)
        if warp_idx == self.epi_warp_id:
            # currently no-op, could use for tma store/reduce
            cute.arch.warpgroup_reg_dealloc(self.num_regs_empty)

        #  LOAD
        # (13)
        if warp_idx == self.load_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)
            self.load(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                mQ,
                mK,
                mV,
                mLSE,
                mdPsum,
                mdO,
                sQ,
                sK,
                sV,
                sLSE,
                sdPsum,
                sdO,
                tma_atom_Q,
                tma_atom_K,
                tma_atom_V,
                tma_atom_dO,
                pipeline_Q,
                pipeline_dO,
                pipeline_LSE,
                pipeline_dPsum,
                cluster_layout_vmnk,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                flashmask_info,
                sStartEndRowIndices,
                sFM_max_min,
                flashmask_loaded_mbar_ptr,
                should_load_Q=True,
                should_load_dO=True,
            )

        #  MMA
        # (12)
        if warp_idx == self.mma_warp_id:
            cute.arch.warpgroup_reg_dealloc(self.num_regs_other)

            # Alloc tmem buffer
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.alloc_tmem(tmem_alloc_cols, storage.tmem_holding_buf)
            cute.arch.sync_warp()

            self.mma(
                tiled_mma_S,
                tiled_mma_dP,
                tiled_mma_dV,
                tiled_mma_dK,
                tiled_mma_dQ,
                sQ,
                sQt,
                sK,
                sV,
                sdO,
                sdOt,
                sdSt,
                sdS,
                sKt,
                tP,
                tdS,
                tStS,
                tdPtdP,
                tdVtdV,
                tdKtdK,
                tdQtdQ,
                pipeline_Q.make_consumer(),
                pipeline_dO,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                flashmask_info,
                sFM_max_min,
                flashmask_loaded_mbar_ptr,
            )
            cute.arch.relinquish_tmem_alloc_permit()
            tmem_ptr = cute.arch.retrieve_tmem_ptr(
                Float32, alignment=16, ptr_to_buffer_holding_addr=storage.tmem_holding_buf
            )

            cute.arch.mbarrier_wait(tmem_dealloc_mbar_ptr, 0)
            tmem_alloc_cols = Int32(self.tmem_alloc_cols)
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols, is_two_cta=False)

        # Compute
        # (4, 5, 6, 7, 8, 9, 10, 11) --> 8 warps
        if warp_idx >= self.compute_warp_ids[0] and warp_idx <= self.compute_warp_ids[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_compute)  # 8 warps
            self.compute_loop(
                thr_mma_S,
                thr_mma_dP,
                thr_mma_dV,
                thr_mma_dK,
                tStS,
                sLSE,
                sdPsum,
                tdVtdV,
                tdKtdK,
                mdV,
                mdK,
                sdS,
                tdPtdP,
                pipeline_LSE,
                pipeline_dPsum,
                pipeline_S_P,
                pipeline_dS,
                pipeline_dKV,
                pipeline_dP,
                softmax_scale,
                softmax_scale_log2,
                block_info,
                SeqlenInfoCls,
                AttentionMaskCls,
                TileSchedulerCls,
                sdV,
                sdK,
                mdV_tma_tensor,
                mdK_tma_tensor,
                tma_atom_dV,
                tma_atom_dK,
                tiled_copy_r2s_dKV,
                mdK_semaphore,
                mdV_semaphore,
                flashmask_info,
                sStartEndRowIndices,
                sFM_max_min,
                flashmask_loaded_mbar_ptr,
            )
            cute.arch.mbarrier_arrive(tmem_dealloc_mbar_ptr)

        # Reduce
        # (0, 1, 2, 3) - dQ
        if warp_idx >= self.reduce_warp_ids[0] and warp_idx <= self.reduce_warp_ids[-1]:
            cute.arch.warpgroup_reg_alloc(self.num_regs_reduce)
            self.dQacc_reduce(
                mdQaccum,
                sdQaccum,
                thr_mma_dQ,
                tdQtdQ,
                pipeline_dQ,
                block_info,
                SeqlenInfoCls,
                TileSchedulerCls,
                mdQ_semaphore,
                flashmask_info,
                sFM_max_min,
                flashmask_loaded_mbar_ptr,
            )

        return

    @cute.jit
    def load(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mLSE: cute.Tensor,
        mdPsum: cute.Tensor,
        mdO: cute.Tensor,
        sQ: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        sdO: cute.Tensor,
        tma_atom_Q: cute.CopyAtom,
        tma_atom_K: cute.CopyAtom,
        tma_atom_V: cute.CopyAtom,
        tma_atom_dO: cute.CopyAtom,
        pipeline_Q: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        cluster_layout_vmnk: cute.Layout,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        flashmask_info: FlashMaskInfo,
        sStartEndRowIndices: cute.Tensor,
        sFM_max_min: cute.Tensor,
        flashmask_loaded_mbar_ptr: cute.Pointer,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
    ):
        num_load_threads = cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads

        producer_state_Q_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.Q_stage
        )
        producer_state_dO_dPsum = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Producer, self.dO_stage
        )

        # Compute multicast mask for Q & dO buffer full
        cta_rank_in_cluster = cute.arch.make_warp_uniform(cute.arch.block_idx_in_cluster())
        block_in_cluster_coord_vmnk = cluster_layout_vmnk.get_flat_coord(cta_rank_in_cluster)
        q_do_mcast_mask = None
        if const_expr(self.is_q_do_mcast):
            q_do_mcast_mask = cpasync.create_tma_multicast_mask(
                cluster_layout_vmnk, block_in_cluster_coord_vmnk, mcast_mode=1
            )

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            head_idx_kv = head_idx // self.qhead_per_kvhead
            mQ_cur = mQ[None, None, head_idx, batch_idx]
            mK_cur = mK[None, None, head_idx_kv, batch_idx]
            mV_cur = mV[None, None, head_idx_kv, batch_idx]
            mdO_cur = mdO[None, None, head_idx, batch_idx]
            mLSE_cur = mLSE[None, head_idx, batch_idx]
            mPsum_cur = mdPsum[None, head_idx, batch_idx]

            gK = cute.local_tile(mK_cur, cute.select(self.mma_tiler_kq, mode=[0, 2]), (n_block, 0))
            tSgK = thr_mma_S.partition_A(gK)
            gV = cute.local_tile(mV_cur, cute.select(self.mma_tiler_vdo, mode=[0, 2]), (n_block, 0))
            tdPgV = thr_mma_dP.partition_A(gV)
            gQ = cute.local_tile(mQ_cur, cute.select(self.mma_tiler_kq, mode=[1, 2]), (None, 0))
            tSgQ = thr_mma_S.partition_B(gQ)
            gLSE = cute.local_tile(mLSE_cur, (self.tile_n,), (None,))
            gdPsum = cute.local_tile(mPsum_cur, (self.tile_n,), (None,))
            gdO = cute.local_tile(mdO_cur, cute.select(self.mma_tiler_pdo, mode=[1, 2]), (0, None))
            tdPgdO = thr_mma_dV.partition_B(gdO)

            load_K, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_K, 0, cute.make_layout(1), tSgK, sK, single_stage=True
            )
            load_V, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_V,
                0,
                cute.make_layout(1),
                tdPgV,
                sV,
                single_stage=True,
            )
            b_cta_layout = cute.make_layout(cute.slice_(cluster_layout_vmnk, (0, None, 0, 0)).shape)
            load_Q, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_Q,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tSgQ,
                dst_tensor=sQ,
                mcast_mask=q_do_mcast_mask,
            )
            load_Q = copy_utils.tma_producer_copy_fn(load_Q, pipeline_Q)
            load_dO, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_dO,
                cta_coord=block_in_cluster_coord_vmnk[1],
                cta_layout=b_cta_layout,
                src_tensor=tdPgdO,
                dst_tensor=sdO,
                mcast_mask=q_do_mcast_mask,
            )
            load_dO = copy_utils.tma_producer_copy_fn(load_dO, pipeline_dO)
            copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SOp(), Float32)
            copy_stats = partial(cute.copy, copy_atom_stats)
            # copy_atom_stats = cute.make_copy_atom(cpasync.CopyBulkG2SMulticastOp(), Float32)
            # sLSE = cute.logical_divide(sLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gLSE = cute.logical_divide(gLSE, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # sdPsum = cute.logical_divide(sdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # gdPsum = cute.logical_divide(gdPsum, (64,))[(None, block_in_cluster_coord_vmnk[1]), None]
            # copy_stats = partial(cute.copy, copy_atom_stats, mcast_mask=q_do_mcast_mask)


            load_step = partial(
                self.load_step,
                gLSE=gLSE,
                sLSE=sLSE,
                gdPsum=gdPsum,
                sdPsum=sdPsum,
                pipeline_Q=pipeline_Q,
                pipeline_LSE=pipeline_LSE,
                pipeline_dO=pipeline_dO,
                pipeline_dPsum=pipeline_dPsum,
                load_Q=load_Q,
                load_dO=load_dO,
                copy_stats=copy_stats,
                should_load_Q=should_load_Q,
                should_load_dO=should_load_dO,
            )

            if const_expr(self.enable_flashmask):
                self.load_fm(flashmask_info, sStartEndRowIndices, sFM_max_min, seqlen, mQ.shape[2], n_block, head_idx, batch_idx)
                cute.arch.mbarrier_arrive(flashmask_loaded_mbar_ptr)

                zero_block = False
                prefetch_m_block = m_block_min
                prefetch_lte = False
                if const_expr(not self.is_causal):
                    has_uts = const_expr(flashmask_info.UTS_nblock_max is not None)
                    if not has_uts or prefetch_m_block > sFM_max_min[4]:
                        prefetch_m_block = sFM_max_min[7]
                if prefetch_m_block > sFM_max_min[0]:
                    has_lte = const_expr(flashmask_info.LTE_nblock_max is not None)
                    if has_lte:
                        prefetch_m_block = max(m_block_min, sFM_max_min[3])
                        prefetch_lte = True
                    else:
                        # masked whole n_block
                        prefetch_m_block = m_block_max
                if prefetch_m_block >= m_block_max:
                    zero_block = True

                # First iteration: load K together w Q & LSE, then V together w dO & dPsum
                if not zero_block and should_load_Q:
                    # K & Q
                    pipeline_Q.producer_acquire(
                        producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"]
                    )
                    load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                    load_Q(prefetch_m_block, producer_state=producer_state_Q_LSE)
                    pipeline_Q.producer_commit(producer_state_Q_LSE)
                    # LSE
                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                    with cute.arch.elect_one():
                        copy_stats(
                            gLSE[None, prefetch_m_block],
                            sLSE[None, producer_state_Q_LSE.index],
                            mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                        )
                    producer_state_Q_LSE.advance()
                if not zero_block and should_load_dO:
                    if tidx == 0 and self.debug_print:
                        cute.printf('n_block: %d, before load_step prefetch_m_block: %d', n_block, prefetch_m_block)
                    # V & dO
                    pipeline_dO.producer_acquire(
                        producer_state_dO_dPsum, extra_tx_count=self.tma_copy_bytes["V"]
                    )
                    load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
                    load_dO(prefetch_m_block, producer_state=producer_state_dO_dPsum)
                    pipeline_dO.producer_commit(producer_state_dO_dPsum)
                    # dPsum
                    pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                    with cute.arch.elect_one():
                        copy_stats(
                            gdPsum[None, prefetch_m_block],
                            sdPsum[None, producer_state_dO_dPsum.index],
                            mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                        )
                    producer_state_dO_dPsum.advance()
                    if tidx == 0 and self.debug_print:
                        cute.printf('n_block: %d, after load_step prefetch_m_block: %d', n_block, prefetch_m_block)

                if not zero_block:
                    loop_start = m_block_min
                    loop_end = m_block_max
                    if const_expr(not self.is_causal):
                        has_uts = const_expr(flashmask_info.UTS_nblock_max is not None)
                        if has_uts and prefetch_m_block <= sFM_max_min[4]:
                            loop_end = sFM_max_min[4] + 1
                            # 0 ~ UTS
                            for m_block in cutlass.range(loop_start + 1, loop_end, unroll=1):
                                if tidx == 0 and self.debug_print:
                                    cute.printf('n_block: %d, before load_step 0 ~ UTS: %d', n_block, m_block)
                                producer_state_Q_LSE, producer_state_dO_dPsum = load_step(
                                    m_block,
                                    producer_state_Q_LSE=producer_state_Q_LSE,
                                    producer_state_dO_dPsum=producer_state_dO_dPsum,
                                )
                                if tidx == 0 and self.debug_print:
                                    cute.printf('n_block: %d, after load_step 0 ~ UTS: %d', n_block, m_block)
                            # Subtract 1 beforehand to use loop_start + 1 uniformly in the for loop.
                            loop_start = sFM_max_min[7] - 1
                        else:
                            loop_start = sFM_max_min[7]

                    # UTE ~ LTS
                    #loop_end = m_block_max if m_block_max < sFM_max_min[0] + 1 else sFM_max_min[0] + 1
                    loop_end = min(m_block_max, sFM_max_min[0] + 1)
                    for m_block in cutlass.range(loop_start + 1, loop_end, unroll=1):
                        if tidx == 0 and self.debug_print:
                            cute.printf('n_block: %d, before load_step UTE ~ LTS: %d', n_block, m_block)
                        producer_state_Q_LSE, producer_state_dO_dPsum = load_step(
                            m_block,
                            producer_state_Q_LSE=producer_state_Q_LSE,
                            producer_state_dO_dPsum=producer_state_dO_dPsum,
                        )
                        if tidx == 0 and self.debug_print:
                            cute.printf('n_block: %d, after load_step UTE ~ LTS: %d', n_block, m_block)

                    # LTE ~ seqlen_q
                    has_lte = const_expr(flashmask_info.LTE_nblock_max is not None)
                    if has_lte:
                        loop_start = max(sFM_max_min[0], sFM_max_min[3])
                        #if prefetch_m_block == sFM_max_min[3]:
                        if not prefetch_lte and sFM_max_min[3] > sFM_max_min[0]:
                            # Subtract 1 beforehand to use loop_start + 1 uniformly in the for loop.
                            loop_start = sFM_max_min[3] - 1
                        loop_start = max(m_block_min, loop_start)

                        loop_end = m_block_max
                        #cute.printf('>>>>>>>>>>>>>>n_block: %d, loop_start: %d, load_step: %d, m_block_max: %d', n_block, loop_start, loop_end, m_block_max)
                        for m_block in cutlass.range(loop_start + 1, loop_end, unroll=1):
                            if tidx == 0 and self.debug_print:
                                cute.printf('n_block: %d, before load_step LTE ~ seqlen_q: %d', n_block, m_block)
                            producer_state_Q_LSE, producer_state_dO_dPsum = load_step(
                                m_block,
                                producer_state_Q_LSE=producer_state_Q_LSE,
                                producer_state_dO_dPsum=producer_state_dO_dPsum,
                            )
                            if tidx == 0 and self.debug_print:
                                cute.printf('n_block: %d, after load_step LTE ~ seqlen_q: %d', n_block, m_block)

                    if not zero_block and should_load_Q:
                        pipeline_Q.producer_tail(
                            producer_state_Q_LSE.clone()
                        )  # will hang if we don't clone
                        pipeline_LSE.producer_tail(producer_state_Q_LSE)
                    if not zero_block and should_load_dO:
                        pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                        pipeline_dPsum.producer_tail(producer_state_dO_dPsum)
                    
            else:
                # First iteration: load K together w Q & LSE, then V together w dO & dPsum
                if const_expr(should_load_Q):
                    # K & Q
                    pipeline_Q.producer_acquire(
                        producer_state_Q_LSE, extra_tx_count=self.tma_copy_bytes["K"]
                    )
                    load_K(tma_bar_ptr=pipeline_Q.producer_get_barrier(producer_state_Q_LSE))
                    load_Q(m_block_min, producer_state=producer_state_Q_LSE)
                    pipeline_Q.producer_commit(producer_state_Q_LSE)
                    # LSE
                    pipeline_LSE.producer_acquire(producer_state_Q_LSE)
                    with cute.arch.elect_one():
                        copy_stats(
                            gLSE[None, m_block_min],
                            sLSE[None, producer_state_Q_LSE.index],
                            mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                        )
                    producer_state_Q_LSE.advance()
                if const_expr(should_load_dO):
                    # V & dO
                    pipeline_dO.producer_acquire(
                        producer_state_dO_dPsum, extra_tx_count=self.tma_copy_bytes["V"]
                    )
                    load_V(tma_bar_ptr=pipeline_dO.producer_get_barrier(producer_state_dO_dPsum))
                    load_dO(m_block_min, producer_state=producer_state_dO_dPsum)
                    pipeline_dO.producer_commit(producer_state_dO_dPsum)
                    # dPsum
                    pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
                    with cute.arch.elect_one():
                        copy_stats(
                            gdPsum[None, m_block_min],
                            sdPsum[None, producer_state_dO_dPsum.index],
                            mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                        )
                    producer_state_dO_dPsum.advance()

                for m_block in cutlass.range(m_block_min + 1, m_block_max, unroll=1):
                    producer_state_Q_LSE, producer_state_dO_dPsum = load_step(
                        m_block,
                        producer_state_Q_LSE=producer_state_Q_LSE,
                        producer_state_dO_dPsum=producer_state_dO_dPsum,
                    )

                if const_expr(should_load_Q):
                    pipeline_Q.producer_tail(
                        producer_state_Q_LSE.clone()
                    )  # will hang if we don't clone
                    pipeline_LSE.producer_tail(producer_state_Q_LSE)
                if const_expr(should_load_dO):
                    pipeline_dO.producer_tail(producer_state_dO_dPsum.clone())
                    pipeline_dPsum.producer_tail(producer_state_dO_dPsum)

            if tidx == 0 and self.debug_print:
                cute.printf('n_block: %d, EEEEEEEEEEEEEEEEEEEE after load_step EEEEEEEEEEEEEEEEEEEE', n_block)
            tile_scheduler.prefetch_next_work()
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def load_step(
        self,
        m_block: cute.Int32,
        gLSE: cute.Tensor,
        sLSE: cute.Tensor,
        gdPsum: cute.Tensor,
        sdPsum: cute.Tensor,
        pipeline_Q: PipelineAsync,
        pipeline_LSE: PipelineAsync,
        pipeline_dO: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        producer_state_Q_LSE: cutlass.pipeline.PipelineState,
        producer_state_dO_dPsum: cutlass.pipeline.PipelineState,
        load_Q: Callable,
        load_dO: Callable,
        copy_stats: Callable,
        should_load_Q: bool = True,
        should_load_dO: bool = True,
    ):
        if const_expr(should_load_Q):
            # Q
            pipeline_Q.producer_acquire(producer_state_Q_LSE)
            load_Q(m_block, producer_state=producer_state_Q_LSE)
            pipeline_Q.producer_commit(producer_state_Q_LSE)
            # LSE
            pipeline_LSE.producer_acquire(producer_state_Q_LSE)
            with cute.arch.elect_one():
                copy_stats(
                    gLSE[None, m_block],
                    sLSE[None, producer_state_Q_LSE.index],
                    mbar_ptr=pipeline_LSE.producer_get_barrier(producer_state_Q_LSE),
                )
            producer_state_Q_LSE.advance()
        if const_expr(should_load_dO):
            # dO
            pipeline_dO.producer_acquire(producer_state_dO_dPsum)
            load_dO(m_block, producer_state=producer_state_dO_dPsum)
            pipeline_dO.producer_commit(producer_state_dO_dPsum)
            # dPsum
            pipeline_dPsum.producer_acquire(producer_state_dO_dPsum)
            with cute.arch.elect_one():
                copy_stats(
                    gdPsum[None, m_block],
                    sdPsum[None, producer_state_dO_dPsum.index],
                    mbar_ptr=pipeline_dPsum.producer_get_barrier(producer_state_dO_dPsum),
                )
            producer_state_dO_dPsum.advance()

        return producer_state_Q_LSE, producer_state_dO_dPsum

    @cute.jit
    def load_fm(
        self,
        flashmask_info: FlashMaskInfo,
        sStartEndRowIndices: cute.Tensor,
        sFM_max_min: cute.Tensor,
        seqlen_info: SeqlenInfoQK,
        num_heads: Int32,
        n_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        # (13) warp_idx == self.load_warp_id
        #num_load_threads = len([self.load_warp_id]) * cute.arch.WARP_SIZE
        num_load_threads = cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        nblock_seqlen = ((seqlen_info.seqlen_k + self.tile_n - 1) // self.tile_n + 3) // 4 * 4
        ntimes_copy = (self.tile_n + num_load_threads - 1) // num_load_threads
        bsz, fm_heads, seqlen_k, num_vec = flashmask_info.startend_row_indices.shape
        fm_batch_idx = batch_idx if bsz > 1 else 0
        fm_head_idx = head_idx // (num_heads // fm_heads)
        bh_offset = fm_batch_idx * fm_heads + fm_head_idx;
        bh_offset_block = bh_offset * nblock_seqlen;

        if tidx == 0:
            # LTS is always valid, otherwise this is not a valid flashmask computation instance
            LTS_nblock_max = cute.make_tensor(flashmask_info.LTS_nblock_max.iterator + bh_offset_block, cute.make_layout((cutlass.Int32(nblock_seqlen)), stride=(cutlass.Int32(1))))
            LTS_nblock_min = cute.make_tensor(flashmask_info.LTS_nblock_min.iterator + bh_offset_block, cute.make_layout((cutlass.Int32(nblock_seqlen)), stride=(cutlass.Int32(1))))
            sFM_max_min[0] = (LTS_nblock_max[n_block] - 1) // self.tile_m
            sFM_max_min[1] = LTS_nblock_min[n_block] // self.tile_m
            if const_expr(flashmask_info.LTE_nblock_max is not None):
                LTE_nblock_max = cute.make_tensor(flashmask_info.LTE_nblock_max.iterator + bh_offset_block, cute.make_layout((cutlass.Int32(nblock_seqlen)), stride=(cutlass.Int32(1))))
                LTE_nblock_min = cute.make_tensor(flashmask_info.LTE_nblock_min.iterator + bh_offset_block, cute.make_layout((cutlass.Int32(nblock_seqlen)), stride=(cutlass.Int32(1))))
                sFM_max_min[2] = (LTE_nblock_max[n_block] - 1) // self.tile_m
                sFM_max_min[3] = LTE_nblock_min[n_block] // self.tile_m
            if const_expr(flashmask_info.UTS_nblock_max is not None):
                UTS_nblock_max = cute.make_tensor(flashmask_info.UTS_nblock_max.iterator + bh_offset_block, cute.make_layout((cutlass.Int32(nblock_seqlen)), stride=(cutlass.Int32(1))))
                UTS_nblock_min = cute.make_tensor(flashmask_info.UTS_nblock_min.iterator + bh_offset_block, cute.make_layout((cutlass.Int32(nblock_seqlen)), stride=(cutlass.Int32(1))))
                sFM_max_min[4] = (UTS_nblock_max[n_block] - 1) // self.tile_m
                sFM_max_min[5] = UTS_nblock_min[n_block] // self.tile_m
            if const_expr(flashmask_info.UTE_nblock_max is not None):
                UTE_nblock_max = cute.make_tensor(flashmask_info.UTE_nblock_max.iterator + bh_offset_block, cute.make_layout((cutlass.Int32(nblock_seqlen)), stride=(cutlass.Int32(1))))
                UTE_nblock_min = cute.make_tensor(flashmask_info.UTE_nblock_min.iterator + bh_offset_block, cute.make_layout((cutlass.Int32(nblock_seqlen)), stride=(cutlass.Int32(1))))
                sFM_max_min[6] = (UTE_nblock_max[n_block] - 1) // self.tile_m
                sFM_max_min[7] = UTE_nblock_min[n_block] // self.tile_m

        for i in cutlass.range_constexpr(ntimes_copy):
            copy_offset = i * num_load_threads + tidx
            sStartEndRowIndices[copy_offset, 0] = 2147483647
            sStartEndRowIndices[copy_offset, 1] = 2147483647
            if (copy_offset < self.tile_n and n_block * self.tile_n + copy_offset < seqlen_k):
                LTS = flashmask_info.startend_row_indices[fm_batch_idx, fm_head_idx, None, 0]
                sStartEndRowIndices[copy_offset, 0] = LTS[n_block * self.tile_n + copy_offset]
                #assert const_expr(num_vec <= 2), "only support num_vec == 2 now"
                if const_expr(flashmask_info.LTE_nblock_max is not None):
                    LTE = flashmask_info.startend_row_indices[fm_batch_idx, fm_head_idx, None, 1]
                    sStartEndRowIndices[copy_offset, 1] = LTE[n_block * self.tile_n + copy_offset]
                if const_expr(flashmask_info.UTE_nblock_max is not None):
                    UTE = flashmask_info.startend_row_indices[fm_batch_idx, fm_head_idx, None, 1]
                    sStartEndRowIndices[copy_offset, 1] = UTE[n_block * self.tile_n + copy_offset]
                #cute.printf("%d, %d", copy_offset, sStartEndRowIndices[copy_offset, 0])
                #cute.print_tensor(LTS)
        cute.arch.sync_warp()

    @cute.jit
    def mma(
        self,
        tiled_mma_S: cute.TiledMma,
        tiled_mma_dP: cute.TiledMma,
        tiled_mma_dV: cute.TiledMma,
        tiled_mma_dK: cute.TiledMma,
        tiled_mma_dQ: cute.TiledMma,
        sQ: cute.Tensor,
        sQt: cute.Tensor,
        sK: cute.Tensor,
        sV: cute.Tensor,
        sdO: cute.Tensor,
        sdOt: cute.Tensor,
        sdSt: cute.Tensor,
        sdS: cute.Tensor,
        sKt: cute.Tensor,
        tP: cute.Tensor,
        tdS: cute.Tensor,
        tStS: cute.Tensor,
        tdPtdP: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        tdQtdQ: cute.Tensor,
        pipeline_Q_consumer: PipelineConsumer,
        pipeline_dO: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        flashmask_info: FlashMaskInfo,
        sFM_max_min: cute.Tensor,
        flashmask_loaded_mbar_ptr: cute.Pointer,
    ):
        # [2025-10-21] For reasons I don't understand, putting these partitioning in the main
        # kernel (before warp specialization) is a lot slower tha putting them here.
        # Partition smem / tmem tensors
        # S = K @ Q.T
        num_load_threads = cute.arch.WARP_SIZE
        tidx = cute.arch.thread_idx()[0] % num_load_threads
        tSrK = tiled_mma_S.make_fragment_A(sK)
        tSrQ = tiled_mma_S.make_fragment_B(sQ)
        # dP = V @ dO.T
        tdPrV = tiled_mma_dP.make_fragment_A(sV)
        tdPrdOt = tiled_mma_dP.make_fragment_B(sdOt)
        # dK = dS.T @ Q
        if const_expr(self.use_smem_dS_for_mma_dK):
            tdKrdS = tiled_mma_dK.make_fragment_A(sdSt)
        else:
            tdKrdS = tiled_mma_dK.make_fragment_A(tdS)
        tdKrQ = tiled_mma_dK.make_fragment_B(sQt)
        # dQ = dS @ K
        tdQrdS = tiled_mma_dQ.make_fragment_A(sdS)
        tdQrK = tiled_mma_dQ.make_fragment_B(sKt)
        # dV = P @ dO.T
        tdVrdO = tiled_mma_dV.make_fragment_B(sdO)
        tdVrP = tiled_mma_dV.make_fragment_A(tP)

        # mma_qk_fn = partial(gemm_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, zero_init=True)
        mma_qk_fn = partial(
            gemm_ptx_w_idx, tiled_mma_S, tStS, tSrK, tSrQ, sA=sK, sB=sQ, zero_init=True
        )
        # mma_dov_fn = partial(gemm_w_idx, tiled_mma_dP, tdPtdP, tdPrV, tdPrdOt, zero_init=True)
        mma_dov_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dP,
            tdPtdP,
            tdPrV,
            tdPrdOt,
            sA=sV,
            sB=sdOt,
            zero_init=True,
        )
        # mma_pdo_fn = partial(gemm_w_idx, tiled_mma_dV, tdVtdV, tdVrP, tdVrdO)
        mma_pdo_fn = partial(
            gemm_ptx_w_idx,
            tiled_mma_dV,
            tdVtdV,
            tdVrP,
            tdVrdO,
            sA=None,
            sB=sdO,
            tA_addr=self.tmem_P_offset,
        )
        mma_dsk_fn = partial(gemm_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, zero_init=True)
        # mma_dsk_fn = partial(
        #     gemm_ptx_w_idx, tiled_mma_dQ, tdQtdQ, tdQrdS, tdQrK, sA=sdS, sB=sKt, zero_init=True
        # )
        if const_expr(self.use_smem_dS_for_mma_dK):
            mma_dsq_fn = partial(gemm_w_idx, tiled_mma_dK, tdKtdK, tdKrdS, tdKrQ)
        else:
            # Need to explicitly pass in tA_addr for correctness
            mma_dsq_fn = partial(
                gemm_ptx_w_idx,
                tiled_mma_dK,
                tdKtdK,
                tdKrdS,
                tdKrQ,
                sA=None,
                sB=sQt,
                tA_addr=self.tmem_dS_offset,
            )

        consumer_state_dO = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        producer_phase_acc = Int32(1)  # For S & P, dP, dQ
        consumer_state_dS = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # producer_state_dKV = cutlass.pipeline.make_pipeline_state(
        #     cutlass.pipeline.PipelineUserType.Producer, 2
        # )
        producer_phase_dKV = Int32(1)
        cta_group = pipeline_S_P.cta_group

        if const_expr(self.enable_flashmask):
            flashmask_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)  # must be seqlen_k
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )

            num_blocks = m_block_max - m_block_min
            if const_expr(self.enable_flashmask):
                cute.arch.mbarrier_wait(flashmask_loaded_mbar_ptr, flashmask_phase)

                num_blocks = 0
                loop_start = m_block_min
                loop_end = m_block_max
                if const_expr(not self.is_causal):
                    has_uts = const_expr(flashmask_info.UTS_nblock_max is not None)
                    if has_uts:
                        loop_end = min(m_block_max, sFM_max_min[4] + 1)
                        #  ~ UTS
                        num_blocks = num_blocks + max(0, (loop_end - loop_start))
                        if tidx == 0 and self.debug_print:
                            cute.printf('after uts mma: n_block: %d, %d', n_block, num_blocks)
                    loop_start = sFM_max_min[7]

                # UTE ~ LTS
                #loop_end = m_block_max if m_block_max < sFM_max_min[0] + 1 else sFM_max_min[0] + 1
                loop_end = min(m_block_max, sFM_max_min[0] + 1)
                num_blocks = num_blocks + max(0, (loop_end - loop_start))
                if tidx == 0 and self.debug_print:
                    cute.printf('after ute ~ lts mma: n_block: %d, %d, m_block_min: %d, m_block_max: %d', n_block, num_blocks, m_block_min, m_block_max)

                # LTE ~ seqlen_q
                has_lte = const_expr(flashmask_info.LTE_nblock_max is not None)
                if has_lte:
                    loop_start = max(sFM_max_min[0] + 1, sFM_max_min[3])
                    if sFM_max_min[3] == sFM_max_min[0]:
                        loop_start = sFM_max_min[3] + 1
                    loop_start = max(m_block_min, loop_start)
                    loop_end = m_block_max
                    num_blocks = num_blocks + (loop_end - loop_start)
                    if tidx == 0 and self.debug_print:
                        cute.printf('after lts ~ seqlen_q mma: n_block: %d, %d', n_block, num_blocks)

            if num_blocks > 0:
                accumulate_dK = False
                # -----------------------------------------------------------
                ###### Prologue
                # -----------------------------------------------------------
                # 1. S  = Q0 @ K.T
                # 2. dP = V @ dO.T
                # 3. dV = P @ dO

                # 1) S  = Q0 @ K.T
                m_block_cur = cute.Int32(0)
                if tidx == 0 and self.debug_print:
                    cute.printf('n_block: %d, before mma_step: %d', n_block, m_block_cur)
                handle_Q = pipeline_Q_consumer.wait_and_advance()
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_qk_fn(B_idx=handle_Q.index)
                # Don't release Q yet
                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                # 2) dP = V @ dO.T
                pipeline_dO.consumer_wait(consumer_state_dO)
                pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)
                # dQ uses the same tmem as dP
                pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                mma_dov_fn(B_idx=consumer_state_dO.index)
                # Don't release dO yet
                pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                producer_phase_acc ^= 1
                # 3) dV = P.T @ dO
                # wait for P to be ready, which uses the same tmem as S
                pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=True)
                pipeline_dO.consumer_release(consumer_state_dO)
                consumer_state_dO.advance()
                if tidx == 0 and self.debug_print:
                    cute.printf('n_block: %d, after mma_step: %d', n_block, m_block_cur)
                # -----------------------------------------------------------
                ###### MAIN LOOP
                # -----------------------------------------------------------
                # 1. S  = K    @ Q.T
                # 2. dQ = dS   @ K
                # 3. dK = dS.T @ Q
                # 4. dP = V    @ dO.T
                # 5. dV = P.T  @ dO
                num_blocks = num_blocks - 1

                for m_block in cutlass.range(0, num_blocks, unroll=1):
                    m_block_cur = m_block_cur + 1
                    if tidx == 0 and self.debug_print:
                        cute.printf('n_block: %d, before mma_step: %d', n_block, m_block_cur)

                    # 1) S = K @ Q_i
                    handle_Q_next = pipeline_Q_consumer.wait_and_advance()
                    # Don't need to wait for S, as P must have been ready ealier, i.e., S is ready
                    mma_qk_fn(B_idx=handle_Q_next.index)
                    pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                    # 2-3)
                    # Do dK = dS.T @ Q, then dQ = dS @ K if dS in tmem for first mma
                    # Otherwise, reverse order
                    pipeline_dS.consumer_wait(consumer_state_dS)

                    if const_expr(self.use_smem_dS_for_mma_dK):
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                        mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                        accumulate_dK = True
                        handle_Q.release()
                    else:
                        mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                        accumulate_dK = True
                        handle_Q.release()
                        mma_dsk_fn()
                        pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)

                    # dP uses the same tmem as dQ
                    # However, if dS is ready, then dP must have been ready,
                    # so we don't need this wait before mma_dsk_fn()
                    # pipeline_dP.sync_object_empty.wait(0, producer_phase_acc)

                    pipeline_dS.consumer_release(consumer_state_dS)
                    consumer_state_dS.advance()

                    # 4) dP = V @ dO.T
                    pipeline_dO.consumer_wait(consumer_state_dO)
                    # dQ uses the same tmem as dP
                    pipeline_dQ.sync_object_empty.wait(0, producer_phase_acc)
                    mma_dov_fn(B_idx=consumer_state_dO.index)
                    pipeline_dP.sync_object_full.arrive(0, pipeline_dP.producer_mask, cta_group)

                    producer_phase_acc ^= 1
                    # 5) dV += P @ dO
                    # wait for P to be ready, which uses the same tmem as S
                    pipeline_S_P.sync_object_empty.wait(0, producer_phase_acc)
                    mma_pdo_fn(B_idx=consumer_state_dO.index, zero_init=False)
                    pipeline_dO.consumer_release(consumer_state_dO)
                    consumer_state_dO.advance()

                    handle_Q = handle_Q_next

                    if tidx == 0 and self.debug_print:
                        cute.printf('n_block: %d, after mma_step: %d', n_block, m_block_cur)

                pipeline_S_P.sync_object_full.arrive(0, pipeline_S_P.producer_mask, cta_group)

                # signal to the epilogue that dV is ready
                # pipeline_dKV.producer_acquire(producer_state_dKV)
                pipeline_dKV.sync_object_empty.wait(0, producer_phase_dKV)
                # pipeline_dKV.producer_commit(producer_state_dKV)
                pipeline_dKV.sync_object_full.arrive(0, pipeline_dKV.producer_mask, cta_group)
                # producer_state_dKV.advance()
                # pipeline_dKV.producer_acquire(producer_state_dKV)
                pipeline_dKV.sync_object_empty.wait(1, producer_phase_dKV)

                # -----------------------------------------------------------
                ###### Remaining 2
                # -----------------------------------------------------------
                # 1) dK += dS.T @ Q
                pipeline_dS.consumer_wait(consumer_state_dS)
                mma_dsq_fn(B_idx=handle_Q.index, zero_init=not accumulate_dK)
                # signal to the epilogue that dK is ready
                # pipeline_dKV.producer_commit(producer_state_dKV)
                pipeline_dKV.sync_object_full.arrive(1, pipeline_dKV.producer_mask, cta_group)
                # producer_state_dKV.advance()
                producer_phase_dKV ^= 1

                # 2) dQ = dS @ K
                # dS is done, so dP must have been ready, we don't need to wait
                mma_dsk_fn()
                pipeline_dQ.sync_object_full.arrive(0, pipeline_dQ.producer_mask, cta_group)
                # Wait until dQ is done before releasing Q, since K and Q0 uses the same mbarrier
                handle_Q.release()
                pipeline_dS.consumer_release(consumer_state_dS)
                consumer_state_dS.advance()

                producer_phase_acc ^= 1

            if const_expr(self.enable_flashmask):
                flashmask_phase ^= 1

            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

            if tidx == 0 and self.debug_print:
                cute.printf('n_block: %d, EEEEEEEEEEEEEEEEEEEE after mma EEEEEEEEEEEEEEEEEEEE', n_block)

        # Currently it hangs if we have this S_P.producer_tail, will need to understand why
        # pipeline_S_P.producer_tail(producer_state_S_P)
        # pipeline_dP.producer_tail(producer_state_dP)
        # pipeline_dKV.producer_tail(producer_state_dKV)
        # pipeline_dQ.producer_tail(producer_state_dQ)

    @cute.jit
    def split_wg(
        self,
        t: cute.Tensor,
        wg_idx: cutlass.Int32,
        num_wg: cutlass.Constexpr[int],
    ):
        reduced_shape = cute.product_each(t.shape)
        rank = len(reduced_shape)
        if const_expr(reduced_shape[1] > 1):
            assert rank >= 2, "Need rank >= 2 for t in split_wg"
            t = cute.logical_divide(t, (reduced_shape[0], reduced_shape[1] // num_wg))
            coord = (None, (None, wg_idx)) + (None,) * (rank - 2)
        else:
            assert rank >= 3, "Need rank >= 3 for t in split_wg"
            if const_expr(rank == 3):
                t = cute.logical_divide(
                    t, (reduced_shape[0], reduced_shape[1], reduced_shape[2] // num_wg)
                )
                coord = (
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 3)
            else:
                t = cute.logical_divide(
                    t,
                    (
                        reduced_shape[0],
                        reduced_shape[1],
                        reduced_shape[2],
                        reduced_shape[3] // num_wg,
                    ),
                )
                coord = (
                    None,
                    None,
                    None,
                    (None, wg_idx),
                ) + (None,) * (rank - 4)
        return t[coord]

    @cute.jit
    def compute_loop(
        self,
        thr_mma_S: cute.core.ThrMma,
        thr_mma_dP: cute.core.ThrMma,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        tStS: cute.Tensor,
        sLSE: cute.Tensor,
        sdPsum: cute.Tensor,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        sdS: cute.Tensor,
        tdPtdP: cute.Tensor,
        pipeline_LSE: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dS: PipelineAsync,
        pipeline_dKV: PipelineAsync,
        pipeline_dP: PipelineAsync,
        softmax_scale: cutlass.Float32,
        softmax_scale_log2: cutlass.Float32,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        AttentionMaskCls: Callable,
        TileSchedulerCls: Callable,
        sdV: Optional[cute.Tensor],
        sdK: Optional[cute.Tensor],
        mdV_tma_tensor: Optional[cute.Tensor],
        mdK_tma_tensor: Optional[cute.Tensor],
        tma_atom_dV: Optional[cute.CopyAtom],
        tma_atom_dK: Optional[cute.CopyAtom],
        tiled_copy_r2s_dKV: Optional[cute.TiledCopy],
        mdK_semaphore: Optional[cute.Tensor],
        mdV_semaphore: Optional[cute.Tensor],
        flashmask_info: FlashMaskInfo,
        sStartEndRowIndices: cute.Tensor,
        sFM_max_min: cute.Tensor,
        flashmask_loaded_mbar_ptr: cute.Pointer,
    ):
        sLSE_2D = cute.make_tensor(
            sLSE.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.Q_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        sdPsum_2D = cute.make_tensor(
            sdPsum.iterator,
            cute.make_layout(
                (self.tile_m, self.tile_n, self.dO_stage),
                stride=(1, 0, cute.round_up(self.tile_m, 64)),
            ),
        )
        # if const_expr(self.SdP_swapAB):
        if const_expr(True):
            sLSE_2D = utils.transpose_view(sLSE_2D)
            sdPsum_2D = utils.transpose_view(sdPsum_2D)

        # tix: [128...384]  8 warps
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())  # 4-11
        tidx = cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        # tidx = cute.arch.thread_idx()[0] - (cute.arch.WARP_SIZE * self.compute_warp_ids[0])
        dp_idx = tidx % 128
        num_wg = len(self.compute_warp_ids) // 4  # 2
        # wg_idx:
        # 0: [256...384]
        # 1: [128...256]

        tileP_f32_like = self.mma_tiler_kq[0] // 32 * self.v_dtype.width  # (128, 64)
        # tStS has shape ((128, 128), 1, 1), tStP has shape ((128, 64), 1, 1)
        # tP overlap with tS
        # cute.printf(tStS)
        # ((128,128),1,1):((65536,1),0,0)
        # (128,64):(1,128)
        tStP = cute.composition(tStS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        # cute.printf(tStP)
        # ((128,128),1,1):((65536,1),0,0) o (128,64):(1,128) => ((128,64),1,1):((65536,1),0,0)
        tStP = cute.make_tensor(tStS.iterator, tStP.layout)  # Otherwise the tmem address is wrong
        tScS = thr_mma_S.partition_C(cute.make_identity_tensor(self.mma_tiler_kq[:2]))
        tScP = cute.composition(tScS, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        # tdS overlap with tdP
        tdPtdS = cute.composition(tdPtdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))
        tdPcdP = thr_mma_dP.partition_C(cute.make_identity_tensor(self.mma_tiler_vdo[:2]))
        tdPcdS = cute.composition(tdPcdP, (cute.make_layout((self.tile_n, tileP_f32_like)), 1, 1))


        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )
        tmem_store_atom = cute.make_copy_atom(
            tcgen05.copy.St32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )

        # tmem -> rmem
        thr_copy_t2r = copy_utils.make_tmem_copy(tmem_load_atom, num_wg).get_slice(tidx)
        tStS_t2r = thr_copy_t2r.partition_S(tStS)  # (((32, 32), 1), 2, 1, 1)
        tdPtdP_t2r = thr_copy_t2r.partition_S(tdPtdP)
        tScS_t2r = thr_copy_t2r.partition_D(tScS)  # ((32, 1), 2, 1, 1)
        t0ScS_t2r = thr_copy_t2r.get_slice(0).partition_D(tScS)  # ((32, 1), 2, 1, 1)
        # ((32, 1), 2, 1, 1, STAGE)
        tSsLSE = thr_copy_t2r.partition_D(thr_mma_S.partition_C(sLSE_2D))
        tSsdPsum = thr_copy_t2r.partition_D(thr_mma_dP.partition_C(sdPsum_2D))
        # rmem -> tmem
        thr_copy_r2t = copy_utils.make_tmem_copy(tmem_store_atom, num_wg).get_slice(tidx)
        tScP_r2t = thr_copy_r2t.partition_S(tScP)
        tStP_r2t = thr_copy_r2t.partition_D(tStP)
        tdPcdS_r2t = thr_copy_r2t.partition_S(tdPcdS)
        tdPtdS_r2t = thr_copy_r2t.partition_D(tdPtdS)
        # rmem -> smem
        # This part is a bit iffy, we might be making a lot of assumptions here
        copy_atom_r2s = sm100_utils_basic.get_smem_store_op(
            LayoutEnum.ROW_MAJOR, self.ds_dtype, Float32, thr_copy_t2r
        )
        thr_copy_r2s = cute.make_tiled_copy_D(copy_atom_r2s, thr_copy_t2r).get_slice(tidx)
        # We assume the swizzle (i.e. layout.inner) stays the same
        sdS_layout = sm100_utils_basic.make_smem_layout_epi(
            self.ds_dtype, LayoutEnum.ROW_MAJOR, (self.tile_n, self.tile_m), 1
        ).outer  # ((8,16), (64,2), (1, 1))
        sdS_layout = cute.slice_(sdS_layout, (None, None, 0))  # ((8,16), (64,2))
        # Need to group into 1 mode to be compatible w thr_copy_r2s
        sdS_layout = cute.make_layout((sdS_layout.shape,), stride=(sdS_layout.stride,))
        sdS_epi = cute.make_tensor(sdS.iterator, sdS_layout)
        tRS_sdS = thr_copy_r2s.partition_D(sdS_epi)

        consumer_state_S_P_dP = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        # consumer_phase_S_P_dP = Int32(0)
        producer_state_dS = pipeline.make_pipeline_state(  # Our impl has shortcut for stage==1
            cutlass.pipeline.PipelineUserType.Producer, 1
        )
        consumer_state_dKV = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 2
        )
        consumer_state_LSE = cutlass.pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.Q_stage
        )
        # consumer_state_dPsum = cutlass.pipeline.make_pipeline_state(
        consumer_state_dPsum = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, self.dO_stage
        )
        if const_expr(self.enable_flashmask):
            flashmask_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            mask = AttentionMaskCls(seqlen.seqlen_q, seqlen.seqlen_k)
            # TODO: condition mask_seqlen
            mask_fn = partial(
                mask.apply_mask_sm100_transposed,
                tScS_t2r=tScS_t2r,
                t0ScS_t2r=t0ScS_t2r,
                n_block=n_block,
                mask_seqlen=True,
                mask_causal=self.is_causal,
                mask_local=self.is_local,
                sStartEndRowIndices=sStartEndRowIndices,
            )

            # prefetch_LSE = not self.is_causal
            prefetch_LSE = False

            compute_step = partial(
                self.compute_step,
                thr_copy_t2r=thr_copy_t2r,
                thr_copy_r2t=thr_copy_r2t,
                tScS_t2r=tScS_t2r,
                tStS_t2r=tStS_t2r,
                tScP_r2t=tScP_r2t,
                tStP_r2t=tStP_r2t,
                tSsLSE=tSsLSE,
                tRS_sdS=tRS_sdS,
                tdPtdS_r2t=tdPtdS_r2t,
                tdPtdP_t2r=tdPtdP_t2r,
                tSsdPsum=tSsdPsum,
                prefetch_LSE=prefetch_LSE,
                pipeline_LSE=pipeline_LSE,
                pipeline_S_P=pipeline_S_P,
                pipeline_dPsum=pipeline_dPsum,
                pipeline_dP=pipeline_dP,
                pipeline_dS=pipeline_dS,
                softmax_scale_log2=softmax_scale_log2,
                mask_fn=mask_fn,
            )

            zero_block = m_block_max <= m_block_min
            if const_expr(self.enable_flashmask):
                cute.arch.mbarrier_wait(flashmask_loaded_mbar_ptr, flashmask_phase)
                loop_start = m_block_min
                loop_end = m_block_max
                zero_block = True
                # 0: 0 ~ UTS_min, no mask
                # 1: UTS_min ~ UTS_max, partially mask
                # 2: UTE_min ~ UTE_max, partially mask
                # 3: UTE_max ~ LTS_min, no mask
                # 4: LTS_min ~ LTS_max, partially mask
                # 5: LTE_min ~ LTE_max, partially mask
                # 6: LTE_max ~ max_seq_k, no mask
                if const_expr(not self.is_causal):
                    has_uts = const_expr(flashmask_info.UTS_nblock_max is not None)
                    if has_uts:
                        # 0 ~ UTS
                        loop_end = sFM_max_min[5] # UTS_min
                        for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                            zero_block = False
                            if tidx == 0 and self.debug_print:
                                cute.printf('n_block: %d, before compute_step 0 ~ UTS_min: %d', n_block, m_block)
                            consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS = compute_step(
                                m_block=m_block,
                                consumer_state_LSE=consumer_state_LSE,
                                consumer_state_S_P_dP=consumer_state_S_P_dP,
                                consumer_state_dPsum=consumer_state_dPsum,
                                producer_state_dS=producer_state_dS,
                                partially_masked=False,
                            )
                            if tidx == 0 and self.debug_print:
                                cute.printf('n_block: %d, after compute_step 0 ~ UTS_min: %d', n_block, m_block)

                        loop_start = sFM_max_min[5] # UTS_min
                        loop_end = sFM_max_min[4] + 1 # UTS_max
                        for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                            zero_block = False
                            if tidx == 0 and self.debug_print:
                                cute.printf('n_block: %d, before compute_step UTS_min ~ UTS_max: %d', n_block, m_block)
                            consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS = compute_step(
                                m_block=m_block,
                                consumer_state_LSE=consumer_state_LSE,
                                consumer_state_S_P_dP=consumer_state_S_P_dP,
                                consumer_state_dPsum=consumer_state_dPsum,
                                producer_state_dS=producer_state_dS,
                                partially_masked=True,
                            )
                            if tidx == 0 and self.debug_print:
                                cute.printf('n_block: %d, after compute_step UTS_min ~ UTS_max: %d', n_block, m_block)

                    loop_start = max(loop_start, sFM_max_min[7]) # UTE_min
                    loop_end = min(sFM_max_min[6] + 1, m_block_max) # UTE_max
                    for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                        zero_block = False
                        if tidx == 0 and self.debug_print:
                            cute.printf('n_block: %d, before compute_step UTE_min ~ UTE_max: %d', n_block, m_block)
                        consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS = compute_step(
                            m_block=m_block,
                            consumer_state_LSE=consumer_state_LSE,
                            consumer_state_S_P_dP=consumer_state_S_P_dP,
                            consumer_state_dPsum=consumer_state_dPsum,
                            producer_state_dS=producer_state_dS,
                            partially_masked=True,
                        )
                        if tidx == 0 and self.debug_print:
                            cute.printf('n_block: %d, after compute_step UTE_min ~ UTE_max: %d', n_block, m_block)
                    loop_start = max(loop_start, loop_end)

                # UTE ~ LTS
                loop_end = min(m_block_max, sFM_max_min[1])
                for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                    zero_block = False
                    if tidx == 0 and self.debug_print:
                        cute.printf('n_block: %d, before compute_step UTE_max ~ LTS_min: %d', n_block, m_block)
                    consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS = compute_step(
                        m_block=m_block,
                        consumer_state_LSE=consumer_state_LSE,
                        consumer_state_S_P_dP=consumer_state_S_P_dP,
                        consumer_state_dPsum=consumer_state_dPsum,
                        producer_state_dS=producer_state_dS,
                        partially_masked=False,
                    )
                    if tidx == 0 and self.debug_print:
                        cute.printf('n_block: %d, after compute_step UTE_max ~ LTS_min: %d', n_block, m_block)

                loop_start = max(loop_start, loop_end)
                loop_end = min(m_block_max, sFM_max_min[0] + 1)
                for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                    zero_block = False
                    if tidx == 0 and self.debug_print:
                        cute.printf('n_block: %d, before compute_step LTS_min ~ LTS_max: %d', n_block, m_block)
                    consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS = compute_step(
                        m_block=m_block,
                        consumer_state_LSE=consumer_state_LSE,
                        consumer_state_S_P_dP=consumer_state_S_P_dP,
                        consumer_state_dPsum=consumer_state_dPsum,
                        producer_state_dS=producer_state_dS,
                        partially_masked=True,
                    )
                    if tidx == 0 and self.debug_print:
                        cute.printf('n_block: %d, after compute_step LTS_min ~ LTS_max: %d', n_block, m_block)

                # LTE ~ seqlen_q
                has_lte = const_expr(flashmask_info.LTE_nblock_max is not None)
                if has_lte:
                    loop_start = max(sFM_max_min[0] + 1, sFM_max_min[3])
                    if sFM_max_min[3] == sFM_max_min[0]:
                        loop_start = sFM_max_min[3] + 1
                    loop_start = max(loop_start, m_block_min)
                    loop_end = min(m_block_max, sFM_max_min[2] + 1)
                    #loop_end = m_block_max
                    for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                        zero_block = False
                        if tidx == 0 and self.debug_print:
                            cute.printf('n_block: %d, before compute_step LTE_min ~ LTE_max: %d', n_block, m_block)
                        consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS = compute_step(
                            m_block=m_block,
                            consumer_state_LSE=consumer_state_LSE,
                            consumer_state_S_P_dP=consumer_state_S_P_dP,
                            consumer_state_dPsum=consumer_state_dPsum,
                            producer_state_dS=producer_state_dS,
                            partially_masked=True,
                        )
                        if tidx == 0 and self.debug_print:
                            cute.printf('n_block: %d, after compute_step LTE_min ~ LTE_max: %d', n_block, m_block)

                    loop_start = max(loop_start, loop_end)
                    loop_end = m_block_max
                    for m_block in cutlass.range(loop_start, loop_end, unroll=1):
                        zero_block = False
                        if tidx == 0 and self.debug_print:
                            cute.printf('n_block: %d, before compute_step LTE_max ~ seqlen_q: %d', n_block, m_block)
                        consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS = compute_step(
                            m_block=m_block,
                            consumer_state_LSE=consumer_state_LSE,
                            consumer_state_S_P_dP=consumer_state_S_P_dP,
                            consumer_state_dPsum=consumer_state_dPsum,
                            producer_state_dS=producer_state_dS,
                            partially_masked=False,
                        )
                        if tidx == 0 and self.debug_print:
                            cute.printf('n_block: %d, after compute_step LTE_max ~ seqlen_q: %d', n_block, m_block)
            else:
                # Mainloop
                for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                    consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS = compute_step(
                        m_block=m_block,
                        consumer_state_LSE=consumer_state_LSE,
                        consumer_state_S_P_dP=consumer_state_S_P_dP,
                        consumer_state_dPsum=consumer_state_dPsum,
                        producer_state_dS=producer_state_dS
                    )

            if not zero_block:
                if const_expr(not self.use_tma_store):
                    consumer_state_dKV = self.epilogue_dKV(
                        dp_idx,
                        warp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        thr_mma_dV,
                        thr_mma_dK,
                        tdVtdV,
                        tdKtdK,
                        mdV,
                        mdK,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale,
                    )
                else:
                    thr_copy_r2s_dKV = tiled_copy_r2s_dKV.get_slice(dp_idx)
                    #### STORE dV
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        thr_mma_dV,
                        tdVtdV,
                        mdV_tma_tensor,
                        sdV,
                        tma_atom_dV,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        None,  # Don't scale
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdV_semaphore,
                    )
                    #### STORE dK
                    consumer_state_dKV = self.epilogue_dK_or_dV_tma(
                        dp_idx,
                        batch_idx,
                        head_idx,
                        n_block,
                        thr_mma_dK,
                        tdKtdK,
                        mdK_tma_tensor,
                        sdK,
                        tma_atom_dK,
                        thr_copy_r2s_dKV,
                        pipeline_dKV,
                        consumer_state_dKV,
                        softmax_scale if const_expr(self.qhead_per_kvhead == 1) else None,
                        int(NamedBarrierBwdSm100.EpilogueWG1),  # barrier_id
                        mdK_semaphore,
                    )
            if const_expr(self.enable_flashmask):
                flashmask_phase ^= 1

            if tidx == 0 and self.debug_print:
                cute.printf('n_block: %d, EEEEEEEEEEEEEEEEEEEE after compute_loop EEEEEEEEEEEEEEEEEEEE', n_block)
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def compute_step(
        self,
        m_block: cute.Int32,
        thr_copy_t2r: cute.TiledCopy,
        thr_copy_r2t: cute.TiledCopy,
        tScS_t2r: cute.Tensor,
        tStS_t2r: cute.Tensor,
        tScP_r2t: cute.Tensor,
        tStP_r2t: cute.Tensor,
        tSsLSE: cute.Tensor,
        tRS_sdS: cute.Tensor,
        tdPtdS_r2t: cute.Tensor,
        tdPtdP_t2r: cute.Tensor,
        tSsdPsum: cute.Tensor,
        prefetch_LSE: bool,
        pipeline_LSE: PipelineAsync,
        pipeline_S_P: PipelineAsync,
        pipeline_dPsum: PipelineAsync,
        pipeline_dP: PipelineAsync,
        pipeline_dS: PipelineAsync,
        softmax_scale_log2: cutlass.Float32,
        consumer_state_LSE: cutlass.pipeline.PipelineState,
        consumer_state_S_P_dP: cutlass.pipeline.PipelineState,
        consumer_state_dPsum: cutlass.pipeline.PipelineState,
        producer_state_dS: cutlass.pipeline.PipelineState, 
        mask_fn: Callable,
        partially_masked: bool = False,
    ):
        # Prefetch 1 stage of LSE
        pipeline_LSE.consumer_wait(consumer_state_LSE)
        tSrLSE_s2r = cute.make_fragment(tScS_t2r[None, 0, 0, 0].shape, Float32)
        if const_expr(prefetch_LSE and not self.shuffle_LSE):
            cute.autovec_copy(tSsLSE[None, 0, 0, 0, consumer_state_LSE.index], tSrLSE_s2r)
    
        pipeline_S_P.consumer_wait(consumer_state_S_P_dP)
        # pipeline_S_P.sync_object_full.wait(0, consumer_phase_S_P_dP)
        #### TMEM->RMEM (Load S from TMEM)
        tSrS_t2r = cute.make_fragment(tScS_t2r.shape, Float32)
        cute.copy(thr_copy_t2r, tStS_t2r, tSrS_t2r)
    
        #### APPLY MASK
        mask_fn(tSrS_t2r, m_block=m_block, partially_masked=partially_masked)
    
        num_stages = cute.size(tScS_t2r, mode=[1])
    
        # ---------------------------------------------
        #### P = exp(S - LSE)
        # ---------------------------------------------
        lane_idx = cute.arch.lane_idx()
        tSrP_r2t_f32 = cute.make_fragment(tScP_r2t.shape, Float32)  # 64
        tSrP_r2t = cute.recast_tensor(tSrP_r2t_f32, self.q_dtype)
        for stage in cutlass.range_constexpr(num_stages):
            tSrS_cur = tSrS_t2r[None, stage, 0, 0]
            tSsLSE_cur = tSsLSE[None, stage, 0, 0, consumer_state_LSE.index]
            if const_expr(not self.shuffle_LSE):
                if const_expr(stage > 0 or not prefetch_LSE):
                    cute.autovec_copy(tSsLSE_cur, tSrLSE_s2r)
                tSrLSE = tSrLSE_s2r
            else:
                tSrLSE = tSsLSE_cur[lane_idx]
            for v in cutlass.range_constexpr(cute.size(tSrS_t2r, mode=[0]) // 2):
                if const_expr(not self.shuffle_LSE):
                    lse_pair = (tSrLSE[2 * v], tSrLSE[2 * v + 1])
                else:
                    lse_pair = (
                        utils.shuffle_sync(tSrLSE, offset=2 * v),
                        utils.shuffle_sync(tSrLSE, offset=2 * v + 1),
                    )
                tSrS_cur[2 * v], tSrS_cur[2 * v + 1] = utils.fma_packed_f32x2(
                    ((tSrS_cur[2 * v], tSrS_cur[2 * v + 1])),
                    (softmax_scale_log2, softmax_scale_log2),
                    (-lse_pair[0], -lse_pair[1]),
                )
                tSrS_cur[2 * v] = cute.math.exp2(tSrS_cur[2 * v], fastmath=True)
                tSrS_cur[2 * v + 1] = cute.math.exp2(tSrS_cur[2 * v + 1], fastmath=True)
            utils.cvt_f16(tSrS_cur, tSrP_r2t[None, stage, 0, 0])
            if const_expr(stage == 0):
                cute.arch.fence_view_async_tmem_load()
                # Without this barrier, we could have 1 warp writing to P in tmem while
                # another warp is still reading S from tmem.
                self.compute_sync_barrier.arrive_and_wait()
            cute.copy(
                thr_copy_r2t,
                tSrP_r2t_f32[None, stage, None, None],
                tStP_r2t[None, stage, None, None],
            )
    
        cute.arch.fence_view_async_tmem_store()
        self.compute_sync_barrier.arrive_and_wait()
    
        with cute.arch.elect_one():
            pipeline_S_P.consumer_release(consumer_state_S_P_dP)
            # pipeline_S_P.sync_object_empty.arrive(0, pipeline_S_P.consumer_mask)
        pipeline_LSE.consumer_release(consumer_state_LSE)
        # consumer_state_S_P_dP.advance()
        consumer_state_LSE.advance()
    
        # ---------------------------------------------
        # dS.T = P.T * (dP.T - D)
        # ---------------------------------------------
        pipeline_dPsum.consumer_wait(consumer_state_dPsum)
    
        pipeline_dP.consumer_wait(consumer_state_S_P_dP)
        # pipeline_dP.sync_object_full.wait(0, consumer_phase_S_P_dP)
        consumer_state_S_P_dP.advance()
        # consumer_phase_S_P_dP ^= 1
    
        ##### dS.T = P.T * (dP.T - Psum)
        for stage in cutlass.range_constexpr(num_stages):
            tdPrdP_t2r = cute.make_fragment(tScS_t2r[None, 0, None, None].shape, Float32)
            cute.copy(thr_copy_t2r, tdPtdP_t2r[None, stage, None, None], tdPrdP_t2r)
            cute.arch.fence_view_async_tmem_load()
            self.compute_sync_barrier.arrive_and_wait()
            tdPrdP_cur = tdPrdP_t2r[None, 0, 0]
            tSrS_cur = tSrS_t2r[None, stage, 0, 0]
            tSsdPsum_cur = tSsdPsum[None, stage, 0, 0, consumer_state_dPsum.index]
            if const_expr(not self.shuffle_dPsum):
                tSrdPsum = cute.make_fragment_like(tSsdPsum_cur, Float32)
                cute.autovec_copy(tSsdPsum_cur, tSrdPsum)
            else:
                tSrdPsum = tSsdPsum_cur[lane_idx]
            for v in cutlass.range_constexpr(cute.size(tdPrdP_t2r, mode=[0]) // 2):
                if const_expr(not self.shuffle_dPsum):
                    dPsum_pair = (tSrdPsum[2 * v], tSrdPsum[2 * v + 1])
                else:
                    dPsum_pair = (
                        utils.shuffle_sync(tSrdPsum, offset=2 * v),
                        utils.shuffle_sync(tSrdPsum, offset=2 * v + 1),
                    )
                tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = utils.sub_packed_f32x2(
                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]), dPsum_pair
                )
                tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1] = utils.mul_packed_f32x2(
                    (tSrS_cur[2 * v], tSrS_cur[2 * v + 1]),
                    (tdPrdP_cur[2 * v], tdPrdP_cur[2 * v + 1]),
                )
            tdPrdS_cvt = cute.make_fragment_like(tdPrdP_cur, self.ds_dtype)
            utils.cvt_f16(tdPrdP_cur, tdPrdS_cvt)
            if const_expr(stage == 0):
                pipeline_dS.producer_acquire(producer_state_dS)
            cute.autovec_copy(tdPrdS_cvt, tRS_sdS[None, stage])
            if const_expr(not self.use_smem_dS_for_mma_dK):
                tdPrdS_r2t_f32 = cute.recast_tensor(tdPrdS_cvt, Float32)
                cute.copy(thr_copy_r2t, tdPrdS_r2t_f32, tdPtdS_r2t[None, stage, 0, 0])
    
        if const_expr(not self.use_smem_dS_for_mma_dK):
            cute.arch.fence_view_async_tmem_store()
        cute.arch.fence_proxy(
            cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
        )
        self.compute_sync_barrier.arrive_and_wait()
    
        # with cute.arch.elect_one():
        # The mma warp no longer waits for dP (it waits for dS), so we don't have to arrive
        # pipeline_dP.sync_object_empty.arrive(0, pipeline_dP.consumer_mask)
        pipeline_dPsum.consumer_release(consumer_state_dPsum)
        consumer_state_dPsum.advance()
        with cute.arch.elect_one():
            pipeline_dS.producer_commit(producer_state_dS)
        producer_state_dS.advance()

        return consumer_state_LSE, consumer_state_S_P_dP, consumer_state_dPsum, producer_state_dS

    @cute.jit
    def dQacc_reduce(
        self,
        mdQaccum: cute.Tensor,
        sdQaccum: cute.Tensor,
        thr_mma_dQ: cute.core.ThrMma,
        tdQtdQ: cute.Tensor,
        pipeline_dQ: PipelineAsync,
        block_info: BlockInfo,
        SeqlenInfoCls: Callable,
        TileSchedulerCls: Callable,
        mdQ_semaphore: Optional[cute.Tensor],
        flashmask_info: FlashMaskInfo,
        sFM_max_min: cute.Tensor,
        flashmask_loaded_mbar_ptr: cute.Pointer,
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx() % len(self.reduce_warp_ids))
        is_tma_warp = warp_idx == 0
        # TMEM -> RMEM
        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(self.dQ_reduce_ncol)), Float32
        )
        thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdQtdQ).get_slice(tidx)
        tdQtdQ_t2r = thr_copy_t2r.partition_S(tdQtdQ)
        tdQcdQ = thr_mma_dQ.partition_C(cute.make_identity_tensor(self.mma_tiler_dsk[:2]))
        tdQrdQ_t2r_shape = thr_copy_t2r.partition_D(tdQcdQ).shape
        assert cute.size(tdQrdQ_t2r_shape, mode=[1]) == self.dQaccum_reduce_stage, (
            "dQaccum reduce stage mismatch"
        )

        thr_copy_dQaccum_r2s = copy_utils.tiled_copy_1d(
            self.dqaccum_dtype, num_reduce_threads, num_copy_elems=128 // self.dqaccum_dtype.width
        ).get_slice(tidx)
        tdQsdQ = thr_copy_dQaccum_r2s.partition_D(sdQaccum)

        read_flag = const_expr(not self.deterministic)
        if const_expr(self.enable_flashmask):
            flashmask_phase = Int32(0)

        tile_scheduler = TileSchedulerCls()
        work_tile = tile_scheduler.initial_work_tile_info()
        dQ_consumer_state = pipeline.make_pipeline_state(
            cutlass.pipeline.PipelineUserType.Consumer, 1
        )
        dQ_tma_store_producer_state = pipeline.make_pipeline_state(
            pipeline.PipelineUserType.Producer, self.sdQaccum_stage
        )
        while work_tile.is_valid_tile:
            n_block, head_idx, batch_idx, _ = work_tile.tile_idx
            seqlen = SeqlenInfoCls(batch_idx)
            m_block_min, m_block_max = block_info.get_m_block_min_max(
                seqlen, n_block // self.cluster_shape_mnk[0]
            )
            mdQaccum_cur = mdQaccum[None, head_idx, batch_idx]
            gdQaccum_ = cute.local_tile(mdQaccum_cur, (self.tile_m * self.tile_hdim,), (None,))
            # (M * K / STAGE, STAGE, _)
            gdQaccum = cute.flat_divide(
                gdQaccum_, (self.tile_m * self.tile_hdim // self.dQaccum_reduce_stage,)
            )

            if const_expr(self.deterministic):
                mdQ_semaphore_cur = mdQ_semaphore[None, None, head_idx, batch_idx]
            else:
                mdQ_semaphore_cur = None

            delay_semaphore_release = self.is_causal
            n_block_global_max = cute.ceil_div(seqlen.seqlen_k, self.tile_n)

            dQacc_reduce_step = partial(
                self.dQacc_reduce_step,
                m_block_min=m_block_min,
                n_block=n_block,
                n_block_global_max=n_block_global_max,
                tidx=tidx,
                tdQrdQ_t2r_shape=tdQrdQ_t2r_shape,
                tdQtdQ_t2r=tdQtdQ_t2r,
                tdQsdQ=tdQsdQ,
                sdQaccum=sdQaccum,
                gdQaccum=gdQaccum,
                thr_copy_dQaccum_r2s=thr_copy_dQaccum_r2s,
                thr_copy_t2r=thr_copy_t2r,
                pipeline_dQ=pipeline_dQ,
                dQ_consumer_state=dQ_consumer_state,
                dQ_tma_store_producer_state=dQ_tma_store_producer_state,
                seqlen=seqlen,
                delay_semaphore_release=delay_semaphore_release,
                read_flag=read_flag,
                is_tma_warp=is_tma_warp,
                mdQ_semaphore_cur=mdQ_semaphore_cur,
            )

            if const_expr(self.enable_flashmask):
                # 0: 0 ~ UTS_min, no mask
                # 1: UTS_min ~ UTS_max, partially mask
                # 2: UTS_max ~ UTE_min, full mask
                # 3: UTE_min ~ UTE_max, partially mask
                # 4: UTE_max ~ LTS_min, no mask
                # 5: LTS_min ~ LTS_max, partially mask
                # 6: LTS_max ~ LTE_min, full mask
                # 7: LTE_min ~ LTE_max, partially mask
                # 8: LTE_max ~ max_seq_k, no mask

                # sFM_max_min[0], [1] -> LTS_max, LTS_min
                # sFM_max_min[2], [3] -> LTE_max, LTE_min
                # sFM_max_min[4], [5] -> UTS_max, UTS_min
                # sFM_max_min[6], [7] -> UTE_max, UTE_min
                cute.arch.mbarrier_wait(flashmask_loaded_mbar_ptr, flashmask_phase)
                loop_start = m_block_min
                loop_end = m_block_max
                for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                    full_mask = False
                    if const_expr(not self.is_causal):
                        UTS_max = -1
                        if const_expr(flashmask_info.UTS_nblock_max is not None):
                            UTS_max = sFM_max_min[4]
                        UTE_min = sFM_max_min[7]
                        if m_block > UTS_max and m_block < UTE_min:
                            full_mask = True

                    LTS_max = sFM_max_min[0]
                    LTE_min = m_block_max
                    if const_expr(flashmask_info.LTE_nblock_max is not None):
                        LTE_min = sFM_max_min[3]

                    if m_block > LTS_max and m_block < LTE_min:
                        full_mask = True

                    if not full_mask:
                        if tidx == 0 and self.debug_print:
                            cute.printf('n_block: %d, m_block: %d, before reduce_step', n_block, m_block)
                        dQ_consumer_state, dQ_tma_store_producer_state = dQacc_reduce_step(
                            m_block=m_block,
                            dQ_consumer_state=dQ_consumer_state,
                            dQ_tma_store_producer_state=dQ_tma_store_producer_state,
                        )
                        if tidx == 0 and self.debug_print:
                            cute.printf('n_block: %d, m_block: %d, after reduce_step', n_block, m_block)

                    if const_expr(self.deterministic):
                        if full_mask:
                            if tidx == 0 and self.debug_print:
                                cute.printf('n_block: %d, m_block: %d, before reduce_step SKIPPPPPPP', n_block, m_block)

                            if const_expr(self.spt):
                                n_block_max_for_m_block = min(
                                    n_block_global_max,
                                    cute.ceil_div(
                                        (m_block + 1) * self.tile_m + seqlen.seqlen_k - seqlen.seqlen_q,
                                        self.tile_n,
                                    ),
                                )
                                lock_value = n_block_max_for_m_block - 1 - n_block
                            else:
                                lock_value = n_block
                            barrier.wait_eq(
                                mdQ_semaphore_cur[(m_block, None)].iterator, tidx, 0, lock_value
                            )

                            if const_expr(delay_semaphore_release):
                                if m_block > m_block_min:
                                    barrier.arrive_inc(
                                        mdQ_semaphore_cur[(m_block - 1, None)].iterator, tidx, 0, 1
                                    )
                            else:
                                barrier.arrive_inc(mdQ_semaphore_cur[m_block, None].iterator, tidx, 0, 1)

                            if tidx == 0 and self.debug_print:
                                cute.printf('n_block: %d, m_block: %d, after reduce_step SKIPPPPPPP', n_block, m_block)

            else:
                for m_block in cutlass.range(m_block_min, m_block_max, unroll=1):
                    dQ_consumer_state, dQ_tma_store_producer_state = dQacc_reduce_step(
                        m_block=m_block,
                        dQ_consumer_state=dQ_consumer_state,
                        dQ_tma_store_producer_state=dQ_tma_store_producer_state,
                    )

            if is_tma_warp:
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            self.reduce_sync_barrier.arrive_and_wait()
            # final semaphore release
            if const_expr(self.deterministic and delay_semaphore_release):
                barrier.arrive_inc(mdQ_semaphore_cur[(m_block_max - 1, None)].iterator, tidx, 0, 1)

            if const_expr(self.enable_flashmask):
                flashmask_phase ^= 1

            if tidx == 0 and self.debug_print:
                cute.printf('n_block: %d, EEEEEEEEEEEEEEEEEEEE after reduce EEEEEEEEEEEEEEEEEEEE', n_block)
            tile_scheduler.advance_to_next_work()
            work_tile = tile_scheduler.get_current_work()

    @cute.jit
    def dQacc_reduce_step(
        self,
        m_block: cute.Int32,
        m_block_min: cute.Int32,
        n_block: cute.Int32,
        n_block_global_max: cute.Int32,
        tidx: cute.Int32,
        tdQrdQ_t2r_shape: cute.Shape,
        tdQtdQ_t2r: cute.Tensor,
        tdQsdQ: cute.Tensor,
        sdQaccum: cute.Tensor,
        gdQaccum: cute.Tensor,
        thr_copy_dQaccum_r2s: cute.TiledCopy,
        thr_copy_t2r: cute.TiledCopy,
        pipeline_dQ: PipelineAsync,
        dQ_consumer_state: cutlass.pipeline.PipelineState,
        dQ_tma_store_producer_state: cutlass.pipeline.PipelineState,
        seqlen: SeqlenInfoQK,
        delay_semaphore_release: bool,
        read_flag: bool,
        is_tma_warp: bool,
        mdQ_semaphore_cur: Optional[cute.Tensor],
    ):
        num_reduce_threads = cute.arch.WARP_SIZE * len(self.reduce_warp_ids)
        tidx = cute.arch.thread_idx()[0] % num_reduce_threads
        if tidx == 0 and self.debug_print:
            cute.printf('n_block: %d, m_block:%d, reduce_step before pipeline_dQ.consumer_wait', n_block, m_block)
        pipeline_dQ.consumer_wait(dQ_consumer_state)
        # TMEM -> RMEM
        tdQrdQ_t2r = cute.make_fragment(tdQrdQ_t2r_shape, Float32)
        cute.copy(thr_copy_t2r, tdQtdQ_t2r, tdQrdQ_t2r)
        cute.arch.fence_view_async_tmem_load()
        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dQ.consumer_release(dQ_consumer_state)
        dQ_consumer_state.advance()
        if tidx == 0 and self.debug_print:
            cute.printf('n_block: %d, m_block:%d, reduce_step after pipeline_dQ.consumer_wait', n_block, m_block)
    
        gdQaccum_cur = gdQaccum[None, None, m_block]
    
        for stage in cutlass.range_constexpr(cute.size(tdQrdQ_t2r, mode=[1])):  # 4
            if tidx == 0 and self.debug_print:
                cute.printf('n_block: %d, m_block: %d, stage: %d, reduce_step before dQ_tma_store_producer_state.advance', n_block, m_block, stage)
            smem_idx = dQ_tma_store_producer_state.index
            tdQsdQ_r2s = tdQsdQ[None, None, smem_idx]
            tdQrdQ_r2s = cute.make_tensor(
                tdQrdQ_t2r[None, stage, None, None].iterator, tdQsdQ_r2s.shape
            )
            cute.copy(thr_copy_dQaccum_r2s, tdQrdQ_r2s, tdQsdQ_r2s)
            # Fence and barrier to make sure shared memory store is visible to TMA store
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            # semaphore acquire
            if const_expr(self.deterministic and stage == 0):
                if const_expr(self.spt):
                    n_block_max_for_m_block = min(
                        n_block_global_max,
                        cute.ceil_div(
                            (m_block + 1) * self.tile_m + seqlen.seqlen_k - seqlen.seqlen_q,
                            self.tile_n,
                        ),
                    )
                    lock_value = n_block_max_for_m_block - 1 - n_block
                else:
                    lock_value = n_block
                if tidx == 0 and self.debug_print:
                    cute.printf('n_block: %d, m_block: %d, stage: %d, lock_value: %d, reduce_step before barrier.wait_eq', n_block, m_block, stage, lock_value)
                barrier.wait_eq(
                    mdQ_semaphore_cur[(m_block, None)].iterator, tidx, 0, lock_value
                )
                if tidx == 0 and self.debug_print:
                    cute.printf('n_block: %d, m_block: %d, stage: %d, lock_value: %d, reduce_step after barrier.wait_eq', n_block, m_block, stage, lock_value)
            self.reduce_sync_barrier.arrive_and_wait()
            # Copy from shared memory to global memory
            if is_tma_warp:
                with cute.arch.elect_one():
                    copy_utils.cpasync_reduce_bulk_add_f32(
                        sdQaccum[None, smem_idx].iterator,
                        gdQaccum_cur[None, stage].iterator,
                        self.tma_copy_bytes["dQ"] // 1,
                    )
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(self.sdQaccum_stage - 1, read=read_flag)
            self.reduce_sync_barrier.arrive_and_wait()
            if tidx == 0 and self.debug_print:
                cute.printf('n_block: %d, m_block: %d, stage: %d, reduce_step after dQ_tma_store_producer_state.advance', n_block, m_block, stage)
            dQ_tma_store_producer_state.advance()
            # Directly add to gmem, much slower
            # tdQgdQ = thr_copy_dQaccum_r2s.partition_D(gdQaccum[None, stage, m_block])
            # assert cute.size(tdQrdQ_r2s) == cute.size(tdQgdQ)
            # for i in cutlass.range(cute.size(tdQrdQ_r2s) // 4, unroll_full=True):
            #     copy_utils.atomic_add_fp32x4(
            #         tdQrdQ_r2s[4 * i],
            #         tdQrdQ_r2s[4 * i + 1],
            #         tdQrdQ_r2s[4 * i + 2],
            #         tdQrdQ_r2s[4 * i + 3],
            #         utils.elem_pointer(tdQgdQ, 4 * i),
            #     )
            # semaphore release for prior m_block
            if const_expr(self.deterministic and stage == 0 and delay_semaphore_release):
                if m_block > m_block_min:
                    if tidx == 0 and self.debug_print:
                        cute.printf('n_block: %d, m_block: %d, stage: %d, lock_value: %d, reduce_step before barrier.arrive_inc in stage', n_block, m_block, stage, lock_value)
                    barrier.arrive_inc(
                        mdQ_semaphore_cur[(m_block - 1, None)].iterator, tidx, 0, 1
                    )
                    if tidx == 0 and self.debug_print:
                        cute.printf('n_block: %d, m_block: %d, stage: %d, lock_value: %d, reduce_step after barrier.arrive_inc in stage', n_block, m_block, stage, lock_value)
    
        # semaphore release
        # NOTE: arrive_inc calls red_release which issues membar
        if const_expr(self.deterministic and not delay_semaphore_release):
            if tidx == 0 and self.debug_print:
                cute.printf('n_block: %d, m_block: %d, reduce_step before barrier.arrive_inc', n_block, m_block)
            if is_tma_warp:
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            self.reduce_sync_barrier.arrive_and_wait()
            barrier.arrive_inc(mdQ_semaphore_cur[m_block, None].iterator, tidx, 0, 1)
            if tidx == 0 and self.debug_print:
                cute.printf('n_block: %d, m_block: %d, reduce_step after barrier.arrive_inc', n_block, m_block)

        return dQ_consumer_state, dQ_tma_store_producer_state

    @cute.jit
    def epilogue_dKV(
        self,
        tidx: Int32,
        warp_idx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        thr_mma_dV: cute.core.ThrMma,
        thr_mma_dK: cute.core.ThrMma,
        tdVtdV: cute.Tensor,
        tdKtdK: cute.Tensor,
        mdV: cute.Tensor,
        mdK: cute.Tensor,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        softmax_scale: Float32,
    ):
        wg_idx = (
            cute.arch.thread_idx()[0] % (cute.arch.WARP_SIZE * len(self.compute_warp_ids))
        ) // 128
        num_wg = cute.arch.WARP_SIZE * len(self.compute_warp_ids) // 128

        assert self.qhead_per_kvhead == 1, "This epilogue path is only for MHA"
        mdV_cur = mdV[None, None, head_idx, batch_idx]
        mdK_cur = mdK[None, None, head_idx, batch_idx]

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(16)), Float32
        )

        # dV
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dV = tcgen05.make_tmem_copy(tmem_load_atom, tdVtdV)
        thr_tmem_ld_dV = tiled_tmem_ld_dV.get_slice(tidx)

        tdVtdV_t2r_p = thr_tmem_ld_dV.partition_S(tdVtdV)
        tdVtdV_t2r = self.split_wg(tdVtdV_t2r_p, wg_idx, num_wg)

        cdV = cute.make_identity_tensor((self.mma_tiler_pdo[0], self.mma_tiler_pdo[1]))
        tdVcdV = thr_mma_dV.partition_C(cdV)
        tdVcdV_tensor = cute.make_tensor(tdVcdV.iterator, tdVcdV.layout)

        tdVcdV_t2r_p = thr_tmem_ld_dV.partition_D(tdVcdV_tensor)
        tdVcdV_t2r = self.split_wg(tdVcdV_t2r_p, wg_idx, num_wg)
        tdVrdV_t2r = cute.make_fragment(tdVcdV_t2r.shape, Float32)

        cute.copy(thr_tmem_ld_dV, tdVtdV_t2r, tdVrdV_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dv_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        tiled_gmem_store_dV = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dV.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dV.tiler_mn,
        )

        tdVrdV_r2s = cute.make_fragment(tdVrdV_t2r.shape, self.dv_dtype)
        for i in cutlass.range_constexpr(cute.size(tdVrdV_t2r, mode=[1])):
            dV_vec = tdVrdV_t2r[(None, i, 0, 0)].load()
            tdVrdV_r2s[(None, i, 0, 0)].store(dV_vec.to(self.dv_dtype))

        gdV = cute.local_tile(mdV_cur, (self.tile_m, self.tile_hdimv), (None, 0))
        gdV_tile = gdV[None, None, n_block]

        tdVgdV = thr_mma_dV.partition_C(gdV_tile)
        tdVgdV_r2g_p = thr_tmem_ld_dV.partition_D(tdVgdV)
        tdVgdV_r2g = self.split_wg(tdVgdV_r2g_p, wg_idx, num_wg)

        cute.copy(tiled_gmem_store_dV, tdVrdV_r2s, tdVgdV_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()

        # dK
        pipeline_dKV.consumer_wait(consumer_state_dKV)

        tiled_tmem_ld_dK = tcgen05.make_tmem_copy(tmem_load_atom, tdKtdK)
        thr_tmem_ld_dK = tiled_tmem_ld_dK.get_slice(tidx)

        tdKtdK_t2r_p = thr_tmem_ld_dK.partition_S(tdKtdK)
        tdKtdK_t2r = self.split_wg(tdKtdK_t2r_p, wg_idx, num_wg)

        cdK = cute.make_identity_tensor((self.mma_tiler_dsq[0], self.mma_tiler_dsq[1]))
        tdKcdK = thr_mma_dK.partition_C(cdK)
        tdKcdK_tensor = cute.make_tensor(tdKcdK.iterator, tdKcdK.layout)

        tdKcdK_t2r_p = thr_tmem_ld_dK.partition_D(tdKcdK_tensor)
        tdKcdK_t2r = self.split_wg(tdKcdK_t2r_p, wg_idx, num_wg)
        tdKrdK_t2r = cute.make_fragment(tdKcdK_t2r.shape, Float32)

        cute.copy(tiled_tmem_ld_dK, tdKtdK_t2r, tdKrdK_t2r)
        cute.arch.fence_view_async_tmem_load()

        universal_copy_bits = 128
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.dk_dtype,
            num_bits_per_copy=universal_copy_bits,
        )

        tiled_gmem_store_dK = cute.make_tiled_copy(
            atom_universal_copy,
            layout_tv=tiled_tmem_ld_dK.layout_dst_tv_tiled,
            tiler_mn=tiled_tmem_ld_dK.tiler_mn,
        )

        tdKrdK_r2s = cute.make_fragment(tdKrdK_t2r.shape, self.dk_dtype)

        for i in cutlass.range_constexpr(cute.size(tdKrdK_t2r, mode=[1])):
            dK_vec = tdKrdK_t2r[(None, i, 0, 0)].load() * softmax_scale
            tdKrdK_r2s[(None, i, 0, 0)].store(dK_vec.to(self.dk_dtype))

        gdK = cute.local_tile(mdK_cur, (self.tile_n, self.tile_hdimv), (None, 0))
        gdK_tile = gdK[None, None, n_block]

        tdKgdK = thr_mma_dK.partition_C(gdK_tile)
        tdKgdK_r2g_p = thr_tmem_ld_dK.partition_D(tdKgdK)
        tdKgdK_r2g = self.split_wg(tdKgdK_r2g_p, wg_idx, num_wg)

        cute.copy(tiled_gmem_store_dK, tdKrdK_r2s, tdKgdK_r2g)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV

    @cute.jit
    def epilogue_dK_or_dV_tma(
        self,
        tidx: Int32,
        batch_idx: Int32,
        head_idx: Int32,
        n_block: Int32,
        thr_mma: cute.core.ThrMma,
        tdKVtdKV: cute.Tensor,
        mdKV: cute.Tensor,
        sdKV: cute.Tensor,
        tma_atom_dKV: cute.CopyAtom,
        thr_copy_r2s_dKV: cute.TiledCopy,
        pipeline_dKV: PipelineAsync,
        consumer_state_dKV: cutlass.pipeline.PipelineState,
        scale: Optional[Float32],
        barrier_id: Int32,
        mdKV_semaphore: Optional[cute.Tensor],
    ) -> cutlass.pipeline.PipelineState:
        # assumes mma_tiler_pdo = mma_tiler_dsq = (tile_n, head_dim)
        # head_dim = head_dim_v, dk_dtype = dv_dtype
        num_compute_threads = cute.arch.WARP_SIZE * len(self.compute_warp_ids)
        wg_idx = (cute.arch.thread_idx()[0] % num_compute_threads) // 128
        num_wg = num_compute_threads // 128
        leader_warp = (cute.arch.make_warp_uniform(cute.arch.warp_idx()) % 4) == 0

        if const_expr(self.qhead_per_kvhead == 1):
            sdKV = sdKV[None, None, wg_idx]  # (tile_n, 64) for bf16
        else:
            sdKV = sdKV[None, wg_idx]  # (tile_n * 32) for fp32

        # (8, tile_n / 128, 64 / 8) = (8, 1, 8) or (4, tile_n * 32 / (128 * 4)) = (4, 8)
        tdKVsdKV_r2s = thr_copy_r2s_dKV.partition_D(sdKV)

        head_idx_kv = head_idx // self.qhead_per_kvhead
        if const_expr(self.qhead_per_kvhead == 1):
            mdKV_cur = mdKV[None, None, head_idx_kv, batch_idx]  # (seqlen, hdim)
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n, self.tile_hdim), (n_block, 0)
            )  # (tile_n, hdim)
            gdKV = self.split_wg(gdKV_p, wg_idx, num_wg)  # (tile_n, hdim / 2)
            gdKV_epi = cute.local_tile(
                gdKV, self.sdKV_epi_tile, (0, None)
            )  # (tile_n, 64, epi_stage = (hdim / 2) / 64)
        else:
            mdKV_cur = mdKV[None, head_idx_kv, batch_idx]  # (seqlen * hdim)
            gdKV_p = cute.local_tile(
                mdKV_cur, (self.tile_n * self.tile_hdim,), (n_block,)
            )  # (tile_n * hdim)
            gdKV = cute.logical_divide(gdKV_p, (self.tile_n * self.tile_hdim // num_wg,))[
                ((None, wg_idx),)
            ]  # (tile_n * hdim / 2)
            gdKV_epi = cute.flat_divide(
                gdKV, (self.sdKV_flat_epi_tile,)
            )  # (tile_n * hdim / 2 / epi_stage, epi_stage)

        deterministic_KV = self.deterministic and self.qhead_per_kvhead > 1
        if const_expr(deterministic_KV):
            mdKV_semaphore_cur = mdKV_semaphore[n_block, None, head_idx_kv, batch_idx]

        if const_expr(self.qhead_per_kvhead == 1):
            tdKVsdKV, tdKVgdKV = cpasync.tma_partition(
                tma_atom_dKV,
                0,  # no multicast
                cute.make_layout(1),
                cute.group_modes(sdKV, 0, 2),
                cute.group_modes(gdKV_epi, 0, 2),
            )  # (TMA) and (TMA, EPI_STAGE)
            assert len(tdKVsdKV.shape) == 1, "Wrong rank for SMEM fragment tdKVsdKV"
            assert len(tdKVgdKV.shape) == 2, "Wrong rank for GMEM fragment tdKVgdKV"
            num_epi_stages = cute.size(tdKVgdKV.shape[1])
            assert num_epi_stages == self.num_epi_stages, f"Epi stage calculation is wrong. num_epi_stages:{num_epi_stages} != self.num_epi_stages: {self.num_epi_stages}"
        else:
            num_epi_stages = self.num_epi_stages

        tmem_load_atom = cute.make_copy_atom(
            tcgen05.copy.Ld32x32bOp(tcgen05.copy.Repetition(32)), Float32
        )

        read_flag = const_expr(not deterministic_KV)

        pipeline_dKV.consumer_wait(consumer_state_dKV)

        # semaphore acquire
        if const_expr(deterministic_KV):
            barrier.wait_eq(
                mdKV_semaphore_cur.iterator, tidx, wg_idx, head_idx % self.qhead_per_kvhead
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

        for epi_stage in cutlass.range_constexpr(num_epi_stages):
            # TMEM -> RMEM -- setup
            thr_copy_t2r = tcgen05.make_tmem_copy(tmem_load_atom, tdKVtdKV).get_slice(tidx)
            tdKVtdKV_t2r_p = thr_copy_t2r.partition_S(tdKVtdKV)
            tdKVtdKV_t2r = self.split_wg(tdKVtdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVtdKV_t2r = tdKVtdKV_t2r[None, epi_stage]

            cdKV = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
            tdKVcdKV = thr_mma.partition_C(cdKV)
            tdKVcdKV_t2r_p = thr_copy_t2r.partition_D(tdKVcdKV)
            tdKVcdKV_t2r = self.split_wg(tdKVcdKV_t2r_p, wg_idx, num_wg)[None, None, 0, 0]
            if const_expr(num_epi_stages > 1):
                tdKVcdKV_t2r = tdKVcdKV_t2r[None, epi_stage]

            tdKVrdKV_t2r = cute.make_fragment(tdKVcdKV_t2r.shape, Float32)

            assert cute.size(tdKVrdKV_t2r) == cute.size(tdKVtdKV_t2r) // cute.arch.WARP_SIZE, (
                "RMEM<->TMEM fragment size mismatch"
            )

            # TMEM -> RMEM -- copy and fence
            cute.copy(thr_copy_t2r, tdKVtdKV_t2r, tdKVrdKV_t2r)
            cute.arch.fence_view_async_tmem_load()

            # RMEM -- scale and convert
            if const_expr(scale is not None):
                for i in cutlass.range(cute.size(tdKVrdKV_t2r.shape) // 2, unroll_full=True):
                    tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1] = utils.mul_packed_f32x2(
                        (tdKVrdKV_t2r[2 * i], tdKVrdKV_t2r[2 * i + 1]), (scale, scale)
                    )
            tdKVrdKV = cute.make_fragment(tdKVrdKV_t2r.shape, self.dv_dtype)  # (32 columns)
            tdKVrdKV.store(tdKVrdKV_t2r.load().to(self.dv_dtype))

            # RMEM -> SMEM -- copy, fence and barrier
            tdKVrdKV_r2s = cute.make_tensor(tdKVrdKV.iterator, tdKVsdKV_r2s.shape)
            cute.copy(thr_copy_r2s_dKV, tdKVrdKV_r2s, tdKVsdKV_r2s)
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)

            # SMEM -> GMEM
            if leader_warp:
                if const_expr(self.qhead_per_kvhead == 1):
                    cute.copy(tma_atom_dKV, tdKVsdKV, tdKVgdKV[None, epi_stage])
                else:
                    with cute.arch.elect_one():
                        copy_utils.cpasync_reduce_bulk_add_f32(
                            sdKV.iterator,
                            gdKV_epi[None, epi_stage].iterator,
                            self.tma_copy_bytes["dKacc"],
                        )
                if const_expr(epi_stage < num_epi_stages - 1):
                    cute.arch.cp_async_bulk_commit_group()
                    cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
                cute.arch.barrier_arrive(
                    barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
                )

            # Barrier since all warps need to wait for SMEM to be freed
            cute.arch.fence_proxy(
                cute.arch.ProxyKind.async_shared, space=cute.arch.SharedSpace.shared_cta
            )
            cute.arch.barrier(
                barrier_id=barrier_id + wg_idx, number_of_threads=128 + cute.arch.WARP_SIZE
            )

        # semaphore release
        # NOTE: arrive_inc calls red_release which issues membar
        if const_expr(deterministic_KV):
            if leader_warp:
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=read_flag)
            cute.arch.barrier(barrier_id=barrier_id + wg_idx, number_of_threads=128)
            barrier.arrive_inc(mdKV_semaphore_cur.iterator, tidx, wg_idx, 1)

        cute.arch.sync_warp()
        with cute.arch.elect_one():
            pipeline_dKV.consumer_release(consumer_state_dKV)
        consumer_state_dKV.advance()
        return consumer_state_dKV
