# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2025-2026 FlyDSL Project Contributors

"""gfx1250 MLA page-size-64 FP8 stage-1 kernel."""

import math

import torch

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl.expr import const_expr, gpu, range_constexpr, rocdl, tdm_ops
from flydsl.expr import math as fmath
from flydsl.expr.typing import T
from flydsl.expr.typing import Vector as Vec
from flydsl.runtime.device import get_rocm_arch

from ..gemm_common_gfx1250 import make_lds_copy_ops

BLOCK_THREADS = 128
WAVE_SIZE = 32
NUM_WAVES = BLOCK_THREADS // WAVE_SIZE
NUM_HEAD_GROUPS = 2
HEADS_PER_WAVE = 16
HEADS_PER_GROUP = NUM_WAVES * HEADS_PER_WAVE
MAPPING_DEBUG_FIELDS = 4
Q_SAMPLE_DIMS = (0, 511, 512, 575)
Q_DEBUG_FIELDS = 2 * len(Q_SAMPLE_DIMS)
Q_TDM_SAMPLE_COORDS = ((0, 0), (0, 127), (0, 511), (15, 0), (15, 127), (15, 511))
Q_TDM_DEBUG_FIELDS = len(Q_TDM_SAMPLE_COORDS)
Q_TDM_DEBUG_BASE = MAPPING_DEBUG_FIELDS + Q_DEBUG_FIELDS
Q_ROPE_SAMPLE_COORDS = ((0, 0), (0, 15), (0, 63), (15, 0), (15, 15), (15, 63))
Q_ROPE_DEBUG_FIELDS = len(Q_ROPE_SAMPLE_COORDS)
Q_ROPE_DEBUG_BASE = Q_TDM_DEBUG_BASE + Q_TDM_DEBUG_FIELDS
Q_FRAGMENT_DEBUG_BASE = Q_ROPE_DEBUG_BASE + Q_ROPE_DEBUG_FIELDS
Q_NOPE_FRAGMENT_COUNT = 4
Q_NOPE_FRAGMENT_DWORDS = 16
Q_ROPE_FRAGMENT_DWORDS = 8
Q_FRAGMENT_DEBUG_DWORDS = (
    Q_NOPE_FRAGMENT_COUNT * Q_NOPE_FRAGMENT_DWORDS + Q_ROPE_FRAGMENT_DWORDS
)
KV_NOPE_SAMPLE_COORDS = Q_TDM_SAMPLE_COORDS
KV_NOPE_DEBUG_FIELDS = len(KV_NOPE_SAMPLE_COORDS)
KV_NOPE_DEBUG_BASE = Q_FRAGMENT_DEBUG_BASE + Q_FRAGMENT_DEBUG_DWORDS
KV_ROPE_SAMPLE_COORDS = Q_ROPE_SAMPLE_COORDS
KV_ROPE_DEBUG_FIELDS = len(KV_ROPE_SAMPLE_COORDS)
KV_ROPE_DEBUG_BASE = KV_NOPE_DEBUG_BASE + KV_NOPE_DEBUG_FIELDS
QK_N_TILES = 4
QK_ACC_DEBUG_BASE = MAPPING_DEBUG_FIELDS
QK_ACC_DWORDS = 8
QK_ACC_DEBUG_DWORDS = QK_N_TILES * QK_ACC_DWORDS
PAGE_MAX_DEBUG_BASE = QK_ACC_DEBUG_BASE + QK_ACC_DEBUG_DWORDS
PAGE_SUM_DEBUG_BASE = PAGE_MAX_DEBUG_BASE + 1
LSE_DEBUG_BASE = PAGE_SUM_DEBUG_BASE + 1
PROB_DEBUG_BASE = LSE_DEBUG_BASE + 1
PROB_DEBUG_VALUES = QK_ACC_DEBUG_DWORDS
PACKED_PROB_DEBUG_BASE = PROB_DEBUG_BASE + PROB_DEBUG_VALUES
PACKED_PROB_WORDS = PROB_DEBUG_VALUES // 4
PV_ACC_DEBUG_BASE = PACKED_PROB_DEBUG_BASE + PACKED_PROB_WORDS
PV_ACC_DWORDS = 8
PV_D_TILES = 32
DEBUG_FIELDS = PV_ACC_DEBUG_BASE + PV_ACC_DWORDS
NUM_Q_HEADS = 128
QK_NOPE_HEAD_DIM = 512
QK_ROPE_HEAD_DIM = 64
QK_HEAD_DIM = QK_NOPE_HEAD_DIM + QK_ROPE_HEAD_DIM
V_HEAD_DIM = QK_NOPE_HEAD_DIM
Q_HEAD_STRIDE = 768
Q_ROW_STRIDE = NUM_Q_HEADS * Q_HEAD_STRIDE
Q_GROUP_STRIDE = HEADS_PER_GROUP * Q_HEAD_STRIDE
Q_WAVE_STRIDE = HEADS_PER_WAVE * Q_HEAD_STRIDE
PAGE_SIZE = 64
KV_PAGE_ELEMENTS = PAGE_SIZE * QK_HEAD_DIM
KV_NOPE_PAGE_ELEMENTS = PAGE_SIZE * QK_NOPE_HEAD_DIM
LDS_WAVE_BYTES = 0x10000
LDS_TOTAL_BYTES = NUM_WAVES * LDS_WAVE_BYTES
Q_LDS_ROW_BYTES = QK_NOPE_HEAD_DIM + 16
Q_ROPE_LDS_OFFSET = HEADS_PER_WAVE * Q_LDS_ROW_BYTES
KV_NUM_STAGES = 5
KV_STAGE_BYTES = 0x2800
KV_ROPE_STAGE_BASE = 0xC800
KV_ROPE_STAGE_BYTES = 0x800
INSTRUCTION_PREFETCH_PAGES = 14
LOG2E = math.log2(math.e)


def _global_load_async_to_lds_b128(global_ptr, lds_ptr):
    rocdl.global_load_async_to_lds_b128(global_ptr.llvm_ptr, lds_ptr.llvm_ptr, 0, 0)


def _wait_asynccnt(count):
    rocdl.s_wait_asynccnt(count)


def _instruction_prefetch(num_pages):
    from flydsl._mlir.dialects import llvm

    lines = [
        f"s_prefetch_inst_pc_rel 0x{page * 0x1000:x}, $0, 31"
        for page in range(num_pages)
    ]
    llvm.inline_asm(
        None,
        [fx.Int32(0).ir_value()],
        "\n".join(lines),
        "s",
        has_side_effects=True,
    )


def _require(condition, message):
    if not condition:
        raise ValueError(message)


def _validate_stage1_inputs(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    qo_indptr,
    num_kv_splits_indptr,
    q_scale,
    kv_scale,
    softmax_scale,
    num_splits,
    page_size,
):
    arch = str(get_rocm_arch() or "").split(":", 1)[0]
    _require(arch == "gfx1250", f"expected gfx1250, got {arch or 'unknown'}")
    _require(
        isinstance(num_splits, int) and not isinstance(num_splits, bool),
        f"num_splits: expected int, got {type(num_splits).__name__}",
    )
    _require(num_splits > 0, f"num_splits: expected positive value, got {num_splits}")
    _require(
        page_size == PAGE_SIZE, f"page_size: expected {PAGE_SIZE}, got {page_size}"
    )

    _require(
        q.dtype == torch.float8_e4m3fn,
        f"q: expected torch.float8_e4m3fn, got {q.dtype}",
    )
    _require(
        q.ndim == 3 and tuple(q.shape[1:]) == (NUM_Q_HEADS, QK_HEAD_DIM),
        f"q: expected [total_q, {NUM_Q_HEADS}, {QK_HEAD_DIM}], got {list(q.shape)}",
    )
    expected_q_stride = (Q_ROW_STRIDE, Q_HEAD_STRIDE, 1)
    _require(
        tuple(q.stride()) == expected_q_stride,
        f"q: expected padded stride {list(expected_q_stride)}, got {list(q.stride())}",
    )
    batch = q.size(0)
    _require(batch > 0, "q: total_q/batch must be positive")

    _require(
        kv_buffer.dtype == torch.float8_e4m3fn,
        f"kv_buffer: expected torch.float8_e4m3fn, got {kv_buffer.dtype}",
    )
    _require(
        kv_buffer.ndim == 2 and kv_buffer.size(1) == KV_PAGE_ELEMENTS,
        "kv_buffer: expected a segmented 2D page view "
        f"[num_pages, {KV_PAGE_ELEMENTS}]; token-major 4D tensors are not accepted",
    )
    _require(kv_buffer.size(0) > 0, "kv_buffer: num_pages must be positive")
    _require(
        tuple(kv_buffer.stride()) == (KV_PAGE_ELEMENTS, 1),
        "kv_buffer: expected contiguous segmented pages with "
        f"page stride 0x{KV_PAGE_ELEMENTS:x}, got stride {list(kv_buffer.stride())}",
    )

    int32_inputs = {
        "kv_indptr": (kv_indptr, (batch + 1,)),
        "kv_page_indices": (kv_page_indices, None),
        "kv_last_page_lens": (kv_last_page_lens, (batch,)),
        "qo_indptr": (qo_indptr, (batch + 1,)),
        "num_kv_splits_indptr": (num_kv_splits_indptr, (batch + 1,)),
    }
    for name, (tensor, expected_shape) in int32_inputs.items():
        _require(
            tensor.dtype == torch.int32,
            f"{name}: expected torch.int32, got {tensor.dtype}",
        )
        _require(
            tensor.ndim == 1,
            f"{name}: expected a 1D tensor, got shape {list(tensor.shape)}",
        )
        if expected_shape is not None:
            _require(
                tuple(tensor.shape) == expected_shape,
                f"{name}: expected shape {list(expected_shape)}, got {list(tensor.shape)}",
            )
        _require(tensor.is_contiguous(), f"{name}: expected a contiguous tensor")
    _require(kv_page_indices.numel() > 0, "kv_page_indices: must not be empty")

    for name, scale in (("q_scale", q_scale), ("kv_scale", kv_scale)):
        _require(
            scale.dtype == torch.float32,
            f"{name}: expected torch.float32, got {scale.dtype}",
        )
        _require(
            tuple(scale.shape) == (1,),
            f"{name}: expected shape [1], got {list(scale.shape)}",
        )
        _require(scale.is_contiguous(), f"{name}: expected a contiguous tensor")

    if num_splits == 1:
        expected_data_shape = (batch, NUM_Q_HEADS, V_HEAD_DIM)
        expected_data_dtype = torch.bfloat16
    else:
        expected_data_shape = (batch, num_splits, NUM_Q_HEADS, V_HEAD_DIM)
        expected_data_dtype = torch.float32
    _require(
        split_data.dtype == expected_data_dtype,
        f"split_data: expected {expected_data_dtype} for num_splits={num_splits}, got {split_data.dtype}",
    )
    _require(
        tuple(split_data.shape) == expected_data_shape,
        f"split_data: expected shape {list(expected_data_shape)}, got {list(split_data.shape)}",
    )
    _require(split_data.is_contiguous(), "split_data: expected a contiguous tensor")

    expected_lse_shape = (batch, num_splits, NUM_Q_HEADS, 1)
    _require(
        split_lse.dtype == torch.float32,
        f"split_lse: expected torch.float32, got {split_lse.dtype}",
    )
    _require(
        tuple(split_lse.shape) == expected_lse_shape,
        f"split_lse: expected shape {list(expected_lse_shape)}, got {list(split_lse.shape)}",
    )
    _require(split_lse.is_contiguous(), "split_lse: expected a contiguous tensor")
    return batch


@flyc.jit
def launch_mla_pagesize64_fp8_fp8(
    ptr_r: fx.Pointer,
    ptr_lse: fx.Pointer,
    ptr_q: fx.Pointer,
    ptr_kv: fx.Pointer,
    kv_indptr: fx.Pointer,
    kv_page_indices: fx.Pointer,
    kv_last_page_lens: fx.Pointer,
    qo_indptr: fx.Pointer,
    num_kv_splits_indptr: fx.Pointer,
    q_scale: fx.Pointer,
    kv_scale: fx.Pointer,
    softmax_scale: fx.Float32,
    batch: fx.Int32,
    num_splits: fx.Int32,
    out_16_nosplit: fx.Constexpr[int],
    stream: fx.Stream = fx.Stream(None),
):

    @flyc.kernel(known_block_size=[BLOCK_THREADS, 1, 1])
    def kernel(
        ptr_r: fx.Pointer,
        ptr_lse: fx.Pointer,
        ptr_q: fx.Pointer,
        ptr_kv: fx.Pointer,
        kv_indptr: fx.Pointer,
        kv_page_indices: fx.Pointer,
        kv_last_page_lens: fx.Pointer,
        qo_indptr: fx.Pointer,
        num_kv_splits_indptr: fx.Pointer,
        q_scale: fx.Pointer,
        kv_scale: fx.Pointer,
        softmax_scale: fx.Float32,
        num_splits: fx.Int32,
    ):
        rocdl.disable_xdl_arb_stall()
        _instruction_prefetch(INSTRUCTION_PREFETCH_PAGES)

        lds_base = fx.SharedAllocator(static=False).allocate(LDS_TOTAL_BYTES)._ptr
        lds_base_idx = fx.index_cast(T.index, fx.ptrtoint(lds_base))
        lds_load_b128, lds_store_b128 = make_lds_copy_ops(128)

        tid = fx.Int32(fx.thread_idx.x)
        _, batch_id, z = fx.block_idx

        head_group = z & 1
        split_id = z >> 1
        wave_id = rocdl.readfirstlane(T.i32, tid >> 5)
        lane_id = tid & (WAVE_SIZE - 1)
        head_in_wave = lane_id & (HEADS_PER_WAVE - 1)
        lane_half = lane_id >> 4
        head = head_group * HEADS_PER_GROUP + wave_id * HEADS_PER_WAVE + head_in_wave

        qo_start = qo_indptr[batch_id]
        q_row_base = fx.Int64(qo_start) * Q_ROW_STRIDE
        q_tile_base = (
            q_row_base
            + fx.Int64(head_group) * Q_GROUP_STRIDE
            + fx.Int64(wave_id) * Q_WAVE_STRIDE
        )

        q_global = fx.Tensor(
            fx.make_view(
                fx.add_offset(ptr_q, q_tile_base),
                fx.make_layout(
                    (HEADS_PER_WAVE, QK_NOPE_HEAD_DIM), (QK_NOPE_HEAD_DIM, 1)
                ),
            )
        )
        q_tdm_atom = fx.rocdl.make_tdm_atom(
            q_global,
            [HEADS_PER_WAVE, None],
            strides=[Q_HEAD_STRIDE, None],
            num_warps=1,
            pad_interval=QK_NOPE_HEAD_DIM,
            pad_amount=Q_LDS_ROW_BYTES - QK_NOPE_HEAD_DIM,
        )
        q_lds_ptr = fx.recast_iter(
            fx.Int8, fx.add_offset(lds_base, wave_id * LDS_WAVE_BYTES)
        )
        q_lds = fx.Tensor(
            fx.make_view(
                q_lds_ptr,
                fx.make_layout(
                    (HEADS_PER_WAVE, QK_NOPE_HEAD_DIM), (Q_LDS_ROW_BYTES, 1)
                ),
            )
        )
        rocdl.sched_barrier(0)
        fx.copy(q_tdm_atom, q_global, q_lds)

        q_rope_lds_ptr = fx.recast_iter(
            fx.Int8,
            fx.add_offset(lds_base, wave_id * LDS_WAVE_BYTES + Q_ROPE_LDS_OFFSET),
        )
        rope_row = lane_id >> 2
        rope_chunk = lane_id & 3
        for rope_pass in range_constexpr(2):
            src_row = rope_pass * 8 + rope_row
            src_offset = (
                q_tile_base
                + fx.Int64(src_row) * Q_HEAD_STRIDE
                + QK_NOPE_HEAD_DIM
                + fx.Int64(rope_chunk) * 16
            )
            dst_offset = rope_pass * 512 + lane_id * 16
            _global_load_async_to_lds_b128(
                fx.add_offset(ptr_q, src_offset),
                fx.add_offset(q_rope_lds_ptr, dst_offset),
            )
        rocdl.sched_barrier(0)
        _wait_asynccnt(0)
        tdm_ops.tensor_wait(0)
        gpu.barrier()

        q_nope_fragments = []
        for k_fragment in range_constexpr(Q_NOPE_FRAGMENT_COUNT):
            fragment_chunks = []
            fragment_offset = (
                wave_id * LDS_WAVE_BYTES
                + head_in_wave * Q_LDS_ROW_BYTES
                + k_fragment * 128
                + lane_half * 16
            )
            for chunk in range_constexpr(4):
                fragment_chunks.append(
                    Vec(lds_load_b128(lds_base_idx, fragment_offset + chunk * 32))
                )
            q_nope_fragments.append(fragment_chunks)

        q_rope_fragment = []
        rope_fragment_offset = (
            wave_id * LDS_WAVE_BYTES
            + Q_ROPE_LDS_OFFSET
            + head_in_wave * QK_ROPE_HEAD_DIM
            + lane_half * 16
        )
        for chunk in range_constexpr(2):
            q_rope_fragment.append(
                Vec(lds_load_b128(lds_base_idx, rope_fragment_offset + chunk * 32))
            )
        rocdl.s_wait_dscnt(0)

        # ----------------------------------------------
        def _concat_wmma_operand(chunks):
            v01 = chunks[0].shuffle(chunks[1], list(range(8)))
            v23 = chunks[2].shuffle(chunks[3], list(range(8)))
            return v01.shuffle(v23, list(range(16)))

        def _concat_wmma_operand_k64(chunks):
            return chunks[0].shuffle(chunks[1], list(range(8)))

        def _rmem_i32(n, value):
            fragment = fx.make_rmem_tensor(n, fx.Int32)
            fragment.store(value)
            return fragment

        q_nope_operands = []
        for k_fragment in range_constexpr(Q_NOPE_FRAGMENT_COUNT):
            q_nope_operands.append(
                _rmem_i32(
                    Q_NOPE_FRAGMENT_DWORDS,
                    _concat_wmma_operand(q_nope_fragments[k_fragment]),
                )
            )
        q_rope_operand = _rmem_i32(
            Q_ROPE_FRAGMENT_DWORDS,
            _concat_wmma_operand_k64(q_rope_fragment),
        )

        qk_wmma_k128 = fx.make_mma_atom(
            fx.rocdl.WMMA(
                16,
                16,
                128,
                fx.Float8E4M3FN,
                fx.Float32,
            )
        )
        qk_wmma_k64 = fx.make_mma_atom(
            fx.rocdl.WMMA(
                16,
                16,
                64,
                fx.Float8E4M3FN,
                fx.Float32,
            )
        )
        score_scale = softmax_scale * q_scale[0] * kv_scale[0]
        scale_log2 = score_scale * fx.Float32(LOG2E)

        page_begin = kv_indptr[batch_id]
        page_end = kv_indptr[batch_id + fx.Int32(1)]

        split_page_begin = page_begin + split_id
        has_pages = split_page_begin < page_end
        last_page_len = kv_last_page_lens[batch_id]

        token_begin = wave_id * HEADS_PER_WAVE
        negative_inf = fx.Float32(float("-inf"))
        zero_out = Vec.filled(PV_ACC_DWORDS, 0.0, fx.Float32)

        @flyc.jit
        def issue_kv_page(page_pos, kv_stage):
            physical_page = kv_page_indices[page_pos]
            kv_page_base = fx.Int64(physical_page) * KV_PAGE_ELEMENTS
            kv_lds_ptr = fx.recast_iter(
                fx.Int8,
                fx.add_offset(
                    lds_base,
                    wave_id * LDS_WAVE_BYTES + kv_stage * KV_STAGE_BYTES,
                ),
            )
            kv_lds = fx.Tensor(
                fx.make_view(
                    kv_lds_ptr,
                    fx.make_layout(
                        (HEADS_PER_WAVE, QK_NOPE_HEAD_DIM),
                        (Q_LDS_ROW_BYTES, 1),
                    ),
                )
            )
            kv_rope_lds_ptr = fx.recast_iter(
                fx.Int8,
                fx.add_offset(
                    lds_base,
                    wave_id * LDS_WAVE_BYTES
                    + KV_ROPE_STAGE_BASE
                    + kv_stage * KV_ROPE_STAGE_BYTES,
                ),
            )
            kv_nope_global = fx.Tensor(
                fx.make_view(
                    fx.add_offset(
                        ptr_kv,
                        kv_page_base + fx.Int64(token_begin) * QK_NOPE_HEAD_DIM,
                    ),
                    fx.make_layout(
                        (HEADS_PER_WAVE, QK_NOPE_HEAD_DIM),
                        (QK_NOPE_HEAD_DIM, 1),
                    ),
                )
            )
            kv_nope_tdm_atom = fx.rocdl.make_tdm_atom(
                kv_nope_global,
                [HEADS_PER_WAVE, None],
                strides=[QK_NOPE_HEAD_DIM, None],
                num_warps=1,
                pad_interval=QK_NOPE_HEAD_DIM,
                pad_amount=Q_LDS_ROW_BYTES - QK_NOPE_HEAD_DIM,
            )
            rocdl.sched_barrier(0)
            fx.copy(kv_nope_tdm_atom, kv_nope_global, kv_lds)
            for rope_pass in range_constexpr(2):
                src_row = token_begin + rope_pass * 8 + rope_row
                src_offset = (
                    kv_page_base
                    + KV_NOPE_PAGE_ELEMENTS
                    + fx.Int64(src_row) * QK_ROPE_HEAD_DIM
                    + fx.Int64(rope_chunk) * 16
                )
                dst_offset = rope_pass * 512 + lane_id * 16
                _global_load_async_to_lds_b128(
                    fx.add_offset(ptr_kv, src_offset),
                    fx.add_offset(kv_rope_lds_ptr, dst_offset),
                )
            rocdl.sched_barrier(0)

        @flyc.jit
        def wait_kv_page(has_next, has_second_next):
            if has_second_next:
                _wait_asynccnt(4)
                tdm_ops.tensor_wait(2)
            else:
                if has_next:
                    _wait_asynccnt(2)
                    tdm_ops.tensor_wait(1)
                else:
                    _wait_asynccnt(0)
                    tdm_ops.tensor_wait(0)

        if has_pages:
            issue_kv_page(split_page_begin, fx.Int32(0))
            second_page = split_page_begin + num_splits
            if second_page < page_end:
                issue_kv_page(second_page, fx.Int32(1))
            third_page = split_page_begin + num_splits * 2
            if third_page < page_end:
                issue_kv_page(third_page, fx.Int32(2))

        def compute_pending_pv(
            running_outs,
            pending_alpha,
            pending_stage,
            pending_valid_len,
            pending_probability_words,
        ):
            p_operand = _rmem_i32(
                PACKED_PROB_WORDS,
                Vec.from_elements(pending_probability_words, fx.Int32),
            )
            v_row = (lane_id >> 3) * 4 + (lane_id & 3)
            v_col = ((lane_id & 7) >> 2) * 8

            def load_v_vector(dv_tile):
                v_tr8_chunks = []
                for token_tile in range_constexpr(QK_N_TILES):
                    v_byte_offset = (
                        token_tile * LDS_WAVE_BYTES
                        + pending_stage * KV_STAGE_BYTES
                        + v_row * Q_LDS_ROW_BYTES
                        + v_col
                        + dv_tile * 16
                    )
                    v_ptr = fx.add_offset(lds_base, v_byte_offset)
                    v_tr8_chunks.append(
                        Vec(
                            rocdl.ds_load_tr8_b64(
                                T.vec(2, T.i32),
                                v_ptr.llvm_ptr,
                            )
                        )
                    )
                v01 = v_tr8_chunks[0].shuffle(
                    v_tr8_chunks[1],
                    list(range(4)),
                )
                v23 = v_tr8_chunks[2].shuffle(
                    v_tr8_chunks[3],
                    list(range(4)),
                )
                v_vector = v01.shuffle(v23, list(range(PV_ACC_DWORDS)))
                sanitized_v_words = []
                for word in range_constexpr(PV_ACC_DWORDS):
                    token_base = (word // 2) * 16 + lane_half * 8 + (word % 2) * 4
                    word_mask = fx.Int32(0)
                    for byte in range_constexpr(4):
                        valid_token = fx.Int32(token_base + byte) < pending_valid_len
                        byte_mask = valid_token.select(
                            fx.Int32(0xFF),
                            fx.Int32(0),
                        )
                        word_mask = word_mask | (byte_mask << fx.Int32(byte * 8))
                    sanitized_v_words.append(v_vector[word] & word_mask)
                return Vec.from_elements(sanitized_v_words, fx.Int32)

            def finish_pv_tile(dv_tile, v_vector):
                v_operand = _rmem_i32(
                    PV_ACC_DWORDS,
                    v_vector,
                )
                page_acc = fx.make_rmem_tensor(PV_ACC_DWORDS, fx.Float32)
                page_acc.store(fx.constant_vector(0.0, T.vec(PV_ACC_DWORDS, T.f32)))
                fx.gemm(
                    qk_wmma_k64,
                    page_acc,
                    v_operand,
                    p_operand,
                    page_acc,
                )
                return running_outs[dv_tile] * pending_alpha + Vec(page_acc.load())

            updated_outs = []
            for dv_pair in range_constexpr(PV_D_TILES // 2):
                even_tile = dv_pair * 2
                odd_tile = even_tile + 1
                even_v = load_v_vector(even_tile)
                odd_v = load_v_vector(odd_tile)
                rocdl.s_wait_dscnt(4)
                updated_outs.append(finish_pv_tile(even_tile, even_v))
                rocdl.s_wait_dscnt(0)
                updated_outs.append(finish_pv_tile(odd_tile, odd_v))
            return updated_outs

        pending_word_zero = fx.Int32(0)
        init_state = (
            [negative_inf, fx.Float32(0.0)]
            + [zero_out for _ in range_constexpr(PV_D_TILES)]
            + [
                fx.Float32(1.0),
                fx.Int32(0),
                fx.Int32(0),
                fx.Int32(0),
            ]
            + [pending_word_zero for _ in range_constexpr(PACKED_PROB_WORDS)]
        )

        def pipeline_step(page_iter, state):
            running_max = fx.Float32(state[0])
            running_sum = fx.Float32(state[1])
            running_outs = [Vec(state[2 + i]) for i in range_constexpr(PV_D_TILES)]
            pending_base = 2 + PV_D_TILES
            pending_alpha = fx.Float32(state[pending_base])
            pending_stage = fx.Int32(state[pending_base + 1])
            pending_valid_len = fx.Int32(state[pending_base + 2])
            pending_valid = fx.Int32(state[pending_base + 3])
            pending_probability_words = [
                fx.Int32(state[pending_base + 4 + i])
                for i in range_constexpr(PACKED_PROB_WORDS)
            ]

            page_pos = fx.Int32(page_iter)
            is_last_page = page_pos == (page_end - fx.Int32(1))
            page_valid_len = is_last_page.select(
                last_page_len,
                fx.Int32(PAGE_SIZE),
            )
            page_iteration = (page_pos - split_page_begin) // num_splits
            kv_stage = page_iteration % KV_NUM_STAGES
            next_page = page_pos + num_splits
            second_next_page = page_pos + num_splits * 2
            wait_kv_page(
                next_page < page_end,
                second_next_page < page_end,
            )
            rocdl.s_barrier_signal(-1)

            qk_accs = []
            for _ in range_constexpr(QK_N_TILES):
                qk_acc = fx.make_rmem_tensor(QK_ACC_DWORDS, fx.Float32)
                qk_acc.store(fx.constant_vector(0.0, T.vec(QK_ACC_DWORDS, T.f32)))
                qk_accs.append(qk_acc)
            pv_ready_outs = running_outs
            if pending_valid != fx.Int32(0):
                pv_ready_outs = compute_pending_pv(
                    running_outs,
                    pending_alpha,
                    pending_stage,
                    pending_valid_len,
                    pending_probability_words,
                )
            rocdl.s_barrier_wait(-1)

            for n_tile in range_constexpr(QK_N_TILES):
                qk_acc = qk_accs[n_tile]
                k_fragment_groups = []
                for k_fragment in range_constexpr(Q_NOPE_FRAGMENT_COUNT):
                    k_fragment_chunks = []
                    k_fragment_offset = (
                        n_tile * LDS_WAVE_BYTES
                        + kv_stage * KV_STAGE_BYTES
                        + head_in_wave * Q_LDS_ROW_BYTES
                        + k_fragment * 128
                        + lane_half * 16
                    )
                    for chunk in range_constexpr(4):
                        k_fragment_chunks.append(
                            Vec(
                                lds_load_b128(
                                    lds_base_idx,
                                    k_fragment_offset + chunk * 32,
                                )
                            )
                        )
                    k_fragment_groups.append(k_fragment_chunks)

                k_rope_fragment = []
                k_rope_fragment_offset = (
                    n_tile * LDS_WAVE_BYTES
                    + KV_ROPE_STAGE_BASE
                    + kv_stage * KV_ROPE_STAGE_BYTES
                    + head_in_wave * QK_ROPE_HEAD_DIM
                    + lane_half * 16
                )
                for chunk in range_constexpr(2):
                    k_rope_fragment.append(
                        Vec(
                            lds_load_b128(
                                lds_base_idx,
                                k_rope_fragment_offset + chunk * 32,
                            )
                        )
                    )

                for k_fragment in range_constexpr(Q_NOPE_FRAGMENT_COUNT):
                    remaining_ds = (Q_NOPE_FRAGMENT_COUNT - 1 - k_fragment) * 4 + 2
                    rocdl.s_wait_dscnt(remaining_ds)
                    k_operand = _rmem_i32(
                        Q_NOPE_FRAGMENT_DWORDS,
                        _concat_wmma_operand(k_fragment_groups[k_fragment]),
                    )
                    fx.gemm(
                        qk_wmma_k128,
                        qk_acc,
                        k_operand,
                        q_nope_operands[k_fragment],
                        qk_acc,
                    )

                rocdl.s_wait_dscnt(0)
                k_rope_operand = _rmem_i32(
                    Q_ROPE_FRAGMENT_DWORDS,
                    _concat_wmma_operand_k64(k_rope_fragment),
                )
                fx.gemm(
                    qk_wmma_k64,
                    qk_acc,
                    k_rope_operand,
                    q_rope_operand,
                    qk_acc,
                )

            qk_acc_vectors = [
                Vec(qk_accs[i].load()) for i in range_constexpr(QK_N_TILES)
            ]
            masked_logits = []
            for n_tile in range_constexpr(QK_N_TILES):
                masked_tile = []
                for i in range_constexpr(QK_ACC_DWORDS):
                    logical_key = n_tile * 16 + lane_half * 8 + i
                    valid_key = fx.Int32(logical_key) < page_valid_len
                    masked_tile.append(
                        valid_key.select(
                            qk_acc_vectors[n_tile][i],
                            negative_inf,
                        )
                    )
                masked_logits.append(masked_tile)

            local_max = masked_logits[0][0]
            for n_tile in range_constexpr(QK_N_TILES):
                for i in range_constexpr(QK_ACC_DWORDS):
                    local_max = local_max.maximumf(masked_logits[n_tile][i])
            page_max = local_max.maximumf(
                local_max.shuffle_xor(
                    fx.Int32(16),
                    fx.Int32(WAVE_SIZE),
                )
            )
            new_max = running_max.maximumf(page_max)
            alpha = ((running_max - new_max) * scale_log2).exp2()

            probabilities = []
            local_sum = fx.Float32(0.0)
            for n_tile in range_constexpr(QK_N_TILES):
                for i in range_constexpr(QK_ACC_DWORDS):
                    probability = (
                        (masked_logits[n_tile][i] - new_max) * scale_log2
                    ).exp2()
                    probabilities.append(probability)
                    local_sum = local_sum + probability
            page_sum = local_sum + local_sum.shuffle_xor(
                fx.Int32(16),
                fx.Int32(WAVE_SIZE),
            )
            new_sum = running_sum * alpha + page_sum

            packed_probability_words = []
            for word in range_constexpr(PACKED_PROB_WORDS):
                base = word * 4
                packed = rocdl.cvt_pk_fp8_f32(
                    T.i32,
                    probabilities[base],
                    probabilities[base + 1],
                    fx.Int32(0),
                    0,
                )
                packed = rocdl.cvt_pk_fp8_f32(
                    T.i32,
                    probabilities[base + 2],
                    probabilities[base + 3],
                    packed,
                    1,
                )
                packed_probability_words.append(fx.Int32(packed))

            future_page = page_pos + num_splits * 3
            if future_page < page_end:
                future_stage = (page_iteration + 3) % KV_NUM_STAGES
                issue_kv_page(future_page, future_stage)
            gpu.barrier()
            return (
                [new_max, new_sum]
                + pv_ready_outs
                + [
                    alpha,
                    kv_stage,
                    page_valid_len,
                    fx.Int32(1),
                ]
                + packed_probability_words
            )

        remaining_pages = page_end - split_page_begin
        split_page_count = (remaining_pages > fx.Int32(0)).select(
            (remaining_pages + num_splits - fx.Int32(1)) // num_splits,
            fx.Int32(0),
        )
        paired_page_count = (split_page_count // 2) * 2
        pair_end = split_page_begin + paired_page_count * num_splits

        pair_results = init_state
        for page_iter, state in range(
            fx.Int64(split_page_begin),
            fx.Int64(pair_end),
            fx.Int64(num_splits * 2),
            init=init_state,
        ):
            even_state = pipeline_step(fx.Int32(page_iter), state)
            odd_state = pipeline_step(
                fx.Int32(page_iter) + num_splits,
                even_state,
            )
            pair_results = yield odd_state

        tail_results = pair_results
        for page_iter, state in range(
            fx.Int64(pair_end),
            fx.Int64(page_end),
            fx.Int64(num_splits),
            init=pair_results,
        ):
            tail_state = pipeline_step(fx.Int32(page_iter), state)
            tail_results = yield tail_state

        loop_results = tail_results
        running_max = fx.Float32(loop_results[0])
        running_sum = fx.Float32(loop_results[1])
        output_accs = [Vec(loop_results[2 + i]) for i in range_constexpr(PV_D_TILES)]
        if has_pages:
            pending_base = 2 + PV_D_TILES
            output_accs = compute_pending_pv(
                output_accs,
                fx.Float32(loop_results[pending_base]),
                fx.Int32(loop_results[pending_base + 1]),
                fx.Int32(loop_results[pending_base + 2]),
                [
                    fx.Int32(loop_results[pending_base + 4 + i])
                    for i in range_constexpr(PACKED_PROB_WORDS)
                ],
            )
            gpu.barrier()
            inv_page_sum = fx.Float32(rocdl.rcp(T.f32, running_sum))
            output_scale = kv_scale[0] * inv_page_sum
            output_base = (
                (fx.Int64(batch_id) * fx.Int64(num_splits) + fx.Int64(split_id))
                * NUM_Q_HEADS
                + fx.Int64(head)
            ) * V_HEAD_DIM
            if const_expr(out_16_nosplit):
                for dv_tile in range_constexpr(PV_D_TILES):
                    output_values = []
                    for i in range_constexpr(PV_ACC_DWORDS):
                        output_values.append(output_accs[dv_tile][i] * output_scale)
                    packed_bf16 = (
                        Vec.from_elements(output_values, fx.Float32)
                        .to(fx.BFloat16)
                        .bitcast(fx.Int32)
                    )
                    lds_output_offset = (
                        wave_id * LDS_WAVE_BYTES
                        + head_in_wave * (V_HEAD_DIM * 2)
                        + (dv_tile * 16 + lane_half * 8) * 2
                    )
                    lds_store_b128(
                        lds_base_idx,
                        lds_output_offset,
                        packed_bf16,
                    )
                rocdl.s_wait_dscnt(0)
                gpu.barrier()

                output_lds_ptr_type = fx.PointerType.get(
                    elem_ty=fx.BFloat16.ir_type,
                    address_space=fx.AddressSpace.Shared,
                    alignment=16,
                )
                output_lds_ptr = fx.inttoptr(
                    output_lds_ptr_type,
                    fx.ptrtoint(
                        fx.add_offset(
                            lds_base,
                            wave_id * LDS_WAVE_BYTES,
                        )
                    ),
                )
                output_head_begin = (
                    head_group * HEADS_PER_GROUP + wave_id * HEADS_PER_WAVE
                )
                output_tile_base = (
                    fx.Int64(qo_start) * NUM_Q_HEADS + fx.Int64(output_head_begin)
                ) * V_HEAD_DIM
                output_global = fx.Tensor(
                    fx.make_view(
                        fx.add_offset(ptr_r, output_tile_base),
                        fx.make_layout(
                            (HEADS_PER_WAVE, V_HEAD_DIM),
                            (V_HEAD_DIM, 1),
                        ),
                    )
                )
                output_lds = fx.Tensor(
                    fx.make_view(
                        output_lds_ptr,
                        fx.make_layout(
                            (HEADS_PER_WAVE, V_HEAD_DIM),
                            (V_HEAD_DIM, 1),
                        ),
                    )
                )
                output_tdm_atom = fx.rocdl.make_tdm_atom(
                    output_global,
                    [HEADS_PER_WAVE, None],
                    strides=[V_HEAD_DIM, None],
                    num_warps=1,
                )
                fx.copy(
                    output_tdm_atom,
                    output_lds,
                    output_global,
                )
                tdm_ops.tensor_wait(0)
            else:
                for dv_tile in range_constexpr(PV_D_TILES):
                    for i in range_constexpr(PV_ACC_DWORDS):
                        output_dim = dv_tile * 16 + lane_half * 8 + i
                        ptr_r[output_base + output_dim] = (
                            output_accs[dv_tile][i] * output_scale
                        )

            if lane_half == fx.Int32(0):
                lse_offset = (
                    fx.Int64(batch_id) * fx.Int64(num_splits) + fx.Int64(split_id)
                ) * NUM_Q_HEADS + fx.Int64(head)
                lse = running_max * score_scale + fmath.log(running_sum)
                ptr_lse[lse_offset] = lse

    kernel(
        ptr_r,
        ptr_lse,
        ptr_q,
        ptr_kv,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        qo_indptr,
        num_kv_splits_indptr,
        q_scale,
        kv_scale,
        softmax_scale,
        num_splits,
    ).launch(
        grid=(1, batch, NUM_HEAD_GROUPS * num_splits),
        block=(BLOCK_THREADS, 1, 1),
        stream=stream,
    )


def mla_fwd_decode_pagesize64_fp8_fp8_gfx1250(
    split_data,
    split_lse,
    q,
    kv_buffer,
    kv_indptr,
    kv_page_indices,
    kv_last_page_lens,
    qo_indptr,
    num_kv_splits_indptr,
    q_scale,
    kv_scale,
    softmax_scale,
    num_splits,
    *,
    page_size=PAGE_SIZE,
    stream=None,
):
    """
    ``kv_buffer`` must be an explicitly flattened segmented-page view.  Within
    each 0x9000-element page, bytes ``[0, 0x8000)`` contain the 64x512 NOPE
    segment and bytes ``[0x8000, 0x9000)`` contain the 64x64 RoPE segment.
    """
    batch = _validate_stage1_inputs(
        split_data,
        split_lse,
        q,
        kv_buffer,
        kv_indptr,
        kv_page_indices,
        kv_last_page_lens,
        qo_indptr,
        num_kv_splits_indptr,
        q_scale,
        kv_scale,
        softmax_scale,
        num_splits,
        page_size,
    )
    if stream is None:
        stream = torch.cuda.current_stream(q.device)
    output_type = fx.BFloat16 if num_splits == 1 else fx.Float32

    launch_mla_pagesize64_fp8_fp8(
        flyc.from_c_void_p(output_type, split_data.data_ptr()),
        flyc.from_c_void_p(fx.Float32, split_lse.data_ptr()),
        flyc.from_c_void_p(fx.Int8, q.data_ptr()),
        flyc.from_c_void_p(fx.Int8, kv_buffer.data_ptr()),
        flyc.from_c_void_p(fx.Int32, kv_indptr.data_ptr()),
        flyc.from_c_void_p(fx.Int32, kv_page_indices.data_ptr()),
        flyc.from_c_void_p(fx.Int32, kv_last_page_lens.data_ptr()),
        flyc.from_c_void_p(fx.Int32, qo_indptr.data_ptr()),
        flyc.from_c_void_p(fx.Int32, num_kv_splits_indptr.data_ptr()),
        flyc.from_c_void_p(fx.Float32, q_scale.data_ptr()),
        flyc.from_c_void_p(fx.Float32, kv_scale.data_ptr()),
        float(softmax_scale),
        batch,
        num_splits,
        int(num_splits == 1),
        stream=stream,
    )


launch_mla_pagesize64_fp8_fp8.compile_hints = {
    "llvm_options": {
        "amdgpu-expert-scheduling-mode": True,
    },
}
