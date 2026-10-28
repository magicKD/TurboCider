// Original bounded SG-matrix experiment. Not a copied MLX/GGML kernel.
// Actual strides belong to every operand, including packed words and metadata.
constexpr uint BK = 32;
constexpr uint WEIGHT_LD = BN + PAD;
constexpr uint GROUP_THREADS = (BM / 16) * (BN / 16) * 32;
const uint M = X_shape[0], K = X_shape[1], N = S_shape[0];
const uint tid = thread_index_in_threadgroup;
const uint sg = simdgroup_index_in_threadgroup;
const uint rbase = threadgroup_position_in_grid.y * BM;
const uint cbase = threadgroup_position_in_grid.x * BN;
const uint sr = (sg / (BN / 16)) * 16, sc = (sg % (BN / 16)) * 16;
threadgroup half left[BM * BK], right[BK * WEIGHT_LD];
threadgroup float result[BM * BN];
simdgroup_float8x8 c00 = make_filled_simdgroup_matrix<float, 8>(0.f);
simdgroup_float8x8 c01 = make_filled_simdgroup_matrix<float, 8>(0.f);
simdgroup_float8x8 c10 = make_filled_simdgroup_matrix<float, 8>(0.f);
simdgroup_float8x8 c11 = make_filled_simdgroup_matrix<float, 8>(0.f);
for (uint base = 0; base < K; base += BK) {
    for (uint index = tid; index < BM * BK; index += GROUP_THREADS) {
        const uint row = rbase + index / BK, k = base + index % BK;
        left[index] = row < M && k < K ? X[row * X_strides[0] + k * X_strides[1]] : half(0);
    }
    for (uint index = tid; index < BK * BN; index += GROUP_THREADS) {
        const uint local_k = COALESCED ? index % BK : index / BN;
        const uint local_col = COALESCED ? index / BK : index % BN;
        const uint k = base + local_k, col = cbase + local_col;
        half value = 0;
        if (k < K && col < N) {
            constexpr uint factor = 32 / BITS, mask = (1u << BITS) - 1u;
            const uint code = (Q[col * Q_strides[0] + (k / factor) * Q_strides[1]] >> ((k % factor) * BITS)) & mask;
            const uint group = k / GROUP_SIZE;
            const float scale = float(S[col * S_strides[0] + group * S_strides[1]]);
            const float bias = float(B[col * B_strides[0] + group * B_strides[1]]);
            value = half(fma(float(code), scale, bias));
        }
        right[local_k * WEIGHT_LD + local_col] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint step = 0; step < BK; step += 8) {
        simdgroup_half8x8 a0, a1, b0, b1;
        simdgroup_load(a0, left + sr * BK + step, BK);
        simdgroup_load(a1, left + (sr + 8) * BK + step, BK);
        simdgroup_load(b0, right + step * WEIGHT_LD + sc, WEIGHT_LD);
        simdgroup_load(b1, right + step * WEIGHT_LD + sc + 8, WEIGHT_LD);
        simdgroup_multiply_accumulate(c00, a0, b0, c00);
        simdgroup_multiply_accumulate(c01, a0, b1, c01);
        simdgroup_multiply_accumulate(c10, a1, b0, c10);
        simdgroup_multiply_accumulate(c11, a1, b1, c11);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
}
simdgroup_store(c00, result + sr * BN + sc, BN);
simdgroup_store(c01, result + sr * BN + sc + 8, BN);
simdgroup_store(c10, result + (sr + 8) * BN + sc, BN);
simdgroup_store(c11, result + (sr + 8) * BN + sc + 8, BN);
threadgroup_barrier(mem_flags::mem_threadgroup);
for (uint index = tid; index < BM * BN; index += GROUP_THREADS) {
    const uint row = rbase + index / BN, col = cbase + index % BN;
    if (row < M && col < N) Y[row * N + col] = half(result[index]);
}
