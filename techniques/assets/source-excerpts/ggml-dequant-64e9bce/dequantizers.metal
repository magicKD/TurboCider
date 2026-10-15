// Selected unchanged functions from ggml-org/llama.cpp, MIT.
// Copyright (c) 2023-2026 The ggml authors. See LICENSE and PROVENANCE.json.
// Only the probe structs/caller are original; these two bodies are excerpts.
template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        reg_f[i/2][2*(i%2) + 0] = d1 * (qs[i] & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qs[i] & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 4; ++i) {
        reg_f[i] = float4(qs[4*il + i]) * d;
    }

    reg = (type4x4) reg_f;
}
