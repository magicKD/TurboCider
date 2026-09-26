#pragma once

// Shape-specialized runtime Metal kernels. Keep model dispatch separate from
// kernel implementations; each header owns its source and launch geometry.
#include "metal/qkv.hpp"
#include "metal/qkv_projection.hpp"
#include "metal/norm_mod.hpp"
#include "metal/norm_mod_virtual.hpp"
#include "metal/swiglu_gemm.hpp"
#include "metal/projection.hpp"
#include "metal/gate_norm.hpp"
#include "metal/gate_norm_virtual.hpp"
