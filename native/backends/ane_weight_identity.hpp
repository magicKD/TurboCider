#pragma once

#include "ane_runtime.hpp"
#include <tuple>

namespace tc::ane {

inline bool same_weight_generation(const std::weak_ptr<void> &a,const std::weak_ptr<void> &b) {
    return !a.owner_before(b) && !b.owner_before(a);
}
inline bool same_weight_matrix(const DeviceMatrixView &a,const DeviceMatrixView &b) {
    return same_weight_generation(a.allocation_identity,b.allocation_identity) &&
        std::tie(a.buffer,a.buffer_bytes,a.offset_bytes,a.rows,a.cols,a.row_stride_bytes,a.dtype)==
        std::tie(b.buffer,b.buffer_bytes,b.offset_bytes,b.rows,b.cols,b.row_stride_bytes,b.dtype);
}
// A prefetched W8 bank is tied to actual immutable physical producers. Scale
// metadata may cross raw refills, but that never permits activating old codes
// for another allocation/content generation, even at a recycled address.
inline bool same_weight_region(const DeviceWeightRegion &a,const DeviceWeightRegion &b) {
    const auto &x=a.source,&y=b.source;const auto &p=a.selection,&q=b.selection;
    const auto metadata=[](const auto &a,const auto &b) {return bool(a)==bool(b) && (!a || same_weight_matrix(*a,*b));};
    return same_weight_generation(x.allocation_identity,y.allocation_identity) &&
        same_weight_generation(x.logical_content_identity,y.logical_content_identity) && x.immutable_generation==y.immutable_generation &&
        std::tie(x.buffer,x.buffer_bytes,x.offset_bytes,x.row_stride_bytes,x.rows,x.cols,x.encoding,x.dense_dtype,x.group_size)==
        std::tie(y.buffer,y.buffer_bytes,y.offset_bytes,y.row_stride_bytes,y.rows,y.cols,y.encoding,y.dense_dtype,y.group_size) &&
        metadata(x.scales,y.scales) && metadata(x.offsets,y.offsets) &&
        std::tie(p.row_begin,p.rows,p.column_begin,p.columns,p.rotation_block,p.rotation_seed,p.transpose,p.basis,p.activation_group_size)==
        std::tie(q.row_begin,q.rows,q.column_begin,q.columns,q.rotation_block,q.rotation_seed,q.transpose,q.basis,q.activation_group_size);
}

} // namespace tc::ane
