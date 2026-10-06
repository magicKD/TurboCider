#pragma once
#include <algorithm>
#include <stdexcept>

namespace tc::ane {
enum class RowPlacement { Suffix, ImagePrefix, ImageTail };
struct RowPolicy {
    RowPlacement placement=RowPlacement::Suffix;
    int protected_suffix_rows=0;
};
inline const char *row_placement_name(RowPlacement placement) {
    switch(placement) {
        case RowPlacement::Suffix:return "suffix";
        case RowPlacement::ImagePrefix:return "image_prefix";
        case RowPlacement::ImageTail:return "image_tail";
    }
    throw std::invalid_argument("unknown ANE row placement");
}
inline int eligible_ane_rows(int rows,RowPolicy policy) {
    (void)row_placement_name(policy.placement);
    if(rows<=0 || policy.protected_suffix_rows<0 || policy.protected_suffix_rows>rows ||
            (policy.placement==RowPlacement::Suffix && policy.protected_suffix_rows))
        throw std::invalid_argument("invalid ANE row policy/geometry");
    return rows-policy.protected_suffix_rows;
}
struct RowWindow {
    int first=0,count=0,total=0;
    int end() const {return first+count;}
    int gpu_before() const {return first;}
    int gpu_after() const {return total-end();}
};
inline RowWindow plan_row_window(int rows,int ane_rows,RowPolicy policy) {
    const int eligible=eligible_ane_rows(rows,policy);
    if(ane_rows<=0 || ane_rows>eligible)throw std::invalid_argument("ANE row window exceeds eligible rows");
    const int first=policy.placement==RowPlacement::ImagePrefix?0:eligible-ane_rows;
    return {first,ane_rows,rows};
}
} // namespace tc::ane
