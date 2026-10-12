#include "../../native/backends/ane_weight_identity.hpp"
#include <iostream>
#include <stdexcept>

using namespace tc::ane;
void check(bool condition) {if(!condition)throw std::runtime_error("weight generation/physical prefetch identity mismatch");}
int main() {
    auto allocation=std::make_shared<int>(1),content=std::make_shared<int>(2);
    DeviceWeightView source;source.buffer=allocation.get();source.buffer_bytes=1024;source.row_stride_bytes=72;
    source.rows=8;source.cols=128;source.encoding=DeviceWeightEncoding::GgufQ4_0;source.owner=allocation;
    source.allocation_identity=allocation;source.logical_content_identity=content;source.immutable_generation=true;
    DeviceWeightRegion original{source,{0,8,0,128,128}};
    check(same_weight_region(original,original));
    auto different=original;
    auto newer=std::make_shared<int>(3);different.source.allocation_identity=newer;
    check(!same_weight_region(original,different)); // same raw address, another producer
    different=original;different.source.logical_content_identity=newer;check(!same_weight_region(original,different));
    different=original;different.source.immutable_generation=false;check(!same_weight_region(original,different));
    different=original;different.selection.rotation_seed++;check(!same_weight_region(original,different));
    different=original;different.source.row_stride_bytes++;check(!same_weight_region(original,different));
    different=original;different.source.encoding=DeviceWeightEncoding::GgufQ8_0;check(!same_weight_region(original,different));
    different=original;different.source.scales=DeviceMatrixView{};check(!same_weight_region(original,different));
    original.source.scales=DeviceMatrixView{};original.source.scales->allocation_identity=allocation;
    different=original;different.source.scales->allocation_identity=newer;check(!same_weight_region(original,different));
    original.source.scales.reset();different=original;different.source.logical_content_identity.reset();
    content.reset();check(original.source.logical_content_identity.expired());check(!same_weight_region(original,different));
    std::cout<<"PASS shared physical prefetch identity: allocation/content/metadata generations, source recipe, expired vs absent\n";
}
