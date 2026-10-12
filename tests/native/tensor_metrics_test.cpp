#include "tensor_metrics.hpp"
#include <array>
#include <cassert>
#include <limits>

int main() {
    const std::array<float,3> b{1,2,3},a{1.01f,2.02f,3.03f},z{0,0,0};
    auto exact=tc::compare_float32(b,b);
    assert(exact.rel_l2==0 && exact.cosine>.999999 && exact.n1_layer() && exact.n1_final());
    auto error=tc::compare_float32(a,b);
    assert(std::abs(error.rel_l2-.01)<1e-6 && error.cosine>.999999 && error.n1_final());
    const std::array<float,3> bad{2,2,3},small{1e-8f,0,0},large{1e-5f,0,0};
    assert(!tc::compare_float32(bad,b).n1_layer());
    assert(tc::compare_float32(z,z).n1_layer());
    assert(tc::compare_float32(small,z).n1_layer());
    assert(!tc::compare_float32(large,z).n1_final());
    assert(tc::compare_float32(z,b).cosine==0);
    tc::Float32ComparisonAccumulator parts;
    parts.add({a.data(),1},{b.data(),1});parts.add({a.data()+1,2},{b.data()+1,2});
    assert(parts.elements==3 && std::abs(parts.result().rel_l2-error.rel_l2)<1e-12);
    bool nonfinite=false,geometry=false;
    try { const std::array<float,3> n{0,std::numeric_limits<float>::infinity(),0};(void)tc::compare_float32(n,b); }
    catch (const std::invalid_argument &) {nonfinite=true;}
    try { (void)tc::compare_float32({},b); } catch (const std::invalid_argument &) {geometry=true;}
    assert(nonfinite && geometry);
}
