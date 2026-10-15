#include "../../native/backends/metal_graph/native_linear.hpp"

#include <iostream>
#include <limits>

using namespace tc::metal_graph;
using namespace tc::metal_graph::dense_linear;
namespace {
size_t checks = 0;
void check(bool value,const char* message) {
    ++checks;
    if (!value) throw std::runtime_error(message);
}
template<class F> void rejects(F operation) {
    bool rejected=false;
    try { operation(); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"invalid native linear contract accepted");
}
Layout matrix(uint32_t rows,uint32_t columns,ScalarType type=ScalarType::BF16,size_t padding=8,size_t offset=16) {
    size_t element=scalar_bytes(type),pitch=(size_t(columns)+padding)*element;
    return {rows,columns,pitch,size_t(rows-1)*pitch+size_t(columns)*element,offset,type};
}
}

int main() {
    try {
        for (auto g : {Geometry{4096,12288},{4096,24576},{12288,4096},{12288,32768},{3840,10240},
                       Geometry{64,4096},{4096,64},{128,192}}) {
            validate_geometry(g);
            const auto shader=source(g);
            check(shader.find("#define TC_LINEAR_K "+std::to_string(g.input_columns))!=std::string::npos,"source K specialization");
            check(shader.find("#define TC_LINEAR_N "+std::to_string(g.output_columns))!=std::string::npos,"source N specialization");
        }
        for (auto g : {Geometry{0,128},{128,0},{127,128},{128,127},{12416,128},{128,32896}})
            rejects([&] { validate_geometry(g); });

        for (uint32_t rows : {1U,31U,32U,33U,uint32_t(std::numeric_limits<int>::max())}) {
            const auto m=matrix(rows,128);
            check(validate_layout(m)==size_t(rows-1)*272+256,"padded physical extent");
        }
        auto padded=matrix(3,128);
        check(validate_layout(padded)==800,"exact-end retained view accepted");
        auto bad=padded; bad.view_size_bytes-=2;
        rejects([&] { validate_layout(bad); });
        for (auto type : {ScalarType::FP16,ScalarType::UInt32,ScalarType::Bytes,ScalarType::Int8}) {
            bad=padded; bad.type=type;
            rejects([&] { validate_layout(bad); });
        }
        bad=padded; bad.rows=0; rejects([&] { validate_layout(bad); });
        bad=padded; bad.rows=uint32_t(std::numeric_limits<int>::max())+1;
        rejects([&] { validate_layout(bad); });
        bad=padded; bad.columns=0; rejects([&] { validate_layout(bad); });
        bad=padded; bad.columns=uint32_t(std::numeric_limits<int>::max())+1;
        rejects([&] { validate_layout(bad); });
        bad=padded; bad.row_stride_bytes=0; rejects([&] { validate_layout(bad); });
        bad=padded; bad.row_stride_bytes=254; rejects([&] { validate_layout(bad); });
        bad=padded; bad.row_stride_bytes=273; rejects([&] { validate_layout(bad); });
        bad=padded; bad.offset_bytes=1; rejects([&] { validate_layout(bad); });
        bad=padded; bad.row_stride_bytes=(size_t(std::numeric_limits<int>::max())+1)*2;
        rejects([&] { validate_layout(bad); });
        bad=padded; bad.offset_bytes=std::numeric_limits<size_t>::max()-1; bad.view_size_bytes=4;
        rejects([&] { validate_layout(bad); });

        const Geometry g{4096,24576};
        auto x=matrix(33,4096),w=matrix(24576,4096),y=matrix(33,24576);
        auto checked=validate_bindings(g,x,w,y);
        check(checked.parameters.rows==33 && checked.parameters.input_stride==4104 &&
              checked.parameters.weight_stride==4104 && checked.parameters.output_stride==24584,
              "BF16 binding parameters retain physical strides");
        auto y32=matrix(33,24576,ScalarType::FP32);
        checked=validate_bindings(g,x,w,y32);
        check(checked.parameters.output_stride==24584,"explicit FP32 output stride");
        auto bias=matrix(1,24576);
        checked=validate_bindings(g,x,w,y,bias);
        check(checked.bias_bytes==24576*2,"checkpoint bias extent");
        auto wrong_bias=bias; wrong_bias.rows=2;
        rejects([&] { validate_bindings(g,x,w,y,wrong_bias); });
        wrong_bias=bias; wrong_bias.columns--;
        rejects([&] { validate_bindings(g,x,w,y,wrong_bias); });
        wrong_bias=bias; wrong_bias.type=ScalarType::FP32;
        rejects([&] { validate_bindings(g,x,w,y,wrong_bias); });
        auto wrong_x=x; wrong_x.type=ScalarType::FP32;
        rejects([&] { validate_bindings(g,wrong_x,w,y); });
        auto wrong_w=w; wrong_w.type=ScalarType::FP32;
        rejects([&] { validate_bindings(g,x,wrong_w,y); });
        wrong_w=w; wrong_w.rows--;
        rejects([&] { validate_bindings(g,x,wrong_w,y); });
        auto wrong_y=y; wrong_y.rows--;
        rejects([&] { validate_bindings(g,x,w,wrong_y); });
        wrong_y=y; wrong_y.columns--;
        rejects([&] { validate_bindings(g,x,w,wrong_y); });
        wrong_y=y; wrong_y.type=ScalarType::UInt32;
        rejects([&] { validate_bindings(g,x,w,wrong_y); });

        check(ranges_overlap(16,256,16,256),"equal intervals overlap");
        check(ranges_overlap(16,256,270,2),"partial overlap");
        check(ranges_overlap(270,2,16,256),"overlap symmetric");
        check(!ranges_overlap(16,256,272,2),"touching intervals are disjoint");
        check(!ranges_overlap(16,0,16,256),"empty interval is disjoint");
        check(!ranges_overlap(std::numeric_limits<size_t>::max()-31,16,
                              std::numeric_limits<size_t>::max()-15,4),"high offsets avoid sum overflow");
        check(NativeLinear::supports_bias(),"bias capability must be explicit");
        std::cout<<"native_linear_cpu=PASS checks="<<checks<<" device_created=0 model_loads=0\n";
    } catch (const std::exception& e) {
        std::cerr<<"native_linear_cpu=FAIL "<<e.what()<<'\n';
        return 1;
    }
}
