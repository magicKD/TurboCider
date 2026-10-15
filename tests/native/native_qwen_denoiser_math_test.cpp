// CPU-only oracle fixtures against shared sequence geometry and the new native
// constant generators. The schedule reference is documentation of the existing
// pipeline's contract, not a replacement scheduler or a native-device pass.
#include "../../native/models/qwen21/native_denoiser.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <vector>

namespace qwen = tc::qwen21;
namespace native = tc::qwen21::native_graph;
namespace {
void check(bool value, const char* message) {
    if (!value) throw std::runtime_error(message);
}
template<class Function> void rejects(Function&& function) {
    bool rejected=false;
    try { function(); } catch (const std::invalid_argument&) { rejected=true; }
    check(rejected,"invalid denoiser math input accepted");
}
float bf16(float value) {
    uint32_t bits=std::bit_cast<uint32_t>(value);
    if ((bits&0x7fffffffU)>0x7f800000U)
        return std::bit_cast<float>((bits&0xffff0000U)|0x00400000U);
    bits+=0x7fffU+((bits>>16)&1U);
    return std::bit_cast<float>(bits&0xffff0000U);
}

// transformer.cpp:224-237. Time frequencies deliberately use double before a
// float cast. The time*1000 and subsequent frequency multiply are FP32, and
// trig concatenates all cosine channels before all sine channels.
std::array<float,512> time_input_oracle(float sigma) {
    if (!std::isfinite(sigma) || sigma<0.f || sigma>1.f)
        throw std::invalid_argument("sigma outside [0,1]");
    std::array<float,512> result{};
    for (size_t row=0;row<2;++row) for (size_t i=0;i<128;++i) {
        const float frequency=float(std::exp(-std::log(10000.0)*double(i)/128.0));
        const float time=(row==0 ? sigma : 0.f)*1000.f;
        const float angle=time*frequency;
        result[row*256+i]=bf16(std::cos(angle));
        result[row*256+128+i]=bf16(std::sin(angle));
    }
    return result;
}

struct RopeTable { std::vector<float> cosine,sine; };
// transformer.cpp:259-272. This frequency path uses float pow, unlike time
// embedding. Output remains FP32 with axes [16,56,56] contributing [8,28,28].
RopeTable rope_oracle(const qwen::SequenceGeometry& sequence) {
    RopeTable table;
    for (const auto& position:sequence.positions) {
        const std::array<int,3> axes{16,56,56};
        for (size_t axis=0;axis<axes.size();++axis) for (int i=0;i<axes[axis];i+=2) {
            const float omega=1.f/std::pow(10000.f,float(i)/float(axes[axis]));
            const float angle=position[axis]*omega;
            table.cosine.push_back(std::cos(angle));
            table.sine.push_back(std::sin(angle));
        }
    }
    return table;
}

// scheduler.hpp base schedule only. No MLX import, GPU allocation or changed
// student schedule. Exact trig/exp ulps still require device-oracle comparison.
std::vector<float> base_schedule_oracle(int width,int height,int steps) {
    if (width<=0 || height<=0 || width%16 || height%16 || steps<=0)
        throw std::invalid_argument("invalid schedule dimensions/steps");
    if (steps==1) return {1.f,0.f};
    const double slope=(0.9-0.5)/(8192.0-256.0);
    const float mu=float(slope*double(width)*double(height)/256.0+0.5-slope*256.0);
    const float exponential=std::exp(mu), end=1.f/float(steps);
    std::vector<float> complements;
    for (int i=0;i<steps;++i) {
        const float initial=i==steps-1 ? end : 1.f+float(i)*((end-1.f)/float(steps-1));
        const float shifted=exponential/(exponential+(1.f/initial-1.f));
        complements.push_back(1.f-shifted);
    }
    const float scale=complements.back()/0.98f;
    std::vector<float> result;
    for (float value:complements) result.push_back(1.f-value/scale);
    result.push_back(0.f);
    return result;
}

void segment(const qwen::SequenceSegment& actual,int start,int end,bool causal,int text,int image) {
    check(actual.start==start && actual.end==end && actual.causal==causal &&
          actual.text_start==text && actual.image_index==image,"sequence segment contract changed");
}
void position(const qwen::SequenceGeometry& sequence,size_t row,float t,float y,float x) {
    check(sequence.positions.at(row)==std::array<float,3>{t,y,x},"sequence/RoPE position changed");
}

void sequence_fixtures() {
    const auto base=qwen::make_sequence_geometry(3,2,2,{});
    check(base.prefix_length==3 && base.positions.size()==7 && base.segments.size()==2,"base sequence rows");
    segment(base.segments[0],0,3,true,0,-1);
    segment(base.segments[1],3,7,false,-1,0);
    position(base,2,2,2,2); position(base,3,3,-1,-1); position(base,6,3,0,0);

    const auto one=qwen::make_sequence_geometry(4,2,2,{{3,1,2}});
    check(one.prefix_length==7 && one.positions.size()==11 && one.segments.size()==4,"one-reference rows");
    segment(one.segments[0],0,2,true,0,-1);
    segment(one.segments[1],2,5,false,-1,0);
    segment(one.segments[2],5,7,true,2,-1);
    segment(one.segments[3],7,11,false,-1,1);
    position(one,2,2,-1.5f,-0.5f); position(one,4,2,0.5f,-0.5f);
    position(one,5,5,5,5); position(one,7,7,-1,-1);

    const auto two=qwen::make_sequence_geometry(5,2,3,{{2,2,1},{1,3,3}});
    check(two.prefix_length==12 && two.positions.size()==18 && two.segments.size()==6,"two-reference rows");
    segment(two.segments[0],0,1,true,0,-1);
    segment(two.segments[1],1,5,false,-1,0);
    segment(two.segments[2],5,7,true,1,-1);
    segment(two.segments[3],7,10,false,-1,1);
    segment(two.segments[4],10,12,true,3,-1);
    segment(two.segments[5],12,18,false,-1,2);
    position(two,1,1,-1,-1.5f); position(two,4,1,0,-0.5f);
    position(two,5,3,3,3); position(two,7,5,-0.5f,-2);
    position(two,10,8,8,8); position(two,12,10,-1,-2);
    // The causal offset is the physical start 10; its RoPE scalar is 8.
    check(float(two.segments[4].start)!=two.positions[10][0],"mask offset must not become RoPE position");

    const auto same_slot=qwen::make_sequence_geometry(2,2,2,{{1,1,0},{1,1,0}});
    check(same_slot.prefix_length==4 && same_slot.segments.size()==4,"equal reference slots must be accepted");
    segment(same_slot.segments[0],0,1,false,-1,0);
    segment(same_slot.segments[1],1,2,false,-1,1);
    segment(same_slot.segments[2],2,4,true,0,-1);
    const auto wide=qwen::make_sequence_geometry(5,2,3,{{1,4,1}});
    const auto square=qwen::make_sequence_geometry(5,2,3,{{2,2,1}});
    check(wide.prefix_length==square.prefix_length && wide.positions!=square.positions,
          "same prefix row count does not imply same geometry/cache key");
    rejects([] { qwen::make_sequence_geometry(0,2,2,{}); });
    rejects([] { qwen::make_sequence_geometry(5,0,2,{}); });
    rejects([] { qwen::make_sequence_geometry(5,2,2,{{0,1,1}}); });
    rejects([] { qwen::make_sequence_geometry(5,2,2,{{1,1,-1}}); });
    rejects([] { qwen::make_sequence_geometry(5,2,2,{{1,1,6}}); });
    rejects([] { qwen::make_sequence_geometry(5,2,2,{{1,1,4},{1,1,3}}); });
    rejects([] { qwen::make_sequence_geometry(5,2,2,std::vector<qwen::ReferenceGeometry>(11,{1,1,1})); });
    rejects([] { qwen::make_sequence_geometry(5,std::numeric_limits<int>::max(),2,{}); });

    const auto rope=rope_oracle(two);
    for (const auto* geometry : {&base,&one,&two,&same_slot,&wide,&square}) {
        const auto expected=rope_oracle(*geometry);
        const auto actual=native::make_rope_tables(*geometry);
        check(actual.cosine==expected.cosine && actual.sine==expected.sine,
              "native RoPE constants differ from Transformer oracle");
    }
    check(rope.cosine.size()==18*64 && rope.sine.size()==rope.cosine.size(),"RoPE shape [tokens,64]");
    for (size_t i=0;i<64;++i) check(rope.cosine[i]==1.f && rope.sine[i]==0.f,"origin RoPE");
    const size_t target=12*64;
    check(rope.cosine[target]==std::cos(10.f) && rope.sine[target]==std::sin(10.f),"time axis RoPE offset");
    check(rope.cosine[target+8]==std::cos(-1.f) && rope.sine[target+8]==std::sin(-1.f),"height RoPE offset 8");
    check(rope.cosine[target+36]==std::cos(-2.f) && rope.sine[target+36]==std::sin(-2.f),"width RoPE offset 36");
    check(rope.cosine[target+9]!=bf16(rope.cosine[target+9]),"RoPE constants must remain FP32");
    rejects([] { native::make_rope_tables({}); });
    rejects([&] { auto broken=base; broken.prefix_length=-1; native::make_rope_tables(broken); });
    rejects([&] { auto broken=base; broken.prefix_length=int(broken.positions.size()); native::make_rope_tables(broken); });
    rejects([&] { auto broken=base; broken.positions[0][2]=std::numeric_limits<float>::quiet_NaN(); native::make_rope_tables(broken); });
    rejects([] { qwen::SequenceGeometry large; large.positions.resize(65537); native::make_rope_tables(large); });
}

void time_schedule_fixtures() {
    for (float sigma:{-0.f,0.f,0.02f,0.25f,0.37f,0.5f,1.f}) {
        const auto input=time_input_oracle(sigma);
        const auto native_input=native::timestep_input(sigma);
        for (size_t i=0;i<input.size();++i)
            check(native_input[i]==uint16_t(std::bit_cast<uint32_t>(input[i])>>16),
                  "native time input differs from Transformer BF16 oracle");
        for (size_t i=0;i<128;++i)
            check(input[256+i]==1.f && input[384+i]==0.f,"prefix uses zero-time embedding row");
        check(input[0]==bf16(std::cos(sigma*1000.f)) &&
              input[128]==bf16(std::sin(sigma*1000.f)),"time embedding cosine/sine concatenation");
    }
    const auto zero=time_input_oracle(0.f);
    for (size_t i=0;i<256;++i) check(zero[i]==zero[256+i],"sigma-zero rows must match");
    rejects([] { time_input_oracle(-0.01f); });
    rejects([] { time_input_oracle(1.01f); });
    rejects([] { time_input_oracle(std::numeric_limits<float>::quiet_NaN()); });
    for (float sigma:{-0.01f,1.01f,std::numeric_limits<float>::infinity(),
                      std::numeric_limits<float>::quiet_NaN()})
        rejects([&] { native::timestep_input(sigma); });
    check(base_schedule_oracle(512,512,1)==std::vector<float>{1.f,0.f},"one-step schedule bypasses zero division");
    const auto schedule=base_schedule_oracle(512,512,20);
    check(schedule.size()==21 && schedule.front()==1.f && schedule.back()==0.f &&
          std::abs(schedule[19]-0.02f)<1e-6f,"base schedule terminal stretch");
    for (size_t i=1;i<schedule.size();++i)
        check(std::isfinite(schedule[i]) && schedule[i]<schedule[i-1],"base sigma nodes decrease");
    rejects([] { base_schedule_oracle(513,512,20); });
    rejects([] { base_schedule_oracle(512,512,0); });
    // Euler casts the host-computed delta before BF16 product and sum.
    const float delta=bf16(schedule[1]-schedule[0]);
    const float euler=bf16(bf16(0.5f)+bf16(bf16(1.5f)*delta));
    check(std::isfinite(euler) && delta<0.f,"BF16 Euler oracle finite and signed");
}
} // namespace

int main() {
    sequence_fixtures();
    time_schedule_fixtures();
    std::cout << "native_qwen_denoiser_math_oracle=PASS sequence_implementation=checked "
                 "native_time_rope=checked scheduler=existing_pipeline_oracle device_created=0\n";
}
