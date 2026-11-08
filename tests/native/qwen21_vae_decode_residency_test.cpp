#include "../../native/models/qwen21/diagnostic_options.hpp"
#include "../../native/models/qwen21/vae_decode_residency.hpp"
#include <turbocider/turbocider.h>
#include <algorithm>
#include <cstring>
#include <iostream>
#include <sstream>

using namespace tc;
using namespace tc::qwen21;

template<class F> std::string rejected(F fn) {
    try { fn(); } catch (const std::exception &error) { return error.what(); }
    throw std::runtime_error("expected rejection was absent");
}

void policy() {
    Request r;
    r.model="qwen-image-2.1";r.residency="streamed";r.execution="gpu";
    r.operation="image.generate";r.steps=8;r.frames=1;r.audio=false;
    r.hybrid_mlp_mode="auto";r.qwen21_denoiser_backend="mlx";r.qwen21_dit_cache="off";
    auto selected=[](const Request &request,const char *raw,bool prepare=false) {
        return stream_vae_decode_only_request(request,prepare,raw);
    };
    require(selected(r,nullptr) && !selected(r,"0") && selected(r,"1"),"eligible automatic selection or explicit control changed");
    for (const char *bad : {"","true","2","01"," 1","1\n"})
        require(rejected([&]{selected(r,bad);}).find("0 or 1")!=std::string::npos,"invalid flag not rejected");
    require(!selected(r,nullptr,true) && !selected(r,"0",true),"prepare-only automatic path selected");
    require(!rejected([&]{selected(r,"1",true);}).empty(),"prepare-only accepted");
    for (int changed=0;changed<12;++changed) {
        auto bad=r;
        switch(changed) {
            case 0:bad.model="flux2-klein-4b";break;
            case 1:bad.residency="resident";break;
            case 2:bad.residency="component_staged";break;
            case 3:bad.operation="image.edit";break;
            case 4:bad.inputs.push_back({});break;
            case 5:bad.execution="gpu_ane";break;
            case 6:bad.hybrid_mlp_mode="runtime";break;
            case 7:bad.qwen21_denoiser_backend="native";break;
            case 8:bad.ane_manifest="fake";break;
            case 9:bad.frames=2;break;
            case 10:bad.audio=true;break;
            case 11:bad.encoder_ane_manifest="fake";break;
        }
        require(!selected(bad,nullptr) && !selected(bad,"0"),"off altered legacy route");
        require(!rejected([&]{selected(bad,"1");}).empty(),"incompatible route accepted");
    }
    r.execution="auto";r.tiny_vae.emplace();r.tiny_vae->preview=true;
    require(selected(r,nullptr) && selected(r,"1"),"original decoder with Tiny preview rejected");
    r.tiny_vae->decoder="tiny";
    require(!selected(r,nullptr) && !selected(r,"0"),"Tiny final automatically claimed original VAE route");
    require(!rejected([&]{selected(r,"1");}).empty(),"Tiny final claimed original VAE route");
}

struct ReadCounts { size_t payload_reads=0,payload_bytes=0; };
class MemoryReader final : public mx::io::Reader {
    std::vector<char> bytes_;
    size_t cursor_=0,payload_=0;
    std::shared_ptr<ReadCounts> counts_;
  public:
    MemoryReader(const std::string &key,const std::vector<int> &shape,std::shared_ptr<ReadCounts> counts)
        : counts_(std::move(counts)) {
        size_t elements=1;
        std::ostringstream dimensions;
        for (size_t i=0;i<shape.size();++i) { if(i)dimensions<<',';dimensions<<shape[i];elements*=size_t(shape[i]); }
        std::string header="{\""+key+"\":{\"dtype\":\"BF16\",\"shape\":["+dimensions.str()+
            "],\"data_offsets\":[0,"+std::to_string(elements*2)+"]}}";
        while(header.size()%8)header+=' ';
        const uint64_t header_bytes=header.size();payload_=8+header_bytes;
        bytes_.resize(payload_+elements*2);
        std::memcpy(bytes_.data(),&header_bytes,8);std::memcpy(bytes_.data()+8,header.data(),header.size());
    }
    bool is_open() const override { return true; }
    bool good() const override { return true; }
    size_t tell() override { return cursor_; }
    void seek(int64_t off,std::ios_base::seekdir way=std::ios_base::beg) override {
        const int64_t base=way==std::ios_base::beg ? 0 : way==std::ios_base::end ? int64_t(bytes_.size()) : int64_t(cursor_);
        require(off>=-base && base+off<=int64_t(bytes_.size()),"fixture seek range");cursor_=size_t(base+off);
    }
    void read(char *out,size_t n) override { read(out,n,cursor_);cursor_+=n; }
    void read(char *out,size_t n,size_t offset) override {
        require(offset<=bytes_.size() && n<=bytes_.size()-offset,"fixture short read");
        if(n && offset+n>payload_) { ++counts_->payload_reads;counts_->payload_bytes+=offset+n-std::max(payload_,offset); }
        std::memcpy(out,bytes_.data()+offset,n);
    }
    std::string label() const override { return "synthetic VAE metadata reader"; }
};
struct Field { std::string name;std::vector<int> shape; };
struct Fixture {
    Weights weights;
    std::vector<std::weak_ptr<MemoryReader>> readers;
    std::vector<std::shared_ptr<ReadCounts>> counts;
    explicit Fixture(const std::vector<Field> &fields) {
        std::vector<std::string> names;std::vector<Tensor> arrays;
        for(const auto &field:fields) {
            auto count=std::make_shared<ReadCounts>();
            auto reader=std::make_shared<MemoryReader>(field.name,field.shape,count);
            names.push_back(field.name);arrays.push_back(mx::load_safetensors(reader,mx::Device::cpu).first.at(field.name));
            readers.push_back(reader);counts.push_back(count);
        }
        weights.bind_arrays(names,arrays);
        no_payload();
    }
    void no_payload() const {
        for(const auto &count:counts)require(count->payload_reads==0 && count->payload_bytes==0,"metadata evaluated source payload");
    }
};
const std::vector<Field> legacy{
    {"conv2.weight",{2,2,1,1,1}}, {"decoder.conv1.weight",{2,2,1,1}},
    {"decoder.head.0.gamma",{2}}, {"decoder.head.2.weight",{4,2,1,1}},
    {"decoder.upsamples.0.upsamples.3.conv.weight",{2,2,1,1}},
    {"encoder.conv1.weight",{2,2,1,1}}, {"conv1.weight",{2,2,1,1}},
    {"decoder.time_conv.weight",{3}}
};

void validation_and_owners() {
    std::atomic<bool> cancelled{false};
    for(const auto &fields:std::vector<std::vector<Field>>{
        {{"decoder.conv_in.weight",{2,2,1,1}},{"encoder.conv_in.weight",{2,2,1}}},
        {{"decoder.conv_in.weight",{2,2,1,1}},{"encoder.conv_in.weight",{2,2,2,1,1}}},
        {{"decoder.conv_in.weight",{2,2,1,1}},{"encoder.conv1.weight",{2,2,1,1}},{"encoder.conv_in.weight",{2,2,1,1}}}}) {
        Fixture fixture(fields);const auto before=fixture.weights.sorted_keys();
        const auto old_error=rejected([&]{VAE original(fixture.weights,VAE::DecodeConvolution::Original,false);});
        const auto new_error=rejected([&]{retain_streamed_vae_decode_weights(fixture.weights,cancelled);});
        require(old_error==new_error && fixture.weights.sorted_keys()==before,"metadata failure changed or partially erased weights");
        fixture.no_payload();
    }
    Fixture fixture(legacy);
    std::vector<uintptr_t> retained_ids;
    for(size_t i=0;i<5;++i)retained_ids.push_back(fixture.weights.at(legacy[i].name).id());
    std::optional<Tensor> external=fixture.weights.at("encoder.conv1.weight");
    const auto metrics=retain_streamed_vae_decode_weights(fixture.weights,cancelled);
    require(metrics.metadata_preflight && metrics.retained_tensors==5 && metrics.dropped_tensors==3 &&
            metrics.retained_logical_bytes==44 && metrics.dropped_logical_bytes==22,"logical counts differ");
    fixture.no_payload();
    for(size_t i=0;i<5;++i)require(!fixture.readers[i].expired() && fixture.weights.has(legacy[i].name) &&
        !fixture.weights.at(legacy[i].name).is_available() && fixture.weights.at(legacy[i].name).id()==retained_ids[i] &&
        fixture.weights.at(legacy[i].name).dtype()==mx::bfloat16,"retained legacy array changed/evaluated");
    require(!fixture.readers[5].expired() && fixture.readers[6].expired() && fixture.readers[7].expired(),
            "dropped reader retained without external alias");
    external.reset();require(fixture.readers[5].expired(),"preflight retained encoder owner");
    {
        VAE decoder(fixture.weights,VAE::DecodeConvolution::Original,false);
        fixture.weights.clear();
        for(size_t i=0;i<5;++i)require(!fixture.readers[i].expired(),"decoder lost retained lazy source");
    }
    for(const auto &reader:fixture.readers)require(reader.expired(),"decoder owner leaked");
    fixture.no_payload();

    Fixture cancelled_fixture(legacy);const auto before=cancelled_fixture.weights.sorted_keys();cancelled=true;
    require(rejected([&]{retain_streamed_vae_decode_weights(cancelled_fixture.weights,cancelled);})=="generation cancelled",
            "cancellation was swallowed");
    require(cancelled_fixture.weights.sorted_keys()==before,"cancelled preflight mutated weights");
    cancelled_fixture.no_payload();cancelled=false;

    auto canonical=legacy;
    for(auto &field:canonical)field.name=VAE::canonical_key(field.name);
    Fixture canonical_fixture(canonical);
    const auto canonical_metrics=retain_streamed_vae_decode_weights(canonical_fixture.weights,cancelled);
    require(canonical_metrics.retained_logical_bytes==44 && canonical_metrics.dropped_logical_bytes==22 &&
            canonical_metrics.retained_tensors==5 && canonical_metrics.dropped_tensors==3,"canonical/legacy retention differs");
    canonical_fixture.no_payload();

    Fixture materialized(legacy);materialized.weights.set_affine_quantization(64,4);
    retain_streamed_vae_decode_weights(materialized.weights,cancelled);
    materialized.weights.set_affine_quantization(64,4);
    require(rejected([&]{materialized.weights.set_affine_quantization(32,8);})=="conflicting saved affine geometry",
            "filter reset saved weight metadata");
    materialized.no_payload();materialized.weights.materialize();
    for(size_t i=0;i<5;++i)require(materialized.counts[i]->payload_bytes==materialized.weights.at(legacy[i].name).nbytes(),
        "retained materialization read differs");
    for(size_t i=5;i<8;++i)require(materialized.counts[i]->payload_reads==0,"dropped payload was read");
}

int main() {try {
    char *identity=tc_runtime_build_identity();require(identity,"runtime identity absent");
    const std::string actual(identity);tc_string_free(identity);
    require(actual==TC_TEST_BOUND_NATIVE_ID,"VAE metadata fixture/native source mismatch");
    mx::set_default_device(mx::Device::cpu);
    policy();validation_and_owners();
    std::cout<<"{\"status\":\"passed\",\"runtime_build_id\":\""<<actual<<"\",\"synthetic_source_only\":true,"
        "\"metadata_preflight_no_payload_reads\":true,\"invalid_unused_metadata_cases\":3,"
        "\"legacy_and_owner_cases_passed\":true,\"canonical_saved_metadata_passed\":true,"
        "\"cancellation_passed\":true,\"gpu_used\":false}\n";
    return 0;
} catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;}}
