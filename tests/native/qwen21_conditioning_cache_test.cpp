#include "models/qwen21/conditioning_cache.hpp"
#include <iostream>

int main() {
    using namespace tc;
    qwen21::ConditioningCache cache;
    auto populate = [&] {
        cache.text = Tensor(1.f); cache.prompt = "generation";
        cache.edit = qwen21::ConditioningCache::Edit{
            "edit",512,{"image-a","image-b"},Tensor(2.f),{1,2},{}};
    };
    require(!cache.select_encoder("gpu"),"first encoder must be a miss");
    populate();
    require(cache.select_encoder("gpu") && cache.text_hit("generation") &&
        cache.edit_hit("edit",512,{"image-a","image-b"}),"same encoder lost conditioning");
    require(!cache.text_hit("other") && !cache.edit_hit("edit",1024,{"image-a","image-b"}) &&
        !cache.edit_hit("edit",512,{"image-b","image-a"}) &&
        !cache.edit_hit("edit",512,{"image-a","new-bytes"}) &&
        !cache.edit_hit("edit",512,{}),"edit identity mismatch admitted");
    // Actual shared cache transitions: GPU -> runtime -> GPU, manifest bytes,
    // runtime policy, and a T2I request immediately following an edit request.
    for (const auto &identity : {"manifest-path:sha-a:chunks1", "gpu",
                                "manifest-path:sha-a:chunks1", "manifest-path:sha-b:chunks1",
                                "manifest-path:sha-b:chunks2"}) {
        require(!cache.select_encoder(identity) && !cache.text && !cache.edit && cache.prompt.empty(),
            "encoder switch retained stale text/edit conditions");
        cache.text = Tensor(3.f); cache.prompt = "generation";
        require(!cache.edit_hit("edit",512,{"image-a","image-b"}),
            "T2I encoder switch relabeled previous edit conditions");
        populate();
        require(cache.select_encoder(identity) && cache.edit_hit("edit",512,{"image-a","image-b"}),
            "cache hit unexpectedly invalidated");
    }
    cache.clear();
    require(cache.encoder_identity.empty() && !cache.text && !cache.edit,"unload retained conditioning");
    std::cout << "PASS Qwen conditioning cache: encoder/source/policy switches and T2I/edit crossover\n";
}
