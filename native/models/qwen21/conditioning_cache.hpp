#pragma once
#include "transformer.hpp"

namespace tc::qwen21 {
// Text-to-image and edit conditions belong to the same encoder generation.
// Switching a manifest, its bytes, or executor policy invalidates BOTH banks,
// even if the first request on the new encoder is text-to-image.
struct ConditioningCache {
    struct Edit {
        std::string prompt;
        int reference_size = 0;
        std::vector<std::string> image_sha256;
        Tensor text;
        std::vector<int> image_slots;
        std::vector<ReferenceLatents> reference_latents;
    };
    std::optional<Tensor> text;
    std::string prompt;
    std::optional<Edit> edit;
    std::string encoder_identity;

    void clear() {
        text.reset(); prompt.clear(); edit.reset(); encoder_identity.clear();
    }
    bool select_encoder(const std::string &identity) {
        require(!identity.empty(), "Qwen21 conditioning requires an encoder identity");
        if (encoder_identity == identity) return true;
        clear(); encoder_identity = identity;
        return false;
    }
    bool text_hit(const std::string &value) const {
        return text && prompt == value;
    }
    bool edit_hit(const std::string &value, int size,
                  const std::vector<std::string> &digests) const {
        return !digests.empty() && edit && edit->prompt == value &&
            edit->reference_size == size && edit->image_sha256 == digests;
    }
};
} // namespace tc::qwen21
