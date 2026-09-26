#include "models/z_image/coreml_bundle.hpp"
#include "core/common.hpp"
#include <iostream>
namespace fs = std::filesystem;
using namespace tc::z_image;
void check(bool value) { tc::require(value, "bundle test assertion"); }
int main(int argc, char **argv) {
    check(argc >= 5);
    if (std::string(argv[1]) == "--real") {
        check(argc == 5);
        tc::streaming::SourceFileIdentity file; file.logical_id = "denoiser"; file.path = argv[2];
        auto parent = tc::streaming::SourceLease::capture_verified({file});
        fs::path manifest(argv[3]);
        auto generation = CoreMLGeneration::import_tree(manifest.parent_path(),argv[4]);
        auto bundle = VerifiedCoreMLBundleLease::bind(generation,manifest.filename(),parent,"denoiser");
        bundle->revalidate();
        auto path=generation->root();
        std::cout << "parent_sha256=" << bundle->partition().parent_checkpoint_digest
                  << " bundle_digest=" << bundle->partition().artifact_content_digest
                  << " partition_identity=" << bundle->partition().identity
                  << " copied_bytes=" << generation->copied_bytes() << " models=" << bundle->models().size() << "\n";
        generation.reset(); parent.reset(); bundle->revalidate(); bundle.reset(); check(!fs::exists(path));
        return 0;
    }
    tc::streaming::SourceFileIdentity file; file.logical_id = "denoiser"; file.path = argv[1];
    auto unverified = tc::streaming::SourceLease::capture({file});
    auto parent = tc::streaming::SourceLease::capture_verified({file});
    std::string identity;
    for (int i = 3; i < argc; ++i) {
        parent = tc::streaming::SourceLease::capture_verified({file});
        fs::path source(argv[i]);
        auto generation = CoreMLGeneration::import_tree(source, argv[2]);
        auto path = generation->root();
        bool expected = source.filename().string().starts_with("pass");
        bool rejected = false;
        std::shared_ptr<const VerifiedCoreMLBundleLease> bundle;
        try { bundle = VerifiedCoreMLBundleLease::bind(generation,"manifest.json",parent,"denoiser"); }
        catch (const std::exception &) { rejected = true; }
        check(rejected != expected); check(bool(bundle) == expected);
        if (bundle) {
            check(bundle->models().size() == 32); check(bundle->partition().ane_end == 5120);
            check(bundle->partition().bucket_rows == 1088); check(bundle->partition().output_scale == 32);
            if (identity.empty()) identity = bundle->partition().identity;
            else check(identity == bundle->partition().identity);
            bool refused = false;
            try { VerifiedCoreMLBundleLease::bind(generation,"manifest.json",unverified,"denoiser"); }
            catch (const std::exception &) { refused = true; }
            check(refused);
            for (const auto &name : {"../manifest.json", "/manifest.json"}) {
                refused = false;
                try { VerifiedCoreMLBundleLease::bind(generation,name,parent,"denoiser"); }
                catch (const std::exception &) { refused = true; }
                check(refused);
            }
            generation.reset(); check(fs::exists(path)); bundle->revalidate();
            auto original = fs::path(argv[1]);
            fs::path moved = original.string()+".moved";
            fs::rename(original,moved); refused = false;
            try { bundle->revalidate(); } catch (const std::exception &) { refused = true; }
            fs::rename(moved,original); check(refused);
        }
        bundle.reset(); generation.reset(); check(!fs::exists(path));
        std::cout << source.filename() << ": pass\n";
    }
}
