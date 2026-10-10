#include "../../native/runtime/streaming/source_lease.hpp"
#include <fstream>
#include <iostream>
#include <stdexcept>

using namespace tc::streaming;

int main(int argc, char **argv) {
    if (argc < 3) return 2;
    try {
        SourceFileIdentity source;
        source.logical_id = "weights";
        source.path = argv[2];
        if (argc > 3) source.content_digest = argv[3];
        const std::string mode = argv[1];
        std::atomic<bool> cancelled{mode == "cancel"};
        std::vector<SourceFileIdentity> files{source};
        if (argc > 4) {
            auto second = source;
            second.logical_id = "zz-second";
            second.path = argv[4];
            second.content_digest = std::string(64, '0');
            files.push_back(second);
        }
        std::shared_ptr<const SourceLease> lease;
        if (mode == "pending" || mode == "pending-stop") {
            PendingSourceVerification pending(files, &cancelled, true);
            if (pending.metadata()->has_verified_content()) throw std::runtime_error("metadata became proof");
            if (mode == "pending-stop") pending.cancel_and_drain();
            lease = pending.finish();
        } else if (mode == "verified") lease = SourceLease::capture_verified(files);
        else if (mode == "preverified") lease = SourceLease::capture_preverified(files);
        else if (mode == "metadata") {
            lease = SourceLease::capture(files);
            (void)lease->artifact_digest(); // A caller digest is deliberately not proof.
        } else lease = SourceLease::capture_registered(files, &cancelled);
        if (mode == "held-mutate") {
            // Even a persistent hit continues holding and revalidating the fd.
            std::ofstream output(argv[2], std::ios::binary | std::ios::trunc);
            output << "replacement";
            output.close();
            lease->revalidate_after_drain();
        }
        auto warm = SourceLease::capture_registered({source});
        if (warm->verification_bytes_read() || warm->registration_records_written() ||
            warm->registration_cache_hits() || warm->artifact_digest() != lease->artifact_digest())
            throw std::runtime_error("warm proof was rehashed/rewritten or changed: bytes=" +
                std::to_string(warm->verification_bytes_read()) + ", writes=" +
                std::to_string(warm->registration_records_written()) + ", persistent_hits=" +
                std::to_string(warm->registration_cache_hits()));
        std::cout << "{\"sha256\":\"" << lease->file("weights").content_digest
            << "\",\"artifact_digest\":\"" << lease->artifact_digest()
            << "\",\"verification_bytes\":" << lease->verification_bytes_read()
            << ",\"verification_cache_hits\":" << lease->verification_cache_hits()
            << ",\"registration_cache_hits\":" << lease->registration_cache_hits()
            << ",\"registration_records_written\":" << lease->registration_records_written()
            << ",\"verification_seconds\":" << lease->verification_seconds() << "}\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
