// Explicit real-source Qwen private-prefix client. Whole consumer is never a toy substitute.
#include "qwen_shared_abi.h"
#include "qwen_shared_compute.h"

int main(int argc, const char **argv) {
    @autoreleasepool {
        if (argc != 3) { std::fprintf(stderr, "usage: qwen-shared manifest.json new-native.json\n"); return 2; }
        NSMutableDictionary *report = [@{@"schema": @"tc-qwen-shared-native-v1", @"status": @"starting",
            @"production_qualified": @NO, @"physical_overlap_qualified": @NO, @"runtime_internal_copy_behavior": @"unknown",
            @"samples": [NSMutableArray array], @"errors": [NSMutableArray array]} mutableCopy];
        tcq::Private session;
        std::vector<IOSurfaceRef> surfaces; NSMutableArray *owners = [NSMutableArray array];
        auto adoption = std::make_shared<tcq::Adoption>(); std::atomic<bool> evaluation_unsafe{false};
        bool gpu_pending = false, mlx_pending = false;
        @try { try {
            NSData *data = [NSData dataWithContentsOfFile:[NSString stringWithUTF8String:argv[1]]];
            tcq::check(data && data.length <= 1024 * 1024, "bounded explicit source manifest required");
            NSDictionary *manifest = [NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
            tcq::check([manifest[@"schema"] isEqual:@"tc-qwen-shared-manifest-v1"] &&
                ([manifest[@"mode"] isEqual:@"inspect"] || [manifest[@"mode"] isEqual:@"benchmark"]), "explicit supported mode required");
            report[@"manifest_sha256"] = tcq::sha(data.bytes, data.length);
            session.load(manifest[@"model_directory"], report, ![manifest[@"skip_compile"] boolValue]);
            report[@"mode"] = manifest[@"mode"];
            if ([manifest[@"mode"] isEqual:@"inspect"]) { report[@"status"] = @"ports-inspected-no-execution"; report[@"formal_performance_score"] = @NO; }
            else tcq::benchmark(session,manifest,report,argv[2],surfaces,owners,adoption,evaluation_unsafe,gpu_pending,mlx_pending);
        } catch (const std::exception &error) { report[@"status"] = @"failed"; [report[@"errors"] addObject:[NSString stringWithUTF8String:error.what()]]; }
        } @catch (NSException *error) { report[@"status"] = @"objc-exception"; [report[@"errors"] addObject:error.name]; }
        bool unloaded = false;
        @try { unloaded = session.unload(); } @catch (NSException *error) { [report[@"errors"] addObject:error.name]; }
        report[@"unload_succeeded"] = @(unloaded); report[@"restore_adoption_releases"] = @(adoption->releases.load());
        report[@"restore_deleter_pointer_matches"] = @(adoption->pointer_ok.load());
        const bool release = unloaded && !evaluation_unsafe.load() && !gpu_pending && !mlx_pending &&
            (surfaces.empty() || (adoption->releases.load() == 2 && adoption->pointer_ok.load()));
        if (release) { [owners removeAllObjects]; for (auto value : surfaces) CFRelease(value); }
        else { (void)CFBridgingRetain(owners); }
        report[@"owned_mappings_released_after_completion"] = @(release); report[@"mapping_quarantined_until_child_exit"] = @(!release);
        report[@"mapping_reuse_after_failure"] = @NO; tcq::save(report, argv[2]);
        std::puts([report[@"status"] UTF8String]);
        return release && ([report[@"status"] isEqual:@"ports-inspected-no-execution"] || [report[@"status"] isEqual:@"measured"]) ? 0 : 1;
    }
}
