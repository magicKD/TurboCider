#pragma once
#include "worker_protocol.hpp"
#include <chrono>
#include <mutex>
#include <cstdio>
#include <cstring>

namespace tc_worker {
// Best-effort telemetry only. stdout remains the single authoritative terminal.
// Bounded lifetime output and cadence prevent diagnostics from growing with steps.
class Events {
    NSDictionary *input_;
    NSString *runtime_;
    std::mutex mutex_;
    size_t bytes_=0;
    uint64_t sequence_=0;
    std::chrono::steady_clock::time_point last_{};
    NSString *phase_=nil;
    bool resolved_sent_=false;
    bool emit(NSString *kind, NSDictionary *payload) {
        NSMutableDictionary *wire=[@{@"event_schema_version":@1,@"kind":kind,
            @"sequence":@(sequence_+1),@"execution_container":@"cli_worker",
            @"runtime_fingerprint":runtime_,@"payload":payload} mutableCopy];
        for(NSString *key in @[@"job_id",@"request_id",@"request_digest"])wire[key]=input_[key];
        NSData *data=canonical_request(wire);
        constexpr size_t prefix_size=9; // TC_EVENT + tab
        if(data.length>128*1024 || bytes_+data.length+prefix_size+1>512*1024)return false;
        ++sequence_;bytes_+=data.length+prefix_size+1;
        flockfile(stderr);
        fputs("TC_EVENT\t",stderr);fwrite(data.bytes,1,data.length,stderr);fputc('\n',stderr);fflush(stderr);
        funlockfile(stderr);
        return true;
    }
public:
    Events(NSDictionary *input,const char *runtime):input_(input),runtime_(@(runtime)){}
    void resolved(NSDictionary *resolution) {
        std::lock_guard lock(mutex_);resolved_sent_=emit(@"resolved",resolution);
    }
    static void progress(const char *json,void *context) noexcept {
        @autoreleasepool {
            try {
                auto &self=*static_cast<Events *>(context);
                std::lock_guard lock(self.mutex_);
                if(!self.resolved_sent_)return;
                auto now=std::chrono::steady_clock::now();
                if(!json || strnlen(json,16*1024+1)>16*1024)return;
                NSData *data=[@(json) dataUsingEncoding:NSUTF8StringEncoding];
                id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
                if(![value isKindOfClass:NSDictionary.class])return;
                NSString *phase=value[@"phase"];
                if(![phase isKindOfClass:NSString.class] || [phase isEqual:@"transformer_block"] || [phase isEqual:@"z_image_denoise_block"])return;
                if([phase isEqual:self.phase_] && now-self.last_<std::chrono::milliseconds(250))return;
                self.emit(@"progress",value);self.last_=now;self.phase_=phase;
            } catch(...) {} // Diagnostics cannot change execution/receipt outcome.
        }
    }
};
}
