#pragma once
#include "worker_protocol.hpp"
#include "../../native/runtime/build_identity.hpp"
#include "turbocider/turbocider.h"
#include <atomic>
#include <csignal>
#include <iostream>
#include <memory>
#include <thread>
#include <chrono>

namespace tc_worker {
inline std::atomic<bool> cancelled{false};
static_assert(std::atomic<bool>::is_always_lock_free);
inline void signal_stop(int) { cancelled.store(true,std::memory_order_relaxed); }
inline NSDictionary *consume(int code, char *output, char *error) {
    std::unique_ptr<char,decltype(&tc_string_free)> owned_output(output,tc_string_free),owned_error(error,tc_string_free);
    if(code)throw std::runtime_error(error?error:"worker_native_query_failed");
    require(output!=nullptr,"worker_native_result_missing");
    NSData *data=[[@(output) copy] dataUsingEncoding:NSUTF8StringEncoding];
    id value=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];
    require([value isKindOfClass:NSDictionary.class],"worker_native_result_invalid");return value;
}
inline int query(const char *path) {
    NSDictionary *input=nil; NSMutableDictionary *reply=nil; int exit_code=1;
    try {
        input=read_input(path); reply=terminal(input,tc::runtime_build_identity());
        NSDictionary *request=input[@"native_request_v2"];
        tc_engine *raw=nullptr;char *error=nullptr;
        int status=tc_engine_create_model_worker([request[@"model"] UTF8String],
            [input[@"model_installation_ref"] fileSystemRepresentation],&raw,&error);
        std::unique_ptr<tc_engine,decltype(&tc_engine_free)> engine(raw,tc_engine_free);
        std::unique_ptr<char,decltype(&tc_string_free)> creation_error(error,tc_string_free);
        if(status)throw std::runtime_error(error?error:"worker_engine_create_failed");
        cancelled=false;
        struct Signals {
            using Handler=void(*)(int);Handler interrupt,terminate;
            Signals():interrupt(std::signal(SIGINT,signal_stop)),terminate(std::signal(SIGTERM,signal_stop)){}
            ~Signals(){std::signal(SIGINT,interrupt);std::signal(SIGTERM,terminate);}
        } signals;
        // Source verification has no event callback. Relay cancellation from a
        // normal thread; never invoke the C API inside an async signal handler.
        std::jthread cancellation([pointer=engine.get()](std::stop_token stop) {
            while(!stop.stop_requested()) {
                if(cancelled.load(std::memory_order_relaxed)) tc_engine_cancel(pointer);
                std::this_thread::sleep_for(std::chrono::milliseconds(25));
            }
        });
        char *result=nullptr;error=nullptr;
        status=tc_engine_verify_streaming_sources_json(engine.get(),&result,&error);
        auto sources=consume(status,result,error);
        require([sources[@"status"] isEqual:@"verified"],"worker_source_verification_missing");
        if(cancelled)throw std::runtime_error("worker_cancelled");
        NSData *encoded=canonical_request(request);
        NSString *json=[[NSString alloc] initWithData:encoded encoding:NSUTF8StringEncoding];
        result=nullptr;error=nullptr;
        status=tc_engine_resolve_streaming_json(engine.get(),json.UTF8String,&result,&error);
        auto resolution=consume(status,result,error);
        require([resolution[@"status"] isEqual:@"resolved"] &&
            [resolution[@"selection"][@"execution_container"] isEqual:@"cli_worker"] &&
            [resolution[@"requested_selector"] isEqual:request[@"execution"][@"streaming"]],"worker_resolution_mismatch");
        if(cancelled)throw std::runtime_error("worker_cancelled");
        reply[@"status"]=@"resolved";
        reply[@"resolution_digest"]=resolution[@"resolution_digest"];
        reply[@"record_digest"]=resolution[@"selection"][@"record_digest"];
        reply[@"layout_digest"]=resolution[@"selection"][@"layout_digest"];
        reply[@"resolution"]=resolution;reply[@"source_verification"]=sources;
        exit_code=0;
    } catch(const std::exception &error) {
        if(!reply) {std::cerr<<error.what()<<'\n';return 1;}
        const bool stopped=cancelled.load(std::memory_order_relaxed);
        reply[@"status"]=stopped?@"cancelled":@"error";
        reply[@"error"]=@{@"code":stopped?@"worker_cancelled":@"worker_query_failed",@"message":@(error.what())};
        exit_code=stopped?2:1;
    }
    NSData *bytes=canonical_request(reply);
    std::cout.write(static_cast<const char *>(bytes.bytes),std::streamsize(bytes.length));std::cout<<'\n';
    return exit_code;
}
}
