// Complete original source hash diagnostics, not model inference timings.
#include "../../native/runtime/streaming/source_lease.hpp"
#include "../../native/runtime/memory_manifest.hpp"
#include "../../native/core/common.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <pthread/qos.h>
#include <fcntl.h>
#include <unistd.h>
#include <time.h>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <sstream>

using namespace tc;
int main(int argc, char **argv) { try {
    require(argc == 6, "usage: qwen21-source-hash-probe source native|1m|8m default|initiated cached|nocache expected-sha256");
    const std::string mode=argv[2], qos=argv[3], cache=argv[4];
    require(mode=="native" || mode=="1m" || mode=="8m", "invalid bounded SHA recipe");
    require(qos=="default" || qos=="initiated", "invalid thread-local QoS");
    require(cache=="cached" || cache=="nocache", "invalid owned-fd cache option");
    const auto quality=qos=="initiated" ? QOS_CLASS_USER_INITIATED : QOS_CLASS_DEFAULT;
    require(pthread_set_qos_class_self_np(quality,0)==0, "thread-local QoS setup failed");
    streaming::SourceFileIdentity file;file.logical_id="probe";file.path=argv[1];
    auto lease=streaming::SourceLease::capture({file});auto fd=lease->duplicate_fd("probe");
    require(::fcntl(fd.get(),F_NOCACHE,cache=="nocache" ? 1 : 0)==0, "owned source cache flag failed");
    const auto bytes=lease->file("probe").bytes;
    require(bytes>0, "empty source");
    auto elapsed=[](auto t){return std::chrono::duration<double>(Clock::now()-t).count();};
    auto cpu=[] {timespec value{};require(clock_gettime(CLOCK_THREAD_CPUTIME_ID,&value)==0,"CPU clock failed");return double(value.tv_sec)+double(value.tv_nsec)*1e-9;};
    const auto start=Clock::now();const double cpu_start=cpu();
    std::string digest;double read_seconds=0,hash_seconds=0;uint64_t count=0;
    const size_t capacity=(mode=="8m" ? 8u : 1u)<<20;
    if(mode=="native") {digest=memory_sha256_fd(fd.get(),bytes);count=bytes;}
    else {
        std::vector<unsigned char> buffer(capacity);CC_SHA256_CTX hash;
        require(CC_SHA256_Init(&hash)==1,"SHA init failed");
        while(count<bytes) {
            const auto read_start=Clock::now();
            const auto n=::pread(fd.get(),buffer.data(),size_t(std::min<uint64_t>(capacity,bytes-count)),off_t(count));
            read_seconds+=elapsed(read_start);
            if(n<0 && errno==EINTR)continue;
            require(n>0,"source truncated/read failed");
            const auto hash_start=Clock::now();
            require(CC_SHA256_Update(&hash,buffer.data(),CC_LONG(n))==1,"SHA update failed");
            hash_seconds+=elapsed(hash_start);count+=uint64_t(n);
        }
        unsigned char result[CC_SHA256_DIGEST_LENGTH];require(CC_SHA256_Final(result,&hash)==1,"SHA final failed");
        std::ostringstream encoded;encoded<<std::hex<<std::setfill('0');
        for(auto value:result)encoded<<std::setw(2)<<unsigned(value);
        digest=encoded.str();
    }
    const double wall=elapsed(start),thread_cpu=cpu()-cpu_start;
    lease->revalidate_after_drain();require(digest==argv[5],"original source SHA differs");
    std::cout<<std::setprecision(12)<<"{\"schema\":\"tc-qwen21-source-hash-probe-v1\",\"qualification_passed\":false,"
        <<"\"scope\":\"complete held-fd original SHA; thread-local QoS/owned-fd cache; no encoder/model performance claim\","
        <<"\"recipe\":\""<<mode<<"\",\"qos\":\""<<qos<<"\",\"cache\":\""<<cache
        <<"\",\"sha256\":\""<<digest<<"\",\"source_bytes\":"<<bytes<<",\"bytes_read\":"<<count
        <<",\"buffer_bytes\":"<<capacity<<",\"wall_seconds\":"<<wall<<",\"thread_cpu_seconds\":"<<thread_cpu
        <<",\"split_read_seconds\":"<<read_seconds<<",\"split_hash_seconds\":"<<hash_seconds<<"}\n";
    return 0;
} catch(const std::exception &error) {std::cerr<<error.what()<<'\n';return 1;} }
