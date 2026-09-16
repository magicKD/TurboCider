#include "ltx_streaming_descriptor.hpp"
#include "ltx_streaming_plan.hpp"
#include <cassert>
#include <fcntl.h>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

static tc::StreamingConfig config(uint32_t slots=3) {
    tc::StreamingConfig c;
    c.enabled=true;c.schema_version=1;c.selection="manual";c.retention="request";
    c.stages["denoiser"]={"streamed",1,slots,1,slots-1,1};return c;
}
template<class F> static void rejects(F fn, const char *part) {
    try { fn(); } catch (const std::invalid_argument &e) {
        assert(std::string(e.what()).find(part)!=std::string::npos); return;
    }
    assert(false);
}
int main(int argc,char **argv) {
    assert(argc==2 || (argc==3 && std::string(argv[2])=="--mutable-fixture"));
    try {
        tc::ltx::StreamingMetadata metadata(argv[1]);
        assert(!metadata.mapping().address && metadata.mapping().descriptor_open);
        const tc::ltx::StreamingWorkload workload{64,64,9,24,16,true,true,false,"synthetic"};
        auto d=metadata.describe(workload);
        const auto plan=tc::streaming::compile_layout(config(),d);
        assert(plan.materializations_complete);
        assert(d.artifacts.size()==1 && d.artifacts[0].identity_kind==tc::streaming::SourceIdentityKind::snapshot);
        assert(d.stages[0].passes.size()==11 && d.stages[0].passes[8].step==8);
        assert(d.stages[0].passes[0].phase=="av_stage1" && d.stages[0].passes[8].phase=="av_stage2");
        uint64_t source_bytes=0,content_bytes=0;
        for(uint32_t b=1;b<48;++b) {
            source_bytes+=metadata.block(b).source_read_bytes;
            content_bytes+=metadata.block(b).gpu_bytes+metadata.block(b).cpu_bytes;
        }
        for(uint32_t k=1;k<=3;++k) {
            const auto p=tc::streaming::compile_layout(config(k),d);
            const auto &s=p.stages[0];
            assert(s.source_read_bytes_per_pass==source_bytes);
            assert(s.prefix_source_read_bytes==metadata.block(0).source_read_bytes);
            assert(s.suffix_content_bytes_per_pass==content_bytes);
            assert(s.peak_pool_bytes==k*(metadata.block(0).gpu_bytes+metadata.block(0).cpu_bytes));
            tc::ltx::StreamingPlanView view(
                argv[1], config(k), workload, 100u+k);
            const auto &native=view.native_options();
            assert(native.version==2u && native.base.version==1u);
            assert(native.base.resident_prefix_blocks==1u);
            assert(native.base.plan==&view.c_plan());
            assert(native.metadata_header==&view.metadata().header());
            assert(native.metadata_mapping==&view.metadata().mapping());
            assert(view.c_plan().request_generation==100u+k);
            assert(view.c_plan().slot_count==k);
            assert(view.c_plan().group_count==47u);
            assert(view.c_plan().pass_count==11u);
            assert(view.layout().digest==p.digest);
        }
        rejects([&]{tc::ltx::StreamingPlanView bad(
            argv[1],config(),workload,0);},"generation");
        assert(tc::streaming::compile_layout(config(),metadata.describe(workload)).digest==plan.digest);
        auto changed=workload;changed.text_rows=32;
        assert(tc::streaming::compile_layout(config(),metadata.describe(changed)).digest!=plan.digest);
        changed=workload;changed.width=128;
        assert(tc::streaming::compile_layout(config(),metadata.describe(changed)).digest!=plan.digest);
        changed=workload;changed.conditioning="connected";
        assert(tc::streaming::compile_layout(config(),metadata.describe(changed)).digest!=plan.digest);
        changed=workload;changed.width=65;
        rejects([&]{metadata.describe(changed);},"normalized");
        changed=workload;changed.width=0xffffffc0u;changed.height=0xffffffc0u;
        rejects([&]{metadata.describe(changed);},"token count");
        changed=workload;changed.text_rows=4097;
        rejects([&]{metadata.describe(changed);},"text conditioning");
        rejects([&]{metadata.block(48);},"block index");
        metadata.check_unchanged();
        std::cout<<"PASS LTX generic projection: 48 blocks/11 passes; fd-only metadata; layout="<<plan.digest
                 <<" source_bytes_per_pass="<<source_bytes<<" content_bytes_per_pass="<<content_bytes
                 <<" quant_metadata_read_bytes="<<metadata.quant_metadata_read_bytes()<<'\n';
        if(argc==3) {
            // Only the runner's disposable fixture is ever modified; real
            // checkpoint invocation never passes --mutable-fixture.
            const int fd=open(argv[1],O_WRONLY|O_CLOEXEC);assert(fd>=0);
            struct stat status{};assert(fstat(fd,&status)==0);
            assert(ftruncate(fd,status.st_size-1)==0);close(fd);
            rejects([&]{metadata.check_unchanged();},"checkpoint_changed");
            rejects([&]{metadata.describe(workload);},"checkpoint_changed");
            std::cout<<"PASS truncated checkpoint invalidates live snapshot before reuse\n";
        }
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
