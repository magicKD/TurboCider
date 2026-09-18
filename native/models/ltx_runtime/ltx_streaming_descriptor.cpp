#include "ltx_streaming_descriptor.hpp"
#include "../../runtime/memory_manifest.hpp"
extern "C" {
#include "ltx.h"
}
#include <array>
#include <bit>
#include <fcntl.h>
#include <limits>
#include <locale>
#include <sstream>
#include <stdexcept>
#include <sys/stat.h>
#include <unistd.h>

namespace tc::ltx {
namespace {
void require_metadata(bool ok, const std::string &why) {
    if (!ok) throw std::invalid_argument("ltx_streaming_metadata: " + why);
}
void checked(int ok, const char *error) {
    // Delay copying the error buffer until after the C operation returns.
    require_metadata(ok!=0,error);
}
std::string fingerprint(const struct stat &s) {
    std::ostringstream out; out.imbue(std::locale::classic());
    out << s.st_dev << ':' << s.st_ino << ':' << s.st_size << ':';
#if defined(__APPLE__)
    out << s.st_mtimespec.tv_sec << ':' << s.st_mtimespec.tv_nsec << ':'
        << s.st_ctimespec.tv_sec << ':' << s.st_ctimespec.tv_nsec;
#else
    out << s.st_mtim.tv_sec << ':' << s.st_mtim.tv_nsec << ':'
        << s.st_ctim.tv_sec << ':' << s.st_ctim.tv_nsec;
#endif
    return memory_sha256_hex(out.str());
}
struct stat file_status(const std::string &path) {
    struct stat status{};
    require_metadata(::stat(path.c_str(), &status)==0 && S_ISREG(status.st_mode) && status.st_size>=8,
                     "missing/invalid regular checkpoint");
    return status;
}
std::string key(uint32_t kind, uint32_t object, uint32_t part) {
    return std::to_string(kind)+":"+std::to_string(object)+":"+std::to_string(part);
}
std::string sigmas(const float *values, size_t count) {
    std::string out;
    for (size_t i=0; i<count; ++i) out += std::to_string(std::bit_cast<uint32_t>(values[i]))+":";
    return out;
}
}

struct StreamingMetadata::State {
    std::string path, identity;
    std::shared_ptr<const streaming::SourceLease> lease;
    streaming::OwnedSourceFd source_fd;
    ltx_st_header header{};
    ltx_st_mapping mapping{};
    std::array<ltx_stream_block_layout,LTX_STREAM_BLOCKS> blocks{};
    uint64_t quant_bytes=0;
    ~State() { ltx_st_map_close(&mapping); ltx_st_free_header(&header); }
};
StreamingMetadata::StreamingMetadata(const std::string &checkpoint) : state_(std::make_unique<State>()) {
    auto &s=*state_; s.path=checkpoint;
    require_metadata(!checkpoint.empty() && checkpoint.find('\0')==std::string::npos, "invalid checkpoint path");
    char error[1024]={};
    const int fd=::open(checkpoint.c_str(),O_RDONLY|O_CLOEXEC);
    require_metadata(fd>=0,"checkpoint open failed");
    // The reader supports fd-only views; do not map a 20GiB checkpoint just to
    // inspect metadata. The existing C cleanup also supports address=null.
    s.mapping.descriptor=fd; s.mapping.descriptor_open=1;
    struct stat before{};
    require_metadata(::fstat(fd,&before)==0 && S_ISREG(before.st_mode) && before.st_size>=8,
                     "invalid opened checkpoint");
    s.identity=fingerprint(before);
    checked(ltx_st_read_header_fd(fd,checkpoint.c_str(),&s.header,error,sizeof(error)),error);
    require_metadata(s.header.file_size<=SIZE_MAX,"checkpoint exceeds address space");
    s.mapping.bytes=static_cast<size_t>(s.header.file_size);
    check_unchanged();
    require_metadata(s.header.file_size==static_cast<uint64_t>(before.st_size),"header/file identity mismatch");
    for (uint32_t b=0; b<LTX_STREAM_BLOCKS; ++b) {
        checked(ltx_stream_describe_block(&s.header,&s.mapping,b,&s.blocks[b],error,sizeof(error)),error);
        require_metadata(ltx_stream_blocks_compatible(&s.blocks[0],&s.blocks[b])!=0,
                         "heterogeneous LTX blocks are not supported by the current adapter");
        require_metadata(s.blocks[b].metadata_read_bytes<=UINT64_MAX-s.quant_bytes,"metadata byte overflow");
        s.quant_bytes+=s.blocks[b].metadata_read_bytes;
    }
    check_unchanged();
}
StreamingMetadata::StreamingMetadata(
        std::shared_ptr<const streaming::SourceLease> lease,
        std::string logical_id)
    : state_(std::make_unique<State>()) {
    auto &s = *state_;
    require_metadata(lease != nullptr, "source lease is missing");
    require_metadata(!logical_id.empty(), "checkpoint logical id is empty");
    const auto &file = lease->file(logical_id);
    s.lease = std::move(lease);
    s.path = file.path.string();
    s.source_fd = s.lease->duplicate_fd(logical_id);
    char error[1024] = {};
    struct stat before{};
    require_metadata(::fstat(s.source_fd.get(), &before) == 0 &&
                         S_ISREG(before.st_mode) && before.st_size >= 8,
                     "invalid opened checkpoint lease");
    s.identity = fingerprint(before);
    checked(ltx_st_read_header_fd(s.source_fd.get(), s.path.c_str(),
                                  &s.header, error, sizeof(error)), error);
    require_metadata(s.header.file_size == static_cast<uint64_t>(before.st_size),
                     "header/file identity mismatch");
    auto mapping_fd = s.lease->duplicate_fd(logical_id);
    s.mapping.descriptor = mapping_fd.release();
    s.mapping.descriptor_open = 1;
    s.mapping.bytes = static_cast<size_t>(s.header.file_size);
    require_metadata(ltx_st_validate_snapshot_fd(
                         &s.header, s.mapping.descriptor, s.path.c_str(),
                         error, sizeof(error)), error);
    for (uint32_t b = 0; b < LTX_STREAM_BLOCKS; ++b) {
        checked(ltx_stream_describe_block(&s.header, &s.mapping, b,
                                          &s.blocks[b], error, sizeof(error)),
                error);
        require_metadata(ltx_stream_blocks_compatible(&s.blocks[0],
                                                       &s.blocks[b]) != 0,
                         "heterogeneous LTX blocks are not supported by the current adapter");
        require_metadata(s.blocks[b].metadata_read_bytes <=
                             UINT64_MAX - s.quant_bytes,
                         "metadata byte overflow");
        s.quant_bytes += s.blocks[b].metadata_read_bytes;
    }
    check_unchanged();
}
StreamingMetadata::~StreamingMetadata() = default;
void StreamingMetadata::check_unchanged() const {
    const auto &s=*state_;
    if (s.lease) {
        s.lease->revalidate_after_drain();
        char error[1024] = {};
        require_metadata(ltx_st_validate_snapshot_fd(
                             &s.header, s.mapping.descriptor, s.path.c_str(),
                             error, sizeof(error)),
                         error[0] ? error : "checkpoint snapshot is stale");
        return;
    }
    struct stat opened{};
    require_metadata(s.mapping.descriptor_open && ::fstat(s.mapping.descriptor,&opened)==0,
                     "checkpoint descriptor unavailable");
    require_metadata(fingerprint(opened)==s.identity && fingerprint(file_status(s.path))==s.identity,
                     "checkpoint_changed: metadata snapshot is stale");
}
const ltx_st_header &StreamingMetadata::header() const { return state_->header; }
const ltx_st_mapping &StreamingMetadata::mapping() const { return state_->mapping; }
const ltx_stream_block_layout &StreamingMetadata::block(uint32_t b) const {
    require_metadata(b<LTX_STREAM_BLOCKS,"invalid block index"); return state_->blocks[b];
}
uint64_t StreamingMetadata::quant_metadata_read_bytes() const { return state_->quant_bytes; }
const std::shared_ptr<const streaming::SourceLease> &
StreamingMetadata::source_lease() const noexcept {
    return state_->lease;
}

streaming::Descriptor StreamingMetadata::describe(const StreamingWorkload &w) const {
    check_unchanged();
    require_metadata(w.text_rows && w.text_rows<=4096 &&
        (w.conditioning=="connected" || w.conditioning=="synthetic"),"unsupported text conditioning");
    require_metadata(w.width>=64 && w.height>=64 && w.width%64==0 && w.height%64==0 &&
                     w.frames && w.frames%8==1 && w.frames<=UINT32_MAX-7 && w.fps,
                     "workload must be normalized before describe");
    const uint64_t plane=uint64_t(w.width/32)*(w.height/32);
    const uint64_t latent_frames=(uint64_t(w.frames)+7)/8;
    require_metadata(plane && latent_frames<=UINT32_MAX/plane,"video token count exceeds native GPU range");
    ltx_workload geometry{}; char error[1024]={};
    checked(ltx_workload_init(&geometry,w.width,w.height,w.frames,w.fps,error,sizeof(error)),error);
    // No silently snapped dimensions in an exact metadata descriptor.
    require_metadata(geometry.output_width==w.width && geometry.output_height==w.height &&
                     geometry.frames==w.frames && geometry.fps==w.fps,"workload must be normalized before describe");
    streaming::Descriptor out{"ltx-2.5-distilled","snapshot:"+state_->identity,"ltx-c-metal-streaming-v1",{}};
    out.artifacts.push_back({"transformer",out.checkpoint_identity,state_->header.file_size,
                            streaming::SourceIdentityKind::snapshot});
    out.workload={{"operation","denoiser.two-stage"},{"format","convrot-int8-g256"},
        {"width",std::to_string(w.width)},{"height",std::to_string(w.height)},
        {"frames",std::to_string(w.frames)},{"fps",std::to_string(w.fps)},
        {"text_rows",std::to_string(w.text_rows)},{"conditioning",w.conditioning},
        {"parallel_av",std::to_string(w.parallel_av)},
        {"batch_audio_commands",std::to_string(w.batch_audio_commands)},
        {"video_attention_batch",std::to_string(w.video_attention_batch)},
        {"split_stages",std::to_string(w.split_stages)},
        {"conditioning_recipe","scalar-conditioning-v1"},
        {"reader_revision",std::to_string(LTX_STREAM_READER_REVISION)},
        {"upsample_boundary",w.split_stages?
            "after-stage1-pool-released":"after-stage1-pool-retained"}};
    size_t count1=0,count2=0;
    const float *s1=ltx_distilled_stage1_sigmas(&count1), *s2=ltx_distilled_stage2_sigmas(&count2);
    require_metadata(count1>=2 && count2>=2 && count1+count2-2<=streaming::max_passes,"invalid sigma schedule");
    out.workload["stage1_sigma_bits"]=sigmas(s1,count1);
    out.workload["stage2_sigma_bits"]=sigmas(s2,count2);
    std::vector<streaming::BlockSpec> blocks;
    blocks.reserve(LTX_STREAM_BLOCKS);
    for (uint32_t b=0; b<LTX_STREAM_BLOCKS; ++b) {
        const auto &meta=state_->blocks[b];
        streaming::BlockSpec block; block.id=b; block.layout_class="ltx-convrot-g256-fixed";
        for (uint32_t fi=0; fi<meta.field_count; ++fi) {
            const auto &f=meta.fields[fi]; const auto &t=*f.source;
            streaming::Materialization m;
            m.storage_mode=f.kind==LTX_STREAM_TABLE_BASE?"cpu":"metal-shared";
            m.format=ltx_dtype_name(t.dtype); m.conversion="copy";
            m.shape.assign(t.shape,t.shape+t.ndim);
            if (f.kind==LTX_STREAM_TABLE_BASE || f.kind==LTX_STREAM_TABLE_ROW) {
                m.format="BF16";
                m.conversion=f.kind==LTX_STREAM_TABLE_BASE?"f32-to-bf16-rne-v1":"copy-bf16-row";
            } else if (f.kind==LTX_STREAM_LINEAR && f.part==0) m.format="I8-convrot-g256";
            if (f.kind==LTX_STREAM_TABLE_ROW) {
                m.shape={meta.table_columns[f.object]};
                m.derived_from=key(LTX_STREAM_TABLE_BASE,f.object,0);
                m.derived_offset=uint64_t(f.part)*f.bytes;
            } else {
                m.reads.push_back({0,t.file_offset,t.data_end-t.data_begin,t.name,ltx_dtype_name(t.dtype),
                    std::vector<uint64_t>(t.shape,t.shape+t.ndim)});
            }
            const auto name=key(f.kind,f.object,f.part);
            block.fields.push_back({name,std::to_string(b)+":"+name,f.bytes,1,std::move(m)});
        }
        blocks.push_back(std::move(block));
    }
    auto make_stage = [&](std::string id, std::string pass_kind,
                          uint32_t pass_count, uint32_t video_tokens,
                          std::vector<streaming::BlockSpec> stage_blocks) {
        streaming::StageDescriptor stage;
        stage.id=std::move(id);
        stage.adapter_revision=w.split_stages?
            "ltx-native-exact-stage-v2":"ltx-native-exact-v1";
        stage.min_prefix=1; stage.max_slots=3; stage.max_group_size=1;
        stage.pass_count=pass_count;
        for (uint32_t pass=0; pass<pass_count; ++pass)
            stage.passes.push_back({pass,pass_kind,
                {video_tokens,geometry.audio_tokens,w.text_rows}});
        stage.blocks=std::move(stage_blocks);
        out.stages.push_back(std::move(stage));
    };
    if (w.split_stages) {
        auto stage2_blocks = blocks;
        make_stage("ltx-stage1-denoiser","av_stage1",
            static_cast<uint32_t>(count1-1),
            static_cast<uint32_t>(geometry.stage1_video_tokens),
            std::move(blocks));
        make_stage("ltx-stage2-denoiser","av_stage2",
            static_cast<uint32_t>(count2-1),
            static_cast<uint32_t>(geometry.stage2_video_tokens),
            std::move(stage2_blocks));
    } else {
        streaming::StageDescriptor stage;
        stage.id="denoiser"; stage.adapter_revision="ltx-native-exact-v1";
        stage.min_prefix=1; stage.max_slots=3; stage.max_group_size=1;
        stage.pass_count=static_cast<uint32_t>(count1+count2-2);
        for (uint32_t pass=0; pass<stage.pass_count; ++pass) {
            const bool first=pass<count1-1;
            stage.passes.push_back({pass,first?"av_stage1":"av_stage2",
                {first?geometry.stage1_video_tokens:geometry.stage2_video_tokens,
                 geometry.audio_tokens,w.text_rows}});
        }
        stage.blocks=std::move(blocks);
        out.stages.push_back(std::move(stage));
    }
    return out;
}
} // namespace tc::ltx
