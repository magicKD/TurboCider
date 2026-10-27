#pragma once
// Isolated serial DiT weight-residency experiment. Public MLX lazy loader and
// a native verified held descriptor; no custom tensor decoder or App selector.
#include "../../native/models/seedvr2/model.hpp"
#include "../../native/backends/mlx_fd_reader.hpp"
#include "../../native/runtime/streaming/source_lease.hpp"
#include <fcntl.h>
#include <mutex>

#if !defined(TURBOCIDER_ENABLE_SEEDVR2_BLOCK_STREAM_EXPERIMENT) || !TURBOCIDER_ENABLE_SEEDVR2_BLOCK_STREAM_EXPERIMENT
#error "SeedVR2 block weight source is an isolated experiment"
#endif

namespace tc::seedvr2 {
inline constexpr const char *kDiTCheckpointSHA="98669fd2c06df5eca88baf68cd5c478775c8e61fc110e598c52b350145ea2660";
inline constexpr const char *kSharedRoPE="blocks.0.attn.rope.rope.freqs";

// The shared block0 RoPE is used to build every later layout, so it belongs
// to globals, not block0's evictable group. Reject noncanonical block numbers.
inline int weight_group(const std::string &name){
    if(name==kSharedRoPE || !name.starts_with("blocks."))return -1;
    const auto stop=name.find('.',7);require(stop!=std::string::npos,"invalid SeedVR2 block weight name");
    const auto number=name.substr(7,stop-7);
    require(!number.empty() && number.size()<=2 && number.find_first_not_of("0123456789")==std::string::npos,
        "invalid SeedVR2 block weight number");
    const int index=std::stoi(number);require(index<32 && number==std::to_string(index),"noncanonical SeedVR2 block weight number");
    return index;
}
struct BlockReadRange{int group;uint64_t offset,bytes;};
struct BlockReadStats{
    std::atomic<uint64_t> header_bytes{0},payload_bytes{0},payload_reads{0};
    std::atomic<int> group{-2};
    bool record_ranges=false;
    std::mutex mutex;
    std::vector<BlockReadRange> ranges;
#if defined(TURBOCIDER_ENABLE_SEEDVR2_BLOCK_STREAM_TEST_HOOKS) && TURBOCIDER_ENABLE_SEEDVR2_BLOCK_STREAM_TEST_HOOKS
    std::atomic<bool> inject_read_failure{false};
#endif
};
class BlockLeaseReader final:public mx::io::Reader{
    std::shared_ptr<const streaming::SourceLease> lease_;
    std::shared_ptr<MlxLeaseFdReader> base_;
    std::shared_ptr<BlockReadStats> stats_;
    std::atomic<bool> &cancelled_;
    uint64_t file_bytes_;
    std::mutex failure_mutex_;
    std::exception_ptr failure_;
  public:
    BlockLeaseReader(std::shared_ptr<const streaming::SourceLease> lease,const std::string &id,
        std::shared_ptr<BlockReadStats> stats,std::atomic<bool> &cancelled):lease_(std::move(lease)),stats_(std::move(stats)),
        cancelled_(cancelled),file_bytes_(lease_->file(id).bytes){
        auto owned=lease_->duplicate_fd(id);
        require(::fcntl(owned.get(),F_NOCACHE,1)==0,"cannot configure uncached SeedVR2 group reads");
        base_=std::make_shared<MlxLeaseFdReader>(MlxOwnedFd(owned.release()),id);
    }
    bool is_open()const override{return base_->is_open();}
    bool good()const override{return base_->good();}
    size_t tell()override{return base_->tell();}
    void seek(int64_t offset,std::ios_base::seekdir direction=std::ios_base::beg)override{base_->seek(offset,direction);}
    std::string label()const override{return "SeedVR2 verified block stream "+base_->label();}
    void rethrow_read_failure(){
        std::exception_ptr failure;{std::lock_guard lock(failure_mutex_);failure=std::exchange(failure_,{});}
        if(failure)std::rethrow_exception(failure);
    }
    void read(char *destination,size_t bytes)override{
        checkpoint(cancelled_);require(bytes<=16ull<<20,"unbounded SeedVR2 safetensors header read");
        base_->read(destination,bytes);stats_->header_bytes+=bytes;checkpoint(cancelled_);
    }
    void read(char *destination,size_t bytes,size_t offset)override{
      try{
        require(offset<=file_bytes_ && bytes<=file_bytes_-offset,"SeedVR2 lazy read exceeds verified source");
        lease_->revalidate_after_drain();
        size_t done=0;
        while(done<bytes){
            checkpoint(cancelled_);const size_t chunk=std::min<size_t>(bytes-done,4ull<<20);
#if defined(TURBOCIDER_ENABLE_SEEDVR2_BLOCK_STREAM_TEST_HOOKS) && TURBOCIDER_ENABLE_SEEDVR2_BLOCK_STREAM_TEST_HOOKS
            if(stats_->inject_read_failure)throw std::runtime_error("seedvr2-block-source-injected-read-failure");
#endif
            base_->read(destination+done,chunk,offset+done);
            stats_->payload_bytes+=chunk;++stats_->payload_reads;
            if(stats_->record_ranges){std::lock_guard lock(stats_->mutex);stats_->ranges.push_back({stats_->group.load(),offset+done,chunk});}
            done+=chunk;checkpoint(cancelled_);
        }
        lease_->revalidate_after_drain();
      }catch(...){
        // Load::eval_cpu uses future.wait(), not get(), in pinned MLX0.32.
        // A Reader exception alone therefore does not stop materialization.
        // Preserve the original typed error for the adoption owner to fetch.
        std::lock_guard lock(failure_mutex_);if(!failure_)failure_=std::current_exception();throw;
      }
    }
};

class BlockWeightSource final{
  public:
    struct Record{mx::Shape shape;mx::Dtype dtype;uint64_t bytes;int group;};
  private:
    std::shared_ptr<const streaming::SourceLease> lease_;
    std::shared_ptr<BlockReadStats> stats_=std::make_shared<BlockReadStats>();
    std::shared_ptr<BlockLeaseReader> reader_;
    std::atomic<bool> &cancelled_;
    std::map<std::string,Record> records_;
    std::array<uint64_t,33> group_bytes_{};
    std::array<size_t,33> group_counts_{};
    std::mutex mutex_;
    WeightGroup read_header(){reader_->seek(0);return mx::load_safetensors(reader_).first;}
  public:
    BlockWeightSource(std::shared_ptr<const streaming::SourceLease> lease,const std::string &id,
        std::atomic<bool> &cancelled,bool ranges=false):lease_(std::move(lease)),cancelled_(cancelled){
        require(lease_ && lease_->has_verified_content(),"block streaming requires a native verified source lease");
        checkpoint(cancelled_);lease_->revalidate_after_drain();stats_->record_ranges=ranges;
        reader_=std::make_shared<BlockLeaseReader>(lease_,id,stats_,cancelled_);
        auto values=read_header();require(!values.empty(),"empty SeedVR2 streamed checkpoint");
        for(const auto &[name,value]:values){
            const int group=weight_group(name);
            records_.emplace(name,Record{value.shape(),value.dtype(),value.nbytes(),group});
            group_bytes_[group+1]+=value.nbytes();++group_counts_[group+1];
        }
        require(records_.count(kSharedRoPE) && group_counts_[0]>1,"missing SeedVR2 shared/global weights");
        for(int group=0;group<32;++group)require(group_counts_[group+1]>0,"missing SeedVR2 block group");
        // The public loader must only have parsed metadata. A changed MLX
        // eager-loading implementation fails closed before claiming residency.
        require(stats_->payload_bytes==0,"MLX safetensors loader eagerly read unselected payload");
        lease_->revalidate_after_drain();
    }
    void revalidate()const{reader_->rethrow_read_failure();lease_->revalidate_after_drain();}
    void materialized()const{revalidate();checkpoint(cancelled_);}
    WeightGroup load(int group){
        require(group>=-1 && group<32,"invalid SeedVR2 weight group");std::lock_guard lock(mutex_);
        checkpoint(cancelled_);revalidate();stats_->group=group;
        auto values=read_header();
        require(values.size()==records_.size(),"SeedVR2 checkpoint metadata count changed");
        for(auto it=values.begin();it!=values.end();){
            const auto found=records_.find(it->first);
            require(found!=records_.end() && found->second.shape==it->second.shape() && found->second.dtype==it->second.dtype(),
                "SeedVR2 checkpoint metadata changed");
            if(found->second.group!=group)it=values.erase(it);else ++it;
        }
        require(values.size()==group_counts_[group+1],"incomplete SeedVR2 selected weight group");revalidate();return values;
    }
    const auto &records()const{return records_;}
    const auto &group_bytes()const{return group_bytes_;}
    const auto &group_counts()const{return group_counts_;}
    const auto &stats()const{return stats_;}
};
} // namespace tc::seedvr2
