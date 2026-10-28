#pragma once
#include "model.hpp"
#include <CommonCrypto/CommonDigest.h>
#include <sys/stat.h>
#include <sys/statvfs.h>
#include <fcntl.h>
#include <unistd.h>
#include <array>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cstring>
#include <limits>
#include <mutex>

namespace tc::seedvr2 {
struct SpatialFileStats {
    uint64_t written_bytes=0,read_bytes=0,disk_live_bytes=0,disk_peak_bytes=0;
    uint64_t tile_convolutions=0,global_normalizations=0;
    uint64_t cleanup_failures=0;
    std::atomic<uint64_t> host_live_bytes{0},host_peak_bytes{0};
    std::atomic<uint64_t> host_allocations{0},host_reuses{0};
};
// Private owned scratch only. Never recursively delete a caller path, cache,
// source, or directory containing unregistered files. No model approximation.
class SpatialFiles : public std::enable_shared_from_this<SpatialFiles> {
    std::filesystem::path directory_;
    int directory_fd_=-1;
    dev_t device_=0;ino_t inode_=0;
    uint64_t serial_=0,budget_;
    bool reuse_read_windows_=false;
    struct Allocation {
        void *data=nullptr;size_t bytes=0;std::shared_ptr<SpatialFileStats> stats;
        ~Allocation(){if(data){free(data);stats->host_live_bytes.fetch_sub(bytes);}}
    };
    // Only the cache's sole reference makes a slot reusable. The Tensor's
    // deleter retains its allocation through the existing completed-GPU boundary.
    // At most two <=128-MiB windows are retained; extra simultaneous consumers
    // get independent uncached owners, preserving the old read contract.
    std::vector<std::shared_ptr<Allocation>> read_windows_;
    std::mutex read_windows_mutex_;
    std::shared_ptr<Allocation> allocate_window(size_t aligned){
        std::lock_guard<std::mutex> lock(read_windows_mutex_);
        if(reuse_read_windows_){
            auto best=read_windows_.end();
            for(auto it=read_windows_.begin();it!=read_windows_.end();++it)
                if(it->use_count()==1 && (*it)->bytes>=aligned && (best==read_windows_.end() || (*it)->bytes<(*best)->bytes))best=it;
            if(best!=read_windows_.end()){stats->host_reuses.fetch_add(1);return *best;}
            // Replace an idle undersized slot before allocating its successor.
            for(auto it=read_windows_.begin();it!=read_windows_.end();++it)
                if(it->use_count()==1){read_windows_.erase(it);break;}
        }
        auto allocation=std::make_shared<Allocation>();
        require(posix_memalign(&allocation->data,getpagesize(),aligned)==0,"cannot allocate aligned spatial read window");
        allocation->bytes=aligned;allocation->stats=stats;
        stats->host_allocations.fetch_add(1);
        const auto live=stats->host_live_bytes.fetch_add(aligned)+aligned;
        auto peak=stats->host_peak_bytes.load();while(peak<live && !stats->host_peak_bytes.compare_exchange_weak(peak,live)){}
        if(reuse_read_windows_ && read_windows_.size()<2)read_windows_.push_back(allocation);
        return allocation;
    }
  public:
    struct Frame {
        std::shared_ptr<SpatialFiles> owner;
        std::string name;
        int fd=-1,height=0,width=0,channels=0,next_row=0;
        dev_t device=0;ino_t inode=0;
        uint64_t bytes=0;
        bool sealed=false;
        std::vector<std::array<unsigned char,CC_SHA256_DIGEST_LENGTH>> digests;
        ~Frame(){
            if(fd<0)return;
            struct stat current{};
            if(fstatat(owner->directory_fd_,name.c_str(),&current,AT_SYMLINK_NOFOLLOW)==0 &&
                S_ISREG(current.st_mode) && current.st_dev==device && current.st_ino==inode) {
                if(unlinkat(owner->directory_fd_,name.c_str(),0))++owner->stats->cleanup_failures;
            } else ++owner->stats->cleanup_failures;
            close(fd);owner->stats->disk_live_bytes-=bytes;
        }
        Frame(const Frame &)=delete;
        Frame()=default;
    };
    std::shared_ptr<SpatialFileStats> stats=std::make_shared<SpatialFileStats>();
    explicit SpatialFiles(uint64_t budget,bool reuse_read_windows=false):budget_(budget),reuse_read_windows_(reuse_read_windows){
        require(budget>=1 && budget<=(8ull<<30),"spatial spill budget must be positive and at most 8 GiB");
        // POSIX mkdtemp creates a mode-0700 directory; its exact inode is pinned.
        std::string pattern=(std::filesystem::temp_directory_path()/"turbocider-vae-spatial-XXXXXX").string();
        std::vector<char> writable(pattern.begin(),pattern.end());writable.push_back(0);
        char *created=mkdtemp(writable.data());require(created!=nullptr,"cannot create private VAE scratch directory");
        directory_=created;
        directory_fd_=open(directory_.c_str(),O_RDONLY|O_DIRECTORY|O_CLOEXEC|O_NOFOLLOW);
        struct stat value{};
        if(directory_fd_<0 || fstat(directory_fd_,&value)){
            if(directory_fd_>=0)close(directory_fd_);rmdir(directory_.c_str());
            throw std::runtime_error("cannot pin private VAE scratch directory");
        }
        device_=value.st_dev;inode_=value.st_ino;
    }
    ~SpatialFiles(){
        clear_read_windows();
        struct stat value{};
        const bool original=lstat(directory_.c_str(),&value)==0 && S_ISDIR(value.st_mode) &&
            value.st_dev==device_ && value.st_ino==inode_;
        if(directory_fd_>=0)close(directory_fd_);
        if(!original || rmdir(directory_.c_str()))++stats->cleanup_failures; // only removes an owned empty directory
    }
    SpatialFiles(const SpatialFiles &)=delete;
    const std::filesystem::path &directory()const{return directory_;}
    void clear_read_windows(){std::lock_guard<std::mutex> lock(read_windows_mutex_);read_windows_.clear();}
    std::shared_ptr<Frame> create(int height,int width,int channels){
        require(height>0 && height<=8192 && width>0 && width<=8192 && channels>0 && channels<=512,
                "invalid spatial frame geometry");
        const uint64_t bytes=uint64_t(height)*width*channels*2;
        require(bytes<=budget_ && stats->disk_live_bytes<=budget_-bytes,"VAE scratch live-file budget exceeded");
        struct statvfs space{};
        require(fstatvfs(directory_fd_,&space)==0 && uint64_t(space.f_bavail)*space.f_frsize>=bytes+(256ull<<20),
                "insufficient free disk for VAE scratch plus reserve");
        auto frame=std::make_shared<Frame>();frame->owner=shared_from_this();
        frame->name="frame-"+std::to_string(++serial_)+".fp16";
        const int fd=openat(directory_fd_,frame->name.c_str(),O_RDWR|O_CREAT|O_EXCL|O_CLOEXEC|O_NOFOLLOW,0600);
        require(fd>=0,"cannot create private spatial frame");
        struct stat value{};
        if(fstat(fd,&value) || ftruncate(fd,off_t(bytes)) || fcntl(fd,F_NOCACHE,1)){
            close(fd);unlinkat(directory_fd_,frame->name.c_str(),0);
            throw std::runtime_error("cannot prepare uncached VAE scratch frame");
        }
        frame->fd=fd;frame->device=value.st_dev;frame->inode=value.st_ino;
        frame->height=height;frame->width=width;frame->channels=channels;frame->bytes=bytes;
        stats->disk_live_bytes+=bytes;stats->disk_peak_bytes=std::max(stats->disk_peak_bytes,stats->disk_live_bytes);
        frame->digests.resize(height);
        return frame;
    }
    void append_rows(const std::shared_ptr<Frame> &frame,int first,int rows,const uint16_t *values,std::atomic<bool> &cancelled){
        require(frame && frame->owner.get()==this && !frame->sealed && first==frame->next_row && rows>0 &&
            first>=0 && rows<=frame->height-first && values,"invalid or overlapping spatial frame write");
        const size_t row_bytes=size_t(frame->width)*frame->channels*2;
        const size_t bytes=size_t(rows)*row_bytes;
        size_t written=0;
        while(written<bytes){
            checkpoint(cancelled);
            const auto count=pwrite(frame->fd,reinterpret_cast<const char *>(values)+written,
                std::min(bytes-written,size_t(4ull<<20)),off_t(size_t(first)*row_bytes+written));
            if(count<0 && errno==EINTR)continue;
            require(count>0,"VAE scratch write failed");written+=size_t(count);stats->written_bytes+=size_t(count);
        }
        for(int row=0;row<rows;++row){
            checkpoint(cancelled);const char *data=reinterpret_cast<const char *>(values)+size_t(row)*row_bytes;
            CC_SHA256(data,CC_LONG(row_bytes),frame->digests[first+row].data());
        }
        frame->next_row+=rows;
    }
    void seal(const std::shared_ptr<Frame> &frame){
        require(frame && frame->owner.get()==this && !frame->sealed && frame->next_row==frame->height,
                "spatial frame is incomplete or already sealed");frame->sealed=true;
    }
    Tensor read_rows(const std::shared_ptr<Frame> &frame,int first,int rows,std::atomic<bool> &cancelled){
        require(frame && frame->owner.get()==this && frame->sealed && first>=0 && rows>0 && rows<=frame->height-first,
                "invalid spatial frame read extent/state");
        checkpoint(cancelled);
        struct stat value{};require(fstat(frame->fd,&value)==0 && value.st_dev==frame->device && value.st_ino==frame->inode &&
            uint64_t(value.st_size)==frame->bytes,"spatial frame inode/extent changed");
        const size_t row_bytes=size_t(frame->width)*frame->channels*2,bytes=size_t(rows)*row_bytes,page=getpagesize();
        require(bytes<=(128ull<<20) && bytes<=SIZE_MAX-page,"spatial read exceeds bounded host workspace");
        const size_t aligned=(bytes+page-1)/page*page;
        auto allocation=allocate_window(aligned);
        size_t done=0;
        while(done<bytes){checkpoint(cancelled);const auto count=pread(frame->fd,static_cast<char *>(allocation->data)+done,
            std::min(bytes-done,size_t(4ull<<20)),off_t(size_t(first)*row_bytes+done));
            if(count<0 && errno==EINTR)continue;require(count>0,"VAE scratch read failed");done+=size_t(count);}
        for(int row=0;row<rows;++row){
            checkpoint(cancelled);std::array<unsigned char,CC_SHA256_DIGEST_LENGTH> digest{};
            CC_SHA256(static_cast<const char *>(allocation->data)+size_t(row)*row_bytes,CC_LONG(row_bytes),digest.data());
            require(digest==frame->digests[first+row],"lossless spatial frame checksum differs");
        }
        stats->read_bytes+=bytes;
        auto backing=mx::allocator::make_buffer(allocation->data,allocation->bytes);
        require(backing.ptr()!=nullptr,"MLX refused the owned aligned spatial read window");
        try {
            return Tensor(backing,{1,1,rows,frame->width,frame->channels},mx::float16,
                [allocation](mx::allocator::Buffer value){mx::allocator::release(value);});
        }catch(...){mx::allocator::release(backing);throw;}
    }
};
using SpatialHistory=std::vector<std::shared_ptr<SpatialFiles::Frame>>;
struct SpatialTileOptions {
    int rows=64;
    uint64_t threshold_bytes=256ull<<20;
    uint64_t disk_budget_bytes=8ull<<30;
    bool reuse_read_windows=false;
};
struct SpatialTileCache {
    SpatialTileOptions options;
    std::shared_ptr<SpatialFileStats> stats=std::make_shared<SpatialFileStats>();
    std::shared_ptr<SpatialFiles> files;
    std::unordered_map<std::string,SpatialHistory> history;
    explicit SpatialTileCache(SpatialTileOptions value):options(value){
        require(value.rows>=8 && value.rows<=128 && value.rows%8==0 && value.threshold_bytes>=1 &&
            value.disk_budget_bytes>=1 && value.disk_budget_bytes<=(8ull<<30),"invalid spatial tile options");
    }
    void prepare_files(){if(!files){files=std::make_shared<SpatialFiles>(options.disk_budget_bytes,options.reuse_read_windows);files->stats=stats;}}
    bool eligible(const Tensor &input)const{return input.ndim()==5 && input.shape(0)==1 &&
        uint64_t(input.shape(2))*input.shape(3)*input.shape(4)*2>=options.threshold_bytes;}
};
} // namespace tc::seedvr2
