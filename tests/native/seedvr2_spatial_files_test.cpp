#include "../../native/models/seedvr2/spatial_files.hpp"
#include "../../native/models/seedvr2/metal/spatial_norm.hpp"
#include "../../native/models/seedvr2/metal/group_norm.hpp"
#include <mlx/random.h>
#include <fstream>
#include <iostream>
namespace mx=mlx::core;
using tc::Tensor;
template<class F>void rejects(F call){bool failed=false;try{call();}catch(const std::exception &){failed=true;}tc::require(failed,"invalid spatial file/norm operation accepted");}
int main(){
    try {
        mx::set_default_device(mx::Device::gpu);std::atomic<bool> cancelled{false};
        constexpr int h=17,w=13,c=32;std::vector<uint16_t> bits(h*w*c);
        const std::array<uint16_t,8> special{0,0x8000,0x7c00,0xfc00,0x7e01,0x7e21,0x0001,0x3c00};
        for(size_t i=0;i<bits.size();++i)bits[i]=i%17==0?special[(i/17)%8]:uint16_t(i*79);
        std::filesystem::path directory;
        {
            auto files=std::make_shared<tc::seedvr2::SpatialFiles>(8ull<<20);directory=files->directory();
            struct stat metadata{};tc::require(stat(directory.c_str(),&metadata)==0 && (metadata.st_mode&0777)==0700,"scratch directory is not private");
            auto frame=files->create(h,w,c);
            rejects([&]{files->append_rows(frame,1,1,bits.data(),cancelled);});
            files->append_rows(frame,0,8,bits.data(),cancelled);rejects([&]{files->seal(frame);});
            files->append_rows(frame,8,5,bits.data()+8*w*c,cancelled);files->append_rows(frame,13,4,bits.data()+13*w*c,cancelled);files->seal(frame);
            auto tensor=files->read_rows(frame,0,h,cancelled);mx::eval(tensor);
            tc::require(tensor.shape()==mx::Shape{1,1,h,w,c} && std::memcmp(bits.data(),tensor.data<uint16_t>(),bits.size()*2)==0,
                        "raw FP16 file roundtrip changed NaN/signed-zero/finite payloads");
            auto window=files->read_rows(frame,7,9,cancelled);mx::eval(window);
            tc::require(std::memcmp(bits.data()+7*w*c,window.data<uint16_t>(),size_t(9)*w*c*2)==0,"row halo read changed bytes");
            rejects([&]{files->read_rows(frame,16,2,cancelled);});rejects([&]{files->append_rows(frame,h,1,bits.data(),cancelled);});
            cancelled=true;bool typed=false;try{files->read_rows(frame,0,1,cancelled);}catch(const tc::Cancelled &){typed=true;}
            tc::require(typed,"spatial read cancellation lost its native type");cancelled=false;
            uint16_t altered=bits[0]^1;tc::require(pwrite(frame->fd,&altered,2,0)==2,"cannot corrupt owned test frame");
            rejects([&]{files->read_rows(frame,0,1,cancelled);});
            // Deleting backing files is safe while owned GPU read windows live.
            frame.reset();tc::require(files->stats->disk_live_bytes==0,"frame file accounting leaked");
            tc::require(std::memcmp(bits.data()+7*w*c,window.data<uint16_t>(),size_t(9)*w*c*2)==0,"loaded window lost its owner after backing retirement");
            tensor=Tensor(0);window=Tensor(0);mx::synchronize();
            tc::require(files->stats->host_live_bytes==0,"aligned read buffer owner leaked");
            auto limited=std::make_shared<tc::seedvr2::SpatialFiles>(1);rejects([&]{limited->create(h,w,c);});
        }
        tc::require(!std::filesystem::exists(directory),"owned empty spatial scratch directory leaked");
        {
            auto files=std::make_shared<tc::seedvr2::SpatialFiles>(8ull<<20,true);directory=files->directory();
            auto frame=files->create(h,w,c);files->append_rows(frame,0,h,bits.data(),cancelled);files->seal(frame);
            auto first=files->read_rows(frame,0,8,cancelled);mx::eval(first);const void *first_address=first.data<uint16_t>();
            auto held=files->read_rows(frame,8,8,cancelled);mx::eval(held);const void *held_address=held.data<uint16_t>();
            tc::require(first_address!=held_address,"pool aliased two live read windows");
            first=Tensor(0);mx::synchronize();
            auto recycled=files->read_rows(frame,2,6,cancelled);mx::eval(recycled);
            tc::require(recycled.data<uint16_t>()==first_address && files->stats->host_reuses.load()==1,
                "pool did not reuse an idle sufficient-capacity window");
            tc::require(std::memcmp(bits.data()+8*w*c,held.data<uint16_t>(),size_t(8)*w*c*2)==0 &&
                std::memcmp(bits.data()+2*w*c,recycled.data<uint16_t>(),size_t(6)*w*c*2)==0,
                "reused read overwrote a live window or changed payload");
            auto third=files->read_rows(frame,0,1,cancelled);mx::eval(third);
            tc::require(third.data<uint16_t>()!=held_address && third.data<uint16_t>()!=first_address,
                "extra simultaneous consumer reused a live pooled slot");
            files->clear_read_windows();frame.reset();
            tc::require(std::memcmp(bits.data()+8*w*c,held.data<uint16_t>(),size_t(8)*w*c*2)==0,
                "trim/file retirement invalidated a live GPU-owned window");
            held=Tensor(0);recycled=Tensor(0);third=Tensor(0);mx::synchronize();
            tc::require(files->stats->host_live_bytes==0,"trimmed pool retained an allocation after consumers retired");
            constexpr int larger_height=40;std::vector<uint16_t> larger_bits(larger_height*w*c);
            for(size_t i=0;i<larger_bits.size();++i)larger_bits[i]=bits[i%bits.size()];
            auto again=files->create(larger_height,w,c);files->append_rows(again,0,larger_height,larger_bits.data(),cancelled);files->seal(again);
            {auto idle=files->read_rows(again,0,8,cancelled);mx::eval(idle);}
            mx::synchronize();const auto allocations=files->stats->host_allocations.load();
            {auto larger=files->read_rows(again,0,larger_height,cancelled);mx::eval(larger);}
            mx::synchronize();tc::require(files->stats->host_allocations.load()==allocations+1,"undersized slot was not replaced");
            auto saved_stats=files->stats;again.reset();files.reset();
            tc::require(saved_stats->host_live_bytes==0 && !std::filesystem::exists(directory),"pool/session teardown leaked owned memory/scratch");
        }
        {
            auto files=std::make_shared<tc::seedvr2::SpatialFiles>(8ull<<20);directory=files->directory();
            const auto foreign=directory/"unregistered-test-marker";{std::ofstream out(foreign);out<<"preserve unrelated child";}
            files.reset();tc::require(std::filesystem::exists(foreign),"scratch cleanup removed an unregistered child");
            std::filesystem::remove(foreign);std::filesystem::remove(directory);
        }
        int cases=0;
        for(int mode=0;mode<5;++mode)for(bool activate:{false,true}){
            const int channels=mode==4?256:128,t=mode==3?2:1;
            auto input=mx::random::normal({1,t,17,19,channels},mx::float32,mx::random::key(123+mode));
            if(mode==1)input=input*.001f+1.f;if(mode==2)input=mx::ones(input.shape(),mx::float32);
            input=mx::astype(input,mx::float16);if(mode==4)input=mx::transpose(input,{0,1,3,2,4});
            auto weight=mx::astype(mx::random::normal({channels},mx::float32,mx::random::key(223+mode))*.1f+1.f,mx::float16);
            auto bias=mx::random::normal({channels},mx::float16,mx::random::key(323+mode));
            auto expected=tc::seedvr2::metal::group_norm(input,weight,bias,activate);
            auto stats=tc::seedvr2::metal::spatial_norm_statistics(input);std::vector<Tensor> tiles;
            for(int start=0;start<input.shape(2);start+=8){auto row=mx::slice(input,{0,0,start,0,0},{1,t,std::min(start+8,input.shape(2)),input.shape(3),channels});
                tiles.push_back(tc::seedvr2::metal::spatial_norm_affine(row,stats,weight,bias,activate));}
            auto actual=mx::concatenate(tiles,2);mx::eval({expected,actual});
            expected=mx::contiguous(expected);actual=mx::contiguous(actual);mx::eval({expected,actual});
            tc::require(std::memcmp(expected.data<uint16_t>(),actual.data<uint16_t>(),expected.nbytes())==0,
                        "global-statistics row normalization is not bitwise equal");++cases;
        }
        std::cout<<"PASS lossless row files, checksum/cancellation/quota/ownership/cleanup, global norm cases="<<cases<<'\n';
    }catch(const std::exception &error){std::cerr<<error.what()<<'\n';return 1;}
}
