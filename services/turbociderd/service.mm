#import <Foundation/Foundation.h>
#include "turbocider/turbocider.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <poll.h>
#include <csignal>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <map>
#include <chrono>
#include <string>
#include <stdexcept>
#include <iostream>
namespace {
volatile std::sig_atomic_t stopping=0;
void stop_service(int){stopping=1;}
std::string encode(id object) {NSData *data=[NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingSortedKeys error:nil];if(!data)throw std::runtime_error("cannot encode response");return {(const char*)data.bytes,data.length};}
id decode(const std::string& text) {return [NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:text.data() length:text.size()] options:0 error:nil];}
std::string take(char *text){if(!text)return {};std::string value(text);tc_string_free(text);return value;}
void check(bool condition,const char *message){if(!condition)throw std::invalid_argument(message);}
std::string field(NSDictionary *object,NSString *key){id v=object[key];check([v isKindOfClass:NSString.class],"missing or invalid string field");return [v UTF8String];}
struct File {int fd=-1;~File(){if(fd>=0)close(fd);}};
struct Job {
    NSMutableDictionary *value;
    bool cancellation=false;
    std::chrono::steady_clock::time_point persisted{};
};
class Service {
    std::string directory_;
    std::map<std::string,Job> jobs_;
    std::deque<std::string> pending_;
    std::mutex mutex_;
    std::condition_variable available_;
    std::thread worker_;
    bool closing_=false;
    tc_engine *engine_=nullptr;
    std::string loaded_,active_;
    void persist(Job& job) {
        auto path=directory_+"/"+field(job.value,@"id")+".json";
        auto text=encode(job.value);NSError *error=nil;
        check([[NSData dataWithBytes:text.data() length:text.size()] writeToFile:@(path.c_str()) options:NSDataWritingAtomic error:&error],"cannot persist job state");
        job.persisted=std::chrono::steady_clock::now();
    }
    static void event(const char *text,void *opaque) noexcept {
        auto self=static_cast<Service*>(opaque);
        @autoreleasepool {try {
            std::lock_guard<std::mutex> lock(self->mutex_);auto& job=self->jobs_.at(self->active_);
            if(job.cancellation)tc_engine_cancel(self->engine_);
            job.value[@"progress"]=decode(text)?:@{};
            if(std::chrono::steady_clock::now()-job.persisted>std::chrono::seconds(1))self->persist(job);
        }catch(...){tc_engine_cancel(self->engine_);}}
    }
    void run() {
        for(;;){@autoreleasepool {
            std::unique_lock<std::mutex> lock(mutex_);available_.wait(lock,[&]{return closing_||!pending_.empty();});if(closing_)break;
            active_=pending_.front();pending_.pop_front();auto& job=jobs_.at(active_);
            if(job.cancellation)continue;
            job.value[@"state"]=@"running";
            auto request=encode(job.value[@"request"]);auto path=field(job.value,@"model_path");
            auto model=field(job.value[@"request"],@"model");auto identity=model+"\n"+path;
            char *error=nullptr,*result=nullptr;int status=0;
            try {persist(job);}catch(const std::exception& e){job.value[@"state"]=@"failed";job.value[@"error"]=@(e.what());continue;}
            if(loaded_!=identity){tc_engine_free(engine_);engine_=nullptr;status=tc_engine_create_model(model.c_str(),path.c_str(),&engine_,&error);if(!status)loaded_=identity;}
            lock.unlock();
            if(!status)status=tc_engine_generate(engine_,request.c_str(),event,this,&result,&error);
            auto response=take(result),failure=take(error);lock.lock();
            job.value[@"state"]=status==0?@"succeeded":status==2?@"cancelled":@"failed";
            if(!response.empty())job.value[@"result"]=decode(response);
            if(!failure.empty())job.value[@"error"]=@(failure.c_str());
            try{persist(job);}catch(const std::exception& e){job.value[@"storage_error"]=@(e.what());}
            active_.clear();
        }}
    }
public:
    explicit Service(const std::string& directory):directory_(directory) {
        NSError *error=nil;check([NSFileManager.defaultManager createDirectoryAtPath:@(directory.c_str()) withIntermediateDirectories:YES attributes:@{NSFilePosixPermissions:@0700} error:&error],"cannot create service store");
        for(NSString *name in [NSFileManager.defaultManager contentsOfDirectoryAtPath:@(directory.c_str()) error:nil]) {
            if(![name.pathExtension isEqual:@"json"])continue;
            NSData *data=[NSData dataWithContentsOfFile:[@(directory.c_str()) stringByAppendingPathComponent:name]];
            id value=[NSJSONSerialization JSONObjectWithData:data options:NSJSONReadingMutableContainers error:nil];
            if(![value isKindOfClass:NSDictionary.class])continue;
            auto id=field(value,@"id");check([name isEqual:@((id+".json").c_str())],"job file identity mismatch");
            Job job{value};
            if([@[@"queued",@"running",@"cancelling"] containsObject:value[@"state"]]){value[@"state"]=@"interrupted";persist(job);}
            jobs_.emplace(id,std::move(job));
        }
        worker_=std::thread([this]{run();});
    }
    ~Service(){ {std::lock_guard<std::mutex> lock(mutex_);closing_=true;tc_engine_cancel(engine_);for(auto& id:pending_){auto&job=jobs_.at(id);job.value[@"state"]=@"interrupted";try{persist(job);}catch(...){}}}available_.notify_one();worker_.join();tc_engine_free(engine_);}
    id rpc(NSDictionary *request) {
        auto action=field(request,@"action");
        if(action=="models")return decode(take(tc_models_json()));
        if(action=="doctor")return decode(take(tc_system_json()));
        if(action=="plan") {char *result=nullptr,*error=nullptr;auto text=encode(request[@"request"]);auto status=tc_plan_json(text.c_str(),&result,&error);auto value=take(result),message=take(error);check(status==0,message.c_str());return decode(value);}
        std::lock_guard<std::mutex> lock(mutex_);
        if(action=="submit") {
            check(pending_.size()<32,"job queue is full");check(jobs_.size()<10000,"job history limit reached");
            check([request[@"request"] isKindOfClass:NSDictionary.class],"request must be object");
            NSMutableDictionary *inference=[request[@"request"] mutableCopy];if(!inference[@"model"])inference[@"model"]=@"flux2-klein-4b";
            char *plan=nullptr,*error=nullptr;auto text=encode(inference);auto status=tc_plan_json(text.c_str(),&plan,&error);auto message=take(error),prepared=take(plan);check(status==0,message.c_str());
            check([decode(prepared)[@"executable"] boolValue],"model executor unavailable");
            auto path=field(request,@"model_path");check(!path.empty(),"model path required");
            auto id=std::string(NSUUID.UUID.UUIDString.UTF8String);
            Job job{[@{@"schema_version":@1,@"id":@(id.c_str()),@"state":@"queued",@"created_at":@(NSDate.date.timeIntervalSince1970),@"model_path":@(path.c_str()),@"request":inference} mutableCopy]};
            persist(job);jobs_.emplace(id,std::move(job));pending_.push_back(id);available_.notify_one();return @{@"id":@(id.c_str()),@"state":@"queued"};
        }
        if(action=="jobs") {
            int offset=request[@"offset"]?[request[@"offset"] intValue]:0;
            int limit=request[@"limit"]?[request[@"limit"] intValue]:20;
            check(offset>=0&&limit>0&&limit<=100,"invalid history page");
            NSMutableArray *values=[NSMutableArray array];for(auto& [id,job]:jobs_)[values addObject:job.value];
            [values sortUsingComparator:^NSComparisonResult(NSDictionary *a,NSDictionary *b){return [b[@"created_at"] compare:a[@"created_at"]];}];
            NSUInteger start=std::min(NSUInteger(offset),values.count),count=std::min(NSUInteger(limit),values.count-start);
            return decode(encode(@{@"jobs":[values subarrayWithRange:NSMakeRange(start,count)],@"total":@(values.count),@"next_offset":start+count<values.count?@(start+count):[NSNull null]}));
        }
        auto id=field(request,@"id");auto found=jobs_.find(id);check(found!=jobs_.end(),"unknown job");auto& job=found->second;
        if(action=="cancel") {
            if([@[@"queued",@"running",@"cancelling"] containsObject:job.value[@"state"]]){
                job.cancellation=true;job.value[@"state"]=active_==id?@"cancelling":@"cancelled";if(active_==id)tc_engine_cancel(engine_);persist(job);
            }
        }else check(action=="status","unknown action");
        return decode(encode(job.value));
    }
};
sockaddr_un address(const char *path){sockaddr_un a{};a.sun_family=AF_UNIX;check(strlen(path)<sizeof(a.sun_path),"socket path too long");strcpy(a.sun_path,path);return a;}
std::string receive(int fd){std::string text;char buffer[4096];while(text.size()<=1<<20){ssize_t n=read(fd,buffer,sizeof(buffer));check(n>0,"connection ended before newline");text.append(buffer,n);auto end=text.find('\n');if(end!=std::string::npos){text.resize(end);return text;}}throw std::invalid_argument("request exceeds 1 MiB");}
void send_all(int fd,const std::string& text){size_t offset=0;while(offset<text.size()){auto n=write(fd,text.data()+offset,text.size()-offset);check(n>0,"connection write failed");offset+=n;}}
void configure(int fd){int one=1;setsockopt(fd,SOL_SOCKET,SO_NOSIGPIPE,&one,sizeof(one));timeval timeout{5,0};setsockopt(fd,SOL_SOCKET,SO_RCVTIMEO,&timeout,sizeof(timeout));setsockopt(fd,SOL_SOCKET,SO_SNDTIMEO,&timeout,sizeof(timeout));}
}
int tc_service_main(const char *socket_path,const char *directory) {@autoreleasepool{try {
    check([NSFileManager.defaultManager createDirectoryAtPath:@(directory) withIntermediateDirectories:YES attributes:@{NSFilePosixPermissions:@0700} error:nil],"cannot create service state directory");
    File lock{open((std::string(directory)+"/.service.lock").c_str(),O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600)};check(lock.fd>=0&&flock(lock.fd,LOCK_EX|LOCK_NB)==0,"service store is locked");
    File socket_lock{open((std::string(socket_path)+".lock").c_str(),O_CREAT|O_RDWR|O_CLOEXEC|O_NOFOLLOW,0600)};
    check(socket_lock.fd>=0&&flock(socket_lock.fd,LOCK_EX|LOCK_NB)==0,"socket belongs to an active service");
    struct stat stale{};
    if(lstat(socket_path,&stale)==0){
        check(S_ISSOCK(stale.st_mode)&&stale.st_uid==geteuid(),"socket path is not an owned Unix socket");
        File probe{socket(AF_UNIX,SOCK_STREAM,0)};auto existing=address(socket_path);
        check(connect(probe.fd,(sockaddr*)&existing,sizeof(existing))!=0&&errno==ECONNREFUSED,"socket is already active");
        check(unlink(socket_path)==0,"cannot remove stale socket");
    }
    auto a=address(socket_path);File server{socket(AF_UNIX,SOCK_STREAM,0)};check(server.fd>=0,"cannot create socket");
    check(bind(server.fd,(sockaddr*)&a,sizeof(a))==0,"cannot bind socket (already running or stale socket)");
    struct SocketPath {const char *path;~SocketPath(){unlink(path);}} cleanup{socket_path};chmod(socket_path,0600);check(listen(server.fd,8)==0,"cannot listen");
    Service service(directory);stopping=0;std::signal(SIGINT,stop_service);std::signal(SIGTERM,stop_service);
    std::cout<<"{\"ready\":true}"<<std::endl;
    while(!stopping){pollfd p{server.fd,POLLIN,0};if(poll(&p,1,200)<=0)continue;File client{accept(server.fd,nullptr,nullptr)};if(client.fd<0)continue;configure(client.fd);
        @autoreleasepool {try{id request=decode(receive(client.fd));check([request isKindOfClass:NSDictionary.class],"RPC must be object");id result=service.rpc(request);send_all(client.fd,encode(@{@"ok":@YES,@"result":result})+"\n");}catch(const std::exception& e){try{send_all(client.fd,encode(@{@"ok":@NO,@"error":@(e.what())})+"\n");}catch(...){}}}
    }
    std::signal(SIGINT,SIG_DFL);std::signal(SIGTERM,SIG_DFL);return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}}
int tc_rpc_main(const char *socket_path,const char *file) {@autoreleasepool{try{
    NSData *data=[NSData dataWithContentsOfFile:@(file)];check(data!=nil,"cannot read RPC request");auto a=address(socket_path);File client{socket(AF_UNIX,SOCK_STREAM,0)};check(client.fd>=0,"cannot create socket");configure(client.fd);check(connect(client.fd,(sockaddr*)&a,sizeof(a))==0,"cannot connect to service");
    id payload=[NSJSONSerialization JSONObjectWithData:data options:0 error:nil];check([payload isKindOfClass:NSDictionary.class],"RPC request must be object");send_all(client.fd,encode(payload)+"\n");std::cout<<receive(client.fd)<<std::endl;return 0;
}catch(const std::exception& e){std::cerr<<e.what()<<std::endl;return 1;}}}
