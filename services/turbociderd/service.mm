#import <Foundation/Foundation.h>
#include "turbocider/turbocider.h"
#include <sys/socket.h>
#include <sys/un.h>
#include <sys/stat.h>
#include <sys/file.h>
#include <fcntl.h>
#include <unistd.h>
#include <spawn.h>
#include <sys/wait.h>
#include <poll.h>
#include <csignal>
#include <cerrno>
#include <cstring>
#include <cstdlib>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <deque>
#include <map>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <string>
#include <stdexcept>
#include <iostream>
#include <vector>
extern char **environ;
extern "C" int tc_engine_create_model_candidate(
    const char*, const char*, tc_engine**, char**);
namespace {
volatile std::sig_atomic_t stopping=0;
void stop_service(int){stopping=1;}
std::string encode(id object) {NSData *data=[NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingSortedKeys error:nil];if(!data)throw std::runtime_error("cannot encode response");return {(const char*)data.bytes,data.length};}
id decode(const std::string& text) {return [NSJSONSerialization JSONObjectWithData:[NSData dataWithBytes:text.data() length:text.size()] options:0 error:nil];}
std::string take(char *text){if(!text)return {};std::string value(text);tc_string_free(text);return value;}
void check(bool condition,const char *message){if(!condition)throw std::invalid_argument(message);}
std::string field(NSDictionary *object,NSString *key){id v=object[key];check([v isKindOfClass:NSString.class],"missing or invalid string field");return [v UTF8String];}
std::string read_text(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream.good()) return {};
    return {std::istreambuf_iterator<char>(stream),
            std::istreambuf_iterator<char>()};
}
std::string final_line(std::string text) {
    while (!text.empty() &&
           (text.back() == '\n' || text.back() == '\r' ||
            text.back() == ' ' || text.back() == '\t'))
        text.pop_back();
    auto start = text.find_last_of("\r\n");
    return start == std::string::npos ? text : text.substr(start + 1);
}
struct LtxServiceRequest {
    bool matches = false;
    bool audio = false;
    std::string residency;
};
LtxServiceRequest inspect_ltx_request(NSDictionary *request) {
    LtxServiceRequest result;
    if (![request isKindOfClass:NSDictionary.class] ||
        ![request[@"model"] isKindOfClass:NSString.class] ||
        ![request[@"model"] isEqual:@"ltx-2.5-distilled"])
        return result;
    result.matches = true;
    NSString *residency = nil;
    NSInteger version = request[@"schema_version"] ?
        [request[@"schema_version"] integerValue] : 1;
    if (version == 2) {
        NSDictionary *execution = [request[@"execution"] isKindOfClass:NSDictionary.class] ?
            request[@"execution"] : @{};
        if ([execution[@"residency"] isKindOfClass:NSString.class])
            residency = execution[@"residency"];
        NSArray *outputs = [request[@"outputs"] isKindOfClass:NSArray.class] ?
            request[@"outputs"] : @[];
        NSDictionary *output = outputs.count &&
            [outputs[0] isKindOfClass:NSDictionary.class] ? outputs[0] : @{};
        if ([output[@"audio"] isKindOfClass:NSNumber.class])
            result.audio = [output[@"audio"] boolValue];
    } else {
        if ([request[@"residency"] isKindOfClass:NSString.class])
            residency = request[@"residency"];
        if ([request[@"audio"] isKindOfClass:NSNumber.class])
            result.audio = [request[@"audio"] boolValue];
    }
    if (residency) result.residency = residency.UTF8String;
    return result;
}
bool external_ltx_request(NSDictionary *request) {
    auto value = inspect_ltx_request(request);
    return value.matches && value.residency == "component_staged" &&
        !value.audio;
}
bool resident_ltx_candidate_request(NSDictionary *request) {
    const char *enabled = std::getenv("TURBOCIDER_LTX_RESIDENT_CANDIDATE");
    if (!enabled || std::strcmp(enabled, "1") != 0) return false;
    auto value = inspect_ltx_request(request);
    /* The resident path intentionally keeps the native Transformer, Gemma,
     * MPSGraph and MLX objects in one daemon Session.  Audio remains outside
     * this candidate until its end-to-end provenance/parity gate is complete. */
    return value.matches && value.residency == "resident" && !value.audio;
}
struct File {int fd=-1;~File(){if(fd>=0)close(fd);}};
struct Job {
    NSMutableDictionary *value;
    bool cancellation=false;
    bool external_worker=false;
    bool resident_candidate=false;
    std::chrono::steady_clock::time_point persisted{};
};
class Service {
    std::string directory_;
    std::string executable_;
    std::filesystem::path ltx_conditioning_cache_;
    std::map<std::string,Job> jobs_;
    std::deque<std::string> pending_;
    std::mutex mutex_;
    std::condition_variable available_;
    std::thread worker_;
    bool closing_=false;
    tc_engine *engine_=nullptr;
    pid_t active_child_=-1;
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
    int run_ltx_worker(const std::string& model_path,
                       const std::string& request,
                       std::string& response,
                       std::string& failure) {
        auto template_path = std::filesystem::path(directory_) / ".ltx-worker-XXXXXX";
        std::string template_text = template_path.string();
        std::vector<char> mutable_path(template_text.begin(), template_text.end());
        mutable_path.push_back('\0');
        int request_fd = ::mkstemp(mutable_path.data());
        if (request_fd < 0) {
            failure = "cannot create LTX worker request: " +
                std::string(std::strerror(errno));
            return 1;
        }
        auto request_path = std::filesystem::path(mutable_path.data());
        auto error_path = request_path.string() + ".stderr";
        ::fchmod(request_fd, 0600);
        size_t offset = 0;
        while (offset < request.size()) {
            ssize_t count = ::write(request_fd, request.data() + offset,
                                    request.size() - offset);
            if (count > 0) { offset += static_cast<size_t>(count); continue; }
            if (count < 0 && errno == EINTR) continue;
            ::close(request_fd);
            std::error_code ignored; std::filesystem::remove(request_path, ignored);
            failure = "cannot write LTX worker request";
            return 1;
        }
        ::close(request_fd);

        int output_pipe[2] = {-1, -1};
        if (::pipe(output_pipe) != 0) {
            std::error_code ignored; std::filesystem::remove(request_path, ignored);
            failure = "cannot create LTX worker output pipe: " +
                std::string(std::strerror(errno));
            return 1;
        }
        posix_spawn_file_actions_t actions;
        posix_spawn_file_actions_init(&actions);
        posix_spawn_file_actions_adddup2(&actions, output_pipe[1], STDOUT_FILENO);
        posix_spawn_file_actions_addclose(&actions, output_pipe[0]);
        posix_spawn_file_actions_addclose(&actions, output_pipe[1]);
        posix_spawn_file_actions_addopen(&actions, STDERR_FILENO,
                                         error_path.c_str(),
                                         O_WRONLY | O_CREAT | O_TRUNC, 0600);
        std::vector<std::string> arguments = {
            executable_, "ltx-worker", model_path, request_path.string()
        };
        std::vector<char*> argv;
        for (auto& argument : arguments) argv.push_back(argument.data());
        argv.push_back(nullptr);
        std::vector<std::string> environment;
        for (char **entry = environ; entry && *entry; ++entry) {
            std::string value(*entry);
            if (value.starts_with("TURBOCIDER_LTX_EXEC_FINALIZER=") ||
                value.starts_with("TURBOCIDER_LTX_CONDITIONING_CACHE_DIR="))
                continue;
            environment.push_back(std::move(value));
        }
        environment.emplace_back("TURBOCIDER_LTX_EXEC_FINALIZER=1");
        environment.emplace_back("TURBOCIDER_LTX_CONDITIONING_CACHE_DIR=" +
                                 ltx_conditioning_cache_.string());
        std::vector<char*> envp;
        for (auto& value : environment) envp.push_back(value.data());
        envp.push_back(nullptr);
        pid_t child = -1;
        int spawn_status = ::posix_spawn(&child, executable_.c_str(), &actions,
                                         nullptr, argv.data(), envp.data());
        posix_spawn_file_actions_destroy(&actions);
        ::close(output_pipe[1]);
        if (spawn_status != 0 || child <= 0) {
            ::close(output_pipe[0]);
            std::error_code ignored; std::filesystem::remove(request_path, ignored);
            ignored.clear(); std::filesystem::remove(error_path, ignored);
            failure = "cannot spawn LTX worker: " +
                std::string(std::strerror(spawn_status));
            return 1;
        }
        {
            std::lock_guard<std::mutex> lock(mutex_);
            active_child_ = child;
        }
        int status = 0;
        bool termination_sent = false;
        bool force_sent = false;
        bool waited_for_child = false;
        std::chrono::steady_clock::time_point termination_time{};
        for (;;) {
            pid_t waited = ::waitpid(child, &status, WNOHANG);
            if (waited == child) { waited_for_child = true; break; }
            if (waited < 0 && errno == EINTR) continue;
            if (waited < 0) {
                failure = "cannot wait for LTX worker: " +
                    std::string(std::strerror(errno));
                break;
            }
            bool cancel = false;
            {
                std::lock_guard<std::mutex> lock(mutex_);
                auto found = jobs_.find(active_);
                cancel = closing_ || (found != jobs_.end() && found->second.cancellation);
            }
            if (cancel && !termination_sent) {
                ::kill(child, SIGTERM);
                termination_sent = true;
                termination_time = std::chrono::steady_clock::now();
            } else if (cancel && !force_sent &&
                       std::chrono::steady_clock::now() - termination_time >
                           std::chrono::seconds(2)) {
                ::kill(child, SIGKILL);
                force_sent = true;
            }
            ::usleep(10000);
        }
        if (!waited_for_child) {
            ::kill(child, SIGKILL);
            while (::waitpid(child, &status, 0) < 0 && errno == EINTR) {}
        }
        std::string captured;
        char buffer[4096];
        for (;;) {
            ssize_t count = ::read(output_pipe[0], buffer, sizeof(buffer));
            if (count > 0) { captured.append(buffer, static_cast<size_t>(count)); continue; }
            if (count < 0 && errno == EINTR) continue;
            break;
        }
        ::close(output_pipe[0]);
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (active_child_ == child) active_child_ = -1;
        }
        auto worker_error = final_line(read_text(error_path));
        std::error_code ignored; std::filesystem::remove(request_path, ignored);
        ignored.clear(); std::filesystem::remove(error_path, ignored);
        bool cancelled = false;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            auto found = jobs_.find(active_);
            cancelled = closing_ || (found != jobs_.end() && found->second.cancellation);
        }
        if (cancelled) return 2;
        if (!failure.empty()) return 1;
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0) {
            failure = worker_error.empty() ?
                "LTX worker exited with status " + std::to_string(status) :
                worker_error;
            return 1;
        }
        while (!captured.empty() &&
               (captured.back() == '\n' || captured.back() == '\r' ||
                captured.back() == ' ' || captured.back() == '\t'))
            captured.pop_back();
        if (captured.empty()) {
            failure = "LTX worker returned no result";
            return 1;
        }
        response = std::move(captured);
        return 0;
    }
    void run() {
        for(;;){@autoreleasepool {
            std::unique_lock<std::mutex> lock(mutex_);available_.wait(lock,[&]{return closing_||!pending_.empty();});if(closing_)break;
            active_=pending_.front();pending_.pop_front();auto& job=jobs_.at(active_);
            if(job.cancellation){active_.clear();continue;}
            job.value[@"state"]=@"running";
            auto request=encode(job.value[@"request"]);auto path=field(job.value,@"model_path");
            auto model=field(job.value[@"request"],@"model");auto identity=model+"\n"+path;
            bool external_worker=job.external_worker ||
                external_ltx_request(job.value[@"request"]);
            bool resident_candidate=job.resident_candidate ||
                resident_ltx_candidate_request(job.value[@"request"]);
            bool resident_reuse=!external_worker && resident_candidate &&
                engine_ != nullptr && loaded_ == identity;
            char *error=nullptr,*result=nullptr;int status=0;
            try {
                if(external_worker)
                    job.value[@"progress"]=@{@"schema_version":@1,
                        @"phase":@"external_worker",@"completed":@0,
                        @"total":@1};
                persist(job);
            }catch(const std::exception& e){job.value[@"state"]=@"failed";job.value[@"error"]=@(e.what());continue;}
            if(external_worker){tc_engine_free(engine_);engine_=nullptr;loaded_.clear();}
            else if(loaded_!=identity){
                tc_engine_free(engine_);engine_=nullptr;
                status=resident_candidate ?
                    tc_engine_create_model_candidate(
                        model.c_str(),path.c_str(),&engine_,&error) :
                    tc_engine_create_model(
                        model.c_str(),path.c_str(),&engine_,&error);
                if(!status)loaded_=identity;
            }
            lock.unlock();
            std::string worker_response,worker_failure;
            if(!status){
                if(external_worker)status=run_ltx_worker(path,request,worker_response,worker_failure);
                else status=tc_engine_generate(engine_,request.c_str(),event,this,&result,&error);
            }
            auto response=take(result),failure=take(error);
            if(!worker_response.empty())response=std::move(worker_response);
            if(!worker_failure.empty())failure=std::move(worker_failure);
            lock.lock();
            job.value[@"state"]=status==0?@"succeeded":status==2?@"cancelled":@"failed";
            if(!response.empty()){
                id decoded=decode(response);
                if(decoded)job.value[@"result"]=decoded;
                else if(status==0){status=1;job.value[@"state"]=@"failed";failure="worker returned invalid JSON";}
            }
            if(!failure.empty())job.value[@"error"]=@(failure.c_str());
            if (status == 0 && [job.value[@"result"] isKindOfClass:NSDictionary.class]) {
                NSMutableDictionary *result_value =
                    [job.value[@"result"] mutableCopy];
                result_value[@"service_execution_path"] =
                    external_worker ? @"disposable_worker" :
                    (resident_candidate ? @"resident_session" : @"native_session");
                result_value[@"service_session_reused"] = @(resident_reuse);
                job.value[@"result"] = result_value;
            }
            try{persist(job);}catch(const std::exception& e){job.value[@"storage_error"]=@(e.what());}
            active_.clear();
        }}
    }
public:
    Service(const std::string& directory,const std::string& executable):
        directory_(directory),executable_(executable) {
        NSError *error=nil;check([NSFileManager.defaultManager createDirectoryAtPath:@(directory.c_str()) withIntermediateDirectories:YES attributes:@{NSFilePosixPermissions:@0700} error:&error],"cannot create service store");
        std::error_code cache_error;
        ltx_conditioning_cache_ = std::filesystem::absolute(
            std::filesystem::path(directory_), cache_error) /
            "ltx-conditioning-cache";
        check(!cache_error, "cannot resolve LTX conditioning cache");
        std::filesystem::create_directories(
            ltx_conditioning_cache_, cache_error);
        check(!cache_error, "cannot create LTX conditioning cache");
        check(::setenv("TURBOCIDER_LTX_CONDITIONING_CACHE_DIR",
                       ltx_conditioning_cache_.c_str(), 1) == 0,
              "cannot configure LTX conditioning cache");
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
    ~Service(){ {std::lock_guard<std::mutex> lock(mutex_);closing_=true;tc_engine_cancel(engine_);if(active_child_>0)::kill(active_child_,SIGTERM);for(auto& id:pending_){auto&job=jobs_.at(id);job.value[@"state"]=@"interrupted";try{persist(job);}catch(...){}}}available_.notify_one();worker_.join();tc_engine_free(engine_);}
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
            id plan_value=decode(prepared);
            /* The planner returns a flattened execution summary, not the
             * request schema.  In particular, schema-v2 residency lives in
             * request.execution.residency but the plan exposes it at the top
             * level.  Route from the validated original request so both
             * schema versions select the same disposable LTX worker. */
            bool external_worker=external_ltx_request(inference);
            bool resident_candidate=resident_ltx_candidate_request(inference);
            check([plan_value[@"executable"] boolValue] || external_worker ||
                  resident_candidate,"model executor unavailable");
            auto path=field(request,@"model_path");check(!path.empty(),"model path required");
            auto id=std::string(NSUUID.UUID.UUIDString.UTF8String);
            Job job{[@{@"schema_version":@1,@"id":@(id.c_str()),@"state":@"queued",@"created_at":@(NSDate.date.timeIntervalSince1970),@"model_path":@(path.c_str()),@"request":inference} mutableCopy]};
            job.external_worker=external_worker;
            job.resident_candidate=resident_candidate;
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
                job.cancellation=true;job.value[@"state"]=active_==id?@"cancelling":@"cancelled";if(active_==id){if(active_child_>0)::kill(active_child_,SIGTERM);else tc_engine_cancel(engine_);}persist(job);
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
int tc_service_main(const char *socket_path,const char *directory,const char *executable) {@autoreleasepool{try {
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
    check(executable&&*executable,"service executable path is required");
    Service service(directory,executable);stopping=0;std::signal(SIGINT,stop_service);std::signal(SIGTERM,stop_service);
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
