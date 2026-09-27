#include "../../apps/cli/worker_events.hpp"
int main(int argc,char **) {
    @autoreleasepool {
        tc_worker::Events events(@{@"job_id":@"job",@"request_id":@"request",@"request_digest":@"digest"},"runtime");
        if(argc>1) {
            events.resolved(@{@"large":[@"x" stringByPaddingToLength:128*1024 withString:@"x" startingAtIndex:0]});
        } else events.resolved(@{@"resolution_digest":@"fixture"});
        for(int i=0;i<10000;++i) {
            auto value=tc_worker::canonical_request(@{@"sequence":@(i+1),@"phase":i%2?@"load":@"denoise",@"completed":@1,@"total":@2,@"elapsed_seconds":@1});
            NSString *text=[[NSString alloc] initWithData:value encoding:NSUTF8StringEncoding];
            tc_worker::Events::progress(text.UTF8String,&events);
        }
    }
}
