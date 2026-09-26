#pragma once
#include <unistd.h>
#include <cerrno>
namespace tc_worker {
// Optional supervised launch barrier. The parent closes its pipe on failure or
// exit, so EOF cannot start model work. This grants no model/catalog authority.
inline bool await_admission() {
    unsigned char value=0;
    ssize_t count;
    do { count=read(STDIN_FILENO,&value,1); } while(count<0 && errno==EINTR);
    return count==1 && value==1;
}
}
