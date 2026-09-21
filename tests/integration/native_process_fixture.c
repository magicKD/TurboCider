#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <fcntl.h>
#include <sys/types.h>
static void stop(int value) { (void)value; const char text[]="cancelled\n"; (void)write(1,text,sizeof(text)-1); _exit(2); }
static void ready(const char *path,pid_t other) {
    int fd=open(path,O_WRONLY|O_CREAT|O_EXCL,0600);if(fd<0)_exit(90);
    dprintf(fd,"%d %d %d\n",getpid(),getpgrp(),other);close(fd);
}
int main(int argc,char **argv) {
    if(argc!=3)return 91;
    const char *mode=argv[1];
    if(!strcmp(mode,"ignore") || !strcmp(mode,"child") || !strcmp(mode,"early_parent"))signal(SIGTERM,SIG_IGN);
    if(!strcmp(mode,"cooperative"))signal(SIGTERM,stop);
    pid_t child=0;
    if(!strcmp(mode,"child") || !strcmp(mode,"early_parent")) {
        child=fork();if(child<0)return 92;
        if(!child){for(;;)pause();}
    }
    ready(argv[2],child);
    if(!strcmp(mode,"early_parent"))return 0;
    if(!strcmp(mode,"normal")){printf("terminal\n");fprintf(stderr,"diagnostic\n");return 7;}
    if(!strcmp(mode,"large") || !strcmp(mode,"flood")) {
        char buffer[4096];memset(buffer,'x',sizeof(buffer));
        unsigned count=0;
        while(!strcmp(mode,"flood") || count<768){ssize_t wrote=write(1,buffer,sizeof(buffer));if(wrote<=0)return 93;++count;}
        for(unsigned i=0;i<128;++i)if(write(2,buffer,sizeof(buffer))<=0)return 94;
        return 0;
    }
    for(;;)pause();
}
