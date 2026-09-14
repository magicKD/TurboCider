/* Synthetic full video-self branch attribution, not Stage-2/video evidence. */
#include "../../native/models/ltx_runtime/ltx_gpu.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {IN,WQ,WK,WV,WO,SQ,SK,SV,SO,BQ,BK,BV,BO,NQ,NK,GW,GB,
      Q,K,V,COS,SIN,GL,GATE,CORE,ROT,OUT,REF,PQ,PK,PV,PO,QC,KC,VS,TH,RO,
      SAVED_POOL,SAVED_EXACT,COUNT};
enum {VARIANTS=7};
static ltx_gpu *gpu;
static ltx_gpu_buffer *b[COUNT];
static unsigned rows;
static char error[2048];
static uint32_t rng = 42;
static uint32_t random_bits(void) {
    rng ^= rng << 13; rng ^= rng >> 17; rng ^= rng << 5; return rng;
}
static uint16_t bf16(float x) {
    uint32_t bits; memcpy(&bits, &x, 4);
    return (uint16_t)((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}
static float f32(uint16_t x) {
    uint32_t bits = (uint32_t)x << 16; float y; memcpy(&y, &bits, 4); return y;
}
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC, &t);
    return t.tv_sec + t.tv_nsec * 1e-9;
}
static int order(const void *a, const void *c) {
    double x = *(const double *)a, y = *(const double *)c;
    return (x > y) - (x < y);
}
static double median(double *values, unsigned count) {
    double copy[100]; memcpy(copy, values, count * sizeof(double));
    qsort(copy, count, sizeof(double), order);
    return count % 2 ? copy[count/2] : (copy[count/2-1]+copy[count/2])*.5;
}
static int parse(const char *text, unsigned maximum, unsigned *value) {
    char *end; errno = 0;
    unsigned long result = strtoul(text, &end, 10);
    if (errno || end == text || *end || text[0] == '-' || !result || result > maximum)
        return 0;
    *value = (unsigned)result; return 1;
}
static int full(void) {
    return ltx_gpu_self_attention_int8_mps_bf16(gpu,b[REF],b[IN],
        b[WQ],b[SQ],b[BQ],b[WK],b[SK],b[BK],b[WV],b[SV],b[BV],
        b[NQ],b[NK],b[GW],b[GB],b[WO],b[SO],b[BO],b[COS],b[SIN],
        rows,32,128,256,1e-6f,error,sizeof(error));
}
static int qkv(void) {
    return ltx_gpu_qkv_int8_convrot_mps_bf16(gpu,b[Q],b[K],b[V],b[IN],
        b[WQ],b[SQ],b[BQ],b[WK],b[SK],b[BK],b[WV],b[SV],b[BV],
        b[NQ],b[NK],rows,4096,4096,256,1e-6f,error,sizeof(error));
}
static int gate(void) {
    return ltx_gpu_linear_bf16(gpu,b[GL],b[IN],b[GW],b[GB],
        rows,4096,32,error,sizeof(error)) &&
        ltx_gpu_sigmoid2_bf16(gpu,b[GATE],b[GL],rows*32,error,sizeof(error));
}
static int core(unsigned variant) {
    if (variant == 1)
        return ltx_gpu_self_attention_core_mps_bf16(gpu,b[CORE],b[Q],b[K],b[V],
            b[COS],b[SIN],b[GATE],rows,32,128,1.f/sqrtf(128.f),error,sizeof(error));
    ltx_sparse_pattern pattern = {variant == 2 ? 4u : (variant == 3 || variant == 5) ? 5u : 1u,
        (variant == 4 || variant == 6) ? 256u : 0u,0,0,32};
    return ltx_gpu_self_attention_core_sparse_bf16(gpu,b[CORE],b[Q],b[K],b[V],
        b[COS],b[SIN],b[GATE],b[PQ],b[PK],b[PV],b[PO],b[QC],b[KC],b[VS],
        b[TH],b[RO],rows,32,128,1.f/sqrtf(128.f),.5f,0,0,0,0,&pattern,error,sizeof(error));
}
static int post(unsigned variant) {
    if (variant >= 5)
        return ltx_gpu_linear_int8_convrot_mps_bf16(gpu,b[OUT],b[CORE],b[WO],b[SO],b[BO],
            rows,4096,4096,256,error,sizeof(error));
    return ltx_gpu_convrot_bf16(gpu,b[ROT],b[CORE],rows,4096,256,error,sizeof(error)) &&
        ltx_gpu_linear_int8_weight_mps_bf16(gpu,b[OUT],b[ROT],b[WO],b[SO],b[BO],
            rows,4096,4096,error,sizeof(error));
}

int main(int argc, char **argv) {
    unsigned runs;
    if (argc != 4 || !parse(argv[2],16384,&rows) || !parse(argv[3],100,&runs)) {
        fprintf(stderr,"usage: %s ABSOLUTE_SHADER_PATH ROWS RUNS\n",argv[0]); return 2;
    }
    size_t elements = (size_t)rows * 4096, weights = (size_t)4096 * 4096;
    size_t blocks = (rows + 63u)/64u, summaries = blocks * 4096;
    size_t sizes[COUNT] = {0};
    sizes[IN] = elements*2;
    for (unsigned i=WQ; i<=WO; ++i) sizes[i] = weights;
    for (unsigned i=SQ; i<=SO; ++i) sizes[i] = 4096*4;
    for (unsigned i=BQ; i<=NK; ++i) sizes[i] = 4096*2;
    sizes[GW] = 4096*32*2; sizes[GB] = 32*2;
    for (unsigned i=Q; i<=V; ++i) sizes[i] = elements*2;
    sizes[COS] = sizes[SIN] = elements;
    sizes[GL] = sizes[GATE] = (size_t)rows*32*2;
    for (unsigned i=CORE; i<=PO; ++i) sizes[i] = elements*2;
    sizes[QC] = summaries*4; sizes[KC] = sizes[VS] = summaries*2;
    sizes[TH] = blocks*32*4;
    sizes[RO] = ltx_sparse_route_scratch_words(rows,32)*4;
    sizes[SAVED_POOL] = sizes[SAVED_EXACT] = elements*2;
    int status = 1;
    gpu = ltx_gpu_create(argv[1],error,sizeof(error));
    if (!gpu) goto done;
    for (unsigned i=0; i<COUNT; ++i) {
        b[i] = ltx_gpu_buffer_new(gpu,sizes[i],error,sizeof(error));
        if (!b[i]) goto done;
        memset(ltx_gpu_buffer_contents(b[i]),0,sizes[i]);
    }
    uint16_t *input = ltx_gpu_buffer_contents(b[IN]);
    for (size_t i=0; i<elements; ++i)
        input[i] = bf16(((float)(random_bits()>>8)/16777216.f-.5f)*2.f);
    for (unsigned p=0; p<4; ++p) {
        int8_t *w = ltx_gpu_buffer_contents(b[WQ+p]);
        float *s = ltx_gpu_buffer_contents(b[SQ+p]);
        uint16_t *bias = ltx_gpu_buffer_contents(b[BQ+p]);
        for (size_t i=0; i<weights; ++i) w[i] = (int8_t)((int)(random_bits()%255)-127);
        for (unsigned i=0; i<4096; ++i) {s[i]=.0002f; bias[i]=bf16(.01f);}
    }
    for (unsigned p=NQ; p<=NK; ++p) {
        uint16_t *norm = ltx_gpu_buffer_contents(b[p]);
        for (unsigned i=0; i<4096; ++i) norm[i]=bf16(1.f);
    }
    uint16_t *gw = ltx_gpu_buffer_contents(b[GW]);
    for (unsigned i=0; i<4096*32; ++i)
        gw[i]=bf16(((float)(random_bits()>>8)/16777216.f-.5f)*.02f);
    for (size_t i=0; i<elements/2; ++i) {
        ((uint16_t *)ltx_gpu_buffer_contents(b[COS]))[i]=bf16(cosf((float)(i%97)*.01f));
        ((uint16_t *)ltx_gpu_buffer_contents(b[SIN]))[i]=bf16(sinf((float)(i%97)*.01f));
    }
    double times[VARIANTS][5][100] = {{{0}}}, relative[VARIANTS]={0}, maximum[VARIANTS]={0};
    double pair_relative[VARIANTS]={0}, pair_maximum[VARIANTS]={0};
    size_t pair_changed[VARIANTS]={0};
    for (unsigned iteration=0; iteration<runs+2; ++iteration) {
        for (unsigned slot=0; slot<VARIANTS; ++slot) {
            unsigned variant = iteration == 0 ? slot : (slot+iteration)%VARIANTS;
            double start = now(), parts[5]={0}, previous = start;
            if (!variant) {if (!full()) goto done;}
            else {
                if (!qkv()) goto done;
                parts[1]=now()-previous; previous=now();
                if (!gate()) goto done;
                parts[2]=now()-previous; previous=now();
                if (!core(variant)) goto done;
                parts[3]=now()-previous; previous=now();
                if (!post(variant)) goto done;
                parts[4]=now()-previous;
            }
            parts[0]=now()-start;
            if (iteration>=2)
                for (unsigned p=0; p<5; ++p) times[variant][p][iteration-2]=parts[p];
            if (variant && iteration==runs+1) {
                const uint16_t *reference=ltx_gpu_buffer_contents(b[REF]);
                const uint16_t *output=ltx_gpu_buffer_contents(b[OUT]);
                double error2=0,norm2=0;
                for (size_t i=0; i<elements; ++i) {
                    double x=f32(reference[i]),y=f32(output[i]),delta=x-y;
                    if (!isfinite(x)||!isfinite(y)) {
                        snprintf(error,sizeof(error),"nonfinite output"); goto done;
                    }
                    error2+=delta*delta; norm2+=x*x;
                    maximum[variant]=fmax(maximum[variant],fabs(delta));
                }
                relative[variant]=sqrt(error2/fmax(norm2,1e-30));
                if (variant >= 5) {
                    /* Untimed comparison: both output paths consume this exact
                     * CORE buffer, excluding QKV/attention inter-run drift. */
                    ltx_gpu_buffer *saved=b[variant == 5 ? SAVED_POOL : SAVED_EXACT];
                    memcpy(ltx_gpu_buffer_contents(saved),output,elements*2);
                    if (!post(variant == 5 ? 3u : 4u)) goto done;
                    reference=ltx_gpu_buffer_contents(b[OUT]);
                    output=ltx_gpu_buffer_contents(saved);
                    error2=norm2=0;
                    for (size_t i=0; i<elements; ++i) {
                        double x=f32(reference[i]),y=f32(output[i]),delta=x-y;
                        if (!isfinite(x)) {snprintf(error,sizeof(error),"nonfinite saved output"); goto done;}
                        error2+=delta*delta; norm2+=x*x;
                        pair_changed[variant]+=reference[i]!=output[i];
                        pair_maximum[variant]=fmax(pair_maximum[variant],fabs(delta));
                    }
                    pair_relative[variant]=sqrt(error2/fmax(norm2,1e-30));
                }
            }
        }
    }
    const char *names[]={"fused_dense","split_dense","direct_topk32","pooled_topk32","all_exact",
                        "pooled_topk32_fused_post","all_exact_fused_post"};
    const char *fields[]={"total_seconds","qkv_seconds","gate_seconds","core_seconds","post_seconds"};
    printf("{\"schema\":\"ltx-attention-branch-probe-v2\",\"rows\":%u,\"heads\":32,"
           "\"dim\":128,\"runs\":%u,\"synthetic\":true,\"batch_commands\":false,"
           "\"warmups\":2,\"variants\":[",rows,runs);
    for (unsigned v=0; v<VARIANTS; ++v) {
        printf("%s{\"name\":\"%s\"",v?",":"",names[v]);
        for (unsigned p=0; p<(v?5:1); ++p)
            printf(",\"%s\":%.9g",fields[p],median(times[v][p],runs));
        printf(",\"relative_l2_vs_fused\":%.9g,\"max_abs_vs_fused\":%.9g",relative[v],maximum[v]);
        if (v>=5)
            printf(",\"relative_l2_vs_unfused_post\":%.9g,\"max_abs_vs_unfused_post\":%.9g,"
                   "\"changed_bf16_elements_vs_unfused_post\":%zu",pair_relative[v],pair_maximum[v],pair_changed[v]);
        printf(",\"samples\":[");
        for (unsigned i=0; i<runs; ++i) printf("%s%.9g",i?",":"",times[v][0][i]);
        printf("]}");
    }
    printf("]}\n"); status=0;
done:
    if (status) fprintf(stderr,"%s\n",error);
    for (unsigned i=0; i<COUNT; ++i) ltx_gpu_buffer_free(b[i]);
    if (gpu) ltx_gpu_free(gpu);
    return status;
}
