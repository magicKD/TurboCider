// Complete causal attention suffix: QK^T, global causal softmax, PV.
class HeadBranch {
 public:
    HeadBranch(const Options &o, void *input, void *output)
      : m(o.M),h(o.H),count(o.headSuffix),dim(o.H/o.heads),
        first(o.heads-o.headSuffix),in((__fp16*)input),out((__fp16*)output),
        packed(size_t(m)*3*h),scores(size_t(m)*m),result(size_t(m)*dim) {
        if(o.headBackend=="ane") ml=std::make_unique<CoreMLGemm>(o.headModel,m,3*h,count*dim,input,output);
    }
    double run() {
        const auto t=Clock::now();
        if(ml) {
            ml->run();
            if(!ml->usedOutputBacking()) {
                MLMultiArray *a=ml->output();const __fp16 *p=(const __fp16*)a.dataPointer;
                for(int i=0;i<m;++i) for(int j=0;j<count*dim;++j)
                    out[size_t(i)*count*dim+j]=p[size_t(i)*a.strides[0].unsignedLongLongValue+size_t(j)*a.strides[1].unsignedLongLongValue];
            }
        } else {
            for(size_t i=0;i<packed.size();++i)packed[i]=float(in[i]);
            for(int head=first;head<first+count;++head) {
                const float *q=packed.data()+head*dim,*k=q+h,*v=k+h;
                cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasTrans,m,m,dim,1/std::sqrt(float(dim)),q,3*h,k,3*h,0,scores.data(),m);
                for(int i=0;i<m;++i) {
                    float mx=-INFINITY,sum=0;
                    for(int j=0;j<=i;++j)mx=std::max(mx,scores[size_t(i)*m+j]);
                    for(int j=0;j<m;++j){float e=j<=i?std::exp(scores[size_t(i)*m+j]-mx):0;scores[size_t(i)*m+j]=e;sum+=e;}
                    for(int j=0;j<m;++j)scores[size_t(i)*m+j]/=sum;
                }
                cblas_sgemm(CblasRowMajor,CblasNoTrans,CblasNoTrans,m,dim,m,1,scores.data(),m,v,3*h,0,result.data(),dim);
                for(int i=0;i<m;++i)for(int j=0;j<dim;++j)out[size_t(i)*count*dim+(head-first)*dim+j]=__fp16(result[size_t(i)*dim+j]);
            }
        }
        return milliseconds(Clock::now()-t);
    }
 private:
    int m,h,count,dim,first;__fp16 *in,*out;
    std::vector<float> packed,scores,result;
    std::unique_ptr<CoreMLGemm> ml;
};
