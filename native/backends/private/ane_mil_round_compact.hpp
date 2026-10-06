#pragma once

#include <string>

namespace tc::ane::private_api {
// Explicit numerical BF16-RNE values in an FP16 carrier, canonicalizing -0.
// Binary normalization uses exact powers of two. Unlike log2 bin estimation,
// no near-power-of-two approximation decides a rounding boundary. Callers
// must guard the 8 finite-half encodings whose BF16 value overflows FP16.
inline std::string emit_bf16_value_round_compact(std::string &body,const std::string &input,
                                              const std::string &prefix,const std::string &shape,bool magic=false) {
    const auto type="tensor<fp16, "+shape+">",boolean="tensor<bool, "+shape+">";
    auto line=[&](const std::string &name,const std::string &expr,bool predicate=false) {
        const auto symbol=prefix+"_"+name;
        body+="        "+(predicate?boolean:type)+" "+symbol+" = "+expr+";\n";
        return symbol;
    };
    const auto magnitude=line("abs","abs(x = "+input+")");
    auto normalized=magnitude;
    std::string inverses[5];
    const int powers[]{8,4,2,1,0};
    for(int i=0;i<5;++i) {
        const auto name=std::to_string(i);
        const auto condition=line("ge1_"+name,"greater_equal(x = "+normalized+", y = fp16(1))",true);
        // Last step maps exponent -1/0 to [1,2). Other steps halve the
        // exponent search interval: [-16,15] -> [-8,7] -> ... -> [-1,0].
        const auto down=i==4?"1":"0x1p-"+std::to_string(powers[i]);
        const auto up=i==4?"2":"0x1p+"+std::to_string(powers[i]);
        const auto factor=line("factor_"+name,"select(cond = "+condition+", a = fp16("+down+"), b = fp16("+up+"))");
        inverses[i]=i==4?line("inverse_last","select(cond = "+condition+", a = fp16(1), b = fp16(0.5))"):
            line("inverse_"+name,"select(cond = "+condition+", a = fp16("+up+"), b = fp16("+down+"))");
        normalized=line("normalize_"+name,"mul(x = "+normalized+", y = "+factor+")");
    }
    std::string restored;
    if(magic) {
        // Negative diagnostic: the current actual compiler/driver does not
        // retain the required intermediate FP16 rounding of this add/sub.
        const auto shifted=line("shifted","add(x = "+normalized+", y = fp16(8))");
        restored=line("rounded","sub(x = "+shifted+", y = fp16(8))");
    } else {
        const auto scaled=line("scaled","mul(x = "+normalized+", y = fp16(128))");
        const auto lower=line("lower","floor(x = "+scaled+")");
        const auto fraction=line("fraction","sub(x = "+scaled+", y = "+lower+")");
        const auto half=line("half","mul(x = "+lower+", y = fp16(0.5))");
        const auto parity_floor=line("parity_floor","floor(x = "+half+")");
        const auto twice=line("twice","add(x = "+parity_floor+", y = "+parity_floor+")");
        const auto parity=line("parity","sub(x = "+lower+", y = "+twice+")");
        const auto above=line("above","greater(x = "+fraction+", y = fp16(0.5))",true);
        const auto tie=line("tie","equal(x = "+fraction+", y = fp16(0.5))",true);
        const auto increment=line("increment","select(cond = "+above+", a = fp16(1), b = fp16(0))");
        const auto tie_increment=line("tie_increment","select(cond = "+tie+", a = "+parity+", b = fp16(0))");
        const auto first=line("round_first","add(x = "+lower+", y = "+increment+")");
        const auto integer=line("integer","add(x = "+first+", y = "+tie_increment+")");
        restored=line("rounded","mul(x = "+integer+", y = fp16(0x1p-7))");
    }
    for(int i=4;i>=0;--i)
        restored=line("restore_"+std::to_string(i),"mul(x = "+restored+", y = "+inverses[i]+")");
    const auto negated=line("negative_value","mul(x = "+restored+", y = fp16(-1))");
    const auto negative=line("negative","less(x = "+input+", y = fp16(0))",true);
    const auto signed_value=line("signed","select(cond = "+negative+", a = "+negated+", b = "+restored+")");
    const auto tiny=line("tiny","less(x = "+magnitude+", y = fp16(0x1p-16))",true);
    return line("out","select(cond = "+tiny+", a = "+input+", b = "+signed_value+")");
}
} // namespace tc::ane::private_api
