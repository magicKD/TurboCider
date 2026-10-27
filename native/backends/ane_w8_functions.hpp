#pragma once
#include "ane_runtime.hpp"
#include <algorithm>
#include <charconv>
#include <cstdlib>

namespace tc::ane {
inline std::vector<int> parse_w8_function_rows(const char *raw) {
    if(!raw || !*raw || std::string_view(raw)=="0")return {};
    const std::string_view text(raw);
    if(text.size()>128)throw CapabilityError("W8 function row list exceeds bound");
    std::vector<int> rows;size_t first=0;
    while(first<text.size()) {
        const auto end=text.find(',',first),last=end==std::string_view::npos?text.size():end;
        int value=0;const auto parsed=std::from_chars(text.data()+first,text.data()+last,value);
        if(parsed.ec!=std::errc{} || parsed.ptr!=text.data()+last || first==last || rows.size()>=16 ||
            value<32 || value>4224 || value%32 || (!rows.empty() && value<=rows.back()))
            throw CapabilityError("W8 function rows require2...16 increasing aligned32 integers");
        rows.push_back(value);
        if(end==std::string_view::npos)break;
        first=end+1;if(first==text.size())throw CapabilityError("W8 function row list has an empty tail");
    }
    if(rows.size()<2)throw CapabilityError("W8 function list needs at least two rows");
    for(int value:rows)if(value%rows.front())throw CapabilityError("W8 function rows must share the minimum alignment");
    return rows;
}
inline std::vector<int> configured_w8_function_rows() {
    return parse_w8_function_rows(std::getenv("TURBOCIDER_PRIVATE_ANE_FUNCTION_ROWS"));
}
inline std::string w8_function_identity(std::span<const int> rows) {
    if(rows.empty())return {};
    std::string result=":w8-functions-v1";for(int value:rows)result+="-"+std::to_string(value);return result;
}
inline std::vector<int> checked_w8_function_rows(const GraphShape &shape,std::span<const int> rows) {
    if(shape.kind!=Kind::SwiGLU || shape.lora_inputs || rows.size()<2 || rows.size()>16 ||
       rows.front()<32 || rows.front()%32 || rows.back()!=shape.rows || shape.rows>4224)
        throw CapabilityError("W8 function family requires2...16 fixed base SwiGLU rows, aligned32 and ending at physical maximum");
    int previous=0;
    for(int value:rows) {
        if(value<=previous || value%rows.front())
            throw CapabilityError("W8 function rows must increase strictly and be multiples of the smallest bucket");
        previous=value;
    }
    return {rows.begin(),rows.end()};
}
inline int w8_function_chunk(int remaining,std::span<const int> rows) {
    if(rows.empty() || remaining<=0 || remaining%rows.front())
        throw CapabilityError("W8 function launch rows require smallest-bucket alignment");
    auto p=std::upper_bound(rows.begin(),rows.end(),remaining);
    if(p==rows.begin())throw CapabilityError("W8 function family has no legal chunk");
    return *--p;
}
inline std::string w8_function_name(int rows) {return "ffn"+std::to_string(rows);}
} // namespace tc::ane
