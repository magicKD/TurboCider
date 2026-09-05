#include "runtime.hpp"
#include <cmath>
namespace tc {
void require(bool b,const std::string& s){if(!b)throw std::invalid_argument(s);}
std::string json(id object) {
    NSError *error=nil;
    NSData *data=[NSJSONSerialization dataWithJSONObject:object options:NSJSONWritingSortedKeys error:&error];
    if(!data)throw std::runtime_error(error.localizedDescription.UTF8String);
    return std::string((const char*)data.bytes,data.length);
}
static NSDictionary *dictionary(id value,const char *context) {
    require([value isKindOfClass:NSDictionary.class],std::string(context)+" must be an object");
    return value;
}
NSDictionary *parse_json(const char *s) {
    require(s,"missing JSON");
    NSData *data=[NSData dataWithBytes:s length:strlen(s)];
    return dictionary([NSJSONSerialization JSONObjectWithData:data options:0 error:nil],"JSON");
}
NSDictionary *read_json(const std::filesystem::path& path) {
    NSData *data=[NSData dataWithContentsOfFile:@(path.c_str())];
    require(data!=nil,"missing file: "+path.string());
    return dictionary([NSJSONSerialization JSONObjectWithData:data options:0 error:nil],path.c_str());
}
std::string string_value(NSDictionary*d,NSString*k,const std::string& fallback) {
    dictionary(d,"container");id v=d[k];if(!v)return fallback;
    require([v isKindOfClass:NSString.class],std::string(k.UTF8String)+" must be string");
    const char *s=[v UTF8String];require(s,"invalid UTF-8");
    require(strlen(s)==[v lengthOfBytesUsingEncoding:NSUTF8StringEncoding],"embedded NUL is not allowed");
    return s;
}
static double numeric(NSDictionary*d,NSString*k,double fallback) {
    id v=d[k];if(!v)return fallback;
    require([v isKindOfClass:NSNumber.class]&&CFGetTypeID((__bridge CFTypeRef)v)!=CFBooleanGetTypeID(),std::string(k.UTF8String)+" must be numeric");
    double x=[v doubleValue];require(std::isfinite(x),"nonfinite number");return x;
}
static int number(NSDictionary*d,NSString*k,int fallback) {
    double x=numeric(d,k,fallback);require(x==std::floor(x)&&x>=0&&x<=2147483647,"invalid integer");return int(x);
}
static bool boolean(NSDictionary*d,NSString*k,bool fallback) {
    id v=d[k];if(!v)return fallback;
    require(CFGetTypeID((__bridge CFTypeRef)v)==CFBooleanGetTypeID(),std::string(k.UTF8String)+" must be bool");return [v boolValue];
}
static void keys(NSDictionary*d,NSArray *allowed) {
    NSSet *set=[NSSet setWithArray:allowed];
    for(NSString *k in d)require([set containsObject:k],"unknown field: "+std::string(k.UTF8String));
}
Request request_from_json(NSDictionary*d) {
    int version=number(d,@"schema_version",1);require(version==1||version==2,"unsupported schema_version");
    Request r;
    if(version==1) {
        keys(d,@[@"schema_version",@"model",@"prompt",@"output",@"execution",@"width",@"height",@"steps",@"seed",@"frames",@"dynamic_text",@"dump_tensors",@"ane_manifest",@"allow_approximation",@"operation",@"inputs",@"fps",@"residency",@"profile"]);
        r.model=string_value(d,@"model",r.model);
        auto descriptor=module_for(r.model).describe();
        r.operation=string_value(d,@"operation",[descriptor[@"output"] isEqual:@"image"]?"image.generate":"video.generate");
        r.prompt=string_value(d,@"prompt");r.output=string_value(d,@"output");
        r.execution=string_value(d,@"execution","gpu");r.ane_manifest=string_value(d,@"ane_manifest");
        r.width=number(d,@"width",512);r.height=number(d,@"height",512);r.frames=number(d,@"frames",1);
        r.steps=number(d,@"steps",4);r.seed=number(d,@"seed",42);r.fps=number(d,@"fps",24);
        r.dynamic_text=boolean(d,@"dynamic_text",true);r.allow_approximation=boolean(d,@"allow_approximation",false);
        r.residency=string_value(d,@"residency",r.residency);r.profile=string_value(d,@"profile");
    } else {
        keys(d,@[@"schema_version",@"model",@"operation",@"inputs",@"outputs",@"sampling",@"execution",@"parameters",@"dump_tensors"]);
        r.model=string_value(d,@"model",r.model);
        auto descriptor=module_for(r.model).describe();
        r.operation=string_value(d,@"operation");require(!r.operation.empty(),"operation is required");
        NSArray *outputs=d[@"outputs"];require([outputs isKindOfClass:NSArray.class]&&outputs.count==1,"one primary output is required");
        auto output=dictionary(outputs[0],"output");keys(output,@[@"kind",@"path",@"width",@"height",@"frames",@"fps",@"audio"]);
        require([@(string_value(output,@"kind").c_str()) isEqual:descriptor[@"output"]],"output kind does not match model");
        r.output=string_value(output,@"path");r.width=number(output,@"width",[descriptor[@"default_width"] intValue]);
        r.height=number(output,@"height",[descriptor[@"default_height"] intValue]);r.frames=number(output,@"frames",[descriptor[@"default_frames"] intValue]);
        r.fps=number(output,@"fps",24);r.audio=boolean(output,@"audio",true);
        auto sampling=d[@"sampling"]?dictionary(d[@"sampling"],"sampling"):@{};
        keys(sampling,@[@"seed",@"steps"]);r.seed=number(sampling,@"seed",42);r.steps=number(sampling,@"steps",[descriptor[@"default_steps"] intValue]);
        auto execution=d[@"execution"]?dictionary(d[@"execution"],"execution"):@{};
        keys(execution,@[@"policy",@"profile",@"ane_manifest",@"allow_approximation",@"residency"]);
        r.execution=string_value(execution,@"policy","gpu");r.profile=string_value(execution,@"profile");
        r.ane_manifest=string_value(execution,@"ane_manifest");r.allow_approximation=boolean(execution,@"allow_approximation",false);
        r.residency=string_value(execution,@"residency",r.residency);
        auto parameters=d[@"parameters"]?dictionary(d[@"parameters"],"parameters"):@{};
        keys(parameters,@[@"dynamic_text"]);r.dynamic_text=boolean(parameters,@"dynamic_text",true);
    }
    r.dump=string_value(d,@"dump_tensors");
    if(d[@"inputs"]) {
        NSArray *inputs=d[@"inputs"];require([inputs isKindOfClass:NSArray.class]&&inputs.count<=32,"inputs must be an array of at most 32 assets");
        bool text_seen=!r.prompt.empty();
        for(id value in inputs) {
            auto input=dictionary(value,"input");keys(input,@[@"kind",@"role",@"path",@"text",@"audio_path",@"include_embedded_audio",@"strength"]);
            InputAsset a;a.kind=string_value(input,@"kind");a.role=string_value(input,@"role");
            if(a.kind=="text") {require(a.role=="prompt"&&!text_seen,"one prompt input is required");r.prompt=string_value(input,@"text");text_seen=true;continue;}
            a.path=string_value(input,@"path");require(!a.path.empty()&&!a.role.empty(),"media input requires path and role");
            a.audio_path=string_value(input,@"audio_path");a.include_audio=boolean(input,@"include_embedded_audio",true);
            a.strength=float(numeric(input,@"strength",a.role=="first_frame"?1.:.75));require(a.strength>=0&&a.strength<=1,"strength must be 0...1");
            require(a.kind=="image"||a.kind=="audio"||a.kind=="video","unsupported media kind");r.inputs.push_back(std::move(a));
        }
    }
    resolve_profile(r);
    return r;
}
void checkpoint(std::atomic<bool>& c){if(c.load())throw Cancelled();}
}
