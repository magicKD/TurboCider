#pragma once
#import <Foundation/Foundation.h>
#include "../../runtime/session.hpp"
#include "../../core/tokenizer.hpp"
namespace tc {
std::string json(id);
NSDictionary *parse_json(const char *);
NSDictionary *read_json(const std::filesystem::path &);
std::string string_value(NSDictionary *, NSString *, const std::string &fallback = "");
Request request_from_json(NSDictionary *);
void resolve_profile(Request &);
NSDictionary *to_dictionary(const ExecutionPlan &);
NSDictionary *to_dictionary(const LoadResult &);
NSDictionary *to_dictionary(const RunResult &);
NSDictionary *to_dictionary(const HybridMetrics &);
NSDictionary *to_dictionary(const ModelDescriptor &);
NSDictionary *to_dictionary(const std::vector<ModelDescriptor> &);
NSDictionary *system_info();
NSDictionary *compile_artifact(const std::filesystem::path &, const std::filesystem::path &);
NSDictionary *coreml_resources(NSDictionary *, const Event &, std::atomic<bool> &);
NSDictionary *manage_coreml_cache(NSDictionary *, const Event &, std::atomic<bool> &);
} // namespace tc
