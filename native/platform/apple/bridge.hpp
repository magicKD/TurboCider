#pragma once
#import <Foundation/Foundation.h>
#include "../../runtime/session.hpp"
#include "../../core/tokenizer.hpp"
namespace tc {
struct MemoryExecutionReport;
std::string json(id);
NSDictionary *parse_json(const char *);
NSDictionary *read_json(const std::filesystem::path &);
NSDictionary *read_config_json(const std::filesystem::path &);
std::string string_value(NSDictionary *, NSString *, const std::string &fallback = "");
Request request_from_json(NSDictionary *);
void parse_memory_constrained(NSDictionary *, MemoryConstrainedConfig &);
void parse_streaming_config(NSDictionary *, StreamingConfig &, const char *origin);
void parse_streaming_input(NSDictionary *, StreamingConfig &,
                           std::optional<StreamingSelector> &, const char *origin);
NSDictionary *streaming_config_dictionary(const StreamingConfig &);
NSDictionary *streaming_selector_dictionary(const StreamingSelector &);
void resolve_profile(Request &);
NSDictionary *to_dictionary(const ExecutionPlan &);
NSDictionary *to_dictionary(const LoadResult &);
NSDictionary *to_dictionary(const RunResult &);
NSDictionary *to_dictionary(const MemoryExecutionReport &);
RunResult native_run_result(NSDictionary *, const Request &, const ExecutionPlan &);
NSDictionary *to_dictionary(const HybridMetrics &);
NSDictionary *to_dictionary(const ModelDescriptor &);
NSDictionary *to_dictionary(const std::vector<ModelDescriptor> &);
NSDictionary *system_info();
NSDictionary *compile_artifact(const std::filesystem::path &, const std::filesystem::path &);
NSDictionary *coreml_resources(NSDictionary *, const Event &, std::atomic<bool> &);
NSDictionary *manage_coreml_cache(NSDictionary *, const Event &, std::atomic<bool> &);
NSDictionary *preflight_ltx_lora(const std::filesystem::path &, const LoRAAsset &);
NSDictionary *preflight_wan_lora(const std::filesystem::path &, const LoRAAsset &);
NSDictionary *preflight_ltx_audio(const std::filesystem::path &);
std::string validate_ltx_ane_profile(const std::filesystem::path &,
                                     uint32_t, uint32_t, uint32_t, uint32_t);
std::string validate_wan_ane_manifest(const std::filesystem::path &);
#ifdef TURBOCIDER_ENABLE_TEST_HOOKS
size_t ltx_exact_process_quarantine_count_for_test() noexcept;
bool ltx_exact_retry_process_quarantine_for_test(std::string &error) noexcept;
#endif
} // namespace tc
