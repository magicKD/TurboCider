#ifndef TURBOCIDER_NATIVE_H
#define TURBOCIDER_NATIVE_H
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* ABI 1. Returned strings belong to the caller; release with tc_string_free.
 * An engine serializes generation. cancel is thread-safe. Free only after all
 * calls return. Events execute synchronously on the generation thread and the
 * supplied UTF-8 JSON is valid only for the duration of the callback.
 * No exceptions cross this boundary. Requests accept schema_version 1 or 2; results use schema_version=1.
 */
typedef struct tc_engine tc_engine;
typedef struct tc_coreml_ffn tc_coreml_ffn;
typedef void (*tc_event_callback)(const char *event_json, void *context);
/* Prepare current prompt/shape/backend; warmup executes without exporting. */
int tc_engine_prepare(tc_engine *,const char *request_json,int warmup,tc_event_callback,void *,char **result,char **error);
/* Managed Core ML cache actions: inspect, compile_manifest, clear. */
int tc_engine_cache(tc_engine *,const char *request_json,tc_event_callback,void *,char **result,char **error);
/* Offline resource inventory/export/compile/deletion preview. No model session needed.
 * Before applying deletion, callers must unload their idle model sessions. */
int tc_coreml_resources_json(const char *,tc_event_callback,void *,char **result,char **error);
void tc_coreml_resources_cancel(void);
uint32_t tc_abi_version(void);
char *tc_system_json(void);
char *tc_models_json(void);
int tc_plan_json(const char *request_json, char **plan_json, char **error);
int tc_engine_create(const char *model_path, tc_engine **engine, char **error);
/* Additive ABI: select a registered model module. Model paths are local only. */
int tc_engine_create_model(const char *model_id, const char *model_path,
                           tc_engine **engine, char **error);
int tc_engine_generate(tc_engine *, const char *request_json,
                       tc_event_callback callback, void *context,
                       char **result_json, char **error);
void tc_engine_cancel(tc_engine *);
/* Explicit image-weight preparation and resource release; idle engine only.
 * Loading does not perform inference or warm a prompt/shape. */
int tc_engine_load(tc_engine *, tc_event_callback, void *, char **result, char **error);
int tc_engine_unload(tc_engine *, char **result, char **error);
void tc_engine_free(tc_engine *);
void tc_string_free(char *);
int tc_compile_coreml_json(const char *source, const char *cache, char **result, char **error);
int tc_native_self_test(char **report_json, char **error);
int tc_tokenize_json(const char *model_path, const char *prompt, char **tokens, char **error);
int tc_ltx_gemma_tokenize_json(const char *tokenizer_json, const char *prompt,
                               uint32_t max_length, char **tokens, char **error);
int tc_ltx_gemma_inspect_json(const char *checkpoint, char **result, char **error);
int tc_ltx_lora_preflight_json(const char *model_path, const char *lora_path,
                               float strength, char **result, char **error);
int tc_fastmetal_lora_preflight_json(const char *model_path, const char *lora_path,
                                     float strength, char **result, char **error);
/* Low-level fixed-shape Core ML FFN bridge used by managed persistent workers.
 * Input/output are IEEE FP16 bit patterns in contiguous [1, rows, hidden]
 * row-major order. The manifest remains checkpoint/provenance verified. */
int tc_coreml_ffn_create(const char *manifest, const char *checkpoint,
                         int minimum_rows, int warmups,
                         tc_coreml_ffn **bridge, char **error);
int tc_coreml_ffn_predict(tc_coreml_ffn *bridge, int block,
                          const uint16_t *input, int rows,
                          uint16_t *output, char **error);
int tc_coreml_ffn_metrics_json(tc_coreml_ffn *bridge, char **result, char **error);
void tc_coreml_ffn_free(tc_coreml_ffn *bridge);
/* Read-only LTX Audio VAE/vocoder provenance check.  A successful response
 * does not enable native audio execution; it reports whether the required
 * artifact and manifest are present and verified. */
int tc_ltx_audio_preflight_json(const char *model_path, char **result,
                                char **error);
#ifdef __cplusplus
}
#endif
#endif
