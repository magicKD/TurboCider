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
typedef void (*tc_event_callback)(const char *event_json, void *context);
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
void tc_engine_free(tc_engine *);
void tc_string_free(char *);
int tc_compile_coreml_json(const char *source, const char *cache, char **result, char **error);
int tc_native_self_test(char **report_json, char **error);
int tc_tokenize_json(const char *model_path, const char *prompt, char **tokens, char **error);
#ifdef __cplusplus
}
#endif
#endif
