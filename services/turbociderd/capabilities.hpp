#pragma once
#import <Foundation/Foundation.h>

// Discovery describes the RPC envelope. Native request admission stays in
// tc_plan_json, so this does not publish a second, drifting model validator.
inline NSDictionary *tc_service_capabilities() {
    auto action = [](NSString *name, NSString *summary, bool mutates,
                     NSDictionary *fields, NSArray *required) -> NSDictionary * {
        NSMutableDictionary *properties = [fields mutableCopy];
        properties[@"action"] = @{@"const": name};
        NSMutableArray *keys = [NSMutableArray arrayWithObject:@"action"];
        [keys addObjectsFromArray:required];
        return @{@"name": name, @"description": summary, @"mutates": @(mutates),
                 @"input_schema": @{@"type": @"object", @"properties": properties,
                     @"required": keys, @"additionalProperties": @NO}};
    };
    NSDictionary *string = @{@"type": @"string", @"minLength": @1};
    NSDictionary *request = @{@"type": @"object", @"description":
        @"Native request schema 1 or 2. Use models for model capabilities and plan for authoritative validation before submit."};
    NSArray *actions = @[
        action(@"capabilities", @"Describe this protocol without loading weights.", false, @{}, @[]),
        action(@"models", @"List registered model capabilities; this is not an installed-model inventory.", false, @{}, @[]),
        action(@"installations", @"Read the service's registered model, LoRA and ANE paths without loading weights or verifying files.", false, @{}, @[]),
        action(@"service_status", @"Read service ownership, active job and session state.", false, @{}, @[]),
        action(@"doctor", @"Read local runtime and device diagnostics.", false, @{}, @[]),
        action(@"plan", @"Validate and resolve a native request without generation; weight compatibility is checked at load time.", false,
               @{@"request": request}, @[@"request"]),
        action(@"submit", @"Persist and enqueue one request. Do not retry automatically after a transport failure: it may already have been accepted.", true,
               @{@"request": request, @"model_path": string}, @[@"request", @"model_path"]),
        action(@"status", @"Read a job, including progress, terminal result or error.", false,
               @{@"id": string}, @[@"id"]),
        action(@"jobs", @"Read newest-first job history. Pages are live, not a snapshot.", false,
               @{@"offset": @{@"type": @"integer", @"minimum": @0, @"maximum": @2147483647, @"default": @0},
                 @"limit": @{@"type": @"integer", @"minimum": @1, @"maximum": @100, @"default": @20}}, @[]),
        action(@"cancel", @"Request cancellation at a safe boundary; poll status until terminal. Terminal jobs remain unchanged.", true,
               @{@"id": string}, @[@"id"])
    ];
    return @{
        @"protocol": @"turbocider.local", @"protocol_version": @1,
        @"transport": @{@"kind": @"unix_socket", @"scope": @"current_user",
                        @"framing": @"one_newline_terminated_json_object_per_connection",
                        @"max_request_bytes": @1048576},
        @"actions": actions, @"native_request_schema_versions": @[@1, @2],
        @"native_request_validation": @"plan",
        @"response_schema": @{@"oneOf": @[
            @{@"type": @"object", @"properties": @{@"ok": @{@"const": @YES}, @"result": @{}},
              @"required": @[@"ok", @"result"], @"additionalProperties": @NO},
            @{@"type": @"object", @"properties": @{@"ok": @{@"const": @NO}, @"error": @{@"type": @"string"}},
              @"required": @[@"ok", @"error"], @"additionalProperties": @NO}
        ]},
        @"job_states": @{@"active": @[@"queued", @"running", @"cancelling"],
                        @"terminal": @[@"succeeded", @"failed", @"cancelled", @"interrupted"]},
        @"limits": @{@"pending_jobs": @32, @"history_records": @10000},
        @"installation_discovery": @{
            @"action": @"installations", @"schema_version": @1,
            @"scope": @"registered_metadata", @"files_verified": @NO,
            @"configuration": @"service environment and model-library settings",
            @"helper": @"matching sibling turbocider-library",
            @"deadline_seconds": @10, @"max_index_bytes": @4194304,
            @"max_settings_bytes": @1048576, @"max_helper_stdout_bytes": @8388608,
            @"max_helper_stderr_bytes": @65536,
            @"missing_registry": @"empty index; no directories created",
            @"invalid_registry": @"error; no fallback scan",
            @"dispatch": @"serial RPC dispatch; inference worker remains unlocked"
        },
        @"workflow": @{
            @"execution": @"client_orchestrated_sequential_jobs",
            @"reference_order": @"inputs array order determines image numbering",
            @"output_paths": @"Use a distinct absolute output path for each job. Submit the next stage only after succeeded.",
            @"file_access": @"Inputs, models, LoRA and outputs are paths on this Mac; files are not uploaded by this protocol.",
            @"submission_retry": @"never_automatic",
            @"app_history": @"API jobs are stored separately from the creation page.",
            @"app_lifecycle": @"An App-owned service stops with the App. Run CLI serve for an independent lifecycle."
        },
        @"generation_result_metadata": @{
            @"service_execution_path": @"native_session, resident_session or disposable_worker; describes service process routing, not physical GPU/ANE placement.",
            @"service_session_reused": @"True when this request reused the existing service-owned engine for the same model and path. Applies to all native model sessions; does not guarantee resident weights or a prompt-cache hit. Disposable workers report false."
        }
    };
}
