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
    NSArray *image_presets = @[@"original", @"automatic", @"fit512", @"portrait512", @"landscape512"];
    NSDictionary *absolute_image_path = @{@"type": @"string", @"minLength": @2,
        @"pattern": @"^/", @"description": @"Absolute local path; embedded NUL is rejected."};
    NSDictionary *image_prepare_input = @{@"type": @"object", @"additionalProperties": @NO,
        @"required": @[@"source_path", @"preset"],
        @"properties": @{
            @"schema_version": @{@"type": @"integer", @"const": @1, @"default": @1},
            @"source_path": absolute_image_path,
            @"preset": @{@"type": @"string", @"enum": image_presets},
            @"output_path": @{@"type": @"string", @"minLength": @5, @"pattern": @"^/[\\s\\S]*\\.png$",
                @"description": @"Required only when downsizing is needed. Parent must exist; atomically creates a new lowercase .png file exclusively. No-op does not write this path."}
        }};
    NSDictionary *dimension = @{@"type": @"integer", @"minimum": @1, @"maximum": @80000000};
    NSDictionary *image_prepare_result = @{@"type": @"object", @"additionalProperties": @NO,
        @"required": @[@"schema_version", @"source_path", @"preset", @"image_path", @"output_created",
                       @"original_width", @"original_height", @"width", @"height", @"changed"],
        @"properties": @{
            @"schema_version": @{@"type": @"integer", @"const": @1}, @"source_path": absolute_image_path,
            @"preset": @{@"type": @"string", @"enum": image_presets}, @"image_path": absolute_image_path,
            @"output_created": @{@"type": @"boolean"}, @"changed": @{@"type": @"boolean"},
            @"original_width": dimension, @"original_height": dimension, @"width": dimension, @"height": dimension
        }};
    NSMutableDictionary *image_prepare = [action(@"image_prepare",
        @"Explicit CPU image downsizing; may create a PNG. Always use result.image_path. Never retry blindly after a lost reply.",
        true, @{@"input": image_prepare_input}, @[@"input"]) mutableCopy];
    image_prepare[@"result_schema"] = image_prepare_result;
    NSArray *actions = @[
        action(@"capabilities", @"Describe this protocol without loading weights.", false, @{}, @[]),
        action(@"models", @"List registered model capabilities; this is not an installed-model inventory.", false, @{}, @[]),
        action(@"workflows", @"Read the shared Qwen image-workflow catalog, ordered roles and prompt prefixes without weights or file access.", false, @{}, @[]),
        action(@"workflow_request", @"Compose a native request from a shared workflow. No plan, file access or submission; call plan separately before submit.", false,
               @{@"input": @{@"type": @"object", @"description": @"See workflows.input_schema: workflow_id, role_paths (absolute paths), instruction (optional), expansion (outpaint only), request (native schema 1 or 2 settings). Operation, prompt and inputs are owned by the workflow."}}, @[@"input"]),
        image_prepare,
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
        @"image_preparation": @{
            @"action": @"image_prepare", @"cli": @"prepare-image INPUT.json", @"schema_version": @1,
            @"scope": @"explicit CPU local-file preparation; no model loading or generation",
            @"presets": @{
                @"original": @"Return source unchanged.",
                @"automatic": @"Fit within 1024x1024 (long side at most 1024).",
                @"fit512": @"Fit within 512x512.",
                @"portrait512": @"Fit within 512x768.",
                @"landscape512": @"Fit within 768x512."
            },
            @"geometry": @"Preserve aspect ratio; never crop, pad or upscale.",
            @"max_source_pixels": @80000000, @"source_frames": @1, @"alpha": @"preserved",
            @"source_mutation": @"never", @"output_write": @"atomic exclusive creation only when downsizing is needed",
            @"no_op": @"image_path is source_path; output_created=false; no output file is written",
            @"result_path": @"Always use returned image_path; do not assume output_path was written.",
            @"dispatch": @"synchronous serial RPC dispatch; inference worker remains unlocked; no immediate cancellation guarantee",
            @"transport_retry": @"never_automatic; a lost reply may follow a successful file write"
        },
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
            @"catalog_action": @"workflows", @"construction_action": @"workflow_request",
            @"construction_scope": @"pure composition only; plan is authoritative for execution admission; files are not checked",
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
