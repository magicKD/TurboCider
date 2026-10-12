#include "image_workflows.hpp"
#include "../platform/apple/bridge.hpp"
#include "image_workflows_catalog_generated.hpp"
#include <cmath>

namespace tc {
namespace {
NSDictionary *object(id value, const char *name) {
    require([value isKindOfClass:NSDictionary.class], std::string(name) + " must be object");
    return value;
}
void only_keys(NSDictionary *value, NSArray *allowed) {
    for (NSString *key in value)
        require([allowed containsObject:key], "unknown workflow field: " + std::string(key.UTF8String));
}
NSString *text(NSDictionary *value, NSString *key, NSString *fallback = @"") {
    return @(string_value(value, key, fallback.UTF8String).c_str());
}
}

NSDictionary *image_workflows_catalog() {
    static NSDictionary *catalog = parse_json(image_workflow_catalog_json);
    return catalog;
}

NSDictionary *image_workflow_request(NSDictionary *input) {
    input = object(input, "workflow input");
    only_keys(input, @[@"workflow_id", @"role_paths", @"instruction", @"expansion", @"request"]);
    NSString *workflow_id = text(input, @"workflow_id");
    NSDictionary *definition = nil;
    for (NSDictionary *candidate in image_workflows_catalog()[@"workflows"])
        if ([candidate[@"id"] isEqual:workflow_id]) { definition = candidate; break; }
    require(definition != nil, "unknown workflow_id");
    NSDictionary *paths = input[@"role_paths"] ? object(input[@"role_paths"], "role_paths") : @{};
    NSArray *definitions = definition[@"roles"];
    NSMutableArray *allowed_roles = [NSMutableArray array];
    for (NSDictionary *role in definitions) [allowed_roles addObject:role[@"id"]];
    for (NSString *key in paths) {
        require([allowed_roles containsObject:key], "unknown workflow role: " + std::string(key.UTF8String));
        NSString *path = text(paths, key);
        require(path.length > 1 && [path hasPrefix:@"/"], "workflow role paths must be nonempty absolute local paths");
    }
    for (NSDictionary *role in definitions)
        require(![role[@"required"] boolValue] || paths[role[@"id"]],
                "missing workflow role: " + std::string([role[@"id"] UTF8String]));

    NSDictionary *mode = nil;
    for (NSDictionary *candidate in definition[@"modes"]) {
        bool matches = true;
        for (NSString *role in candidate[@"required_roles"]) if (!paths[role]) matches = false;
        for (NSString *role in candidate[@"absent_roles"]) if (paths[role]) matches = false;
        if (matches) { mode = candidate; break; }
    }
    require(mode != nil, "workflow role combination has no supported mode");
    NSString *prefix = mode[@"prompt_prefix"];
    if ([workflow_id isEqual:@"playground.outpaint"]) {
        NSNumber *value = input[@"expansion"] ?: definition[@"parameters"][@"expansion"][@"default"];
        require([value isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)value) != CFBooleanGetTypeID(),
                "expansion must be 1.25, 1.5 or 2");
        const double expansion = value.doubleValue;
        require(std::isfinite(expansion) && (expansion == 1.25 || expansion == 1.5 || expansion == 2),
                "expansion must be 1.25, 1.5 or 2");
        NSString *label = expansion == 1.25 ? @"1.25" : expansion == 1.5 ? @"1.5" : @"2";
        prefix = [prefix stringByReplacingOccurrencesOfString:@"{{expansion}}" withString:label];
    } else require(!input[@"expansion"], "expansion is supported only by playground.outpaint");
    NSString *instruction = text(input, @"instruction", mode[@"default_instruction"] ?: definition[@"default_instruction"]);
    NSString *prompt = [prefix stringByAppendingFormat:@"\n\nAdditional instruction:\n%@", instruction];

    NSMutableDictionary *request = [object(input[@"request"], "request") mutableCopy];
    id version = request[@"schema_version"] ?: @1;
    require([version isKindOfClass:NSNumber.class] && CFGetTypeID((__bridge CFTypeRef)version) != CFBooleanGetTypeID() &&
                ([version doubleValue] == 1 || [version doubleValue] == 2), "unsupported workflow native schema_version");
    NSString *model = text(request, @"model", image_workflows_catalog()[@"model"]);
    require([model isEqual:image_workflows_catalog()[@"model"]], "image workflows require qwen-image-2.1");
    request[@"schema_version"] = version; request[@"model"] = model;
    request[@"operation"] = mode[@"operation"];
    NSMutableArray *inputs = [NSMutableArray array], *roles = [NSMutableArray array];
    if ([version intValue] == 2) {
        [inputs addObject:@{@"kind": @"text", @"role": @"prompt", @"text": prompt}];
        [request removeObjectForKey:@"prompt"];
    } else request[@"prompt"] = prompt;
    for (NSDictionary *definition in definitions) {
        NSString *role = definition[@"id"], *path = paths[role];
        if (!path) continue;
        [roles addObject:@{@"role": role, @"image_number": @(roles.count + 1), @"path": path}];
        [inputs addObject:@{@"kind": @"image", @"role": @"reference", @"path": path}];
    }
    request[@"inputs"] = inputs;
    return @{@"schema_version": @1, @"workflow_id": workflow_id, @"roles": roles,
             @"operation": mode[@"operation"], @"prompt": prompt, @"request": request};
}
}
