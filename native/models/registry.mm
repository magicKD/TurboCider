#include "runtime.hpp"
namespace tc {
ModelModule flux4_module();
ModelModule flux9_module();
ModelModule fastmetal_module();
ModelModule h3_module();
ModelModule ltx_module();
static const std::vector<ModelModule>& modules() {
    static const std::vector<ModelModule> all = {flux4_module(), flux9_module(), fastmetal_module(), h3_module(), ltx_module()};
    return all;
}
const ModelModule& module_for(const std::string& id) {
    for (const auto& module : modules()) if (module.id == id) return module;
    throw std::invalid_argument("unknown model module: " + id);
}
NSDictionary *describe_modules() {
    NSMutableArray *models = [NSMutableArray array];
    for (const auto& module : modules()) [models addObject:module.describe()];
    return @{@"schema_version":@2, @"models":models};
}
Recipe model_recipe(const std::string& id) { return module_for(id).recipe(); }
}
