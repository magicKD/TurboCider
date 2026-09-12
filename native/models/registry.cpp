#include "../runtime/session.hpp"
namespace tc {
ModelModule flux_module();
ModelModule flux9_module();
ModelModule h3_module();
ModelModule h3_mlx_module();
ModelModule h3_mlx_vsa_module();
ModelModule ltx_module();
ModelModule wan_module();
ModelModule z_image_module();
ModelModule z_image_gguf_module();
ModelModule llada_module();
static const std::vector<ModelModule> &modules() {
    static const std::vector<ModelModule> all = {
        flux_module(), flux9_module(), h3_module(), h3_mlx_module(), h3_mlx_vsa_module(),
        ltx_module(), wan_module(),
        z_image_module(), z_image_gguf_module(), llada_module()
    };
    return all;
}
const ModelModule &module_for(const std::string &id) {
    for (const auto &module : modules())
        if (module.id == id)
            return module;
    throw std::invalid_argument("unknown model module: " + id);
}
std::vector<ModelDescriptor> describe_modules() {
    std::vector<ModelDescriptor> models;
    for (const auto &module : modules())
        models.push_back(module.describe());
    return models;
}
Recipe model_recipe(const std::string &id) {
    return module_for(id).recipe();
}
} // namespace tc
