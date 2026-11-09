#include "../../native/models/ltx098/scheduler.hpp"

#include <iostream>

int main(int argc, char **argv) {
    try {
        tc::require(argc == 6, "usage: ltx098-scheduler-probe INPUT OUTPUT STRENGTH SIGMA NEXT_SIGMA");
        auto values = tc::mx::load_safetensors(argv[1]).first;
        const float strength = std::stof(argv[3]), sigma = std::stof(argv[4]), next = std::stof(argv[5]);
        auto model_input = tc::ltx098::conditioned_input(values.at("state"), values.at("first"), strength);
        auto output = tc::ltx098::euler_step(values.at("state"), model_input, values.at("velocity"),
                                             values.at("first"), strength, sigma, next);
        auto original_input = tc::ltx098::original_condition(values.at("state"), values.at("first"), strength);
        auto original_output = tc::ltx098::original_euler_step(values.at("state"), values.at("velocity"), strength, sigma, next);
        auto time_strength = tc::Tensor(tc::ltx098::original_first_frame_strength(strength, sigma), tc::mx::float32);
        tc::mx::eval(model_input, output, original_input, original_output, time_strength);
        tc::mx::save_safetensors(argv[2], {{"model_input", model_input}, {"output", output},
                                          {"original_input", original_input}, {"original_output", original_output},
                                          {"original_time_strength", time_strength}});
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
