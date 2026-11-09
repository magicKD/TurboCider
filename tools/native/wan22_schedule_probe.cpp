#include "../../native/models/wan22/schedule.hpp"

#include <iomanip>
#include <iostream>

int main() {
    std::cout << std::setprecision(10);
    for (auto variant : {tc::wan22::Distillation::fastvideo_3, tc::wan22::Distillation::quanhaol_4}) {
        auto sigmas = tc::wan22::denoising_sigmas(variant);
        for (size_t step = 0; step < sigmas.size(); ++step)
            std::cout << sigmas[step] << ' ' << tc::wan22::transformer_timestep(variant, int(step)) << '\n';
    }
}
