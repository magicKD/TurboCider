#pragma once

#include <cstdlib>
#include <iostream>
#include <string>

namespace tc::test {

template <class Exception>
inline constexpr auto rejects = [](auto &&function, const char *part) {
    try {
        function();
    } catch (const Exception &error) {
        if (std::string(error.what()).find(part) != std::string::npos)
            return;
        std::cerr << error.what() << " expected " << part << '\n';
        std::abort();
    }
    std::cerr << "expected rejection containing " << part << '\n';
    std::abort();
};

} // namespace tc::test
