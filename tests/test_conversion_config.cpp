#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Config.h"

namespace {
void writeConfig(const std::filesystem::path& path, int torus, long long maxEvents) {
    std::ofstream output(path);
    output << "{\"beamEnergy\":10.604,\"torus\":" << torus
           << ",\"maxEvents\":" << maxEvents << "}";
}
}

int main() {
    const auto validPath = std::filesystem::current_path() / "test_config_torus_valid.json";
    const auto invalidPath = std::filesystem::current_path() / "test_config_torus_invalid.json";

    writeConfig(validPath, -1, 50000000);
    const Config valid(validPath.string());
    std::filesystem::remove(validPath);
    if (valid.torus != -1 || valid.maxEvents != 50000000) {
        std::cerr << "conversion config did not retain torus/event limits\n";
        return 1;
    }

    writeConfig(invalidPath, -1, 0);
    bool rejected = false;
    try {
        const Config invalid(invalidPath.string());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    std::filesystem::remove(invalidPath);
    if (!rejected) {
        std::cerr << "conversion config accepted an invalid event limit\n";
        return 1;
    }
    return 0;
}
