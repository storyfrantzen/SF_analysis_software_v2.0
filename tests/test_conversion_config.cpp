#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include "Config.h"

namespace {
void writeConfig(const std::filesystem::path& path,
                 int torus,
                 long long maxEvents,
                 double electronPMin = 2.0) {
    std::ofstream output(path);
    output << "{\"beamEnergy\":10.604,\"torus\":" << torus
           << ",\"maxEvents\":" << maxEvents
           << ",\"electron_p_min\":" << electronPMin << "}";
}
}

int main() {
    const auto validPath = std::filesystem::current_path() / "test_config_torus_valid.json";
    const auto invalidPath = std::filesystem::current_path() / "test_config_torus_invalid.json";
    const auto legacyPath = std::filesystem::current_path() / "test_config_legacy_y.json";
    const auto elasticConfigPath =
        std::filesystem::current_path() / "test_config_elastic_momentum.json";
    const auto elasticParamsPath =
        std::filesystem::current_path() / "test_elastic_momentum_params.json";
    const auto neutralInclusivePath =
        std::filesystem::current_path() / "test_config_neutral_inclusive.json";
    const auto invalidNeutralInclusivePath =
        std::filesystem::current_path() / "test_config_invalid_neutral_inclusive.json";

    writeConfig(validPath, -1, 50000000);
    const Config valid(validPath.string());
    std::filesystem::remove(validPath);
    if (valid.torus != -1 || valid.maxEvents != 50000000 || valid.electronP_min != 2.0) {
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

    {
        std::ofstream output(legacyPath);
        output << "{\"beamEnergy\":10.604,\"y_max\":0.8}";
    }
    rejected = false;
    try {
        const Config invalid(legacyPath.string());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    std::filesystem::remove(legacyPath);
    if (!rejected) {
        std::cerr << "conversion config accepted the removed y_max cut\n";
        return 1;
    }

    {
        std::ofstream params(elasticParamsPath);
        params << "{\"schema\":\"elastic_momentum_correction/v1\","
               << "\"correctionType\":\"fractionalMomentum\","
               << "\"beamEnergyGeV\":10.604,\"torus\":-1,\"regions\":[]}";
    }
    {
        std::ofstream output(elasticConfigPath);
        output << "{\"elasticMomentumCorrections\":\""
               << elasticParamsPath.filename().string() << "\"}";
    }
    const Config elastic(elasticConfigPath.string());
    std::filesystem::remove(elasticConfigPath);
    std::filesystem::remove(elasticParamsPath);
    if (elastic.elasticMomentumCorrections.value("schema", std::string{}) !=
        "elastic_momentum_correction/v1") {
        std::cerr << "conversion config did not resolve elastic momentum parameters\n";
        return 1;
    }

    {
        std::ofstream output(neutralInclusivePath);
        output << "{\"inclusive\":false,"
               << "\"allowAdditionalNeutralParticles\":true}";
    }
    const Config neutralInclusive(neutralInclusivePath.string());
    std::filesystem::remove(neutralInclusivePath);
    if (neutralInclusive.inclusive ||
        !neutralInclusive.allowAdditionalNeutralParticles) {
        std::cerr << "conversion config did not retain neutral-only topology policy\n";
        return 1;
    }

    {
        std::ofstream output(invalidNeutralInclusivePath);
        output << "{\"inclusive\":true,"
               << "\"allowAdditionalNeutralParticles\":true}";
    }
    rejected = false;
    try {
        const Config invalid(invalidNeutralInclusivePath.string());
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    std::filesystem::remove(invalidNeutralInclusivePath);
    if (!rejected) {
        std::cerr << "conversion config accepted an ambiguous additional-particle policy\n";
        return 1;
    }
    return 0;
}
