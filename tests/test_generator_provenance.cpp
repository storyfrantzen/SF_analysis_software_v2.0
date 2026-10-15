#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>

#include "GeneratorProvenance.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;

namespace {

nlohmann::json sourceRecord(const fs::path& hipo,
                            const fs::path& lund,
                            const fs::path& config,
                            const std::string& model = "pi0.amp2614") {
    return {
        {"hipo_path", hipo.string()},
        {"lund_path", lund.string()},
        {"config_path", config.string()},
        {"provider", "exclugen"},
        {"mode", "born"},
        {"channel", "pi0p"},
        {"model", model},
        {"generator_revision", "50a5074da87718d5588d429b57468489aca75e6c"},
        {"generator_branch", "main"},
        {"dirty_state", "clean"},
        {"run_index", 1},
        {"seed", 1001},
        {"requested_events", 5000},
        {"observed_events", 5000},
        {"sigma_value", 0.9},
        {"sigma_units", "nb"},
        {"card_signature", std::string(64, 'a')},
        {"config_sha256", std::string(64, 'b')},
        {"execution_path", "farm"},
        {"beam_energy_GeV", 10.604},
        {"generation_vcut_GeV2", 0.0},
        {"t_nucl_GeV2", -0.3},
        {"physics_card", {{"ebeam", 10.604}}},
    };
}

void writeJson(const fs::path& path, const nlohmann::json& value) {
    std::ofstream output(path);
    output << value.dump(2) << '\n';
}

bool rejects(const fs::path& path, const std::string& message) {
    try {
        const auto ignored = GeneratorSourceCatalog::load(path.string());
        (void)ignored;
    } catch (const std::runtime_error& error) {
        return std::string(error.what()).find(message) != std::string::npos;
    }
    return false;
}

}  // namespace

int main() {
    const fs::path root = fs::current_path() / "test_generator_provenance_tmp";
    fs::remove_all(root);
    fs::create_directories(root);
    const fs::path hipo = root / "sample_born_001.hipo";
    const fs::path secondHipo = root / "sample_copy_born_001.hipo";
    const fs::path lund = root / "sample_born_001.lund";
    const fs::path config = root / "born_001.config.json";
    std::ofstream(hipo).put('\n');
    std::ofstream(secondHipo).put('\n');
    std::ofstream(lund).put('\n');
    std::ofstream(config) << "{}\n";

    const fs::path validPath = root / "valid.json";
    writeJson(
        validPath,
        {{"schema", "sf-generator-source-manifest-v1"},
         {"modes", {"born"}},
         {"sources", {sourceRecord(hipo, lund, config)}}}
    );
    const auto catalog = GeneratorSourceCatalog::load(validPath.string());
    const auto& source = catalog.at(hipo.string());
    if (catalog.size() != 1 || source.provider != "exclugen" ||
        source.mode != "born" || source.runIndex != 1 ||
        source.generatorWeight != 1.0 || source.stratumFlatIndex != -1 ||
        !source.acceptsRadiativeEvent(false) ||
        source.acceptsRadiativeEvent(true)) {
        std::cerr << "valid EXCLUGEN source manifest was not retained\n";
        fs::remove_all(root);
        return 1;
    }
    GeneratorSourceMetadata radiativeSource;
    radiativeSource.mode = "rad";
    if (!radiativeSource.acceptsRadiativeEvent(false) ||
        !radiativeSource.acceptsRadiativeEvent(true)) {
        std::cerr << "full-RC source did not accept both soft and hard events\n";
        fs::remove_all(root);
        return 1;
    }

    bool missingRejected = false;
    try {
        catalog.at((root / "missing.hipo").string());
    } catch (const std::runtime_error& error) {
        missingRejected = std::string(error.what()).find("absent") != std::string::npos;
    }
    if (!missingRejected) {
        std::cerr << "missing HIPO mapping was not rejected\n";
        fs::remove_all(root);
        return 1;
    }

    const fs::path unknownPath = root / "unknown.json";
    writeJson(unknownPath, {{"schema", "unknown"}, {"sources", nlohmann::json::array()}});
    if (!rejects(unknownPath, "Unsupported")) {
        std::cerr << "unknown source schema was not rejected\n";
        fs::remove_all(root);
        return 1;
    }

    const fs::path conflictPath = root / "conflict.json";
    writeJson(
        conflictPath,
        {{"schema", "sf-generator-source-manifest-v1"},
         {"modes", {"born"}},
         {"sources",
          {sourceRecord(hipo, lund, config),
           sourceRecord(secondHipo, lund, config, "pi0.wrong")}}}
    );
    if (!rejects(conflictPath, "conflicting")) {
        std::cerr << "conflicting run metadata was not rejected\n";
        fs::remove_all(root);
        return 1;
    }

    const fs::path aaoPath = root / "aao_chunks.json";
    writeJson(
        aaoPath,
        {{"schema", "aao-osg-stratum-chunks-v1"},
         {"chunks",
          {{{"chunk_file", "aao_born__s00012__g0003__p000017.lund"},
            {"source_lund_file", lund.string()},
            {"flat_index", 12},
            {"pooled_event_weight_microbarn", 0.0025}}}}}
    );
    const auto aao = LegacyAaoChunkCatalog::load(aaoPath.string());
    const auto& chunk = aao.atHipoPath(
        "portal-aao_born__s00012__g0003__p000017-42-0.hipo"
    );
    if (aao.size() != 1 || chunk.stratumFlatIndex != 12 ||
        chunk.generatorWeight != 0.0025 || chunk.sourceLundPath.empty()) {
        std::cerr << "legacy AAO chunk provenance was not retained\n";
        fs::remove_all(root);
        return 1;
    }

    fs::remove_all(root);
    return 0;
}
