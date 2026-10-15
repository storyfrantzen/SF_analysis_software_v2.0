#include "GeneratorProvenance.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <map>
#include <regex>
#include <set>
#include <stdexcept>
#include <tuple>
#include <vector>

#include "nlohmann/json.hpp"

namespace fs = std::filesystem;

namespace {

constexpr const char* SOURCE_MANIFEST_SCHEMA =
    "sf-generator-source-manifest-v1";

std::string normalizedPath(const fs::path& path) {
    std::error_code error;
    const auto canonical = fs::weakly_canonical(path, error);
    if (!error) return canonical.string();
    return fs::absolute(path).lexically_normal().string();
}

std::string resolvedManifestPath(const fs::path& manifestDirectory,
                                 const std::string& rawPath) {
    fs::path path(rawPath);
    if (path.is_relative()) path = manifestDirectory / path;
    return normalizedPath(path);
}

std::string requiredString(const nlohmann::json& value,
                           const char* key,
                           const std::string& context) {
    if (!value.contains(key) || !value.at(key).is_string()) {
        throw std::runtime_error(context + " requires string field " + key);
    }
    const auto result = value.at(key).get<std::string>();
    if (result.empty()) {
        throw std::runtime_error(context + " contains empty field " + key);
    }
    return result;
}

long long requiredNonnegativeInteger(const nlohmann::json& value,
                                     const char* key,
                                     const std::string& context) {
    if (!value.contains(key) || !value.at(key).is_number_integer()) {
        throw std::runtime_error(context + " requires integer field " + key);
    }
    const auto result = value.at(key).get<long long>();
    if (result < 0) {
        throw std::runtime_error(context + " contains negative field " + key);
    }
    return result;
}

double requiredFiniteNumber(const nlohmann::json& value,
                            const char* key,
                            const std::string& context) {
    if (!value.contains(key) || !value.at(key).is_number()) {
        throw std::runtime_error(context + " requires numeric field " + key);
    }
    const double result = value.at(key).get<double>();
    if (!std::isfinite(result)) {
        throw std::runtime_error(context + " contains non-finite field " + key);
    }
    return result;
}

bool isSha256(const std::string& value) {
    static const std::regex pattern(R"([0-9a-fA-F]{64})");
    return std::regex_match(value, pattern);
}

std::string extractCanonicalAaoChunkId(const std::string& filename) {
    static const std::regex pattern(R"(s[0-9]{5}__g[0-9]{4}__p[0-9]{6})");
    std::sregex_iterator match(filename.begin(), filename.end(), pattern);
    const std::sregex_iterator end;
    if (match == end) {
        throw std::runtime_error(
            "Filename does not contain a canonical AAO chunk ID "
            "(sNNNNN__gNNNN__pNNNNNN): " + filename
        );
    }
    const std::string identifier = match->str();
    ++match;
    if (match != end) {
        throw std::runtime_error(
            "Filename contains more than one canonical AAO chunk ID: " + filename
        );
    }
    return identifier;
}

bool sameRun(const GeneratorSourceMetadata& left,
             const GeneratorSourceMetadata& right) {
    return std::tie(left.lundPath,
                    left.configPath,
                    left.provider,
                    left.mode,
                    left.channel,
                    left.model,
                    left.generatorRevision,
                    left.generatorBranch,
                    left.dirtyState,
                    left.runIndex,
                    left.seed,
                    left.requestedEvents,
                    left.observedEvents,
                    left.sigmaValue,
                    left.sigmaUnits,
                    left.cardSignature,
                    left.configSha256,
                    left.executionPath,
                    left.beamEnergyGeV,
                    left.generationVcutGeV2,
                    left.tNuclGeV2) ==
           std::tie(right.lundPath,
                    right.configPath,
                    right.provider,
                    right.mode,
                    right.channel,
                    right.model,
                    right.generatorRevision,
                    right.generatorBranch,
                    right.dirtyState,
                    right.runIndex,
                    right.seed,
                    right.requestedEvents,
                    right.observedEvents,
                    right.sigmaValue,
                    right.sigmaUnits,
                    right.cardSignature,
                    right.configSha256,
                    right.executionPath,
                    right.beamEnergyGeV,
                    right.generationVcutGeV2,
                    right.tNuclGeV2);
}

GeneratorSourceMetadata parseSource(const nlohmann::json& value,
                                    const fs::path& manifestDirectory,
                                    std::size_t index) {
    const std::string context = "generator source record " + std::to_string(index);
    if (!value.is_object()) {
        throw std::runtime_error(context + " must be an object");
    }
    GeneratorSourceMetadata source;
    source.manifestSchema = SOURCE_MANIFEST_SCHEMA;
    source.hipoPath = resolvedManifestPath(
        manifestDirectory, requiredString(value, "hipo_path", context)
    );
    if (fs::path(source.hipoPath).extension() != ".hipo") {
        throw std::runtime_error(context + " hipo_path is not a .hipo file");
    }
    source.lundPath = resolvedManifestPath(
        manifestDirectory, requiredString(value, "lund_path", context)
    );
    source.configPath = resolvedManifestPath(
        manifestDirectory, requiredString(value, "config_path", context)
    );
    source.provider = requiredString(value, "provider", context);
    source.mode = requiredString(value, "mode", context);
    if (source.mode != "born" && source.mode != "rad") {
        throw std::runtime_error(context + " mode must be born or rad");
    }
    source.channel = requiredString(value, "channel", context);
    source.model = requiredString(value, "model", context);
    source.generatorRevision = requiredString(
        value, "generator_revision", context
    );
    source.generatorBranch = requiredString(value, "generator_branch", context);
    source.dirtyState = requiredString(value, "dirty_state", context);
    if (source.dirtyState != "clean" && source.dirtyState != "dirty") {
        throw std::runtime_error(context + " dirty_state must be clean or dirty");
    }
    source.runIndex = requiredNonnegativeInteger(value, "run_index", context);
    source.seed = requiredNonnegativeInteger(value, "seed", context);
    source.requestedEvents = requiredNonnegativeInteger(
        value, "requested_events", context
    );
    source.observedEvents = requiredNonnegativeInteger(
        value, "observed_events", context
    );
    if (source.requestedEvents <= 0) {
        throw std::runtime_error(context + " requested_events must be positive");
    }
    source.sigmaValue = requiredFiniteNumber(value, "sigma_value", context);
    if (source.sigmaValue <= 0.0) {
        throw std::runtime_error(context + " sigma_value must be positive");
    }
    source.sigmaUnits = requiredString(value, "sigma_units", context);
    if (source.sigmaUnits != "nb") {
        throw std::runtime_error(context + " sigma_units must be nb");
    }
    source.cardSignature = requiredString(value, "card_signature", context);
    source.configSha256 = requiredString(value, "config_sha256", context);
    if (!isSha256(source.cardSignature) || !isSha256(source.configSha256)) {
        throw std::runtime_error(
            context + " card_signature and config_sha256 must be SHA-256 hex"
        );
    }
    source.executionPath = requiredString(value, "execution_path", context);
    source.beamEnergyGeV = requiredFiniteNumber(
        value, "beam_energy_GeV", context
    );
    source.generationVcutGeV2 = requiredFiniteNumber(
        value, "generation_vcut_GeV2", context
    );
    source.tNuclGeV2 = requiredFiniteNumber(value, "t_nucl_GeV2", context);

    // Detector response and generated/reconstructed diagnostics count ordinary
    // EXCLUGEN events. Cross-section normalization remains in the source
    // catalog and is never converted into an event weight here.
    source.generatorWeight = 1.0;
    source.stratumFlatIndex = -1;
    return source;
}

}  // namespace

bool GeneratorSourceMetadata::acceptsRadiativeEvent(bool hardRadiative) const {
    if (mode == "born") return !hardRadiative;
    // EXCLUGEN full-RC files contain both the unresolved/soft component and
    // explicit hard-photon events.  The source mode therefore cannot require
    // every event to carry a hard photon.
    if (mode == "rad") return true;
    throw std::runtime_error("generator source mode is neither born nor rad");
}

GeneratorSourceCatalog GeneratorSourceCatalog::load(
    const std::string& manifestPath
) {
    std::ifstream input(manifestPath);
    if (!input.is_open()) {
        throw std::runtime_error(
            "Cannot open generator source manifest: " + manifestPath
        );
    }
    nlohmann::json manifest;
    input >> manifest;
    if (!manifest.is_object()) {
        throw std::runtime_error("generator source manifest must be an object");
    }
    if (manifest.value("schema", std::string()) != SOURCE_MANIFEST_SCHEMA) {
        throw std::runtime_error(
            "Unsupported generator source manifest schema in " + manifestPath
        );
    }
    if (!manifest.contains("sources") || !manifest.at("sources").is_array() ||
        manifest.at("sources").empty()) {
        throw std::runtime_error(
            "generator source manifest requires a non-empty sources array"
        );
    }

    GeneratorSourceCatalog catalog;
    catalog.manifestPath_ = normalizedPath(manifestPath);
    const fs::path manifestDirectory = fs::path(catalog.manifestPath_).parent_path();
    std::map<std::tuple<std::string, std::string, long long>, GeneratorSourceMetadata>
        runs;
    std::set<std::string> actualModes;
    std::size_t index = 0;
    for (const auto& value : manifest.at("sources")) {
        auto source = parseSource(value, manifestDirectory, index++);
        const auto identity = std::make_tuple(
            source.provider, source.mode, source.runIndex
        );
        const auto previous = runs.find(identity);
        if (previous != runs.end() && !sameRun(previous->second, source)) {
            throw std::runtime_error(
                "generator source manifest contains conflicting run metadata"
            );
        }
        runs[identity] = source;
        actualModes.insert(source.mode);
        if (!catalog.sources_.emplace(source.hipoPath, std::move(source)).second) {
            throw std::runtime_error(
                "generator source manifest contains duplicate HIPO paths"
            );
        }
    }
    if (manifest.contains("modes")) {
        if (!manifest.at("modes").is_array()) {
            throw std::runtime_error("generator source manifest modes must be an array");
        }
        const auto recorded = manifest.at("modes").get<std::vector<std::string>>();
        const std::vector<std::string> actual(actualModes.begin(), actualModes.end());
        if (recorded != actual) {
            throw std::runtime_error(
                "generator source manifest modes conflict with source records"
            );
        }
    }
    return catalog;
}

const GeneratorSourceMetadata& GeneratorSourceCatalog::at(
    const std::string& hipoPath
) const {
    const auto normalized = normalizedPath(hipoPath);
    const auto found = sources_.find(normalized);
    if (found == sources_.end()) {
        throw std::runtime_error(
            "HIPO input is absent from generator source manifest: " + normalized
        );
    }
    return found->second;
}

LegacyAaoChunkCatalog LegacyAaoChunkCatalog::load(
    const std::string& manifestPath
) {
    std::ifstream input(manifestPath);
    if (!input.is_open()) {
        throw std::runtime_error(
            "Cannot open generator chunk provenance: " + manifestPath
        );
    }
    nlohmann::json manifest;
    input >> manifest;
    if (!manifest.is_object() ||
        manifest.value("schema", std::string()) != "aao-osg-stratum-chunks-v1") {
        throw std::runtime_error(
            "Unsupported generator chunk provenance schema in " + manifestPath
        );
    }
    if (!manifest.contains("chunks") || !manifest.at("chunks").is_array()) {
        throw std::runtime_error(
            "AAO generator chunk provenance requires a chunks array"
        );
    }

    LegacyAaoChunkCatalog catalog;
    catalog.manifestPath_ = normalizedPath(manifestPath);
    const fs::path manifestDirectory = fs::path(catalog.manifestPath_).parent_path();
    for (const auto& chunk : manifest.at("chunks")) {
        const std::string chunkFile = requiredString(
            chunk, "chunk_file", "AAO generator chunk"
        );
        const std::string chunkId = extractCanonicalAaoChunkId(chunkFile);
        LegacyAaoChunkMetadata metadata;
        metadata.stratumFlatIndex = static_cast<int>(requiredNonnegativeInteger(
            chunk, "flat_index", "AAO generator chunk"
        ));
        metadata.generatorWeight = requiredFiniteNumber(
            chunk, "pooled_event_weight_microbarn", "AAO generator chunk"
        );
        if (metadata.generatorWeight <= 0.0) {
            throw std::runtime_error(
                "Invalid generator weight for " + chunkFile
            );
        }
        const std::string sourceLund = chunk.value("source_lund_file", std::string());
        if (!sourceLund.empty()) {
            metadata.sourceLundPath = resolvedManifestPath(
                manifestDirectory, sourceLund
            );
        }
        if (!catalog.chunks_.emplace(chunkId, std::move(metadata)).second) {
            throw std::runtime_error(
                "Duplicate canonical generator chunk ID in provenance: " + chunkId
            );
        }
    }
    return catalog;
}

const LegacyAaoChunkMetadata& LegacyAaoChunkCatalog::atHipoPath(
    const std::string& hipoPath
) const {
    const std::string chunkId = extractCanonicalAaoChunkId(
        fs::path(hipoPath).filename().string()
    );
    const auto found = chunks_.find(chunkId);
    if (found == chunks_.end()) {
        throw std::runtime_error(
            "HIPO canonical chunk ID is absent from generator chunk provenance: " +
            chunkId
        );
    }
    return found->second;
}
