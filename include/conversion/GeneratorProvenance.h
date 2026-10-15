#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>

struct GeneratorSourceMetadata {
    std::string manifestSchema;
    std::string hipoPath;
    std::string lundPath;
    std::string configPath;
    std::string provider;
    std::string mode;
    std::string channel;
    std::string model;
    std::string generatorRevision;
    std::string generatorBranch;
    std::string dirtyState;
    long long runIndex = -1;
    long long seed = -1;
    long long requestedEvents = -1;
    long long observedEvents = -1;
    double sigmaValue = 0.0;
    std::string sigmaUnits;
    std::string cardSignature;
    std::string configSha256;
    std::string executionPath;
    double beamEnergyGeV = 0.0;
    double generationVcutGeV2 = 0.0;
    double tNuclGeV2 = 0.0;
    int stratumFlatIndex = -1;
    double generatorWeight = 1.0;

    bool acceptsRadiativeEvent(bool hardRadiative) const;
};

class GeneratorSourceCatalog {
public:
    static GeneratorSourceCatalog load(const std::string& manifestPath);

    const GeneratorSourceMetadata& at(const std::string& hipoPath) const;
    std::size_t size() const { return sources_.size(); }
    const std::string& manifestPath() const { return manifestPath_; }

private:
    std::string manifestPath_;
    std::unordered_map<std::string, GeneratorSourceMetadata> sources_;
};

struct LegacyAaoChunkMetadata {
    std::string sourceLundPath;
    int stratumFlatIndex = -1;
    double generatorWeight = 1.0;
};

class LegacyAaoChunkCatalog {
public:
    static LegacyAaoChunkCatalog load(const std::string& manifestPath);

    const LegacyAaoChunkMetadata& atHipoPath(const std::string& hipoPath) const;
    std::size_t size() const { return chunks_.size(); }
    const std::string& manifestPath() const { return manifestPath_; }

private:
    std::string manifestPath_;
    std::unordered_map<std::string, LegacyAaoChunkMetadata> chunks_;
};
