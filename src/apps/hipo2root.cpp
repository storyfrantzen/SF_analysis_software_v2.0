#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <exception>
#include <fstream>
#include <iomanip>
#include <memory>
#include <optional>
#include <regex>
#include <sstream>
#include <unordered_map>

#include <fcntl.h>
#include <sys/wait.h>
#include <unistd.h>

#include "TFile.h"
#include "TParameter.h"
#include "TTree.h"
#include "TVector3.h"

#include "clas12reader.h"

#include "Config.h"
#include "GeneratedEvent.h"
#include "ProtonEnergyLossCorrections.h"
#include "QualityAssurance.h"
#include "Kinematics.h"
#include "ROOTBranches.h"
#include "core/TreeNames.h"

using namespace clas12;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct MatchResult {
    int genIdx = -1;
    double angleDeg = NAN;
};

struct HipoOpenCheck {
    bool ok = true;
    std::string detail;
};

namespace {

struct GeneratorFileMetadata {
    int stratumFlatIndex = -1;
    double weight = 1.0;
};

std::string pidBranchToken(int pid) {
    return pid < 0 ? "Minus" + std::to_string(-pid) : std::to_string(pid);
}

struct ReconstructedEventOutput {
    std::uint64_t sourceFileId = INVALID_SOURCE_ID;
    std::uint64_t sourceEventIndex = INVALID_SOURCE_ID;
    int runNum = -999;
    int eventNum = -999;
    int helicity = -999;
    double charge = NAN;
    int nReconstructedParticles = 0;
    int nWrittenReconstructedParticles = 0;
    int nParticleTreeRows = 0;
    std::vector<int> topologyPids;
    std::vector<int> topologyRequiredCounts;
    std::vector<int> topologyExact;
    std::vector<int> topologyPidCounts;
    std::vector<int> topologyPidCountsFT;
    std::vector<int> topologyPidCountsFD;
    std::vector<int> topologyPidCountsCD;
    std::vector<int> topologyPidCountsOther;

    explicit ReconstructedEventOutput(const std::vector<FinalState>& finalState) {
        for (const auto& requirement : finalState) {
            topologyPids.push_back(requirement.pid);
            topologyRequiredCounts.push_back(requirement.count);
            topologyExact.push_back(requirement.exact ? 1 : 0);
        }
        resetCounts();
    }

    void registerBranches(TTree& tree) {
        tree.Branch("sourceFileId", &sourceFileId, "sourceFileId/l");
        tree.Branch("sourceEventIndex", &sourceEventIndex, "sourceEventIndex/l");
        tree.Branch("runNum", &runNum, "runNum/I");
        tree.Branch("eventNum", &eventNum, "eventNum/I");
        tree.Branch("helicity", &helicity, "helicity/I");
        tree.Branch("charge", &charge, "charge/D");
        tree.Branch("nReconstructedParticles", &nReconstructedParticles,
                    "nReconstructedParticles/I");
        tree.Branch("nWrittenReconstructedParticles", &nWrittenReconstructedParticles,
                    "nWrittenReconstructedParticles/I");
        tree.Branch("nParticleTreeRows", &nParticleTreeRows, "nParticleTreeRows/I");
        tree.Branch("topologyPids", &topologyPids);
        tree.Branch("topologyRequiredCounts", &topologyRequiredCounts);
        tree.Branch("topologyExact", &topologyExact);
        tree.Branch("topologyPidCounts", &topologyPidCounts);
        tree.Branch("topologyPidCountsFT", &topologyPidCountsFT);
        tree.Branch("topologyPidCountsFD", &topologyPidCountsFD);
        tree.Branch("topologyPidCountsCD", &topologyPidCountsCD);
        tree.Branch("topologyPidCountsOther", &topologyPidCountsOther);
        for (size_t index = 0; index < topologyPids.size(); ++index) {
            const std::string base = "nPid" + pidBranchToken(topologyPids[index]);
            tree.Branch(base.c_str(), &topologyPidCounts[index], (base + "/I").c_str());
            tree.Branch((base + "FT").c_str(), &topologyPidCountsFT[index],
                        (base + "FT/I").c_str());
            tree.Branch((base + "FD").c_str(), &topologyPidCountsFD[index],
                        (base + "FD/I").c_str());
            tree.Branch((base + "CD").c_str(), &topologyPidCountsCD[index],
                        (base + "CD/I").c_str());
            tree.Branch((base + "Other").c_str(), &topologyPidCountsOther[index],
                        (base + "Other/I").c_str());
        }
    }

    void fill(const EventBranches& event,
              const std::vector<clas12::region_particle*>& particles,
              int writtenRecParticles,
              int particleTreeRows) {
        sourceFileId = event.sourceFileId;
        sourceEventIndex = event.sourceEventIndex;
        runNum = event.runNum;
        eventNum = event.eventNum;
        helicity = event.helicity;
        charge = event.charge;
        nReconstructedParticles = static_cast<int>(particles.size());
        nWrittenReconstructedParticles = writtenRecParticles;
        nParticleTreeRows = particleTreeRows;
        resetCounts();
        for (const auto* particle : particles) {
            if (!particle) continue;
            const int pid = particle->getPid();
            const auto found = std::find(topologyPids.begin(), topologyPids.end(), pid);
            if (found == topologyPids.end()) continue;
            const size_t index = static_cast<size_t>(found - topologyPids.begin());
            ++topologyPidCounts[index];
            switch (getDetector(particle->par()->getStatus())) {
                case 0: ++topologyPidCountsFT[index]; break;
                case 1: ++topologyPidCountsFD[index]; break;
                case 2: ++topologyPidCountsCD[index]; break;
                default: ++topologyPidCountsOther[index]; break;
            }
        }
    }

private:
    void resetCounts() {
        const size_t size = topologyPids.size();
        if (topologyPidCounts.size() != size) {
            topologyPidCounts.resize(size);
            topologyPidCountsFT.resize(size);
            topologyPidCountsFD.resize(size);
            topologyPidCountsCD.resize(size);
            topologyPidCountsOther.resize(size);
        }
        std::fill(topologyPidCounts.begin(), topologyPidCounts.end(), 0);
        std::fill(topologyPidCountsFT.begin(), topologyPidCountsFT.end(), 0);
        std::fill(topologyPidCountsFD.begin(), topologyPidCountsFD.end(), 0);
        std::fill(topologyPidCountsCD.begin(), topologyPidCountsCD.end(), 0);
        std::fill(topologyPidCountsOther.begin(), topologyPidCountsOther.end(), 0);
    }
};

std::string extractCanonicalChunkId(const std::string& filename) {
    static const std::regex pattern(
        R"(s[0-9]{5}__g[0-9]{4}__p[0-9]{6})"
    );
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

std::unordered_map<std::string, GeneratorFileMetadata>
loadGeneratorWeights(const std::string& provenancePath) {
    std::ifstream source(provenancePath);
    if (!source.is_open()) {
        throw std::runtime_error(
            "Cannot open generator chunk provenance: " + provenancePath
        );
    }
    nlohmann::json provenance;
    source >> provenance;
    if (provenance.value("schema", std::string()) != "aao-osg-stratum-chunks-v1") {
        throw std::runtime_error(
            "Unsupported generator chunk provenance schema in " + provenancePath
        );
    }
    std::unordered_map<std::string, GeneratorFileMetadata> weights;
    for (const auto& chunk : provenance.at("chunks")) {
        const std::string chunkFile = chunk.at("chunk_file").get<std::string>();
        const std::string chunkId = extractCanonicalChunkId(chunkFile);
        const GeneratorFileMetadata metadata{
            chunk.at("flat_index").get<int>(),
            chunk.at("pooled_event_weight_microbarn").get<double>()
        };
        if (!std::isfinite(metadata.weight) || metadata.weight <= 0.0) {
            throw std::runtime_error("Invalid generator weight for " + chunkFile);
        }
        if (!weights.emplace(chunkId, metadata).second) {
            throw std::runtime_error(
                "Duplicate canonical generator chunk ID in provenance: " + chunkId
            );
        }
    }
    return weights;
}

bool isRichDetectorSchemaWarning(const std::string& line) {
    return line.find("hipo::schema getEntryOrder") != std::string::npos &&
           line.find("item :detector not found") != std::string::npos &&
           line.find("bank RICH::Particle") != std::string::npos;
}

void replayWithoutRichDetectorSchemaWarning(FILE* capture, std::ostream& destination) {
    std::rewind(capture);
    char buffer[4096];
    bool skipFileLine = false;
    while (std::fgets(buffer, sizeof(buffer), capture) != nullptr) {
        const std::string line(buffer);
        if (isRichDetectorSchemaWarning(line)) {
            skipFileLine = true;
            continue;
        }
        if (skipFileLine && line.rfind("for file ", 0) == 0) {
            skipFileLine = false;
            continue;
        }
        skipFileLine = false;
        destination << line;
    }
    destination.flush();
}

// CLAS12ROOT constructs every known bank when a file is opened. Its RICH bank
// inherits a helper that probes for a nonexistent "detector" item, producing a
// harmless two-line warning for every file. Capture constructor diagnostics and
// remove only that exact warning; all other stdout/stderr output is replayed.
template <typename Callable>
void withoutRichDetectorSchemaWarning(Callable&& callable) {
    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);

    FILE* stdoutCapture = std::tmpfile();
    FILE* stderrCapture = std::tmpfile();
    const int savedStdout = dup(STDOUT_FILENO);
    const int savedStderr = dup(STDERR_FILENO);
    if (stdoutCapture == nullptr || stderrCapture == nullptr ||
        savedStdout < 0 || savedStderr < 0) {
        if (stdoutCapture != nullptr) std::fclose(stdoutCapture);
        if (stderrCapture != nullptr) std::fclose(stderrCapture);
        if (savedStdout >= 0) close(savedStdout);
        if (savedStderr >= 0) close(savedStderr);
        callable();
        return;
    }

    const bool redirected =
        dup2(fileno(stdoutCapture), STDOUT_FILENO) >= 0 &&
        dup2(fileno(stderrCapture), STDERR_FILENO) >= 0;
    if (!redirected) {
        dup2(savedStdout, STDOUT_FILENO);
        dup2(savedStderr, STDERR_FILENO);
        close(savedStdout);
        close(savedStderr);
        std::fclose(stdoutCapture);
        std::fclose(stderrCapture);
        callable();
        return;
    }

    std::exception_ptr failure;
    try {
        callable();
    } catch (...) {
        failure = std::current_exception();
    }

    std::cout.flush();
    std::cerr.flush();
    std::fflush(nullptr);
    dup2(savedStdout, STDOUT_FILENO);
    dup2(savedStderr, STDERR_FILENO);
    close(savedStdout);
    close(savedStderr);

    replayWithoutRichDetectorSchemaWarning(stdoutCapture, std::cout);
    replayWithoutRichDetectorSchemaWarning(stderrCapture, std::cerr);
    std::fclose(stdoutCapture);
    std::fclose(stderrCapture);

    if (failure) std::rethrow_exception(failure);
}

}  // namespace

std::uint64_t stableSourceFileId(const std::string& fileName) {
    // Stable FNV-1a hash; unlike std::hash this is reproducible across systems.
    std::uint64_t hash = 14695981039346656037ULL;
    for (const unsigned char character : fileName) {
        hash ^= character;
        hash *= 1099511628211ULL;
    }
    return hash;
}

bool isIntegerArgument(const std::string& value) {
    if (value.empty()) return false;
    std::size_t start = 0;
    if (value[0] == '-' || value[0] == '+') {
        if (value.size() == 1) return false;
        start = 1;
    }
    return std::all_of(value.begin() + static_cast<std::string::difference_type>(start),
                       value.end(),
                       [](unsigned char c) { return std::isdigit(c); });
}

bool pathExists(const std::string& value) {
    std::error_code error;
    return fs::exists(fs::path(value), error);
}

bool isTrailingNumericOption(const std::string& value) {
    return isIntegerArgument(value) && !pathExists(value);
}

HipoOpenCheck checkHipoOpenInChild(const std::string& hipoPath) {
    const pid_t pid = fork();
    if (pid < 0) {
        return {false, "fork failed"};
    }

    if (pid == 0) {
        const int devNull = open("/dev/null", O_WRONLY);
        if (devNull >= 0) {
            dup2(devNull, STDOUT_FILENO);
            dup2(devNull, STDERR_FILENO);
            close(devNull);
        }

        try {
            std::optional<clas12::clas12reader> reader;
            withoutRichDetectorSchemaWarning([&]() { reader.emplace(hipoPath); });
        } catch (...) {
            _exit(2);
        }
        _exit(0);
    }

    int status = 0;
    if (waitpid(pid, &status, 0) < 0) {
        return {false, "waitpid failed"};
    }
    if (WIFEXITED(status)) {
        const int exitCode = WEXITSTATUS(status);
        if (exitCode == 0) return {true, ""};
        return {false, "open-check child exited with code " + std::to_string(exitCode)};
    }
    if (WIFSIGNALED(status)) {
        return {false, "open-check child terminated by signal " +
                       std::to_string(WTERMSIG(status))};
    }
    return {false, "open-check child ended in an unknown state"};
}

void printProgress(std::size_t currentFile,
                   std::size_t totalFiles,
                   long long nTotal,
                   long long nWritten,
                   long long nOutputRows,
                   long long nQAFail,
                   long long nFSFail,
                   long long nSkimFail,
                   long long nMatched,
                   bool showMatches,
                   const Clock::time_point& startTime) {
    const double elapsed = std::chrono::duration<double>(Clock::now() - startTime).count();
    const double eventRate = elapsed > 0.0 ? static_cast<double>(nTotal) / elapsed : 0.0;

    std::cout << std::fixed << std::setprecision(1)
              << "[PROGRESS] files " << currentFile << "/" << totalFiles
              << "  events " << nTotal
              << "  accepted " << nWritten
              << "  rows " << nOutputRows
              << "  rejected(qa/fs/skim) " << nQAFail << "/" << nFSFail << "/" << nSkimFail;
    if (showMatches) std::cout << "  matched " << nMatched;
    std::cout << "  rate " << eventRate << " events/s"
              << "  elapsed " << elapsed << " s\n";
}

// ─── Final state filter ───────────────────────────────────────────────────────

bool passesFinalState(const Config& cfg, clas12::clas12reader& c12) {
    if (cfg.finalState.empty()) return true;

    for (const auto& s : cfg.finalState) {
        const int n = static_cast<int>(c12.getByID(s.pid).size());
        if (s.exact && n != s.count) return false;
        if (!s.exact && n < s.count) return false;
    }

    if (!cfg.inclusive) {
        const auto& particles = c12.getDetParticles();
        for (const auto* particle : particles) {
            const int pid = particle->getPid();
            const bool listed = std::any_of(cfg.finalState.begin(), cfg.finalState.end(),
                                            [pid](const FinalState& s) { return s.pid == pid; });
            if (!listed) return false;
        }
    }

    return true;
}

bool shouldWritePid(const Config& cfg, int pid) {
    if (cfg.outputPids.empty()) return true;
    return std::find(cfg.outputPids.begin(), cfg.outputPids.end(), pid) != cfg.outputPids.end();
}

// ─── DIS skim ────────────────────────────────────────────────────────────────

bool passesDISSkim(const Config& cfg, clas12::clas12reader& c12) {
    if (!cfg.enableSkim) return true;

    auto electrons = c12.getByID(11);
    if (electrons.empty()) return false;

    auto* e = electrons[0];
    const TLorentzVector lvE = Kinematics::particle(e->par()->getPx(),
                                                    e->par()->getPy(),
                                                    e->par()->getPz(),
                                                    Kinematics::massForPid(11));
    const Kinematics::DIS dis = Kinematics::dis(lvE, cfg.beamEnergy);

    return (dis.Q2 >= cfg.Q2_min && dis.W >= cfg.W_min && dis.y <= cfg.y_max);
}

bool passesDiphotonMassSkim(const Config& cfg, clas12::clas12reader& c12) {
    if (!cfg.diphotonMassSkim.enabled) return true;

    const auto photons = c12.getByID(22);
    for (std::size_t first = 0; first < photons.size(); ++first) {
        const auto* gamma1 = photons[first];
        const TLorentzVector lvGamma1 = Kinematics::particle(
            gamma1->par()->getPx(), gamma1->par()->getPy(), gamma1->par()->getPz(), 0.0
        );
        for (std::size_t second = first + 1; second < photons.size(); ++second) {
            const auto* gamma2 = photons[second];
            const TLorentzVector lvGamma2 = Kinematics::particle(
                gamma2->par()->getPx(), gamma2->par()->getPy(), gamma2->par()->getPz(), 0.0
            );
            const double mass2 = (lvGamma1 + lvGamma2).M2();
            if (!std::isfinite(mass2) || mass2 < 0.0) continue;
            const double mass = std::sqrt(mass2);
            if (mass >= cfg.diphotonMassSkim.minGeV &&
                mass <= cfg.diphotonMassSkim.maxGeV) {
                return true;
            }
        }
    }
    return false;
}

// ─── MC matching ─────────────────────────────────────────────────────────────

MatchResult findBestGenMatch(clas12::region_particle* rec,
                             clas12::mcparticle* mc,
                             const std::vector<bool>& usedGen,
                             double maxAngleDeg) {
    MatchResult best;
    double bestAngle = maxAngleDeg;
    const TVector3 recVec(rec->par()->getPx(), rec->par()->getPy(), rec->par()->getPz());
    if (recVec.Mag2() <= 0.0) return best;

    for (int i = 0; i < mc->getRows(); ++i) {
        if (usedGen[i]) continue;
        if (mc->getPid(i) != rec->getPid()) continue;

        const TVector3 genVec(mc->getPx(i), mc->getPy(i), mc->getPz(i));
        if (genVec.Mag2() <= 0.0) continue;

        const double angleDeg = recVec.Angle(genVec) * 180.0 / 3.14159265358979323846;
        if (!std::isfinite(angleDeg)) continue;
        if (angleDeg <= bestAngle) {
            best.genIdx = i;
            best.angleDeg = angleDeg;
            bestAngle = angleDeg;
        }
    }

    return best;
}

void fillRecBranch(RecBranches& recBranches,
                   clas12::region_particle* particle,
                   int runNum,
                   int eventNum,
                   int particleIdx,
                   const ProtonEnergyLossCorrections& corrections) {
    if (particle->getPid() == 2212 && corrections.enabled()) {
        const int det = getDetector(particle->par()->getStatus());
        const double p = particle->getP();
        const double theta = particle->getTheta();
        const double phi = particle->getPhi();
        const CorrectedKinematics corrected = corrections.correct(p, theta, phi, det);
        recBranches.fill(particle, runNum, eventNum, particleIdx,
                         corrected.p, corrected.theta, corrected.phi);
        return;
    }

    recBranches.fill(particle, runNum, eventNum, particleIdx);
}

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cerr << "Usage: hipo2root <config.json> <hipo_file_or_directory>... "
                  << "[max_files] [progress_events]\n";
        return 1;
    }

    int maxFiles = -1; // -1 = all
    long long progressEvery = 1000000;
    int inputArgEnd = argc;
    if (argc >= 4 && isTrailingNumericOption(argv[argc - 1])) {
        if (argc >= 5 && isTrailingNumericOption(argv[argc - 2])) {
            progressEvery = std::stoll(argv[argc - 1]);
            if (progressEvery < 0) progressEvery = 0;
            maxFiles = std::stoi(argv[argc - 2]);
            inputArgEnd = argc - 2;
        } else {
            maxFiles = std::stoi(argv[argc - 1]);
            inputArgEnd = argc - 1;
        }
        if (maxFiles <= 0) maxFiles = -1;
    }
    if (inputArgEnd <= 2) {
        std::cerr << "[ERROR] At least one HIPO input file or directory is required.\n";
        return 1;
    }

    Config cfg(argv[1]);
    if (cfg.generatedEventTree.enabled && !cfg.fillMC) {
        std::cerr << "[ERROR] generatedEventTree requires fillMC=true\n";
        return 1;
    }
    if (cfg.generatorWeights.enabled && !cfg.generatedEventTree.enabled) {
        std::cerr << "[ERROR] generatorWeights requires generatedEventTree.enabled=true\n";
        return 1;
    }
    std::unordered_map<std::string, GeneratorFileMetadata> generatorWeights;
    if (cfg.generatorWeights.enabled) {
        try {
            generatorWeights = loadGeneratorWeights(cfg.generatorWeights.chunkProvenance);
        } catch (const std::exception& error) {
            std::cerr << "[ERROR] " << error.what() << "\n";
            return 1;
        }
    }
    const ProtonEnergyLossCorrections corrections(cfg.kinematicCorrections);
    std::unique_ptr<QualityAssurance> qa;
    try {
        qa = std::make_unique<QualityAssurance>(cfg.qadb);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << "\n";
        return 1;
    }
    // ── Collect .hipo files ───────────────────────────────────────────────────
    std::vector<std::string> hipoFiles;
    for (int argIndex = 2; argIndex < inputArgEnd; ++argIndex) {
        const fs::path hipoInput = argv[argIndex];
        if (fs::is_regular_file(hipoInput)) {
            if (hipoInput.extension() != ".hipo") {
                std::cerr << "[ERROR] Input file is not a .hipo file: " << hipoInput << "\n";
                return 1;
            }
            hipoFiles.push_back(hipoInput.string());
        } else if (fs::is_directory(hipoInput)) {
            for (const auto& entry : fs::recursive_directory_iterator(hipoInput)) {
                if (!entry.is_regular_file()) continue;
                if (entry.path().extension() == ".hipo") {
                    hipoFiles.push_back(entry.path().string());
                }
            }
        } else {
            std::cerr << "[ERROR] HIPO input does not exist: " << hipoInput << "\n";
            return 1;
        }
    }
    if (hipoFiles.empty()) {
        std::cerr << "[ERROR] No .hipo files found in the requested inputs.\n";
        return 1;
    }
    std::sort(hipoFiles.begin(), hipoFiles.end());
    if (maxFiles > 0 && static_cast<int>(hipoFiles.size()) > maxFiles) {
        hipoFiles.resize(maxFiles);
    }
    std::unordered_map<std::string, int> sourceBasenameCounts;
    for (const auto& hipoPath : hipoFiles) {
        ++sourceBasenameCounts[fs::path(hipoPath).filename().string()];
    }
    std::cout << "[INFO] Found " << hipoFiles.size() << " hipo file(s) to process.\n";

    // ── Echo active config ────────────────────────────────────────────────────
    std::cout << "[INFO] Output file : " << cfg.outputFile << "\n"
              << "[INFO] REC trees   : " << TreeNames::rEvents
              << ", " << TreeNames::rParticles << "\n"
              << "[INFO] Beam energy : " << cfg.beamEnergy << " GeV\n"
              << "[INFO] Fill MC     : " << (cfg.fillMC ? "yes" : "no") << "\n"
              << "[INFO] Match MC    : " << (cfg.matchMC ? "yes" : "no") << "\n"
              << "[INFO] GEN events  : "
              << (cfg.generatedEventTree.enabled ? TreeNames::gEvents : "disabled")
              << "\n"
              << "[INFO] GEN weights : "
              << (cfg.generatorWeights.enabled
                      ? cfg.generatorWeights.chunkProvenance
                      : "unit weights")
              << "\n"
              << "[INFO] QADB        : " << (qa->enabled() ? "enabled" : "disabled") << "\n"
              << "[INFO] Input check : "
              << (cfg.inputValidation.enabled
                      ? (cfg.inputValidation.skipMalformed ? "enabled, skip malformed"
                                                          : "enabled, fail on malformed")
                      : "disabled")
              << "\n"
              << "[INFO] Progress    : "
              << (progressEvery > 0 ? std::to_string(progressEvery) + " events" : "disabled")
              << "\n"
              << "[INFO] Corrections : "
              << (corrections.enabled() ? "enabled" : "disabled") << "\n";

    if (qa->enabled()) {
        std::cout << "[INFO] QADB source : " << cfg.qadb.database << "\n"
                  << "[INFO] QA defects  :";
        for (const auto& defect : cfg.qadb.rejectDefects) std::cout << " " << defect;
        std::cout << "\n";
    }

    if (!cfg.finalState.empty()) {
        std::cout << "[INFO] Final state filter:\n";
        for (const auto& s : cfg.finalState)
            std::cout << "  PID " << s.pid
                      << "  " << (s.exact ? "==" : ">=") << s.count << "\n";
        std::cout << "  Unlisted PIDs: "
                  << (cfg.inclusive ? "allowed" : "rejected") << "\n";
    }
    if (cfg.enableSkim) {
        std::cout << "[INFO] DIS skim: Q2 >= " << cfg.Q2_min
                  << ", W >= "  << cfg.W_min
                  << ", y <= " << cfg.y_max << "\n";
    }
    if (cfg.diphotonMassSkim.enabled) {
        std::cout << "[INFO] Diphoton mass skim: at least one pair in ["
                  << cfg.diphotonMassSkim.minGeV << ", "
                  << cfg.diphotonMassSkim.maxGeV << "] GeV\n";
    }
    if (!cfg.outputPids.empty()) {
        std::cout << "[INFO] Output PID filter:";
        for (const int pid : cfg.outputPids) std::cout << " " << pid;
        std::cout << "\n";
    }

    // ── Output ROOT file + tree ───────────────────────────────────────────────
    TFile* outFile = TFile::Open(cfg.outputFile.c_str(), "RECREATE");
    if (!outFile || outFile->IsZombie()) {
        std::cerr << "[ERROR] Could not open output file: " << cfg.outputFile << "\n";
        return 1;
    }
    auto* rParticles = new TTree(TreeNames::rParticles, TreeNames::rParticles);
    auto* rEvents = new TTree(TreeNames::rEvents, TreeNames::rEvents);
    ReconstructedEventOutput rEvent(cfg.finalState);
    rEvent.registerBranches(*rEvents);
    TTree* gEvents = nullptr;
    GeneratedEventBranches gEvent;
    if (cfg.generatedEventTree.enabled) {
        gEvents = new TTree(TreeNames::gEvents, TreeNames::gEvents);
        gEvent.registerBranches(*gEvents);
    }
    TTree* sourceFilesTree = nullptr;
    std::uint64_t catalogSourceFileId = INVALID_SOURCE_ID;
    std::string catalogSourceFileName;
    int catalogStratumFlatIndex = -1;
    double catalogGeneratorWeight = 1.0;
    if (cfg.generatedEventTree.enabled) {
        sourceFilesTree = new TTree("SourceFiles", "SourceFiles");
        sourceFilesTree->Branch("sourceFileId", &catalogSourceFileId, "sourceFileId/l");
        sourceFilesTree->Branch("sourceFileName", &catalogSourceFileName);
        sourceFilesTree->Branch(
            "stratumFlatIndex", &catalogStratumFlatIndex, "stratumFlatIndex/I"
        );
        sourceFilesTree->Branch(
            "generatorWeight", &catalogGeneratorWeight, "generatorWeight/D"
        );
    }

    EventBranches evBranches;
    RecBranches   recBranches;
    GenBranches   genBranches;

    rParticles->Branch("event", &evBranches);
    rParticles->Branch("rec",   &recBranches);
    if (cfg.fillMC) rParticles->Branch("gen", &genBranches);

    // ── Event loop ────────────────────────────────────────────────────────────
    long long nTotal = 0, nQAFail = 0, nFSFail = 0, nSkimFail = 0, nWritten = 0;
    long long nDISSkimFail = 0, nDiphotonMassSkimFail = 0;
    long long nOutputRows = 0;
    long long nSkippedOutputPid = 0;
    long long nMatched = 0, nUnmatchedRec = 0, nUnmatchedGen = 0;
    long long nGeneratedEvents = 0, nGeneratedTopologyValid = 0;
    long long nReconstructedEventRows = 0;
    long long nInputFail = 0;
    long long lastProgressEvent = 0;
    const Clock::time_point startTime = Clock::now();
    std::unordered_map<std::uint64_t, std::string> sourceFileCatalog;

    for (std::size_t fileIndex = 0; fileIndex < hipoFiles.size(); ++fileIndex) {
        const auto& hipoPath = hipoFiles[fileIndex];
        std::cout << "[INFO] Processing: " << hipoPath << "\n";

        if (cfg.inputValidation.enabled) {
            const HipoOpenCheck check = checkHipoOpenInChild(hipoPath);
            if (!check.ok) {
                ++nInputFail;
                std::cerr << "[WARN] Skipping malformed HIPO input: " << hipoPath
                          << " (" << check.detail << ")\n";
                if (cfg.inputValidation.skipMalformed) {
                    continue;
                }
                return 1;
            }
        }

        const std::string sourceBasename = fs::path(hipoPath).filename().string();
        int sourceStratumFlatIndex = -1;
        double sourceGeneratorWeight = 1.0;
        if (cfg.generatorWeights.enabled) {
            std::string sourceChunkId;
            try {
                sourceChunkId = extractCanonicalChunkId(sourceBasename);
            } catch (const std::exception& error) {
                std::cerr << "[ERROR] " << error.what() << "\n";
                return 1;
            }
            const auto weightEntry = generatorWeights.find(sourceChunkId);
            if (weightEntry == generatorWeights.end()) {
                std::cerr << "[ERROR] HIPO canonical chunk ID is absent from "
                          << "generator chunk provenance: " << sourceChunkId << "\n";
                return 1;
            }
            sourceStratumFlatIndex = weightEntry->second.stratumFlatIndex;
            sourceGeneratorWeight = weightEntry->second.weight;
        }
        const std::string sourceFileName = sourceBasenameCounts[sourceBasename] > 1
            ? fs::absolute(fs::path(hipoPath)).lexically_normal().string()
            : sourceBasename;
        const std::uint64_t sourceFileId = stableSourceFileId(sourceFileName);
        const auto catalogEntry = sourceFileCatalog.find(sourceFileId);
        if (catalogEntry != sourceFileCatalog.end() && catalogEntry->second != sourceFileName) {
            std::cerr << "[ERROR] Source-file hash collision between "
                      << catalogEntry->second << " and " << sourceFileName << "\n";
            return 1;
        }
        if (catalogEntry == sourceFileCatalog.end()) {
            sourceFileCatalog[sourceFileId] = sourceFileName;
            if (sourceFilesTree) {
                catalogSourceFileId = sourceFileId;
                catalogSourceFileName = sourceFileName;
                catalogStratumFlatIndex = sourceStratumFlatIndex;
                catalogGeneratorWeight = sourceGeneratorWeight;
                sourceFilesTree->Fill();
            }
        }

        std::optional<clas12::clas12reader> reader;
        withoutRichDetectorSchemaWarning([&]() { reader.emplace(hipoPath); });
        auto& c12 = *reader;
        std::uint64_t sourceEventIndex = 0;

        const auto maybePrintProgress = [&]() {
            if (progressEvery <= 0 || nTotal - lastProgressEvent < progressEvery) return;
            printProgress(fileIndex + 1,
                          hipoFiles.size(),
                          nTotal,
                          nWritten,
                          nOutputRows,
                          nQAFail,
                          nFSFail,
                          nSkimFail,
                          nMatched,
                          cfg.fillMC && cfg.matchMC,
                          startTime);
            lastProgressEvent = nTotal;
        };

        while (c12.next()) {
            ++nTotal;
            const std::uint64_t currentSourceEventIndex = sourceEventIndex++;

            const int runNum = c12.runconfig()->getRun();
            const int eventNum = c12.runconfig()->getEvent();
            auto* mc = cfg.fillMC ? c12.mcparts() : nullptr;

            // Preserve the generated denominator before any reconstructed QA,
            // final-state, or DIS decision is made.
            if (gEvents) {
                gEvent.fill(mc, runNum, eventNum,
                            sourceFileId, currentSourceEventIndex,
                            cfg.beamEnergy,
                            sourceStratumFlatIndex,
                            sourceGeneratorWeight);
                gEvents->Fill();
                ++nGeneratedEvents;
                if (gEvent.topologyValid) ++nGeneratedTopologyValid;
            }

            if (!qa->pass(runNum, eventNum)) {
                ++nQAFail;
                maybePrintProgress();
                continue;
            }

            if (!passesFinalState(cfg, c12)) {
                ++nFSFail;
                maybePrintProgress();
                continue;
            }
            if (!passesDISSkim(cfg, c12)) {
                ++nSkimFail;
                ++nDISSkimFail;
                maybePrintProgress();
                continue;
            }
            if (!passesDiphotonMassSkim(cfg, c12)) {
                ++nSkimFail;
                ++nDiphotonMassSkimFail;
                maybePrintProgress();
                continue;
            }

            evBranches.fill(c12);
            evBranches.setSource(sourceFileId, currentSourceEventIndex);
            int rn = evBranches.runNum;
            int en = evBranches.eventNum;

            const auto& particles = c12.getDetParticles();
            std::vector<bool> usedGen(mc ? mc->getRows() : 0, false);
            int writtenRecParticles = 0;
            const long long particleRowsBeforeEvent = nOutputRows;

            for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
                auto* particle = particles[i];
                if (!shouldWritePid(cfg, particle->getPid())) {
                    ++nSkippedOutputPid;
                    continue;
                }

                fillRecBranch(recBranches, particle, rn, en, i, corrections);
                if (cfg.fillMC) genBranches.reset();

                if (cfg.fillMC && cfg.matchMC && mc) {
                    const MatchResult match = findBestGenMatch(particle, mc, usedGen, cfg.matchMaxAngleDeg);
                    if (match.genIdx >= 0) {
                        usedGen[match.genIdx] = true;
                        recBranches.setMatch(match.genIdx, match.angleDeg);
                        genBranches.fill(mc, rn, en, match.genIdx);
                        ++nMatched;
                    } else {
                        ++nUnmatchedRec;
                    }
                }

                rParticles->Fill();
                ++nOutputRows;
                ++writtenRecParticles;
            }

            if (cfg.fillMC && mc && (!cfg.matchMC || cfg.saveUnmatchedMC)) {
                for (int i = 0; i < mc->getRows(); ++i) {
                    if (cfg.matchMC && usedGen[i]) continue;
                    if (!shouldWritePid(cfg, mc->getPid(i))) {
                        ++nSkippedOutputPid;
                        continue;
                    }
                    recBranches.reset();
                    genBranches.fill(mc, rn, en, i);
                    rParticles->Fill();
                    ++nOutputRows;
                    ++nUnmatchedGen;
                }
            }

            rEvent.fill(
                evBranches,
                particles,
                writtenRecParticles,
                static_cast<int>(nOutputRows - particleRowsBeforeEvent)
            );
            rEvents->Fill();
            ++nReconstructedEventRows;

            ++nWritten;
            maybePrintProgress();
        }
    }

    if (progressEvery > 0 && nTotal != lastProgressEvent) {
        printProgress(hipoFiles.size(),
                      hipoFiles.size(),
                      nTotal,
                      nWritten,
                      nOutputRows,
                      nQAFail,
                      nFSFail,
                      nSkimFail,
                      nMatched,
                      cfg.fillMC && cfg.matchMC,
                      startTime);
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    double accumulatedCharge = qa->accumulatedCharge();
    const std::vector<RunChargeRecord> runChargeRecords = qa->runChargeRecords();
    const double elapsed = std::chrono::duration<double>(Clock::now() - startTime).count();
    std::cout << "\n[DONE]\n"
              << "  Total events      : " << nTotal    << "\n"
              << "  Failed QADB       : " << nQAFail   << "\n"
              << "  Failed final state: " << nFSFail   << "\n"
              << "  Failed skim       : " << nSkimFail << "\n"
              << "    DIS             : " << nDISSkimFail << "\n"
              << "    Diphoton mass   : " << nDiphotonMassSkimFail << "\n"
              << "  Written events    : " << nWritten  << "\n"
              << "  Output rows       : " << nOutputRows << "\n"
              << "  PID-filtered rows : " << nSkippedOutputPid << "\n"
              << "  Skipped input files: " << nInputFail << "\n"
              << "  Generated events  : " << nGeneratedEvents << "\n"
              << "  Reconstructed event rows: " << nReconstructedEventRows << "\n"
              << "  Valid GEN topology: " << nGeneratedTopologyValid << "\n"
              << "  Elapsed time      : " << std::fixed << std::setprecision(1)
              << elapsed << " s\n";
    if (qa->enabled()) {
        std::cout << "  Accumulated charge: " << accumulatedCharge << " nC\n";
    }
    if (cfg.fillMC && cfg.matchMC) {
        std::cout << "  Matched REC rows  : " << nMatched << "\n"
                  << "  Unmatched REC rows: " << nUnmatchedRec << "\n"
                  << "  Unmatched GEN rows: " << nUnmatchedGen << "\n";
    }

    TTree summary("Summary", "Processing summary");
    bool qadbEnabled = qa->enabled();
    std::string qadbDatabase = qadbEnabled ? cfg.qadb.database : "";
    std::ostringstream defectStream;
    for (std::size_t i = 0; i < cfg.qadb.rejectDefects.size(); ++i) {
        if (i > 0) defectStream << ",";
        defectStream << cfg.qadb.rejectDefects[i];
    }
    std::string qadbRejectDefects = defectStream.str();
    summary.Branch("QADBEnabled", &qadbEnabled, "QADBEnabled/O");
    summary.Branch("QADBDatabase", &qadbDatabase);
    summary.Branch("QADBRejectDefects", &qadbRejectDefects);
    summary.Branch("TotalEvents", &nTotal, "TotalEvents/L");
    summary.Branch("FailedQADB", &nQAFail, "FailedQADB/L");
    summary.Branch("FailedFinalState", &nFSFail, "FailedFinalState/L");
    summary.Branch("FailedSkim", &nSkimFail, "FailedSkim/L");
    summary.Branch("FailedDISSkim", &nDISSkimFail, "FailedDISSkim/L");
    summary.Branch("FailedDiphotonMassSkim", &nDiphotonMassSkimFail,
                   "FailedDiphotonMassSkim/L");
    bool diphotonMassSkimEnabled = cfg.diphotonMassSkim.enabled;
    double diphotonMassSkimMinGeV = cfg.diphotonMassSkim.minGeV;
    double diphotonMassSkimMaxGeV = cfg.diphotonMassSkim.maxGeV;
    summary.Branch("DiphotonMassSkimEnabled", &diphotonMassSkimEnabled,
                   "DiphotonMassSkimEnabled/O");
    summary.Branch("DiphotonMassSkimMinGeV", &diphotonMassSkimMinGeV,
                   "DiphotonMassSkimMinGeV/D");
    summary.Branch("DiphotonMassSkimMaxGeV", &diphotonMassSkimMaxGeV,
                   "DiphotonMassSkimMaxGeV/D");
    summary.Branch("WrittenEvents", &nWritten, "WrittenEvents/L");
    summary.Branch("OutputRows", &nOutputRows, "OutputRows/L");
    summary.Branch("PidFilteredRows", &nSkippedOutputPid, "PidFilteredRows/L");
    summary.Branch("SkippedInputFiles", &nInputFail, "SkippedInputFiles/L");
    summary.Branch("GeneratedEventRows", &nGeneratedEvents, "GeneratedEventRows/L");
    summary.Branch("ReconstructedEventRows", &nReconstructedEventRows,
                   "ReconstructedEventRows/L");
    summary.Branch("GeneratedTopologyValid", &nGeneratedTopologyValid,
                   "GeneratedTopologyValid/L");
    long long runChargeRows = static_cast<long long>(runChargeRecords.size());
    summary.Branch("RunChargeRows", &runChargeRows, "RunChargeRows/L");
    summary.Fill();

    TTree runCharge("RunCharge", "QADB accumulated charge and event counts by run");
    int chargeRunNum = -999;
    double runAccumulatedChargeNC = 0.0;
    long long runTotalEvents = 0;
    long long runPassedQADBEvents = 0;
    long long runFailedQADBEvents = 0;
    runCharge.Branch("runNum", &chargeRunNum, "runNum/I");
    runCharge.Branch(
        "accumulatedCharge_nC", &runAccumulatedChargeNC, "accumulatedCharge_nC/D"
    );
    runCharge.Branch("totalEvents", &runTotalEvents, "totalEvents/L");
    runCharge.Branch(
        "passedQADBEvents", &runPassedQADBEvents, "passedQADBEvents/L"
    );
    runCharge.Branch(
        "failedQADBEvents", &runFailedQADBEvents, "failedQADBEvents/L"
    );
    for (const auto& record : runChargeRecords) {
        chargeRunNum = record.runNum;
        runAccumulatedChargeNC = record.accumulatedChargeNC;
        runTotalEvents = record.totalEvents;
        runPassedQADBEvents = record.passedQADBEvents;
        runFailedQADBEvents = record.failedQADBEvents;
        runCharge.Fill();
    }

    TParameter<double> chargeMetadata("AccumulatedCharge", accumulatedCharge);
    chargeMetadata.Write();

    outFile->Write();
    outFile->Close();
    return 0;
}
