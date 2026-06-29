#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cctype>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>
#include <unordered_map>

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

using namespace clas12;
namespace fs = std::filesystem;
using Clock = std::chrono::steady_clock;

struct MatchResult {
    int genIdx = -1;
    double angleDeg = NAN;
};

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
    if (cfg.generatedEventTree.enabled && cfg.generatedEventTree.treeName == cfg.treeName) {
        std::cerr << "[ERROR] generatedEventTree.treeName must differ from treeName\n";
        return 1;
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
              << "[INFO] Tree name   : " << cfg.treeName   << "\n"
              << "[INFO] Beam energy : " << cfg.beamEnergy << " GeV\n"
              << "[INFO] Fill MC     : " << (cfg.fillMC ? "yes" : "no") << "\n"
              << "[INFO] Match MC    : " << (cfg.matchMC ? "yes" : "no") << "\n"
              << "[INFO] GEN events  : "
              << (cfg.generatedEventTree.enabled ? cfg.generatedEventTree.treeName : "disabled")
              << "\n"
              << "[INFO] QADB        : " << (qa->enabled() ? "enabled" : "disabled") << "\n"
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
    TTree* tree = new TTree(cfg.treeName.c_str(), cfg.treeName.c_str());
    TTree* generatedTree = nullptr;
    GeneratedEventBranches generatedEvent;
    if (cfg.generatedEventTree.enabled) {
        generatedTree = new TTree(cfg.generatedEventTree.treeName.c_str(),
                                  cfg.generatedEventTree.treeName.c_str());
        generatedEvent.registerBranches(*generatedTree);
    }
    TTree* sourceFilesTree = nullptr;
    std::uint64_t catalogSourceFileId = INVALID_SOURCE_ID;
    std::string catalogSourceFileName;
    if (cfg.generatedEventTree.enabled) {
        sourceFilesTree = new TTree("SourceFiles", "SourceFiles");
        sourceFilesTree->Branch("sourceFileId", &catalogSourceFileId, "sourceFileId/l");
        sourceFilesTree->Branch("sourceFileName", &catalogSourceFileName);
    }

    EventBranches evBranches;
    RecBranches   recBranches;
    GenBranches   genBranches;

    tree->Branch("event", &evBranches);
    tree->Branch("rec",   &recBranches);
    if (cfg.fillMC) tree->Branch("gen", &genBranches);

    // ── Event loop ────────────────────────────────────────────────────────────
    long long nTotal = 0, nQAFail = 0, nFSFail = 0, nSkimFail = 0, nWritten = 0;
    long long nOutputRows = 0;
    long long nSkippedOutputPid = 0;
    long long nMatched = 0, nUnmatchedRec = 0, nUnmatchedGen = 0;
    long long nGeneratedEvents = 0, nGeneratedTopologyValid = 0;
    long long lastProgressEvent = 0;
    const Clock::time_point startTime = Clock::now();
    std::unordered_map<std::uint64_t, std::string> sourceFileCatalog;

    for (std::size_t fileIndex = 0; fileIndex < hipoFiles.size(); ++fileIndex) {
        const auto& hipoPath = hipoFiles[fileIndex];
        std::cout << "[INFO] Processing: " << hipoPath << "\n";

        const std::string sourceBasename = fs::path(hipoPath).filename().string();
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
                sourceFilesTree->Fill();
            }
        }

        clas12::clas12reader c12(hipoPath);
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
            if (generatedTree) {
                generatedEvent.fill(mc, runNum, eventNum,
                                    sourceFileId, currentSourceEventIndex,
                                    cfg.beamEnergy);
                generatedTree->Fill();
                ++nGeneratedEvents;
                if (generatedEvent.topologyValid) ++nGeneratedTopologyValid;
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
                maybePrintProgress();
                continue;
            }

            evBranches.fill(c12);
            evBranches.setSource(sourceFileId, currentSourceEventIndex);
            int rn = evBranches.runNum;
            int en = evBranches.eventNum;

            const auto& particles = c12.getDetParticles();
            std::vector<bool> usedGen(mc ? mc->getRows() : 0, false);

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

                tree->Fill();
                ++nOutputRows;
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
                    tree->Fill();
                    ++nOutputRows;
                    ++nUnmatchedGen;
                }
            }

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
    const double elapsed = std::chrono::duration<double>(Clock::now() - startTime).count();
    std::cout << "\n[DONE]\n"
              << "  Total events      : " << nTotal    << "\n"
              << "  Failed QADB       : " << nQAFail   << "\n"
              << "  Failed final state: " << nFSFail   << "\n"
              << "  Failed skim       : " << nSkimFail << "\n"
              << "  Written events    : " << nWritten  << "\n"
              << "  Output rows       : " << nOutputRows << "\n"
              << "  PID-filtered rows : " << nSkippedOutputPid << "\n"
              << "  Generated events  : " << nGeneratedEvents << "\n"
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
    summary.Branch("WrittenEvents", &nWritten, "WrittenEvents/L");
    summary.Branch("OutputRows", &nOutputRows, "OutputRows/L");
    summary.Branch("PidFilteredRows", &nSkippedOutputPid, "PidFilteredRows/L");
    summary.Branch("GeneratedEventRows", &nGeneratedEvents, "GeneratedEventRows/L");
    summary.Branch("GeneratedTopologyValid", &nGeneratedTopologyValid,
                   "GeneratedTopologyValid/L");
    summary.Fill();

    TParameter<double> chargeMetadata("AccumulatedCharge", accumulatedCharge);
    chargeMetadata.Write();

    outFile->Write();
    outFile->Close();
    return 0;
}
