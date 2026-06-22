#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <iomanip>
#include <memory>
#include <sstream>

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
        std::cerr << "Usage: hipo2root <config.json> <hipo_directory> "
                  << "[max_files] [progress_events]\n";
        return 1;
    }

    int maxFiles = -1; // -1 = all
    if (argc >= 4) {
        maxFiles = std::stoi(argv[3]);
        if (maxFiles <= 0) maxFiles = -1;
    }
    long long progressEvery = 1000000;
    if (argc >= 5) {
        progressEvery = std::stoll(argv[4]);
        if (progressEvery < 0) progressEvery = 0;
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
    const std::string hipoDir = argv[2];

    // ── Collect .hipo files ───────────────────────────────────────────────────
    std::vector<std::string> hipoFiles;
    for (const auto& entry : fs::recursive_directory_iterator(hipoDir)) {
        if (!entry.is_regular_file()) continue;
        if (entry.path().extension() == ".hipo") {
            hipoFiles.push_back(entry.path().string());
        }
    }
    if (hipoFiles.empty()) {
        std::cerr << "[ERROR] No .hipo files found in " << hipoDir << "\n";
        return 1;
    }
    std::sort(hipoFiles.begin(), hipoFiles.end());
    if (maxFiles > 0 && static_cast<int>(hipoFiles.size()) > maxFiles) {
        hipoFiles.resize(maxFiles);
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

    EventBranches evBranches;
    RecBranches   recBranches;
    GenBranches   genBranches;

    tree->Branch("event", &evBranches);
    tree->Branch("rec",   &recBranches);
    if (cfg.fillMC) tree->Branch("gen", &genBranches);

    // ── Event loop ────────────────────────────────────────────────────────────
    long long nTotal = 0, nQAFail = 0, nFSFail = 0, nSkimFail = 0, nWritten = 0;
    long long nOutputRows = 0;
    long long nMatched = 0, nUnmatchedRec = 0, nUnmatchedGen = 0;
    long long nGeneratedEvents = 0, nGeneratedTopologyValid = 0;
    long long lastProgressEvent = 0;
    const Clock::time_point startTime = Clock::now();

    for (std::size_t fileIndex = 0; fileIndex < hipoFiles.size(); ++fileIndex) {
        const auto& hipoPath = hipoFiles[fileIndex];
        std::cout << "[INFO] Processing: " << hipoPath << "\n";

        clas12::clas12reader c12(hipoPath);

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

            const int runNum = c12.runconfig()->getRun();
            const int eventNum = c12.runconfig()->getEvent();
            auto* mc = cfg.fillMC ? c12.mcparts() : nullptr;

            // Preserve the generated denominator before any reconstructed QA,
            // final-state, or DIS decision is made.
            if (generatedTree) {
                generatedEvent.fill(mc, runNum, eventNum, cfg.beamEnergy);
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
            int rn = evBranches.runNum;
            int en = evBranches.eventNum;

            const auto& particles = c12.getDetParticles();
            std::vector<bool> usedGen(mc ? mc->getRows() : 0, false);

            for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
                auto* particle = particles[i];

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
