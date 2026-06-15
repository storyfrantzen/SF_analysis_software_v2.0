#include <iostream>
#include <filesystem>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

#include "TFile.h"
#include "TTree.h"

#include "clas12reader.h"

#include "Config.h"
#include "KinematicCorrections.h"
#include "Kinematics.h"
#include "ROOTBranches.h"

using namespace clas12;
namespace fs = std::filesystem;

constexpr double kPi = 3.14159265358979323846;

double normalizePhi(double phi) {
    while (phi > kPi) phi -= 2.0 * kPi;
    while (phi < -kPi) phi += 2.0 * kPi;
    return phi;
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

// ─── Main ────────────────────────────────────────────────────────────────────

int main(int argc, char** argv) {

    if (argc < 3) {
        std::cerr << "Usage: hipo2root <config.json> <hipo_directory> [max_files]\n";
        return 1;
    }

    int maxFiles = -1; // -1 = all
    if (argc >= 4) {
        maxFiles = std::stoi(argv[3]);
        if (maxFiles <= 0) maxFiles = -1;
    }

    Config cfg(argv[1]);
    const KinematicCorrections corrections(cfg.kinematicCorrections);
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
              << "[INFO] Corrections : "
              << (corrections.enabled() ? "enabled" : "disabled") << "\n";

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

    EventBranches evBranches;
    RecBranches   recBranches;
    GenBranches   genBranches;

    tree->Branch("event", &evBranches);
    tree->Branch("rec",   &recBranches);
    if (cfg.fillMC) tree->Branch("gen", &genBranches);

    // ── Event loop ────────────────────────────────────────────────────────────
    long long nTotal = 0, nFSFail = 0, nSkimFail = 0, nWritten = 0;

    for (const auto& hipoPath : hipoFiles) {
        std::cout << "[INFO] Processing: " << hipoPath << "\n";

        clas12::clas12reader c12(hipoPath);

        while (c12.next()) {
            ++nTotal;

            if (!passesFinalState(cfg, c12)) { ++nFSFail;   continue; }
            if (!passesDISSkim(cfg, c12))     { ++nSkimFail; continue; }

            evBranches.fill(c12);
            int rn = evBranches.runNum;
            int en = evBranches.eventNum;

            const auto& particles = c12.getDetParticles();
            for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
                auto* particle = particles[i];
                if (particle->getPid() == 2212 && corrections.enabled()) {
                    const int det = getDetector(particle->par()->getStatus());
                    const double p = particle->getP();
                    const double theta = particle->getTheta();
                    const double phi = particle->getPhi();
                    recBranches.fill(particle,
                                     rn,
                                     en,
                                     i,
                                     p + corrections.deltaP(p, theta, det),
                                     theta + corrections.deltaTheta(p, theta, det),
                                     normalizePhi(phi + corrections.deltaPhi(p, theta, det)));
                } else {
                    recBranches.fill(particle, rn, en, i);
                }
                tree->Fill();
            }

            if (cfg.fillMC) {
                auto* mc = c12.mcparts();
                for (int i = 0; i < mc->getRows(); ++i) {
                    genBranches.fill(mc, rn, en, i);
                    tree->Fill();
                }
            }

            ++nWritten;
        }
    }

    // ── Summary ───────────────────────────────────────────────────────────────
    std::cout << "\n[DONE]\n"
              << "  Total events      : " << nTotal    << "\n"
              << "  Failed final state: " << nFSFail   << "\n"
              << "  Failed skim       : " << nSkimFail << "\n"
              << "  Written           : " << nWritten  << "\n";

    outFile->Write();
    outFile->Close();
    return 0;
}
