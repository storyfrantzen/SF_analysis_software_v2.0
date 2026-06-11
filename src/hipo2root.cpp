#include <iostream>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>
#include <algorithm>

#include "nlohmann/json.hpp"

#include "TFile.h"
#include "TTree.h"

#include "hipo4/reader.h"
#include "clas12reader.h"

#include "Config.h"
#include "PhysicalConstants.h"
#include "ROOTBranches.h"

#include "TLorentzVector.h"
#include "TVector3.h"

using namespace clas12;
namespace fs = std::filesystem;

// ─── Final state filter ───────────────────────────────────────────────────────

bool passesFinalState(const Config& cfg, clas12::clas12reader& c12) {
    if (cfg.finalState.empty()) return true;

    for (const auto& s : cfg.finalState) {
        int n = static_cast<int>(c12.getByID(s.pid).size());
        if ( s.exact && n != s.count) return false;
        if (!s.exact && n <  s.count) return false;
    }
    return true;
}

// ─── DIS skim ────────────────────────────────────────────────────────────────

bool passesSkim(const Config& cfg, clas12::clas12reader& c12) {
    if (!cfg.enableSkim) return true;

    auto electrons = c12.getByID(11);
    if (electrons.empty()) return false;

    auto* e = electrons[0];
    TLorentzVector lv_e(e->par()->getPx(), e->par()->getPy(), e->par()->getPz(), 
                        std::sqrt(e->getP() * e->getP() + M_ELECTRON * M_ELECTRON));  

    TLorentzVector lv_beam(0, 0, cfg.beamEnergy, cfg.beamEnergy);
    TLorentzVector q = lv_beam - lv_e;

    double Q2 = -q.M2();
    double nu  = cfg.beamEnergy - lv_e.E();
    double y   = nu / cfg.beamEnergy;
    double W   = std::sqrt(std::max(0.0,
                     (TLorentzVector(0, 0, 0, M_PROTON) + q).M2()));

    return (Q2 >= cfg.Q2_min && W >= cfg.W_min && y <= cfg.y_max);
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
              << "[INFO] Fill MC     : " << (cfg.fillMC ? "yes" : "no") << "\n";

    if (!cfg.finalState.empty()) {
        std::cout << "[INFO] Final state filter:\n";
        for (const auto& s : cfg.finalState)
            std::cout << "  PID " << s.pid
                      << "  " << (s.exact ? "==" : ">=") << s.count << "\n";
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
            if (!passesSkim(cfg, c12))        { ++nSkimFail; continue; }

            evBranches.fill(c12);
            int rn = evBranches.runNum;
            int en = evBranches.eventNum;

            const auto& particles = c12.getDetParticles();
            for (int i = 0; i < static_cast<int>(particles.size()); ++i) {
                recBranches.fill(particles[i], rn, en, i);
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