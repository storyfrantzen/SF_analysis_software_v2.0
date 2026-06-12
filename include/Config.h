#pragma once
#include <string>
#include <vector>
#include <fstream>
#include <stdexcept>
#include "nlohmann/json.hpp"

// ─── Final state ─────────────────────────────────────────────────────────

struct FinalState {
    int  pid;
    int  count;
    bool exact;   // true = ==count, false = >=count
};

// ─── Config ───────────────────────────────────────────────────────────────────

struct Config {

    // ── Output ────────────────────────────────
    std::string outputFile = "output.root";
    std::string treeName   = "Events";

    // ── Beam ──────────────────────────────────
    double beamEnergy = 10.6;

    // ── Final state filter ────────────────────
    std::vector<FinalState> finalState;
    bool inclusive = false;

    // ── DIS skim ──────────────────────────────
    bool   enableSkim = true;
    double Q2_min     = 1.0;
    double W_min      = 2.0;
    double y_max      = 0.8;

    // ── MC ────────────────────────────────────
    bool fillMC = false;

    // ── Constructors ──────────────────────────

    Config() = default;

    explicit Config(const std::string& filename) {
        std::ifstream f(filename);
        if (!f.is_open())
            throw std::runtime_error("Cannot open config file: " + filename);

        nlohmann::json j;
        f >> j;

        outputFile = j.value("outputFile", outputFile);
        treeName   = j.value("treeName",   treeName);

        beamEnergy = j.value("beamEnergy", beamEnergy);

        enableSkim = j.value("enableSkim", enableSkim);
        Q2_min     = j.value("Q2_min",     Q2_min);
        W_min      = j.value("W_min",      W_min);
        y_max      = j.value("y_max",      y_max);

        fillMC = j.value("fillMC", fillMC);

        inclusive = j.value("inclusive", inclusive);

        if (j.contains("finalState")) {
            for (const auto& p : j["finalState"]) {
                FinalState fs;
                fs.pid   = p.at("pid").get<int>();
                fs.count = p.at("count").get<int>();
                fs.exact = (p.at("mode").get<std::string>() == "exact");
                finalState.push_back(fs);
            }
        }
    }
};
