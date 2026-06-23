#pragma once
#include <filesystem>
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

struct QADBConfig {
    bool enabled = false;
    std::string database = "latest";
    std::vector<std::string> rejectDefects;
    std::vector<int> allowMiscRuns;
};

struct GeneratedEventTreeConfig {
    bool enabled = false;
    std::string treeName = "GeneratedEvents";
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
    std::vector<int> outputPids;

    // ── DIS skim ──────────────────────────────
    bool   enableSkim = true;
    double Q2_min     = 1.0;
    double W_min      = 2.0;
    double y_max      = 0.8;

    // ── MC ────────────────────────────────────
    bool fillMC = false;
    bool matchMC = false;
    bool saveUnmatchedMC = true;
    double matchMaxAngleDeg = 3.0;
    GeneratedEventTreeConfig generatedEventTree;

    // ── Data quality ──────────────────────────
    QADBConfig qadb;

    // ── Kinematic corrections ─────────────────
    nlohmann::json kinematicCorrections;

    // ── Constructors ──────────────────────────

    Config() = default;

    explicit Config(const std::string& filename) {
        std::ifstream f(filename);
        if (!f.is_open())
            throw std::runtime_error("Cannot open config file: " + filename);

        nlohmann::json j;
        f >> j;
        const auto configDir = std::filesystem::path(filename).parent_path();

        outputFile = j.value("outputFile", outputFile);
        treeName   = j.value("treeName",   treeName);

        beamEnergy = j.value("beamEnergy", beamEnergy);

        enableSkim = j.value("enableSkim", enableSkim);
        Q2_min     = j.value("Q2_min",     Q2_min);
        W_min      = j.value("W_min",      W_min);
        y_max      = j.value("y_max",      y_max);

        fillMC = j.value("fillMC", fillMC);
        matchMC = j.value("matchMC", matchMC);
        saveUnmatchedMC = j.value("saveUnmatchedMC", saveUnmatchedMC);
        matchMaxAngleDeg = j.value("matchMaxAngleDeg", matchMaxAngleDeg);

        if (j.contains("generatedEventTree")) {
            const auto& generated = j["generatedEventTree"];
            if (!generated.is_object()) {
                throw std::runtime_error("generatedEventTree must be a JSON object");
            }
            generatedEventTree.enabled = generated.value("enabled", generatedEventTree.enabled);
            generatedEventTree.treeName = generated.value("treeName", generatedEventTree.treeName);
            if (generatedEventTree.treeName.empty()) {
                throw std::runtime_error("generatedEventTree.treeName must not be empty");
            }
        }

        if (j.contains("qadb")) {
            const auto& qa = j["qadb"];
            if (!qa.is_object()) {
                throw std::runtime_error("qadb must be a JSON object");
            }
            qadb.enabled = qa.value("enabled", qadb.enabled);
            qadb.database = qa.value("database", qadb.database);
            qadb.rejectDefects = qa.value("rejectDefects", qadb.rejectDefects);
            qadb.allowMiscRuns = qa.value("allowMiscRuns", qadb.allowMiscRuns);
        }

        inclusive = j.value("inclusive", inclusive);
        outputPids = j.value("outputPids", outputPids);

        if (j.contains("kinematicCorrections")) {
            const auto& corrections = j["kinematicCorrections"];
            if (corrections.is_string()) {
                std::filesystem::path correctionPath = corrections.get<std::string>();
                if (correctionPath.is_relative() && !configDir.empty()) {
                    correctionPath = configDir / correctionPath;
                }

                std::ifstream cf(correctionPath);
                if (!cf.is_open()) {
                    throw std::runtime_error("Cannot open kinematic corrections file: " +
                                             correctionPath.string());
                }
                cf >> kinematicCorrections;
            } else {
                kinematicCorrections = corrections;
            }
        }

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
