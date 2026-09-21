#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "QADB.h"
#include "conversion/Config.h"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;

struct Options {
    fs::path config;
    fs::path runList;
    fs::path outputDir;
};

Options parseOptions(int argc, char** argv) {
    Options options;
    for (int index = 1; index < argc; ++index) {
        const std::string argument = argv[index];
        if (argument == "--config" && index + 1 < argc) {
            options.config = argv[++index];
        } else if (argument == "--run-list" && index + 1 < argc) {
            options.runList = argv[++index];
        } else if (argument == "--output-dir" && index + 1 < argc) {
            options.outputDir = argv[++index];
        } else {
            throw std::runtime_error("unknown or incomplete argument: " + argument);
        }
    }
    if (options.config.empty() || options.runList.empty() || options.outputDir.empty()) {
        throw std::runtime_error(
            "usage: qadb_helicity_audit --config CONFIG --run-list RUNS "
            "--output-dir DIRECTORY"
        );
    }
    return options;
}

std::vector<int> readRuns(const fs::path& path) {
    std::ifstream input(path);
    if (!input) throw std::runtime_error("cannot open run list: " + path.string());
    std::set<int> unique;
    std::string line;
    while (std::getline(input, line)) {
        const auto comment = line.find('#');
        if (comment != std::string::npos) line.erase(comment);
        std::replace(line.begin(), line.end(), ',', ' ');
        std::istringstream fields(line);
        std::string token;
        while (fields >> token) {
            const auto dash = token.find('-');
            if (dash == std::string::npos) {
                unique.insert(std::stoi(token));
                continue;
            }
            const int first = std::stoi(token.substr(0, dash));
            const int last = std::stoi(token.substr(dash + 1));
            if (last < first) throw std::runtime_error("descending run range: " + token);
            for (int run = first; run <= last; ++run) unique.insert(run);
        }
    }
    if (unique.empty()) throw std::runtime_error("run list is empty: " + path.string());
    return {unique.begin(), unique.end()};
}

int main(int argc, char** argv) {
    try {
        const Options options = parseOptions(argc, argv);
        const Config config(options.config.string());
        if (!config.qadb.enabled) {
            throw std::runtime_error("processing configuration has qadb.enabled=false");
        }
        const auto runs = readRuns(options.runList);
        fs::create_directories(options.outputDir);

        QA::QADB qadb(config.qadb.database.c_str());
        for (const auto& defect : config.qadb.rejectDefects) {
            qadb.CheckForDefect(defect.c_str());
        }
        for (const int run : config.qadb.allowMiscRuns) qadb.AllowMiscBit(run);

        std::ofstream intervals(options.outputDir / "helicity_sign_intervals.tsv");
        std::ofstream summaries(options.outputDir / "run_helicity_charge.tsv");
        if (!intervals || !summaries) throw std::runtime_error("cannot create audit tables");
        intervals << "run\tbin\tevent_min\tevent_max\tqadb_pass\thelicity_sign\n";
        summaries
            << "run\tqadb_bins\taccepted_bins\ttotal_charge_nC\t"
            << "raw_plus_charge_nC\traw_minus_charge_nC\traw_zero_charge_nC\t"
            << "sign_plus_bins\tsign_minus_bins\tsign_unknown_bins\trun_sign\t"
            << "physical_plus_charge_nC\tphysical_minus_charge_nC\t"
            << "physical_zero_charge_nC\n";
        intervals << std::setprecision(17);
        summaries << std::setprecision(17);

        int missingRuns = 0;
        int usableRuns = 0;
        int unknownSignRuns = 0;
        double physicalPlusTotal = 0.0;
        double physicalMinusTotal = 0.0;
        double acceptedChargeTotal = 0.0;

        for (const int run : runs) {
            const std::string runKey = std::to_string(run);
            if (!qadb.GetQaTree()->HasMember(runKey.c_str())) {
                ++missingRuns;
                summaries << run << "\t0\t0\t0\t0\t0\t0\t0\t0\t0\t0\tnan\tnan\tnan\n";
                continue;
            }
            const double chargeBefore = qadb.GetAccumulatedCharge();
            const double plusBefore = qadb.GetAccumulatedChargeHL(+1);
            const double minusBefore = qadb.GetAccumulatedChargeHL(-1);
            const double zeroBefore = qadb.GetAccumulatedChargeHL(0);
            int bins = 0;
            int acceptedBins = 0;
            int signPlusBins = 0;
            int signMinusBins = 0;
            int signUnknownBins = 0;
            const int maximumBin = qadb.GetMaxBinnum(run);
            for (int bin = 0; bin <= maximumBin; ++bin) {
                if (!qadb.HasBinnum(run, bin) || !qadb.QueryByBinnum(run, bin)) continue;
                ++bins;
                const int eventMin = qadb.GetEvnumMin();
                const int eventMax = qadb.GetEvnumMax();
                const bool pass = qadb.Pass(run, eventMin);
                const int sign = qadb.CorrectHelicitySign(run, eventMin);
                intervals << run << '\t' << bin << '\t' << eventMin << '\t'
                          << eventMax << '\t' << (pass ? 1 : 0) << '\t' << sign << '\n';
                if (!pass) continue;
                ++acceptedBins;
                if (sign > 0) ++signPlusBins;
                else if (sign < 0) ++signMinusBins;
                else ++signUnknownBins;
                qadb.AccumulateCharge();
                qadb.AccumulateChargeHL();
            }
            const double totalCharge = qadb.GetAccumulatedCharge() - chargeBefore;
            const double rawPlus = qadb.GetAccumulatedChargeHL(+1) - plusBefore;
            const double rawMinus = qadb.GetAccumulatedChargeHL(-1) - minusBefore;
            const double rawZero = qadb.GetAccumulatedChargeHL(0) - zeroBefore;
            int runSign = 0;
            if (signUnknownBins == 0 && signPlusBins > 0 && signMinusBins == 0) runSign = +1;
            if (signUnknownBins == 0 && signMinusBins > 0 && signPlusBins == 0) runSign = -1;
            double physicalPlus = std::numeric_limits<double>::quiet_NaN();
            double physicalMinus = std::numeric_limits<double>::quiet_NaN();
            double physicalZero = std::numeric_limits<double>::quiet_NaN();
            if (runSign != 0 && rawPlus >= 0.0 && rawMinus >= 0.0 && rawZero >= 0.0) {
                physicalPlus = runSign > 0 ? rawPlus : rawMinus;
                physicalMinus = runSign > 0 ? rawMinus : rawPlus;
                physicalZero = rawZero;
                ++usableRuns;
                physicalPlusTotal += physicalPlus;
                physicalMinusTotal += physicalMinus;
                acceptedChargeTotal += totalCharge;
            } else {
                ++unknownSignRuns;
            }
            summaries << run << '\t' << bins << '\t' << acceptedBins << '\t'
                      << totalCharge << '\t' << rawPlus << '\t' << rawMinus << '\t'
                      << rawZero << '\t' << signPlusBins << '\t' << signMinusBins << '\t'
                      << signUnknownBins << '\t' << runSign << '\t' << physicalPlus << '\t'
                      << physicalMinus << '\t' << physicalZero << '\n';
        }

        nlohmann::json summary = {
            {"schema_version", 1},
            {"method", "QADB accepted-bin helicity-sign and helicity-latched charge audit"},
            {"processing_config", fs::absolute(options.config).lexically_normal().string()},
            {"run_list", fs::absolute(options.runList).lexically_normal().string()},
            {"qadb_database", config.qadb.database},
            {"qadb_reject_defects", config.qadb.rejectDefects},
            {"qadb_allow_misc_runs", config.qadb.allowMiscRuns},
            {"runs_requested", runs.size()},
            {"runs_missing_from_qadb", missingRuns},
            {"runs_with_usable_physical_helicity_charge", usableRuns},
            {"runs_with_unknown_or_mixed_sign", unknownSignRuns},
            {"physical_plus_charge_nC", physicalPlusTotal},
            {"physical_minus_charge_nC", physicalMinusTotal},
            {"accepted_total_charge_nC", acceptedChargeTotal},
        };
        std::ofstream jsonOutput(options.outputDir / "audit_summary.json");
        jsonOutput << std::setw(2) << summary << '\n';
        std::cout << "Runs requested: " << runs.size() << '\n';
        std::cout << "Usable runs: " << usableRuns << '\n';
        std::cout << "Missing QADB runs: " << missingRuns << '\n';
        std::cout << "Unknown/mixed-sign runs: " << unknownSignRuns << '\n';
        std::cout << "Physical + charge: " << physicalPlusTotal << " nC\n";
        std::cout << "Physical - charge: " << physicalMinusTotal << " nC\n";
        std::cout << "Wrote " << fs::absolute(options.outputDir) << '\n';
        return (missingRuns == 0 && unknownSignRuns == 0) ? 0 : 2;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
