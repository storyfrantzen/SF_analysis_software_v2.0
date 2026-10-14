#include <cmath>
#include <iostream>
#include <stdexcept>

#include "ElasticMomentumCorrections.h"

namespace {
constexpr double kPi = 3.14159265358979323846;

nlohmann::json sampleParameters() {
    return {
        {"schema", "elastic_momentum_correction/v1"},
        {"correctionType", "fractionalMomentum"},
        {"beamEnergyGeV", 6.535},
        {"torus", 1},
        {"regions", {
            {
                {"pid", 11},
                {"detector", 1},
                {"sector", 2},
                {"thetaRangeDeg", {10.0, 40.0}},
                {"phiRangeDeg", {-30.0, 30.0}},
                {"thetaCenterDeg", 25.0},
                {"thetaScaleDeg", 15.0},
                {"phiCenterDeg", 0.0},
                {"phiScaleDeg", 30.0},
                {"phiVariable", "sectorLocal"},
                {"basis", "polynomial"},
                {"supportCells", {
                    {
                        {"thetaRangeDeg", {10.0, 30.0}},
                        {"phiRangeDeg", {-30.0, 15.0}}
                    }
                }},
                {"terms", {
                    {{"thetaPower", 0}, {"phiPower", 0}, {"coefficient", 0.01}},
                    {{"thetaPower", 1}, {"phiPower", 0}, {"coefficient", 0.02}},
                    {{"thetaPower", 0}, {"phiPower", 1}, {"coefficient", -0.03}}
                }}
            },
            {
                {"pid", 2212},
                {"detector", 2},
                {"sector", 0},
                {"thetaRangeDeg", {35.0, 80.0}},
                {"phiRangeDeg", {-180.0, 180.0}},
                {"thetaCenterDeg", 57.5},
                {"thetaScaleDeg", 22.5},
                {"phiCenterDeg", 0.0},
                {"phiScaleDeg", 180.0},
                {"phiVariable", "global"},
                {"basis", "fourier"},
                {"terms", {
                    {{"thetaPower", 0}, {"component", "constant"}, {"harmonic", 0},
                     {"coefficient", 0.01}},
                    {{"thetaPower", 0}, {"component", "cos"}, {"harmonic", 2},
                     {"coefficient", 0.02}}
                }}
            }
        }}
    };
}

nlohmann::json momentumDependentParameters() {
    return {
        {"schema", "particle_momentum_correction/v2"},
        {"correctionType", "fractionalMomentum"},
        {"beamEnergyGeV", 6.535},
        {"torus", 1},
        {"regions", {{
            {"pid", 2212},
            {"detector", 2},
            {"sector", 0},
            {"momentumRangeGeV", {0.5, 1.5}},
            {"momentumCenterGeV", 1.0},
            {"momentumScaleGeV", 0.5},
            {"thetaRangeDeg", {35.0, 65.0}},
            {"phiRangeDeg", {-180.0, 180.0}},
            {"thetaCenterDeg", 50.0},
            {"thetaScaleDeg", 15.0},
            {"phiCenterDeg", 0.0},
            {"phiScaleDeg", 180.0},
            {"phiVariable", "global"},
            {"basis", "polynomial"},
            {"supportCells", {{
                {"momentumRangeGeV", {0.75, 1.25}},
                {"thetaRangeDeg", {40.0, 60.0}},
                {"phiRangeDeg", {-180.0, 180.0}}
            }}},
            {"terms", {
                {{"momentumPower", 0}, {"thetaPower", 0}, {"phiPower", 0},
                 {"coefficient", 0.01}},
                {{"momentumPower", 1}, {"thetaPower", 0}, {"phiPower", 0},
                 {"coefficient", 0.02}},
                {{"momentumPower", 0}, {"thetaPower", 1}, {"phiPower", 0},
                 {"coefficient", -0.03}},
                {{"momentumPower", 1}, {"thetaPower", 1}, {"phiPower", 0},
                 {"coefficient", 0.04}}
            }}
        }}}
    };
}
}

int main() {
    const ElasticMomentumCorrections corrections(sampleParameters());
    if (!corrections.enabled()) {
        std::cerr << "valid correction parameters were not enabled\n";
        return 1;
    }
    if (std::abs(corrections.beamEnergyGeV() - 6.535) > 1.0e-12 ||
        corrections.torus() != 1) {
        std::cerr << "correction provenance was not retained\n";
        return 1;
    }

    // Sector 2 is centered at 60 degrees. At theta=25 and local phi=10,
    // the fitted fractional correction is 0.01 - 0.03/3 = 0.
    const auto electron = corrections.correct(
        4.0, 25.0 * kPi / 180.0, 70.0 * kPi / 180.0, 11, 1, 2
    );
    if (std::abs(electron.p - 4.0) > 1.0e-12) {
        std::cerr << "sector-local polynomial correction evaluated incorrectly\n";
        return 1;
    }
    if (!electron.applied) {
        std::cerr << "an in-support zero correction was not marked applied\n";
        return 1;
    }

    // At global phi=0, cos(2 phi)=1 and the correction is 3%.
    const auto proton = corrections.correct(
        1.0, 50.0 * kPi / 180.0, 0.0, 2212, 2, 0
    );
    if (std::abs(proton.p - 1.03) > 1.0e-12) {
        std::cerr << "global Fourier correction evaluated incorrectly\n";
        return 1;
    }

    const auto outside = corrections.correct(
        4.0, 45.0 * kPi / 180.0, 70.0 * kPi / 180.0, 11, 1, 2
    );
    if (outside.deltaP != 0.0) {
        std::cerr << "correction extrapolated beyond the calibrated theta range\n";
        return 1;
    }

    const auto unsupportedHole = corrections.correct(
        4.0, 25.0 * kPi / 180.0, 80.0 * kPi / 180.0, 11, 1, 2
    );
    if (unsupportedHole.deltaP != 0.0) {
        std::cerr << "correction interpolated through an unsupported profile cell\n";
        return 1;
    }
    if (unsupportedHole.applied) {
        std::cerr << "an unsupported correction was marked applied\n";
        return 1;
    }

    const ElasticMomentumCorrections momentumDependent(
        momentumDependentParameters()
    );
    // At p=1.25 GeV and theta=57.5 degrees, both normalized coordinates
    // are 0.5, so f = .01 + .02*.5 - .03*.5 + .04*.25 = .015.
    const auto bilinear = momentumDependent.correct(
        1.25, 57.5 * kPi / 180.0, 0.0, 2212, 2, 0
    );
    if (!bilinear.applied || std::abs(bilinear.p - 1.26875) > 1.0e-12) {
        std::cerr << "momentum-theta bilinear correction evaluated incorrectly\n";
        return 1;
    }
    const auto outsideMomentumCell = momentumDependent.correct(
        1.4, 50.0 * kPi / 180.0, 0.0, 2212, 2, 0
    );
    if (outsideMomentumCell.applied || outsideMomentumCell.deltaP != 0.0) {
        std::cerr << "correction extrapolated outside momentum-cell support\n";
        return 1;
    }

    bool rejected = false;
    try {
        nlohmann::json invalid = sampleParameters();
        invalid["schema"] = "unknown";
        const ElasticMomentumCorrections bad(invalid);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "invalid correction schema was accepted\n";
        return 1;
    }
    rejected = false;
    try {
        nlohmann::json overlapping = sampleParameters();
        auto duplicate = overlapping["regions"][0];
        duplicate["sector"] = 0;
        duplicate["phiVariable"] = "global";
        overlapping["regions"].push_back(duplicate);
        const ElasticMomentumCorrections bad(overlapping);
    } catch (const std::runtime_error&) {
        rejected = true;
    }
    if (!rejected) {
        std::cerr << "overlapping wildcard correction regions were accepted\n";
        return 1;
    }
    return 0;
}
