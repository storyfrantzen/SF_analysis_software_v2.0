#include "ElasticMomentumCorrections.h"

#include <cmath>
#include <stdexcept>
#include <string>
#include <utility>

namespace {
constexpr double kPi = 3.14159265358979323846;

double finiteNumber(const nlohmann::json& object, const char* key) {
    const double value = object.at(key).get<double>();
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string("Non-finite elastic momentum parameter: ") + key);
    }
    return value;
}

std::pair<double, double> finiteRange(const nlohmann::json& object, const char* key) {
    const auto values = object.at(key).get<std::vector<double>>();
    if (values.size() != 2 || !std::isfinite(values[0]) || !std::isfinite(values[1]) ||
        values[1] <= values[0]) {
        throw std::runtime_error(std::string("Invalid elastic momentum range: ") + key);
    }
    return {values[0], values[1]};
}
}

ElasticMomentumCorrections::ElasticMomentumCorrections(
    const nlohmann::json& corrections) {
    if (corrections.is_null() || corrections.empty()) return;
    if (corrections.value("schema", std::string{}) != "elastic_momentum_correction/v1") {
        throw std::runtime_error(
            "elasticMomentumCorrections must use schema elastic_momentum_correction/v1"
        );
    }
    if (corrections.value("correctionType", std::string{}) != "fractionalMomentum") {
        throw std::runtime_error(
            "elasticMomentumCorrections correctionType must be fractionalMomentum"
        );
    }
    if (!corrections.contains("regions") || !corrections.at("regions").is_array()) {
        throw std::runtime_error("elasticMomentumCorrections requires a regions array");
    }
    beamEnergyGeV_ = finiteNumber(corrections, "beamEnergyGeV");
    torus_ = corrections.at("torus").get<int>();
    if (beamEnergyGeV_ <= 0.0 || (torus_ != -1 && torus_ != 1)) {
        throw std::runtime_error(
            "elasticMomentumCorrections requires positive beamEnergyGeV and torus +/-1"
        );
    }
    for (const auto& entry : corrections.at("regions")) {
        regions_.push_back(parseRegion(entry));
    }
}

ElasticMomentumCorrections::Region
ElasticMomentumCorrections::parseRegion(const nlohmann::json& entry) {
    Region region;
    region.pid = entry.at("pid").get<int>();
    region.detector = entry.at("detector").get<int>();
    region.sector = entry.value("sector", 0);
    if (region.pid == 0 || region.detector < 0 || region.sector < 0 || region.sector > 6) {
        throw std::runtime_error("Invalid pid, detector, or sector in elastic momentum region");
    }

    const auto thetaRange = finiteRange(entry, "thetaRangeDeg");
    const auto phiRange = finiteRange(entry, "phiRangeDeg");
    region.thetaMinDeg = thetaRange.first;
    region.thetaMaxDeg = thetaRange.second;
    region.phiMinDeg = phiRange.first;
    region.phiMaxDeg = phiRange.second;
    region.thetaCenterDeg = finiteNumber(entry, "thetaCenterDeg");
    region.thetaScaleDeg = finiteNumber(entry, "thetaScaleDeg");
    region.phiCenterDeg = finiteNumber(entry, "phiCenterDeg");
    region.phiScaleDeg = finiteNumber(entry, "phiScaleDeg");
    if (region.thetaScaleDeg <= 0.0 || region.phiScaleDeg <= 0.0) {
        throw std::runtime_error("Elastic momentum normalization scales must be positive");
    }

    const std::string phiVariable = entry.at("phiVariable").get<std::string>();
    if (phiVariable == "sectorLocal") {
        region.phiVariable = PhiVariable::SectorLocal;
        if (region.sector == 0) {
            throw std::runtime_error("sectorLocal elastic momentum regions require sector 1-6");
        }
    } else if (phiVariable == "global") {
        region.phiVariable = PhiVariable::Global;
    } else {
        throw std::runtime_error("Unsupported elastic momentum phiVariable: " + phiVariable);
    }

    const std::string basis = entry.at("basis").get<std::string>();
    if (basis == "polynomial") region.basis = Basis::Polynomial;
    else if (basis == "fourier") region.basis = Basis::Fourier;
    else throw std::runtime_error("Unsupported elastic momentum basis: " + basis);

    if (!entry.contains("terms") || !entry.at("terms").is_array() ||
        entry.at("terms").empty()) {
        throw std::runtime_error("Elastic momentum region requires nonempty terms");
    }
    for (const auto& termEntry : entry.at("terms")) {
        Term term;
        term.thetaPower = termEntry.value("thetaPower", 0);
        term.coefficient = finiteNumber(termEntry, "coefficient");
        if (term.thetaPower < 0) {
            throw std::runtime_error("Elastic momentum theta powers must be nonnegative");
        }
        if (region.basis == Basis::Polynomial) {
            term.phiPower = termEntry.value("phiPower", 0);
            if (term.phiPower < 0) {
                throw std::runtime_error("Elastic momentum phi powers must be nonnegative");
            }
        } else {
            const std::string component = termEntry.value("component", "constant");
            term.harmonic = termEntry.value("harmonic", 0);
            if (component == "constant") {
                term.component = FourierComponent::Constant;
                term.harmonic = 0;
            } else if (component == "cos") {
                term.component = FourierComponent::Cosine;
            } else if (component == "sin") {
                term.component = FourierComponent::Sine;
            } else {
                throw std::runtime_error("Unsupported elastic momentum Fourier component: " +
                                         component);
            }
            if (term.harmonic < 0 ||
                (term.component != FourierComponent::Constant && term.harmonic == 0)) {
                throw std::runtime_error("Invalid elastic momentum Fourier harmonic");
            }
        }
        region.terms.push_back(term);
    }
    return region;
}

double ElasticMomentumCorrections::normalizeDegrees(double degrees) {
    while (degrees > 180.0) degrees -= 360.0;
    while (degrees <= -180.0) degrees += 360.0;
    return degrees;
}

double ElasticMomentumCorrections::sectorLocalPhiDegrees(double phiDeg, int sector) {
    return normalizeDegrees(phiDeg - 60.0 * static_cast<double>(sector - 1));
}

double ElasticMomentumCorrections::evaluate(const Region& region,
                                             double thetaDeg,
                                             double phiDeg) {
    const double theta = (thetaDeg - region.thetaCenterDeg) / region.thetaScaleDeg;
    const double phi = (phiDeg - region.phiCenterDeg) / region.phiScaleDeg;
    double value = 0.0;
    for (const auto& term : region.terms) {
        const double thetaTerm = std::pow(theta, term.thetaPower);
        if (region.basis == Basis::Polynomial) {
            value += term.coefficient * thetaTerm * std::pow(phi, term.phiPower);
            continue;
        }

        double phiTerm = 1.0;
        const double phiRad = phiDeg * kPi / 180.0;
        if (term.component == FourierComponent::Cosine) {
            phiTerm = std::cos(static_cast<double>(term.harmonic) * phiRad);
        } else if (term.component == FourierComponent::Sine) {
            phiTerm = std::sin(static_cast<double>(term.harmonic) * phiRad);
        }
        value += term.coefficient * thetaTerm * phiTerm;
    }
    return value;
}

MomentumCorrectionResult ElasticMomentumCorrections::correct(double p,
                                                              double thetaRad,
                                                              double phiRad,
                                                              int pid,
                                                              int detector,
                                                              int sector) const {
    MomentumCorrectionResult result{p, 0.0};
    if (!std::isfinite(p) || p <= 0.0 || !std::isfinite(thetaRad) ||
        !std::isfinite(phiRad)) {
        return result;
    }

    const double thetaDeg = thetaRad * 180.0 / kPi;
    const double globalPhiDeg = normalizeDegrees(phiRad * 180.0 / kPi);
    for (const auto& region : regions_) {
        if (region.pid != pid || region.detector != detector) continue;
        if (region.sector != 0 && region.sector != sector) continue;

        const double phiDeg = region.phiVariable == PhiVariable::SectorLocal
            ? sectorLocalPhiDegrees(globalPhiDeg, sector)
            : globalPhiDeg;
        if (thetaDeg < region.thetaMinDeg || thetaDeg > region.thetaMaxDeg ||
            phiDeg < region.phiMinDeg || phiDeg > region.phiMaxDeg) {
            continue;
        }

        const double fractionalCorrection = evaluate(region, thetaDeg, phiDeg);
        const double correctedP = p * (1.0 + fractionalCorrection);
        if (!std::isfinite(correctedP) || correctedP <= 0.0) return result;
        result.p = correctedP;
        result.deltaP = correctedP - p;
        return result;
    }
    return result;
}
