#include "ProtonEnergyLossCorrections.h"

#include <cmath>
#include <stdexcept>

namespace {
constexpr double kPi = 3.14159265358979323846;

double normalizePhi(double phi) {
    while (phi > kPi) phi -= 2.0 * kPi;
    while (phi < -kPi) phi += 2.0 * kPi;
    return phi;
}
}

ProtonEnergyLossCorrections::ProtonEnergyLossCorrections(const nlohmann::json& corrections) {
    if (corrections.is_null() || corrections.empty()) return;

    fd_.deltaP = parseTerm(corrections, "p_delta_p_FD");
    fd_.deltaTheta = parseTerm(corrections, "p_delta_theta_FD");
    fd_.deltaPhi = parseTerm(corrections, "p_delta_phi_FD");

    cd_.deltaP = parseTerm(corrections, "p_delta_p_CD");
    cd_.deltaTheta = parseTerm(corrections, "p_delta_theta_CD");
    cd_.deltaPhi = parseTerm(corrections, "p_delta_phi_CD");

    enabled_ = fd_.enabled() || cd_.enabled();
}

ProtonEnergyLossCorrections::Form
ProtonEnergyLossCorrections::parseForm(const std::string& form) {
    if (form == "[0] + [1]/p + [2]/(p^2)") return Form::InvP2;
    if (form == "[0] + [1]/p") return Form::InvP;
    if (form == "[0] + [1]*p + [2]*p^2") return Form::PolyP2;
    if (form == "[0] + [1]*p") return Form::PolyP;
    throw std::runtime_error("Unsupported correction form: " + form);
}

ProtonEnergyLossCorrections::CorrectionTerm
ProtonEnergyLossCorrections::parseTerm(const nlohmann::json& corrections,
                                       const std::string& key) {
    CorrectionTerm term;
    if (!corrections.contains(key)) return term;

    const auto& entry = corrections.at(key);
    term.form = parseForm(entry.at("form").get<std::string>());

    const auto& coeffs = entry.at("coeffs");
    for (auto it = coeffs.begin(); it != coeffs.end(); ++it) {
        term.thetaCoeffs.push_back(it.value().get<std::vector<double>>());
    }
    return term;
}

double ProtonEnergyLossCorrections::evalThetaPolynomial(const std::vector<double>& coeffs,
                                                        double thetaDeg) {
    double value = 0.0;
    double thetaPower = 1.0;
    for (double coeff : coeffs) {
        value += coeff * thetaPower;
        thetaPower *= thetaDeg;
    }
    return value;
}

std::vector<double>
ProtonEnergyLossCorrections::coefficientsAtTheta(const CorrectionTerm& term,
                                                 double thetaDeg) {
    std::vector<double> coeffs;
    for (const auto& coeffSet : term.thetaCoeffs) {
        coeffs.push_back(evalThetaPolynomial(coeffSet, thetaDeg));
    }
    return coeffs;
}

double ProtonEnergyLossCorrections::evaluateTerm(const CorrectionTerm& term,
                                                 double p,
                                                 double thetaDeg) {
    if (!term.enabled()) return 0.0;
    const std::vector<double> c = coefficientsAtTheta(term, thetaDeg);

    switch (term.form) {
        case Form::InvP2:
            return c[0] + c[1] / p + c[2] / (p * p);
        case Form::InvP:
            return c[0] + c[1] / p;
        case Form::PolyP2:
            return c[0] + c[1] * p + c[2] * p * p;
        case Form::PolyP:
            return c[0] + c[1] * p;
        case Form::None:
            return 0.0;
    }

    return 0.0;
}

const ProtonEnergyLossCorrections::DetectorCorrectionSet*
ProtonEnergyLossCorrections::setForDetector(int detector) const {
    if (detector == 1) return &fd_;
    if (detector == 2) return &cd_;
    return nullptr;
}

CorrectedKinematics ProtonEnergyLossCorrections::correct(double p,
                                                         double thetaRad,
                                                         double phiRad,
                                                         int detector) const {
    CorrectedKinematics corrected;
    corrected.p = p;
    corrected.theta = thetaRad;
    corrected.phi = phiRad;

    if (!enabled_) return corrected;
    const DetectorCorrectionSet* set = setForDetector(detector);
    if (!set) return corrected;

    const double thetaDeg = thetaRad * 180.0 / kPi;
    corrected.deltaP = evaluateTerm(set->deltaP, p, thetaDeg);
    corrected.deltaTheta = evaluateTerm(set->deltaTheta, p, thetaDeg) * kPi / 180.0;
    corrected.deltaPhi = evaluateTerm(set->deltaPhi, p, thetaDeg) * kPi / 180.0;

    corrected.p += corrected.deltaP;
    corrected.theta += corrected.deltaTheta;
    corrected.phi = normalizePhi(corrected.phi + corrected.deltaPhi);
    return corrected;
}
