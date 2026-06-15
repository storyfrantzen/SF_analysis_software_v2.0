#include "KinematicCorrections.h"

#include <cmath>
#include <stdexcept>

namespace {
constexpr double kPi = 3.14159265358979323846;

std::string detectorSuffix(int detector) {
    if (detector == 1) return "FD";
    if (detector == 2) return "CD";
    return "";
}
}

KinematicCorrections::KinematicCorrections(const nlohmann::json& corrections) {
    if (corrections.is_null() || corrections.empty()) return;

    for (auto it = corrections.begin(); it != corrections.end(); ++it) {
        CorrEntry entry;
        entry.func = makeProfileFunc(it.value().at("form").get<std::string>());
        entry.coeffsJson = it.value().at("coeffs");
        corrections_[it.key()] = entry;
    }

    enabled_ = !corrections_.empty();
}

KinematicCorrections::CorrFunc
KinematicCorrections::makeProfileFunc(const std::string& form) const {
    if (form == "[0] + [1]/p + [2]/(p^2)") {
        return [](const std::vector<double>& c, double p) { return c[0] + c[1] / p + c[2] / (p * p); };
    }
    if (form == "[0] + [1]/p") {
        return [](const std::vector<double>& c, double p) { return c[0] + c[1] / p; };
    }
    if (form == "[0] + [1]*p + [2]*p^2") {
        return [](const std::vector<double>& c, double p) { return c[0] + c[1] * p + c[2] * p * p; };
    }
    if (form == "[0] + [1]*p") {
        return [](const std::vector<double>& c, double p) { return c[0] + c[1] * p; };
    }
    throw std::runtime_error("Unsupported correction form: " + form);
}

double KinematicCorrections::evalPoly(const std::vector<double>& coeffs,
                                      double thetaDeg) const {
    double value = 0.0;
    double thetaPower = 1.0;
    for (double coeff : coeffs) {
        value += coeff * thetaPower;
        thetaPower *= thetaDeg;
    }
    return value;
}

std::vector<double>
KinematicCorrections::getCoeffsFromTheta(const nlohmann::json& coeffsJson,
                                         double thetaDeg) const {
    std::vector<double> coeffs;
    for (auto it = coeffsJson.begin(); it != coeffsJson.end(); ++it) {
        coeffs.push_back(evalPoly(it.value().get<std::vector<double>>(), thetaDeg));
    }
    return coeffs;
}

double KinematicCorrections::evaluate(const std::string& baseKey,
                                      double p,
                                      double thetaRad,
                                      int detector) const {
    if (!enabled_) return 0.0;
    const std::string suffix = detectorSuffix(detector);
    if (suffix.empty()) return 0.0;

    const auto it = corrections_.find(baseKey + "_" + suffix);
    if (it == corrections_.end()) return 0.0;

    const double thetaDeg = thetaRad * 180.0 / kPi;
    const std::vector<double> coeffs = getCoeffsFromTheta(it->second.coeffsJson, thetaDeg);
    return it->second.func(coeffs, p);
}

double KinematicCorrections::deltaP(double p, double thetaRad, int detector) const {
    return evaluate("p_delta_p", p, thetaRad, detector);
}

double KinematicCorrections::deltaTheta(double p, double thetaRad, int detector) const {
    return evaluate("p_delta_theta", p, thetaRad, detector) * kPi / 180.0;
}

double KinematicCorrections::deltaPhi(double p, double thetaRad, int detector) const {
    return evaluate("p_delta_phi", p, thetaRad, detector) * kPi / 180.0;
}
