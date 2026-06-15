#pragma once

#include <functional>
#include <map>
#include <string>
#include <vector>

#include "nlohmann/json.hpp"

class KinematicCorrections {
public:
    KinematicCorrections() = default;
    explicit KinematicCorrections(const nlohmann::json& corrections);

    bool enabled() const { return enabled_; }

    double deltaP(double p, double thetaRad, int detector) const;
    double deltaTheta(double p, double thetaRad, int detector) const;
    double deltaPhi(double p, double thetaRad, int detector) const;

private:
    using CorrFunc = std::function<double(const std::vector<double>&, double)>;

    struct CorrEntry {
        CorrFunc func;
        nlohmann::json coeffsJson;
    };

    bool enabled_ = false;
    std::map<std::string, CorrEntry> corrections_;

    CorrFunc makeProfileFunc(const std::string& form) const;
    double evalPoly(const std::vector<double>& coeffs, double thetaDeg) const;
    std::vector<double> getCoeffsFromTheta(const nlohmann::json& coeffsJson,
                                           double thetaDeg) const;
    double evaluate(const std::string& baseKey,
                    double p,
                    double thetaRad,
                    int detector) const;
};
