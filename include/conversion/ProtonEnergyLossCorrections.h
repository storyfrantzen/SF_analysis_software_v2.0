#pragma once

#include <string>
#include <vector>

#include "nlohmann/json.hpp"

struct CorrectedKinematics {
    double p = 0.0;
    double theta = 0.0;
    double phi = 0.0;
    double deltaP = 0.0;
    double deltaTheta = 0.0;
    double deltaPhi = 0.0;
};

class ProtonEnergyLossCorrections {
public:
    ProtonEnergyLossCorrections() = default;
    explicit ProtonEnergyLossCorrections(const nlohmann::json& corrections);

    bool enabled() const { return enabled_; }

    CorrectedKinematics correct(double p, double thetaRad, double phiRad, int detector) const;

private:
    struct CorrectionTerm {
        std::vector<int> momentumPowers;
        std::vector<std::vector<double>> thetaCoeffs;

        bool enabled() const {
            return !momentumPowers.empty() && momentumPowers.size() == thetaCoeffs.size();
        }
    };

    struct DetectorCorrectionSet {
        CorrectionTerm deltaP;
        CorrectionTerm deltaTheta;
        CorrectionTerm deltaPhi;

        bool enabled() const {
            return deltaP.enabled() || deltaTheta.enabled() || deltaPhi.enabled();
        }
    };

    bool enabled_ = false;
    DetectorCorrectionSet fd_;
    DetectorCorrectionSet cd_;

    static std::vector<int> parseLegacyForm(const std::string& form);
    static CorrectionTerm parseTerm(const nlohmann::json& corrections,
                                    const std::string& key);
    static double evalThetaPolynomial(const std::vector<double>& coeffs,
                                      double thetaDeg);
    static std::vector<double> coefficientsAtTheta(const CorrectionTerm& term,
                                                   double thetaDeg);
    static double evaluateTerm(const CorrectionTerm& term, double p, double thetaDeg);
    const DetectorCorrectionSet* setForDetector(int detector) const;
};
