#pragma once

#include <vector>

#include "nlohmann/json.hpp"

struct MomentumCorrectionResult {
    double p = 0.0;
    double deltaP = 0.0;
};

class ElasticMomentumCorrections {
public:
    ElasticMomentumCorrections() = default;
    explicit ElasticMomentumCorrections(const nlohmann::json& corrections);

    bool enabled() const { return !regions_.empty(); }
    double beamEnergyGeV() const { return beamEnergyGeV_; }
    int torus() const { return torus_; }

    MomentumCorrectionResult correct(double p,
                                     double thetaRad,
                                     double phiRad,
                                     int pid,
                                     int detector,
                                     int sector) const;

private:
    enum class PhiVariable { SectorLocal, Global };
    enum class Basis { Polynomial, Fourier };
    enum class FourierComponent { Constant, Cosine, Sine };

    struct Term {
        int thetaPower = 0;
        int phiPower = 0;
        int harmonic = 0;
        double coefficient = 0.0;
        FourierComponent component = FourierComponent::Constant;
    };

    struct Region {
        int pid = 0;
        int detector = -1;
        int sector = 0;  // zero is a detector-wide wildcard
        double thetaMinDeg = 0.0;
        double thetaMaxDeg = 0.0;
        double phiMinDeg = 0.0;
        double phiMaxDeg = 0.0;
        double thetaCenterDeg = 0.0;
        double thetaScaleDeg = 1.0;
        double phiCenterDeg = 0.0;
        double phiScaleDeg = 1.0;
        PhiVariable phiVariable = PhiVariable::Global;
        Basis basis = Basis::Polynomial;
        std::vector<Term> terms;
    };

    std::vector<Region> regions_;
    double beamEnergyGeV_ = 0.0;
    int torus_ = 0;

    static Region parseRegion(const nlohmann::json& entry);
    static double normalizeDegrees(double degrees);
    static double sectorLocalPhiDegrees(double phiDeg, int sector);
    static double evaluate(const Region& region, double thetaDeg, double phiDeg);
};
