#pragma once

#include <array>
#include <map>
#include <string>
#include <utility>
#include <vector>

#include "ROOTBranches.h"
#include "json.hpp"

struct CutDecision {
    bool pass = true;
    std::vector<std::string> failed;

    void require(bool condition, const std::string& name);
    void merge(const CutDecision& other);
    std::string failedCsv() const;
};

struct ExclusivityVars {
    double missingEnergy = NAN;
    double thetaEG1Deg = NAN;
    double thetaEG2Deg = NAN;
    double thetaG1G2Deg = NAN;
    double pi0ConeAngleDeg = NAN;
};

struct PrimitiveCutSpec {
    std::string name;
    std::string op;
    double min = NAN;
    double max = NAN;
    int detector = -999;
    std::string refRole;
};

struct ParticleRoleSpec {
    std::string role;
    int pid = -999;
    int count = 1;
    std::vector<PrimitiveCutSpec> cuts;
};

struct CompositeSpec {
    std::string role;
    std::string type;
    std::vector<std::string> daughters;
    double mass = NAN;
    double window = NAN;
};

struct ChannelSpec {
    std::string name = "eppi0";
    std::vector<ParticleRoleSpec> particles;
    std::vector<CompositeSpec> composites;

    const ParticleRoleSpec* findRole(const std::string& role) const;
    const CompositeSpec* findComposite(const std::string& role) const;
};

struct PostCutConfig {
    std::string outputFile = "post_processed.root";
    std::string inputTree = "Events";
    std::string outputTree = "Events";

    double beamEnergy = 10.6;
    int torus = -1;
    bool saveFailedCandidates = false;

    std::vector<std::string> fiducialCuts;

    bool sfEnabled = false;
    double sfNumSigma = 3.5;
    double sfTriangleYScale = 1.0;
    double sfTriangleXScale = 1.0;
    double sfTriangleHypotenuse = 0.2;
    double sfHtccThreshold = 4.5;
    nlohmann::json sfParams;

    double maxAbsMissingEnergy = 2.0;
    double minElectronPhotonAngleDeg = 4.0;
    double minPhotonPhotonAngleDeg = 1.0;
    double maxPi0ConeAngleDeg = 4.0;

    ChannelSpec channel;

    static PostCutConfig fromFile(const std::string& filename);
};

class Cuts {
public:
    explicit Cuts(PostCutConfig cfg);

    const PostCutConfig& config() const { return cfg_; }

    CutDecision evaluateParticle(const RecBranches& p,
                                 const ParticleRoleSpec& role,
                                 const std::map<std::string, const RecBranches*>& selected) const;

    CutDecision evaluateFiducial(const RecBranches& p) const;
    CutDecision evaluateSamplingFraction(const RecBranches& e) const;
    CutDecision evaluateLooseExclusivity(const ExclusivityVars& vars) const;

    bool passesFiducial(const RecBranches& p) const;
    bool passesSamplingFraction(const RecBranches& e) const;

    bool passesLooseExclusivity(double missingEnergy,
                                double thetaEG1Deg,
                                double thetaEG2Deg,
                                double thetaG1G2Deg,
                                double pi0ConeAngleDeg) const;

private:
    struct Hole { double x, y, radius; };
    struct SectorCoeffs {
        std::vector<double> muCoeffs;
        std::vector<double> sigmaCoeffs;
    };

    PostCutConfig cfg_;
    std::vector<Hole> ftHoles_;
    std::map<std::string, std::vector<std::pair<double, double>>> rgaEcalExclusionMap_;
    std::map<std::string, std::vector<std::pair<double, double>>> rgkEcalExclusionMap_;
    std::array<std::array<double, 6>, 9> rgkEdgeParams_;
    std::map<int, SectorCoeffs> sfCoeffs_;

    bool dcEdgesRGA_ = false;
    bool ftRGA_ = false;
    bool ecalRGA_ = false;
    bool ecalRGK_ = false;
    bool ecalEdgesRGK_ = false;
    bool cvtRGA_ = false;

    void enableFiducialTag(const std::string& tag);
    void loadSamplingFractionParams(const nlohmann::json& params);

    bool passesDC(const RecBranches& p) const;
    bool passesFT(const RecBranches& p) const;
    bool passesECAL(const RecBranches& p) const;
    bool passesCVT(const RecBranches& p) const;

    bool passesCVTPhiVeto(const RecBranches& p, double minDeg, double maxDeg) const;
    bool inFTHole(double x, double y) const;
    bool inExcludedECALRegion(int detector, int sector, char coord, double value) const;
    std::pair<double, double> rotateToSector1Frame(double x, double y, int sector) const;

    double evalPoly(const std::vector<double>& coeffs, double p) const;
    double sfMu(int sector, double p) const;
    double sfSigma(int sector, double p) const;
    bool passSFSigmaCut(int sector, double sf, double p) const;
    bool passSFTriangleCut(double ePCAL, double eECIN, double p) const;
};
