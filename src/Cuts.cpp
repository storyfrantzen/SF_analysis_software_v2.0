#include "Cuts.h"

#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>

using json = nlohmann::json;

namespace {
constexpr double kPi = 3.14159265358979323846;

bool isFinite(double value) {
    return std::isfinite(value);
}
}

PostCutConfig PostCutConfig::fromFile(const std::string& filename) {
    std::ifstream f(filename);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open post-processing config file: " + filename);
    }

    json j;
    f >> j;

    PostCutConfig cfg;
    cfg.outputFile = j.value("outputFile", cfg.outputFile);
    cfg.inputTree = j.value("inputTree", cfg.inputTree);
    cfg.outputTree = j.value("outputTree", cfg.outputTree);
    cfg.beamEnergy = j.value("beamEnergy", cfg.beamEnergy);
    cfg.torus = j.value("torus", cfg.torus);
    cfg.saveFailedCandidates = j.value("saveFailedCandidates", cfg.saveFailedCandidates);
    cfg.fiducialCuts = j.value("fiducialCuts", cfg.fiducialCuts);

    if (j.contains("samplingFraction")) {
        const auto& sf = j["samplingFraction"];
        cfg.sfEnabled = sf.value("enabled", cfg.sfEnabled);
        cfg.sfNumSigma = sf.value("numSigma", cfg.sfNumSigma);
        cfg.sfTriangleYScale = sf.value("triangleYScale", cfg.sfTriangleYScale);
        cfg.sfTriangleXScale = sf.value("triangleXScale", cfg.sfTriangleXScale);
        cfg.sfTriangleHypotenuse = sf.value("triangleHypotenuse", cfg.sfTriangleHypotenuse);
        cfg.sfHtccThreshold = sf.value("htccThreshold", cfg.sfHtccThreshold);
        if (sf.contains("params")) cfg.sfParams = sf["params"];
    }

    if (j.contains("eppi0")) {
        const auto& eppi0 = j["eppi0"];
        cfg.electronMinP = eppi0.value("electronMinP", cfg.electronMinP);
        cfg.protonMinP = eppi0.value("protonMinP", cfg.protonMinP);
        cfg.photonMinP = eppi0.value("photonMinP", cfg.photonMinP);
        cfg.photonMinBeta = eppi0.value("photonMinBeta", cfg.photonMinBeta);
        cfg.photonMaxBeta = eppi0.value("photonMaxBeta", cfg.photonMaxBeta);
        cfg.photonMinCalEnergy = eppi0.value("photonMinCalEnergy", cfg.photonMinCalEnergy);
        cfg.rejectPhotonsInCD = eppi0.value("rejectPhotonsInCD", cfg.rejectPhotonsInCD);
        cfg.rejectPhotonsInElectronSector = eppi0.value("rejectPhotonsInElectronSector",
                                                       cfg.rejectPhotonsInElectronSector);
        cfg.pi0MassWindow = eppi0.value("pi0MassWindow", cfg.pi0MassWindow);
        cfg.maxAbsMissingEnergy = eppi0.value("maxAbsMissingEnergy", cfg.maxAbsMissingEnergy);
        cfg.minElectronPhotonAngleDeg = eppi0.value("minElectronPhotonAngleDeg",
                                                   cfg.minElectronPhotonAngleDeg);
        cfg.minPhotonPhotonAngleDeg = eppi0.value("minPhotonPhotonAngleDeg",
                                                 cfg.minPhotonPhotonAngleDeg);
        cfg.maxPi0ConeAngleDeg = eppi0.value("maxPi0ConeAngleDeg", cfg.maxPi0ConeAngleDeg);
    }

    return cfg;
}

Cuts::Cuts(PostCutConfig cfg) : cfg_(std::move(cfg)) {
    ftHoles_ = {
        {-8.42, 9.89, 1.60},
        {-9.89, -5.33, 1.60},
        {-6.15, -13.0, 2.30},
        {3.70, -6.50, 2.00}
    };

    rgaEcalExclusionMap_ = {
        {"1_1_w", {{72.0, 94.5}, {220.5, 234.0}}},
        {"1_2_v", {{99.0, 117.0}}},
        {"1_3_w", {{346.5, 378.0}}},
        {"1_4_w", {{0.0, 13.5}}},
        {"1_4_v", {{229.5, 243.0}}},
        {"1_6_w", {{166.5, 193.5}}},
        {"4_1_v", {{67.5, 94.5}}},
        {"4_5_v", {{0.0, 23.5}}},
        {"7_1_v", {{0.0, 40.5}}},
        {"7_5_u", {{193.5, 216.0}}}
    };

    rgkEcalExclusionMap_ = {
        {"1_1_w", {{72.0, 94.5}, {207.0, 234.0}}},
        {"1_2_v", {{99.0, 112.5}}},
        {"1_3_v", {{346.5, 382.5}}},
        {"1_4_v", {{229.5, 243.0}}},
        {"1_6_w", {{171.0, 193.5}}},
        {"2_1_v", {{67.5, 99.0}}}
    };

    rgkEdgeParams_ = {{
        {87, 82, 85, 77, 78, 82},
        {58.7356, 62.8204, 62.2296, 53.7756, 58.2888, 54.5822},
        {58.7477, 51.2589, 59.2357, 56.2415, 60.8219, 49.8914},
        {0.582053, 0.544976, 0.549788, 0.56899, 0.56414, 0.57343},
        {-0.591876, -0.562926, -0.562246, -0.563726, -0.568902, -0.550729},
        {64.9348, 64.7541, 67.832, 55.9324, 55.9225, 60.0997},
        {65.424, 54.6992, 63.6628, 57.8931, 56.5367, 56.4641},
        {0.745578, 0.606081, 0.729202, 0.627239, 0.503674, 0.717899},
        {-0.775022, -0.633863, -0.678901, -0.612458, -0.455319, -0.692481}
    }};

    for (const auto& tag : cfg_.fiducialCuts) enableFiducialTag(tag);
    if (cfg_.sfEnabled) loadSamplingFractionParams(cfg_.sfParams);
}

void Cuts::enableFiducialTag(const std::string& tag) {
    if (tag == "DCEdges_RGA") dcEdgesRGA_ = true;
    else if (tag == "FT_RGA") ftRGA_ = true;
    else if (tag == "ECAL_RGA") ecalRGA_ = true;
    else if (tag == "ECAL_RGAS19") {
        rgaEcalExclusionMap_["1_2_v"].emplace_back(31.5, 49.5);
        ecalRGA_ = true;
    } else if (tag == "ECAL_RGK") ecalRGK_ = true;
    else if (tag == "ECALEdges_RGK") ecalEdgesRGK_ = true;
    else if (tag == "CVT_RGA") cvtRGA_ = true;
    else std::cerr << "[Cuts] Warning: unrecognized fiducial cut tag '" << tag << "'\n";
}

void Cuts::loadSamplingFractionParams(const json& params) {
    sfCoeffs_.clear();
    if (params.is_null() || params.empty()) return;

    for (int sector = 1; sector <= 6; ++sector) {
        const std::string key = "sector_" + std::to_string(sector);
        if (!params.contains(key)) continue;
        SectorCoeffs coeffs;
        coeffs.muCoeffs = params[key]["mu_coeffs"].get<std::vector<double>>();
        coeffs.sigmaCoeffs = params[key]["sigma_coeffs"].get<std::vector<double>>();
        sfCoeffs_[sector] = coeffs;
    }
}

bool Cuts::passesFiducial(const RecBranches& p) const {
    if (p.det == 0) return passesFT(p);
    if (p.det == 1) return passesDC(p) && passesECAL(p);
    if (p.det == 2) return passesCVT(p);
    return true;
}

bool Cuts::passesElectronPreselection(const RecBranches& e) const {
    if (e.pid != 11) return false;
    if (e.p < cfg_.electronMinP) return false;
    if (!passesFiducial(e)) return false;
    return passesSamplingFraction(e);
}

bool Cuts::passesProtonPreselection(const RecBranches& p, double electronVz) const {
    if (p.pid != 2212) return false;
    if (p.p < cfg_.protonMinP) return false;
    if (isFinite(electronVz) && isFinite(p.vz) && std::abs(p.vz - electronVz) > 20.0) return false;
    return passesFiducial(p);
}

bool Cuts::passesPhotonPreselection(const RecBranches& g, int electronSector) const {
    if (g.pid != 22) return false;
    if (g.p < cfg_.photonMinP) return false;
    if (cfg_.rejectPhotonsInCD && g.det == 2) return false;
    if (cfg_.rejectPhotonsInElectronSector && g.sector == electronSector) return false;
    if (g.beta < cfg_.photonMinBeta || g.beta > cfg_.photonMaxBeta) return false;
    if ((g.E_PCAL + g.E_ECIN + g.E_ECOUT) < cfg_.photonMinCalEnergy) return false;
    return passesFiducial(g);
}

bool Cuts::passesLooseExclusivity(double missingEnergy,
                                  double thetaEG1Deg,
                                  double thetaEG2Deg,
                                  double thetaG1G2Deg,
                                  double pi0ConeAngleDeg) const {
    if (std::abs(missingEnergy) > cfg_.maxAbsMissingEnergy) return false;
    if (thetaEG1Deg < cfg_.minElectronPhotonAngleDeg) return false;
    if (thetaEG2Deg < cfg_.minElectronPhotonAngleDeg) return false;
    if (thetaG1G2Deg < cfg_.minPhotonPhotonAngleDeg) return false;
    if (pi0ConeAngleDeg > cfg_.maxPi0ConeAngleDeg) return false;
    return true;
}

bool Cuts::passesDC(const RecBranches& p) const {
    if (!dcEdgesRGA_) return true;
    if (!isFinite(p.edgeDC1) || !isFinite(p.edgeDC2) || !isFinite(p.edgeDC3)) return false;

    const bool outbending = cfg_.torus * p.charge < 0;
    const int absPid = std::abs(p.pid);

    if (absPid == 11) {
        const double thetaDeg = p.theta * 180.0 / kPi;
        if (p.edgeDC3 <= 10) return false;
        if (outbending) return p.edgeDC1 > 3 && p.edgeDC2 > 3;
        return thetaDeg > 10.0 ? (p.edgeDC1 > 3 && p.edgeDC2 > 3)
                               : (p.edgeDC1 > 10 && p.edgeDC2 > 10);
    }

    if (absPid == 2212) {
        if (p.edgeDC3 <= 5) return false;
        if (!outbending) return p.edgeDC1 > 3 && p.edgeDC2 > 3;
    }

    return true;
}

bool Cuts::passesFT(const RecBranches& p) const {
    if (!ftRGA_) return true;
    if (!isFinite(p.xFT) || !isFinite(p.yFT)) return false;
    const double r = std::sqrt(p.xFT * p.xFT + p.yFT * p.yFT);
    if (r < 8.5 || r > 15.5) return false;
    return !inFTHole(p.xFT, p.yFT);
}

bool Cuts::passesECAL(const RecBranches& p) const {
    if (ecalRGA_ || ecalRGK_) {
        if (p.sector < 1 || p.sector > 6) return false;
        if (!isFinite(p.uPCAL) || !isFinite(p.vPCAL) || !isFinite(p.wPCAL)) return false;
        if (!isFinite(p.uECIN) || !isFinite(p.vECIN) || !isFinite(p.wECIN)) return false;
        if (!isFinite(p.uECOUT) || !isFinite(p.vECOUT) || !isFinite(p.wECOUT)) return false;
        if (p.vPCAL < 9.0 || p.wPCAL < 9.0) return false;

        const std::array<int, 3> layers = {1, 4, 7};
        const std::array<double, 3> u = {p.uPCAL, p.uECIN, p.uECOUT};
        const std::array<double, 3> v = {p.vPCAL, p.vECIN, p.vECOUT};
        const std::array<double, 3> w = {p.wPCAL, p.wECIN, p.wECOUT};
        for (size_t i = 0; i < layers.size(); ++i) {
            if (inExcludedECALRegion(layers[i], p.sector, 'u', u[i])) return false;
            if (inExcludedECALRegion(layers[i], p.sector, 'v', v[i])) return false;
            if (inExcludedECALRegion(layers[i], p.sector, 'w', w[i])) return false;
        }
    }

    if (ecalEdgesRGK_) {
        if (p.sector < 1 || p.sector > 6) return false;
        if (!isFinite(p.xPCAL) || !isFinite(p.yPCAL)) return false;

        auto [xLocal, yLocal] = rotateToSector1Frame(p.xPCAL, p.yPCAL, p.sector);
        const size_t s = static_cast<size_t>(p.sector - 1);
        const double pSplit = rgkEdgeParams_[0][s];
        const double tLeft = rgkEdgeParams_[1][s];
        const double tRight = rgkEdgeParams_[2][s];
        const double sLeft = rgkEdgeParams_[3][s];
        const double sRight = rgkEdgeParams_[4][s];
        const double rLeft = rgkEdgeParams_[5][s];
        const double rRight = rgkEdgeParams_[6][s];
        const double qLeft = rgkEdgeParams_[7][s];
        const double qRight = rgkEdgeParams_[8][s];

        const bool highX = xLocal > pSplit &&
                           yLocal < sLeft * (xLocal - tLeft) &&
                           yLocal > sRight * (xLocal - tRight);
        const bool lowX = xLocal < pSplit &&
                          yLocal < qLeft * (xLocal - rLeft) &&
                          yLocal > qRight * (xLocal - rRight);
        return highX || lowX;
    }

    return true;
}

bool Cuts::passesCVT(const RecBranches& p) const {
    if (!cvtRGA_) return true;
    if (p.edge_cvt1 <= 0 || p.edge_cvt3 <= 0 || p.edge_cvt5 <= 0 ||
        p.edge_cvt7 <= 0 || p.edge_cvt12 <= 0) return false;
    if (!isFinite(p.phi_cvt)) return false;

    double phiDeg = p.phi_cvt * 180.0 / kPi;
    if (phiDeg < 0) phiDeg += 360.0;
    if ((25 <= phiDeg && phiDeg <= 40) ||
        (143 <= phiDeg && phiDeg <= 158) ||
        (265 <= phiDeg && phiDeg <= 280)) return false;
    return true;
}

bool Cuts::passesSamplingFraction(const RecBranches& e) const {
    if (!cfg_.sfEnabled || sfCoeffs_.empty() || e.det != 1) return true;
    if (e.E_PCAL <= 0.07) return false;
    const double sf = (e.E_PCAL + e.E_ECIN + e.E_ECOUT) / e.p;
    return passSFTriangleCut(e.E_PCAL, e.E_ECIN, e.p) &&
           passSFSigmaCut(e.sector, sf, e.p);
}

bool Cuts::inFTHole(double x, double y) const {
    for (const auto& hole : ftHoles_) {
        const double dx = x - hole.x;
        const double dy = y - hole.y;
        if (std::sqrt(dx * dx + dy * dy) < hole.radius) return true;
    }
    return false;
}

bool Cuts::inExcludedECALRegion(int detector, int sector, char coord, double value) const {
    const std::string key = std::to_string(detector) + "_" + std::to_string(sector) + "_" + coord;
    const auto& exclusionMap = ecalRGK_ ? rgkEcalExclusionMap_ : rgaEcalExclusionMap_;
    const auto it = exclusionMap.find(key);
    if (it == exclusionMap.end()) return false;

    for (const auto& range : it->second) {
        if (value >= range.first && value <= range.second) return true;
    }
    return false;
}

std::pair<double, double> Cuts::rotateToSector1Frame(double x, double y, int sector) const {
    if (sector == 1) return {x, y};
    const double angle = -60.0 * (sector - 1) * kPi / 180.0;
    return {x * std::cos(angle) - y * std::sin(angle),
            x * std::sin(angle) + y * std::cos(angle)};
}

double Cuts::evalPoly(const std::vector<double>& coeffs, double p) const {
    double value = 0.0;
    for (size_t i = 0; i < coeffs.size(); ++i) {
        value += coeffs[i] * std::pow(p, coeffs.size() - 1 - i);
    }
    return value;
}

double Cuts::sfMu(int sector, double p) const {
    return evalPoly(sfCoeffs_.at(sector).muCoeffs, p);
}

double Cuts::sfSigma(int sector, double p) const {
    return evalPoly(sfCoeffs_.at(sector).sigmaCoeffs, p);
}

bool Cuts::passSFSigmaCut(int sector, double sf, double p) const {
    const auto it = sfCoeffs_.find(sector);
    if (it == sfCoeffs_.end()) return true;
    const double mu = sfMu(sector, p);
    const double sigma = sfSigma(sector, p);
    return sf > mu - cfg_.sfNumSigma * sigma && sf < mu + cfg_.sfNumSigma * sigma;
}

bool Cuts::passSFTriangleCut(double ePCAL, double eECIN, double p) const {
    if (p < cfg_.sfHtccThreshold) return true;
    const double y = ePCAL / p;
    const double x = eECIN / p;
    return y * cfg_.sfTriangleYScale + x * cfg_.sfTriangleXScale > cfg_.sfTriangleHypotenuse;
}
