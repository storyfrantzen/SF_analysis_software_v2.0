#include "Cuts.h"

#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>

using json = nlohmann::json;

namespace {
constexpr double kPi = 3.14159265358979323846;

bool isFinite(double value) {
    return std::isfinite(value);
}

bool isFirstPidInstance(const RecBranches& p, const std::vector<RecBranches>& eventParticles) {
    bool found = false;
    int firstIdx = -1;

    for (const auto& eventParticle : eventParticles) {
        if (eventParticle.pid != p.pid) continue;
        if (eventParticle.particleIdx < 0) continue;

        if (!found || eventParticle.particleIdx < firstIdx) {
            firstIdx = eventParticle.particleIdx;
            found = true;
        }
    }

    return found && p.particleIdx == firstIdx;
}

bool detectorAllowed(int det, const std::vector<int>& allowedDetectors) {
    if (allowedDetectors.empty()) return true;
    for (const int allowed : allowedDetectors) {
        if (det == allowed) return true;
    }
    return false;
}

PrimitiveCutSpec makeCut(const std::string& name,
                         const std::string& op,
                         double min = NAN,
                         double max = NAN,
                         int detector = -999,
                         const std::string& refRole = "") {
    PrimitiveCutSpec cut;
    cut.name = name;
    cut.op = op;
    cut.min = min;
    cut.max = max;
    cut.detector = detector;
    cut.refRole = refRole;
    return cut;
}

PrimitiveCutSpec parseCutSpec(const json& j) {
    PrimitiveCutSpec cut;
    cut.op = j.value("op", "");
    cut.name = j.value("name", cut.op);
    cut.mode = j.value("mode", cut.mode);
    if (cut.mode != "require" && cut.mode != "tag") {
        throw std::runtime_error("Unsupported cut mode '" + cut.mode +
                                 "' for cut '" + cut.name + "'");
    }
    cut.min = j.value("min", cut.min);
    cut.max = j.value("max", cut.max);
    cut.detector = j.value("detector", cut.detector);
    cut.refRole = j.value("refRole", cut.refRole);
    return cut;
}

ParticleRoleSpec parseRoleSpec(const json& j) {
    ParticleRoleSpec role;
    role.role = j.value("role", "");
    role.pid = j.value("pid", role.pid);
    role.count = j.value("count", role.count);
    role.detectors = j.value("detectors", role.detectors);
    if (j.contains("cuts")) {
        for (const auto& cut : j["cuts"]) role.cuts.push_back(parseCutSpec(cut));
    }
    return role;
}

CompositeSpec parseCompositeSpec(const json& j) {
    CompositeSpec composite;
    composite.role = j.value("role", "");
    composite.type = j.value("type", "");
    composite.mode = j.value("mode", composite.mode);
    if (composite.mode != "require" && composite.mode != "tag") {
        throw std::runtime_error("Unsupported composite mode '" + composite.mode +
                                 "' for composite '" + composite.role + "'");
    }
    composite.daughters = j.value("daughters", composite.daughters);
    composite.mass = j.value("mass", composite.mass);
    composite.window = j.value("window", composite.window);
    return composite;
}

ChannelSpec parseChannelSpec(const json& j) {
    ChannelSpec channel;
    channel.name = j.value("name", channel.name);
    if (j.contains("particles")) {
        for (const auto& particle : j["particles"]) channel.particles.push_back(parseRoleSpec(particle));
    }
    if (j.contains("composites")) {
        for (const auto& composite : j["composites"]) {
            channel.composites.push_back(parseCompositeSpec(composite));
        }
    }
    return channel;
}

CandidateSelectionSpec parseCandidateSelectionSpec(const json& j) {
    CandidateSelectionSpec selection;
    selection.method = j.value("method", selection.method);
    if (selection.method != "compositeDistance" &&
        selection.method != "pi0MassThenMissingPt") {
        throw std::runtime_error(
            "Unsupported candidate-selection method '" + selection.method + "'"
        );
    }
    return selection;
}

void mergeConfig(json& base, const json& override) {
    for (const auto& item : override.items()) {
        if (item.key() == "extends") continue;

        auto existing = base.find(item.key());
        if (existing != base.end() && existing->is_object() && item.value().is_object()) {
            mergeConfig(*existing, item.value());
        } else {
            base[item.key()] = item.value();
        }
    }
}

json loadConfigJson(const std::filesystem::path& filename,
                    std::vector<std::filesystem::path>& includeStack) {
    const auto normalized = std::filesystem::absolute(filename).lexically_normal();
    for (const auto& active : includeStack) {
        if (active == normalized) {
            throw std::runtime_error("Cycle detected while resolving config extends: " +
                                     normalized.string());
        }
    }

    std::ifstream f(normalized);
    if (!f.is_open()) {
        throw std::runtime_error("Cannot open post-processing config file: " + normalized.string());
    }

    json current;
    f >> current;

    if (!current.contains("extends")) return current;
    if (!current["extends"].is_string()) {
        throw std::runtime_error("Config extends must be a string path in: " + normalized.string());
    }

    std::filesystem::path parentPath = current["extends"].get<std::string>();
    if (parentPath.is_relative()) {
        parentPath = normalized.parent_path() / parentPath;
    }

    includeStack.push_back(normalized);
    json merged = loadConfigJson(parentPath, includeStack);
    includeStack.pop_back();
    mergeConfig(merged, current);
    return merged;
}

ChannelSpec legacyEppi0Channel(const json& eppi0) {
    const double electronMinP = eppi0.value("electronMinP", 1.0);
    const double protonMinP = eppi0.value("protonMinP", 0.3);
    const double photonMinP = eppi0.value("photonMinP", 0.4);
    const double photonMinBeta = eppi0.value("photonMinBeta", 0.9);
    const double photonMaxBeta = eppi0.value("photonMaxBeta", 1.1);
    const double photonMinCalEnergy = eppi0.value("photonMinCalEnergy", 0.15);
    const bool rejectPhotonsInCD = eppi0.value("rejectPhotonsInCD", true);
    const bool rejectPhotonsInElectronSector = eppi0.value("rejectPhotonsInElectronSector", true);
    const double maxElectronProtonVertexDiff = eppi0.value("maxElectronProtonVertexDiff", 20.0);
    const double pi0MassWindow = eppi0.value("pi0MassWindow", 0.15);

    ChannelSpec channel;
    channel.name = "eppi0";

    ParticleRoleSpec electron;
    electron.role = "electron";
    electron.pid = 11;
    electron.count = 1;
    electron.cuts = {
        makeCut("electron.trigger", "firstPidInstance"),
        makeCut("electron.min_p", "minP", electronMinP),
        makeCut("electron.fiducial", "fiducial"),
        makeCut("electron.sampling_fraction", "samplingFraction")
    };
    channel.particles.push_back(electron);

    ParticleRoleSpec proton;
    proton.role = "proton";
    proton.pid = 2212;
    proton.count = 1;
    proton.cuts = {
        makeCut("proton.min_p", "minP", protonMinP),
        makeCut("proton.vertex", "vertexDiff", NAN, maxElectronProtonVertexDiff, -999, "electron"),
        makeCut("proton.fiducial", "fiducial")
    };
    channel.particles.push_back(proton);

    ParticleRoleSpec gamma;
    gamma.role = "gamma";
    gamma.pid = 22;
    gamma.count = 2;
    gamma.cuts = {
        makeCut("gamma.min_p", "minP", photonMinP),
        makeCut("gamma.beta", "betaRange", photonMinBeta, photonMaxBeta),
        makeCut("gamma.cal_energy", "minCalEnergy", photonMinCalEnergy),
        makeCut("gamma.fiducial", "fiducial")
    };
    if (rejectPhotonsInCD) gamma.detectors = {0, 1};
    if (rejectPhotonsInElectronSector) {
        gamma.cuts.push_back(makeCut("gamma.not_electron_sector",
                                     "rejectSameSectorAsRole",
                                     NAN,
                                     NAN,
                                     -999,
                                     "electron"));
    }
    channel.particles.push_back(gamma);

    CompositeSpec pi0;
    pi0.role = "pi0";
    pi0.type = "pairMass";
    pi0.daughters = {"gamma", "gamma"};
    pi0.mass = 0.1349768;
    pi0.window = pi0MassWindow;
    channel.composites.push_back(pi0);
    return channel;
}
}

const ParticleRoleSpec* ChannelSpec::findRole(const std::string& role) const {
    for (const auto& particle : particles) {
        if (particle.role == role) return &particle;
    }
    return nullptr;
}

const CompositeSpec* ChannelSpec::findComposite(const std::string& role) const {
    for (const auto& composite : composites) {
        if (composite.role == role) return &composite;
    }
    return nullptr;
}

void CutDecision::require(bool condition, const std::string& name) {
    evaluated.push_back(name);
    if (condition) return;
    pass = false;
    failed.push_back(name);
}

void CutDecision::tag(bool condition, const std::string& name) {
    evaluated.push_back(name);
    tagged.push_back(name);
    if (!condition) taggedFailed.push_back(name);
}

void CutDecision::merge(const CutDecision& other) {
    evaluated.insert(evaluated.end(), other.evaluated.begin(), other.evaluated.end());
    tagged.insert(tagged.end(), other.tagged.begin(), other.tagged.end());
    taggedFailed.insert(taggedFailed.end(), other.taggedFailed.begin(), other.taggedFailed.end());
    if (!other.pass) pass = false;
    failed.insert(failed.end(), other.failed.begin(), other.failed.end());
}

std::string CutDecision::failedCsv() const {
    std::ostringstream out;
    for (size_t i = 0; i < failed.size(); ++i) {
        if (i > 0) out << ",";
        out << failed[i];
    }
    return out.str();
}

PostCutConfig PostCutConfig::fromFile(const std::string& filename) {
    std::vector<std::filesystem::path> includeStack;
    json j = loadConfigJson(filename, includeStack);

    PostCutConfig cfg;
    cfg.outputFile = j.value("outputFile", cfg.outputFile);
    cfg.inputTree = j.value("inputTree", cfg.inputTree);
    cfg.inputEventTree = j.value("inputEventTree", cfg.inputEventTree);
    cfg.outputTree = j.value("outputTree", cfg.outputTree);
    cfg.outputParticleTree = j.value("outputParticleTree", cfg.outputParticleTree);
    cfg.outputMode = j.value("outputMode", cfg.outputMode);
    cfg.beamEnergy = j.value("beamEnergy", cfg.beamEnergy);
    cfg.torus = j.value("torus", cfg.torus);
    cfg.saveFailedCandidates = j.value("saveFailedCandidates", cfg.saveFailedCandidates);
    cfg.fiducialCuts = j.value("fiducialCuts", cfg.fiducialCuts);

    if (j.contains("samplingFraction")) {
        const auto& sf = j["samplingFraction"];
        cfg.sfEnabled = sf.value("enabled", cfg.sfEnabled);
        cfg.sfMinPcalEnergy = sf.value("minPcalEnergy", cfg.sfMinPcalEnergy);
        cfg.sfNumSigma = sf.value("numSigma", cfg.sfNumSigma);
        cfg.sfTriangleYScale = sf.value("triangleYScale", cfg.sfTriangleYScale);
        cfg.sfTriangleXScale = sf.value("triangleXScale", cfg.sfTriangleXScale);
        cfg.sfTriangleHypotenuse = sf.value("triangleHypotenuse", cfg.sfTriangleHypotenuse);
        cfg.sfHtccThreshold = sf.value("htccThreshold", cfg.sfHtccThreshold);

        if (sf.contains("diagonal")) {
            const auto& diagonal = sf["diagonal"];
            cfg.sfTriangleYScale = diagonal.value("yScale", cfg.sfTriangleYScale);
            cfg.sfTriangleXScale = diagonal.value("xScale", cfg.sfTriangleXScale);
            cfg.sfTriangleHypotenuse = diagonal.value("threshold", cfg.sfTriangleHypotenuse);
            cfg.sfHtccThreshold = diagonal.value("momentumThreshold", cfg.sfHtccThreshold);
        }
        if (sf.contains("sigma")) {
            const auto& sigma = sf["sigma"];
            cfg.sfNumSigma = sigma.value("numSigma", cfg.sfNumSigma);
            if (sigma.contains("params")) cfg.sfParams = sigma["params"];
            if (sigma.contains("paramsFile")) {
                std::filesystem::path paramsPath = sigma["paramsFile"].get<std::string>();
                if (paramsPath.is_relative()) {
                    paramsPath = std::filesystem::path(filename).parent_path() / paramsPath;
                }
                std::ifstream paramsFile(paramsPath);
                if (!paramsFile.is_open()) {
                    throw std::runtime_error("Cannot open sampling-fraction parameter file: " +
                                             paramsPath.string());
                }
                paramsFile >> cfg.sfParams;
            }
        }
        if (sf.contains("params")) cfg.sfParams = sf["params"];
    }

    if (j.contains("channel")) {
        cfg.channel = parseChannelSpec(j["channel"]);
    } else if (j.contains("eppi0")) {
        cfg.channel = legacyEppi0Channel(j["eppi0"]);
    }

    if (j.contains("candidateSelection")) {
        cfg.candidateSelection = parseCandidateSelectionSpec(j["candidateSelection"]);
    }

    if (j.contains("exclusivity")) {
        const auto& exclusivity = j["exclusivity"];
        cfg.maxAbsMissingEnergy = exclusivity.value("maxAbsMissingEnergy", cfg.maxAbsMissingEnergy);
        cfg.minElectronPhotonAngleDeg = exclusivity.value("minElectronPhotonAngleDeg",
                                                          cfg.minElectronPhotonAngleDeg);
        cfg.minPhotonPhotonAngleDeg = exclusivity.value("minPhotonPhotonAngleDeg",
                                                        cfg.minPhotonPhotonAngleDeg);
        cfg.maxPi0ConeAngleDeg = exclusivity.value("maxPi0ConeAngleDeg", cfg.maxPi0ConeAngleDeg);
    }

    if (j.contains("eppi0")) {
        const auto& eppi0 = j["eppi0"];
        cfg.maxAbsMissingEnergy = eppi0.value("maxAbsMissingEnergy", cfg.maxAbsMissingEnergy);
        cfg.minElectronPhotonAngleDeg = eppi0.value("minElectronPhotonAngleDeg",
                                                   cfg.minElectronPhotonAngleDeg);
        cfg.minPhotonPhotonAngleDeg = eppi0.value("minPhotonPhotonAngleDeg",
                                                 cfg.minPhotonPhotonAngleDeg);
        cfg.maxPi0ConeAngleDeg = eppi0.value("maxPi0ConeAngleDeg", cfg.maxPi0ConeAngleDeg);
    }

    if (cfg.outputMode == "candidates" && cfg.outputParticleTree == cfg.outputTree) {
        throw std::runtime_error("outputParticleTree must differ from outputTree");
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
    if (!cfg_.sfParams.empty()) loadSamplingFractionParams(cfg_.sfParams);
}

void Cuts::enableFiducialTag(const std::string& tag) {
    if (tag == "DCEdges" || tag == "DCEdges_RGA") dcEdgesRGA_ = true;
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
    return evaluateFiducial(p).pass;
}

CutDecision Cuts::evaluateParticle(const RecBranches& p,
                                   const ParticleRoleSpec& role,
                                   const std::vector<RecBranches>& eventParticles,
                                   const std::map<std::string, const RecBranches*>& selected) const {
    CutDecision decision;
    decision.require(p.pid == role.pid, role.role + ".pid");
    decision.require(detectorAllowed(p.det, role.detectors), role.role + ".detector");

    for (const auto& cut : role.cuts) {
        const std::string name = cut.name.empty() ? (role.role + "." + cut.op) : cut.name;
        const auto apply = [&](bool condition) {
            if (cut.mode == "tag") decision.tag(condition, name);
            else decision.require(condition, name);
        };

        if (cut.op == "minP") {
            apply(isFinite(p.p) && isFinite(cut.min) && p.p >= cut.min);
        } else if (cut.op == "maxP") {
            apply(isFinite(p.p) && isFinite(cut.max) && p.p <= cut.max);
        } else if (cut.op == "pRange") {
            apply(isFinite(p.p) && isFinite(cut.min) && isFinite(cut.max) &&
                  p.p >= cut.min && p.p <= cut.max);
        } else if (cut.op == "betaRange") {
            apply(isFinite(p.beta) && isFinite(cut.min) && isFinite(cut.max) &&
                  p.beta >= cut.min && p.beta <= cut.max);
        } else if (cut.op == "minCalEnergy") {
            const double calEnergy = p.E_PCAL + p.E_ECIN + p.E_ECOUT;
            apply(isFinite(calEnergy) && isFinite(cut.min) && calEnergy >= cut.min);
        } else if (cut.op == "firstPidInstance") {
            apply(isFirstPidInstance(p, eventParticles));
        } else if (cut.op == "matchedGen") {
            apply(p.matchedGenIdx >= 0);
        } else if (cut.op == "rejectDetector") {
            apply(p.det != cut.detector);
        } else if (cut.op == "rejectSameSectorAsRole") {
            const auto ref = selected.find(cut.refRole);
            apply(ref == selected.end() || !ref->second || p.sector != ref->second->sector);
        } else if (cut.op == "vertexDiff") {
            const auto ref = selected.find(cut.refRole);
            const bool pass = ref == selected.end() || !ref->second ||
                              !isFinite(ref->second->vz) || !isFinite(p.vz) ||
                              (isFinite(cut.max) && std::abs(p.vz - ref->second->vz) <= cut.max);
            apply(pass);
        } else if (cut.op == "removeCVTPhi") {
            apply(isFinite(cut.min) && isFinite(cut.max) &&
                  passesCVTPhiVeto(p, cut.min, cut.max));
        } else if (cut.op == "fiducial") {
            apply(evaluateFiducial(p).pass);
        } else if (cut.op == "minPcalEnergy") {
            apply(p.det != 1 || passSFMinPcalCut(p.E_PCAL));
        } else if (cut.op == "samplingFractionDiagonal") {
            apply(p.det != 1 || passSFTriangleCut(p.E_PCAL, p.E_ECIN, p.p));
        } else if (cut.op == "samplingFractionSigma") {
            const double sf = (p.E_PCAL + p.E_ECIN + p.E_ECOUT) / p.p;
            apply(p.det != 1 ||
                  (!sfCoeffs_.empty() && passSFSigmaCut(p.sector, sf, p.p)));
        } else if (cut.op == "samplingFraction") {
            apply(evaluateSamplingFraction(p).pass);
        } else {
            apply(false);
        }
    }

    return decision;
}

CutDecision Cuts::evaluateFiducial(const RecBranches& p) const {
    CutDecision decision;
    if (p.det == 0) {
        decision.require(passesFT(p), "fiducial.ft");
    } else if (p.det == 1) {
        // FD electrons use tracking and calorimetry, protons use tracking,
        // and photons use calorimetry. Requiring both systems for every PID
        // rejects valid proton and photon candidates.
        if (p.pid == 11 || p.pid == 2212) {
            decision.require(passesDC(p), "fiducial.dc_edges");
        }
        if (p.pid == 11 || p.pid == 22) {
            decision.require(passesECAL(p), "fiducial.ecal");
        }
    } else if (p.det == 2) {
        decision.require(passesCVT(p), "fiducial.cvt");
    }
    return decision;
}

bool Cuts::passesLooseExclusivity(double missingEnergy,
                                  double thetaEG1Deg,
                                  double thetaEG2Deg,
                                  double thetaG1G2Deg,
                                  double pi0ConeAngleDeg) const {
    return evaluateLooseExclusivity({missingEnergy,
                                     thetaEG1Deg,
                                     thetaEG2Deg,
                                     thetaG1G2Deg,
                                     pi0ConeAngleDeg}).pass;
}

CutDecision Cuts::evaluateLooseExclusivity(const ExclusivityVars& vars) const {
    CutDecision decision;
    decision.require(std::abs(vars.missingEnergy) <= cfg_.maxAbsMissingEnergy,
                     "exclusivity.missing_energy");
    decision.require(vars.thetaEG1Deg >= cfg_.minElectronPhotonAngleDeg,
                     "exclusivity.theta_e_g1");
    decision.require(vars.thetaEG2Deg >= cfg_.minElectronPhotonAngleDeg,
                     "exclusivity.theta_e_g2");
    decision.require(vars.thetaG1G2Deg >= cfg_.minPhotonPhotonAngleDeg,
                     "exclusivity.theta_g1_g2");
    decision.require(vars.pi0ConeAngleDeg <= cfg_.maxPi0ConeAngleDeg,
                     "exclusivity.pi0_cone");
    return decision;
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
    return true;
}

bool Cuts::passesCVTPhiVeto(const RecBranches& p, double minDeg, double maxDeg) const {
    if (p.det != 2) return true;
    if (!isFinite(p.phi_cvt)) return false;
    double phiDeg = p.phi_cvt * 180.0 / kPi;
    if (phiDeg < 0) phiDeg += 360.0;

    if (minDeg <= maxDeg) return !(minDeg <= phiDeg && phiDeg <= maxDeg);
    return !(phiDeg >= minDeg || phiDeg <= maxDeg);
}

bool Cuts::passesSamplingFraction(const RecBranches& e) const {
    return evaluateSamplingFraction(e).pass;
}

CutDecision Cuts::evaluateSamplingFraction(const RecBranches& e) const {
    CutDecision decision;
    if (!cfg_.sfEnabled || sfCoeffs_.empty() || e.det != 1) return decision;
    decision.require(passSFMinPcalCut(e.E_PCAL), "sampling_fraction.min_pcal");
    const double sf = (e.E_PCAL + e.E_ECIN + e.E_ECOUT) / e.p;
    decision.require(passSFTriangleCut(e.E_PCAL, e.E_ECIN, e.p),
                     "sampling_fraction.triangle");
    decision.require(passSFSigmaCut(e.sector, sf, e.p),
                     "sampling_fraction.sigma");
    return decision;
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

bool Cuts::passSFMinPcalCut(double ePCAL) const {
    return isFinite(ePCAL) && ePCAL > cfg_.sfMinPcalEnergy;
}

bool Cuts::passSFTriangleCut(double ePCAL, double eECIN, double p) const {
    if (!isFinite(ePCAL) || !isFinite(eECIN) || !isFinite(p) || p <= 0) return false;
    if (p < cfg_.sfHtccThreshold) return true;
    const double y = ePCAL / p;
    const double x = eECIN / p;
    return y * cfg_.sfTriangleYScale + x * cfg_.sfTriangleXScale > cfg_.sfTriangleHypotenuse;
}
