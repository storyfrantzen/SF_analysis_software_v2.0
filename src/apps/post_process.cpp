#include <algorithm>
#include <cctype>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <map>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "TFile.h"
#include "TLorentzVector.h"
#include "TTree.h"

#include "Cuts.h"
#include "Kinematics.h"
#include "ROOTBranches.h"
#include "core/TreeNames.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
using Clock = std::chrono::steady_clock;

using Selection = std::map<std::string, std::vector<const RecBranches*>>;

struct EventRows {
    EventBranches event;
    std::vector<RecBranches> recs;

    void clear() {
        event.reset();
        recs.clear();
    }
};

struct ProcessingStats {
    long long eventsWithoutSavedCandidate = 0;
    long long compositeFailures = 0;
    long long exclusivityFailures = 0;
    std::map<std::string, long long> cutFailures;

    void addFailures(const CutDecision& decision) {
        for (const auto& name : decision.failed) ++cutFailures[name];
    }
};

void addCsvFailures(const std::string& failedCuts, ProcessingStats& stats) {
    if (failedCuts.empty()) {
        ++stats.cutFailures["exclusivity"];
        return;
    }

    std::istringstream stream(failedCuts);
    std::string name;
    while (std::getline(stream, name, ',')) {
        if (!name.empty()) ++stats.cutFailures[name];
    }
}

void appendUnique(std::vector<std::string>& destination,
                  const std::vector<std::string>& source) {
    for (const auto& name : source) {
        if (name.empty()) continue;
        if (std::find(destination.begin(), destination.end(), name) == destination.end()) {
            destination.push_back(name);
        }
    }
}

std::string joinCsv(const std::vector<std::string>& names) {
    std::ostringstream out;
    for (size_t i = 0; i < names.size(); ++i) {
        if (i > 0) out << ",";
        out << names[i];
    }
    return out.str();
}

std::vector<std::string> splitCsv(const std::string& value) {
    std::vector<std::string> names;
    std::istringstream stream(value);
    std::string name;
    while (std::getline(stream, name, ',')) {
        if (!name.empty()) names.push_back(name);
    }
    return names;
}

double fraction(long long numerator, long long denominator) {
    if (denominator <= 0) return 0.0;
    return static_cast<double>(numerator) / static_cast<double>(denominator);
}

void printChannelSummary(const PostCutConfig& cfg) {
    std::cout << "[INFO] Channel     : " << cfg.channel.name << "\n"
              << "[INFO] Roles       :";
    for (const auto& role : cfg.channel.particles) {
        std::cout << " " << role.role << "(pid=" << role.pid
                  << ", count=" << role.count;
        if (!role.detectors.empty()) {
            std::cout << ", det=";
            for (size_t i = 0; i < role.detectors.size(); ++i) {
                if (i > 0) std::cout << "/";
                std::cout << role.detectors[i];
            }
        }
        std::cout << ")";
    }
    std::cout << "\n"
              << "[INFO] Candidate selection: " << cfg.candidateSelection.method
              << "\n";
}

void printProgress(Long64_t currentRow,
                   Long64_t totalRows,
                   long long nEvents,
                   long long nWritten,
                   const ProcessingStats& stats,
                   const Clock::time_point& startTime) {
    const auto now = Clock::now();
    const double elapsed = std::chrono::duration<double>(now - startTime).count();
    const double rowRate = elapsed > 0.0 ? static_cast<double>(currentRow) / elapsed : 0.0;
    const double pct = 100.0 * fraction(currentRow, totalRows);
    const long long rejected = stats.eventsWithoutSavedCandidate;

    std::cout << std::fixed << std::setprecision(1)
              << "[PROGRESS] rows " << currentRow << "/" << totalRows
              << " (" << pct << "%)"
              << "  events " << nEvents
              << "  saved " << nWritten
              << "  rejected " << rejected
              << "  rate " << rowRate << " rows/s"
              << "  elapsed " << elapsed << " s\n";
}

struct CandidateOutput {
    std::uint64_t sourceFileId = INVALID_SOURCE_ID;
    std::uint64_t sourceEventIndex = INVALID_SOURCE_ID;
    int runNum = -999;
    int eventNum = -999;
    int helicity = -999;
    int passTopology = 0;
    std::vector<std::string> selectedRoles;
    std::vector<int> selectedIdx;
    std::vector<int> selectedPid;
    std::vector<int> selectedDet;
    std::vector<int> selectedSector;
    std::vector<double> selectedP;
    std::vector<double> selectedTheta;
    std::vector<double> selectedPhi;
    std::vector<int> topologyPids;
    std::vector<int> topologyPidCounts;
    std::vector<int> topologyPidCountsFT;
    std::vector<int> topologyPidCountsFD;
    std::vector<int> topologyPidCountsCD;
    std::vector<int> topologyPidCountsOther;

    double charge = NAN;

    double electronEPCAL = NAN;
    double electronEECIN = NAN;
    double electronEECOUT = NAN;

    int eppi0_eIdx = -999;
    int eppi0_pIdx = -999;
    int eppi0_g1Idx = -999;
    int eppi0_g2Idx = -999;
    int eppi0_eDet = -999;
    int eppi0_pDet = -999;
    int eppi0_g1Det = -999;
    int eppi0_g2Det = -999;
    int eppi0_eSector = -999;
    int eppi0_pSector = -999;
    int eppi0_g1Sector = -999;
    int eppi0_g2Sector = -999;
    int eppi0_passFiducial = 0;
    int eppi0_electronPassFiducial = 0;
    int eppi0_protonPassFiducial = 0;
    int eppi0_gamma1PassFiducial = 0;
    int eppi0_gamma2PassFiducial = 0;
    int eppi0_passSamplingFraction = 0;
    int eppi0_passExclusivity = 0;
    std::string eppi0_evaluatedCuts;
    std::string eppi0_failedCuts;
    double Q2 = NAN;
    double nu = NAN;
    double xB = NAN;
    double y = NAN;
    double W = NAN;
    double t = NAN;
    double t_pi0 = NAN;
    double trentoPhi = NAN;

    double pi0_p = NAN;
    double pi0_theta = NAN;
    double pi0_phi = NAN;
    double pi0_deltaPhi = NAN;
    double pi0_thetaX = NAN;
    double m_gg = NAN;
    double m2_miss = NAN;
    double m2_epX = NAN;
    double m2_epi0X = NAN;
    double m_eggX = NAN;
    double E_miss = NAN;
    double pT_miss = NAN;
    double theta_e_g1 = NAN;
    double theta_e_g2 = NAN;
    double theta_g1_g2 = NAN;

    void reset() { *this = CandidateOutput{}; }

    void registerGenericBranches(TTree& tree) {
        tree.Branch("sourceFileId", &sourceFileId, "sourceFileId/l");
        tree.Branch("sourceEventIndex", &sourceEventIndex, "sourceEventIndex/l");
        tree.Branch("runNum", &runNum, "runNum/I");
        tree.Branch("eventNum", &eventNum, "eventNum/I");
        tree.Branch("helicity", &helicity, "helicity/I");
        tree.Branch("charge", &charge, "charge/D");
        tree.Branch("passTopology", &passTopology, "passTopology/I");
        tree.Branch("selectedRoles", &selectedRoles);
        tree.Branch("selectedIdx", &selectedIdx);
        tree.Branch("selectedPid", &selectedPid);
        tree.Branch("selectedDet", &selectedDet);
        tree.Branch("selectedSector", &selectedSector);
        tree.Branch("selectedP", &selectedP);
        tree.Branch("selectedTheta", &selectedTheta);
        tree.Branch("selectedPhi", &selectedPhi);
        tree.Branch("topologyPids", &topologyPids);
        tree.Branch("topologyPidCounts", &topologyPidCounts);
        tree.Branch("topologyPidCountsFT", &topologyPidCountsFT);
        tree.Branch("topologyPidCountsFD", &topologyPidCountsFD);
        tree.Branch("topologyPidCountsCD", &topologyPidCountsCD);
        tree.Branch("topologyPidCountsOther", &topologyPidCountsOther);
        tree.Branch("electronEPCAL", &electronEPCAL, "electronEPCAL/D");
        tree.Branch("electronEECIN", &electronEECIN, "electronEECIN/D");
        tree.Branch("electronEECOUT", &electronEECOUT, "electronEECOUT/D");
        tree.Branch("Q2", &Q2, "Q2/D");
        tree.Branch("nu", &nu, "nu/D");
        tree.Branch("xB", &xB, "xB/D");
    }

    void registerEppi0Branches(TTree& tree) {
        tree.Branch("eIdx", &eppi0_eIdx, "eIdx/I");
        tree.Branch("pIdx", &eppi0_pIdx, "pIdx/I");
        tree.Branch("g1Idx", &eppi0_g1Idx, "g1Idx/I");
        tree.Branch("g2Idx", &eppi0_g2Idx, "g2Idx/I");
        tree.Branch("eDet", &eppi0_eDet, "eDet/I");
        tree.Branch("pDet", &eppi0_pDet, "pDet/I");
        tree.Branch("g1Det", &eppi0_g1Det, "g1Det/I");
        tree.Branch("g2Det", &eppi0_g2Det, "g2Det/I");
        tree.Branch("eSector", &eppi0_eSector, "eSector/I");
        tree.Branch("pSector", &eppi0_pSector, "pSector/I");
        tree.Branch("g1Sector", &eppi0_g1Sector, "g1Sector/I");
        tree.Branch("g2Sector", &eppi0_g2Sector, "g2Sector/I");
        tree.Branch("passFiducial", &eppi0_passFiducial, "passFiducial/I");
        tree.Branch("electronPassFiducial", &eppi0_electronPassFiducial,
                    "electronPassFiducial/I");
        tree.Branch("protonPassFiducial", &eppi0_protonPassFiducial,
                    "protonPassFiducial/I");
        tree.Branch("gamma1PassFiducial", &eppi0_gamma1PassFiducial,
                    "gamma1PassFiducial/I");
        tree.Branch("gamma2PassFiducial", &eppi0_gamma2PassFiducial,
                    "gamma2PassFiducial/I");
        tree.Branch("passSamplingFraction", &eppi0_passSamplingFraction, "passSamplingFraction/I");
        tree.Branch("passExclusivity", &eppi0_passExclusivity, "passExclusivity/I");
        tree.Branch("evaluatedCuts", &eppi0_evaluatedCuts);
        tree.Branch("failedCuts", &eppi0_failedCuts);
        tree.Branch("y", &y, "y/D");
        tree.Branch("W", &W, "W/D");
        tree.Branch("t", &t, "t/D");
        tree.Branch("t_pi0", &t_pi0, "t_pi0/D");
        tree.Branch("trentoPhi", &trentoPhi, "trentoPhi/D");
        tree.Branch("pi0_p", &pi0_p, "pi0_p/D");
        tree.Branch("pi0_theta", &pi0_theta, "pi0_theta/D");
        tree.Branch("pi0_phi", &pi0_phi, "pi0_phi/D");
        tree.Branch("pi0_deltaPhi", &pi0_deltaPhi, "pi0_deltaPhi/D");
        tree.Branch("pi0_thetaX", &pi0_thetaX, "pi0_thetaX/D");
        tree.Branch("m_gg", &m_gg, "m_gg/D");
        tree.Branch("m2_miss", &m2_miss, "m2_miss/D");
        tree.Branch("m2_epX", &m2_epX, "m2_epX/D");
        tree.Branch("m2_epi0X", &m2_epi0X, "m2_epi0X/D");
        tree.Branch("m_eggX", &m_eggX, "m_eggX/D");
        tree.Branch("E_miss", &E_miss, "E_miss/D");
        tree.Branch("pT_miss", &pT_miss, "pT_miss/D");
        tree.Branch("theta_e_g1", &theta_e_g1, "theta_e_g1/D");
        tree.Branch("theta_e_g2", &theta_e_g2, "theta_e_g2/D");
        tree.Branch("theta_g1_g2", &theta_g1_g2, "theta_g1_g2/D");
    }

    void registerBranches(TTree& tree, bool includeEppi0Branches) {
        registerGenericBranches(tree);
        if (includeEppi0Branches) registerEppi0Branches(tree);
    }
};

struct SelectedParticleOutput {
    std::uint64_t sourceFileId = INVALID_SOURCE_ID;
    std::uint64_t sourceEventIndex = INVALID_SOURCE_ID;
    int runNum = -999;
    int eventNum = -999;
    std::string role;
    int occurrence = 0;
    int particleIdx = -999;
    int pid = -999;
    int det = -999;
    int sector = -999;
    double p = NAN;
    double theta = NAN;
    double phi = NAN;

    void registerBranches(TTree& tree) {
        tree.Branch("sourceFileId", &sourceFileId, "sourceFileId/l");
        tree.Branch("sourceEventIndex", &sourceEventIndex, "sourceEventIndex/l");
        tree.Branch("runNum", &runNum, "runNum/I");
        tree.Branch("eventNum", &eventNum, "eventNum/I");
        tree.Branch("role", &role);
        tree.Branch("occurrence", &occurrence, "occurrence/I");
        tree.Branch("particleIdx", &particleIdx, "particleIdx/I");
        tree.Branch("pid", &pid, "pid/I");
        tree.Branch("det", &det, "det/I");
        tree.Branch("sector", &sector, "sector/I");
        tree.Branch("p", &p, "p/D");
        tree.Branch("theta", &theta, "theta/D");
        tree.Branch("phi", &phi, "phi/D");
    }

    void fillRows(const CandidateOutput& candidate, TTree& tree) {
        sourceFileId = candidate.sourceFileId;
        sourceEventIndex = candidate.sourceEventIndex;
        runNum = candidate.runNum;
        eventNum = candidate.eventNum;
        std::map<std::string, int> occurrences;
        for (size_t index = 0; index < candidate.selectedRoles.size(); ++index) {
            role = candidate.selectedRoles[index];
            occurrence = ++occurrences[role];
            particleIdx = index < candidate.selectedIdx.size()
                ? candidate.selectedIdx[index] : -999;
            pid = index < candidate.selectedPid.size() ? candidate.selectedPid[index] : -999;
            det = index < candidate.selectedDet.size() ? candidate.selectedDet[index] : -999;
            sector = index < candidate.selectedSector.size()
                ? candidate.selectedSector[index] : -999;
            p = index < candidate.selectedP.size() ? candidate.selectedP[index] : NAN;
            theta = index < candidate.selectedTheta.size()
                ? candidate.selectedTheta[index] : NAN;
            phi = index < candidate.selectedPhi.size() ? candidate.selectedPhi[index] : NAN;
            tree.Fill();
        }
    }
};

class ReconstructedEventReader {
public:
    explicit ReconstructedEventReader(TTree* tree) : tree_(tree) {
        if (!tree_) return;
        const std::vector<std::string> required{
            "sourceFileId", "sourceEventIndex", "runNum", "eventNum",
            "topologyPids", "topologyPidCounts", "topologyPidCountsFT",
            "topologyPidCountsFD", "topologyPidCountsCD", "topologyPidCountsOther"
        };
        for (const auto& name : required) {
            if (!tree_->GetBranch(name.c_str())) {
                tree_ = nullptr;
                return;
            }
        }
        tree_->SetBranchAddress("sourceFileId", &sourceFileId_);
        tree_->SetBranchAddress("sourceEventIndex", &sourceEventIndex_);
        tree_->SetBranchAddress("runNum", &runNum_);
        tree_->SetBranchAddress("eventNum", &eventNum_);
        tree_->SetBranchAddress("topologyPids", &topologyPids_);
        tree_->SetBranchAddress("topologyPidCounts", &topologyPidCounts_);
        tree_->SetBranchAddress("topologyPidCountsFT", &topologyPidCountsFT_);
        tree_->SetBranchAddress("topologyPidCountsFD", &topologyPidCountsFD_);
        tree_->SetBranchAddress("topologyPidCountsCD", &topologyPidCountsCD_);
        tree_->SetBranchAddress("topologyPidCountsOther", &topologyPidCountsOther_);
    }

    bool available() const { return tree_ != nullptr; }

    void copyFor(const EventBranches& event, CandidateOutput& candidate) {
        if (!tree_) return;
        while (nextEntry_ < tree_->GetEntries()) {
            tree_->GetEntry(nextEntry_++);
            const bool sourceAware = event.sourceFileId != INVALID_SOURCE_ID &&
                                     sourceFileId_ != INVALID_SOURCE_ID;
            const bool matches = sourceAware
                ? sourceFileId_ == event.sourceFileId &&
                  sourceEventIndex_ == event.sourceEventIndex
                : runNum_ == event.runNum && eventNum_ == event.eventNum;
            if (!matches) continue;
            candidate.topologyPids = *topologyPids_;
            candidate.topologyPidCounts = *topologyPidCounts_;
            candidate.topologyPidCountsFT = *topologyPidCountsFT_;
            candidate.topologyPidCountsFD = *topologyPidCountsFD_;
            candidate.topologyPidCountsCD = *topologyPidCountsCD_;
            candidate.topologyPidCountsOther = *topologyPidCountsOther_;
            return;
        }
        throw std::runtime_error(
            "Could not match selected particle event to input rEvents row"
        );
    }

private:
    TTree* tree_ = nullptr;
    Long64_t nextEntry_ = 0;
    std::uint64_t sourceFileId_ = INVALID_SOURCE_ID;
    std::uint64_t sourceEventIndex_ = INVALID_SOURCE_ID;
    int runNum_ = -999;
    int eventNum_ = -999;
    std::vector<int>* topologyPids_ = nullptr;
    std::vector<int>* topologyPidCounts_ = nullptr;
    std::vector<int>* topologyPidCountsFT_ = nullptr;
    std::vector<int>* topologyPidCountsFD_ = nullptr;
    std::vector<int>* topologyPidCountsCD_ = nullptr;
    std::vector<int>* topologyPidCountsOther_ = nullptr;
};

std::string branchRoleName(const std::string& role) {
    std::string name;
    bool previousUnderscore = false;
    for (const unsigned char character : role) {
        if (std::isalnum(character)) {
            name.push_back(static_cast<char>(character));
            previousUnderscore = false;
        } else if (!name.empty() && !previousUnderscore) {
            name.push_back('_');
            previousUnderscore = true;
        }
    }
    while (!name.empty() && name.back() == '_') name.pop_back();
    if (name.empty()) throw std::runtime_error("Particle role cannot produce an empty branch name");
    if (std::isdigit(static_cast<unsigned char>(name.front()))) name = "role_" + name;
    return name;
}

std::string pidBranchToken(int pid) {
    return pid < 0 ? "Minus" + std::to_string(-pid) : std::to_string(pid);
}

struct PidMultiplicitySlot {
    int pid = -999;
    int total = 0;
    int ft = 0;
    int fd = 0;
    int cd = 0;
    int other = 0;
};

class PidMultiplicityBranches {
public:
    explicit PidMultiplicityBranches(const PostCutConfig& cfg) {
        for (const auto& role : cfg.channel.particles) {
            if (lookup_.count(role.pid)) continue;
            lookup_[role.pid] = slots_.size();
            slots_.push_back({role.pid});
        }
    }

    void registerBranches(TTree& tree) {
        for (auto& slot : slots_) {
            const std::string base = "nPid" + pidBranchToken(slot.pid);
            registerInt(tree, base, slot.total);
            registerInt(tree, base + "FT", slot.ft);
            registerInt(tree, base + "FD", slot.fd);
            registerInt(tree, base + "CD", slot.cd);
            registerInt(tree, base + "Other", slot.other);
        }
    }

    void fill(const CandidateOutput& candidate) {
        for (auto& slot : slots_) slot.total = slot.ft = slot.fd = slot.cd = slot.other = 0;
        for (size_t index = 0; index < candidate.topologyPids.size(); ++index) {
            const auto found = lookup_.find(candidate.topologyPids[index]);
            if (found == lookup_.end()) continue;
            auto& slot = slots_[found->second];
            slot.total = valueAt(candidate.topologyPidCounts, index);
            slot.ft = valueAt(candidate.topologyPidCountsFT, index);
            slot.fd = valueAt(candidate.topologyPidCountsFD, index);
            slot.cd = valueAt(candidate.topologyPidCountsCD, index);
            slot.other = valueAt(candidate.topologyPidCountsOther, index);
        }
    }

private:
    static int valueAt(const std::vector<int>& values, size_t index) {
        return index < values.size() ? values[index] : 0;
    }

    static void registerInt(TTree& tree, const std::string& name, int& value) {
        tree.Branch(name.c_str(), &value, (name + "/I").c_str());
    }

    std::vector<PidMultiplicitySlot> slots_;
    std::map<int, size_t> lookup_;
};

struct SelectedRoleBranch {
    std::string role;
    int occurrence = 1;
    std::string branchBase;
    int idx = -999;
    int pid = -999;
    int det = -999;
    int sector = -999;
    double p = NAN;
    double theta = NAN;
    double phi = NAN;

    void reset() {
        idx = pid = det = sector = -999;
        p = theta = phi = NAN;
    }
};

class SelectedRoleBranches {
public:
    explicit SelectedRoleBranches(const PostCutConfig& cfg) {
        size_t totalRoles = 0;
        for (const auto& role : cfg.channel.particles) {
            totalRoles += static_cast<size_t>(std::max(role.count, 0));
        }
        slots_.reserve(totalRoles);
        std::map<std::string, std::string> branchOwners;
        for (const auto& role : cfg.channel.particles) {
            const std::string roleName = branchRoleName(role.role);
            for (int occurrence = 1; occurrence <= role.count; ++occurrence) {
                const std::string branchBase = role.count == 1
                    ? roleName
                    : roleName + std::to_string(occurrence);
                const auto [owner, inserted] = branchOwners.emplace(branchBase, role.role);
                if (!inserted) {
                    throw std::runtime_error(
                        "Particle roles '" + owner->second + "' and '" + role.role +
                        "' produce the same branch prefix '" + branchBase + "'"
                    );
                }
                lookup_[{role.role, occurrence}] = slots_.size();
                slots_.push_back({role.role, occurrence, branchBase});
            }
        }
    }

    void registerBranches(TTree& tree) {
        for (auto& slot : slots_) {
            registerInt(tree, slot.branchBase + "Idx", slot.idx);
            registerInt(tree, slot.branchBase + "Pid", slot.pid);
            registerInt(tree, slot.branchBase + "Det", slot.det);
            registerInt(tree, slot.branchBase + "Sector", slot.sector);
            registerDouble(tree, slot.branchBase + "P", slot.p);
            registerDouble(tree, slot.branchBase + "Theta", slot.theta);
            registerDouble(tree, slot.branchBase + "Phi", slot.phi);
        }
    }

    void fill(const CandidateOutput& candidate) {
        for (auto& slot : slots_) slot.reset();
        std::map<std::string, int> occurrences;
        for (size_t index = 0; index < candidate.selectedRoles.size(); ++index) {
            const std::string& role = candidate.selectedRoles[index];
            const int occurrence = ++occurrences[role];
            const auto found = lookup_.find({role, occurrence});
            if (found == lookup_.end()) continue;
            SelectedRoleBranch& slot = slots_[found->second];
            if (index < candidate.selectedIdx.size()) slot.idx = candidate.selectedIdx[index];
            if (index < candidate.selectedPid.size()) slot.pid = candidate.selectedPid[index];
            if (index < candidate.selectedDet.size()) slot.det = candidate.selectedDet[index];
            if (index < candidate.selectedSector.size()) slot.sector = candidate.selectedSector[index];
            if (index < candidate.selectedP.size()) slot.p = candidate.selectedP[index];
            if (index < candidate.selectedTheta.size()) slot.theta = candidate.selectedTheta[index];
            if (index < candidate.selectedPhi.size()) slot.phi = candidate.selectedPhi[index];
        }
    }

private:
    static void registerInt(TTree& tree, const std::string& name, int& value) {
        const std::string leaf = name + "/I";
        tree.Branch(name.c_str(), &value, leaf.c_str());
    }

    static void registerDouble(TTree& tree, const std::string& name, double& value) {
        const std::string leaf = name + "/D";
        tree.Branch(name.c_str(), &value, leaf.c_str());
    }

    std::vector<SelectedRoleBranch> slots_;
    std::map<std::pair<std::string, int>, size_t> lookup_;
};

const RecBranches* firstParticle(const Selection& selection, const std::string& role) {
    const auto it = selection.find(role);
    if (it == selection.end() || it->second.empty()) return nullptr;
    return it->second.front();
}

void fillSelectedParticleBranches(const Selection& selection,
                                  const PostCutConfig& cfg,
                                  CandidateOutput& out) {
    for (const auto& roleSpec : cfg.channel.particles) {
        const auto it = selection.find(roleSpec.role);
        if (it == selection.end()) continue;
        const auto& particles = it->second;
        for (const auto* particle : particles) {
            if (!particle) continue;
            out.selectedRoles.push_back(roleSpec.role);
            out.selectedIdx.push_back(particle->particleIdx);
            out.selectedPid.push_back(particle->pid);
            out.selectedDet.push_back(particle->det);
            out.selectedSector.push_back(particle->sector);
            out.selectedP.push_back(particle->p);
            out.selectedTheta.push_back(particle->theta);
            out.selectedPhi.push_back(particle->phi);
        }
    }
}

void fillElectronCalorimeterBranches(const Selection& selection,
                                     CandidateOutput& out) {
    if (const RecBranches* electron = firstParticle(selection, "electron")) {
        out.electronEPCAL = electron->E_PCAL;
        out.electronEECIN = electron->E_ECIN;
        out.electronEECOUT = electron->E_ECOUT;
    }
}

void fillDISBranches(const Selection& selection,
                     const PostCutConfig& cfg,
                     CandidateOutput& out) {
    const RecBranches* electron = firstParticle(selection, "electron");
    if (!electron || electron->pid != 11) return;

    const TLorentzVector lvE = Kinematics::particle(*electron);
    const Kinematics::DIS dis = Kinematics::dis(lvE, cfg.beamEnergy);

    out.Q2 = dis.Q2;
    out.nu = dis.nu;
    out.xB = dis.xB;
}

bool evaluateCompositeRank(const Selection& selection,
                           const PostCutConfig& cfg,
                           double& rank,
                           CutDecision& decision) {
    rank = 0.0;
    bool usedComposite = false;

    for (const auto& composite : cfg.channel.composites) {
        if (composite.type != "pairMass" || composite.daughters.size() != 2) continue;

        const std::string& leftRole = composite.daughters[0];
        const std::string& rightRole = composite.daughters[1];
        const RecBranches* left = nullptr;
        const RecBranches* right = nullptr;

        if (leftRole == rightRole) {
            const auto it = selection.find(leftRole);
            if (it == selection.end() || it->second.size() < 2) return false;
            left = it->second[0];
            right = it->second[1];
        } else {
            left = firstParticle(selection, leftRole);
            right = firstParticle(selection, rightRole);
            if (!left || !right) return false;
        }

        const TLorentzVector lvLeft = Kinematics::particle(*left);
        const TLorentzVector lvRight = Kinematics::particle(*right);
        const double mass = (lvLeft + lvRight).M();
        const double delta = std::abs(mass - composite.mass);
        const bool passesWindow = !std::isfinite(composite.window) || delta <= composite.window;
        const std::string cutName = composite.role + ".mass_window";
        if (composite.mode == "tag") decision.tag(passesWindow, cutName);
        else decision.require(passesWindow, cutName);
        if (!decision.pass) return false;
        rank += delta;
        usedComposite = true;
    }

    if (!usedComposite) rank = 0.0;
    return true;
}

struct CandidateChoiceKey {
    double compositeDistance = std::numeric_limits<double>::max();
    double missingPt = 0.0;
    std::vector<int> particleIndices;
};

bool makeCandidateChoiceKey(const CandidateOutput& candidate,
                            const PostCutConfig& cfg,
                            double compositeDistance,
                            CandidateChoiceKey& key) {
    if (!std::isfinite(compositeDistance)) return false;
    key.compositeDistance = compositeDistance;
    key.missingPt = 0.0;
    if (cfg.candidateSelection.method == "pi0MassThenMissingPt") {
        if (!std::isfinite(candidate.pT_miss)) return false;
        key.missingPt = candidate.pT_miss;
    }
    key.particleIndices = candidate.selectedIdx;
    return true;
}

bool candidateChoiceLess(const CandidateChoiceKey& left,
                         const CandidateChoiceKey& right) {
    if (left.compositeDistance != right.compositeDistance) {
        return left.compositeDistance < right.compositeDistance;
    }
    if (left.missingPt != right.missingPt) return left.missingPt < right.missingPt;
    return std::lexicographical_compare(
        left.particleIndices.begin(), left.particleIndices.end(),
        right.particleIndices.begin(), right.particleIndices.end()
    );
}

bool supportsEppi0Logic(const PostCutConfig& cfg) {
    const auto countForRole = [&](const std::string& role) {
        for (const auto& roleSpec : cfg.channel.particles) {
            if (roleSpec.role == role) return roleSpec.count;
        }
        return 0;
    };

    return countForRole("electron") >= 1 &&
           countForRole("proton") >= 1 &&
           countForRole("gamma") >= 2;
}

void fillGenericCandidate(const EventRows& rows,
                          const Selection& selection,
                          const PostCutConfig& cfg,
                          CandidateOutput& out) {
    out.reset();
    out.sourceFileId = rows.event.sourceFileId;
    out.sourceEventIndex = rows.event.sourceEventIndex;
    out.runNum = rows.event.runNum;
    out.eventNum = rows.event.eventNum;
    out.helicity = rows.event.helicity;
    out.charge = rows.event.charge;
    out.passTopology = 1;
    for (const auto& role : cfg.channel.particles) {
        if (std::find(out.topologyPids.begin(), out.topologyPids.end(), role.pid) ==
            out.topologyPids.end()) {
            out.topologyPids.push_back(role.pid);
        }
    }
    const size_t topologySize = out.topologyPids.size();
    out.topologyPidCounts.assign(topologySize, 0);
    out.topologyPidCountsFT.assign(topologySize, 0);
    out.topologyPidCountsFD.assign(topologySize, 0);
    out.topologyPidCountsCD.assign(topologySize, 0);
    out.topologyPidCountsOther.assign(topologySize, 0);
    for (const auto& particle : rows.recs) {
        const auto found = std::find(
            out.topologyPids.begin(), out.topologyPids.end(), particle.pid
        );
        if (found == out.topologyPids.end()) continue;
        const size_t index = static_cast<size_t>(found - out.topologyPids.begin());
        ++out.topologyPidCounts[index];
        switch (particle.det) {
            case 0: ++out.topologyPidCountsFT[index]; break;
            case 1: ++out.topologyPidCountsFD[index]; break;
            case 2: ++out.topologyPidCountsCD[index]; break;
            default: ++out.topologyPidCountsOther[index]; break;
        }
    }
    fillSelectedParticleBranches(selection, cfg, out);
    fillElectronCalorimeterBranches(selection, out);
    fillDISBranches(selection, cfg, out);
}

void runEppi0Logic(const Selection& selection,
                   const Cuts& cuts,
                   CandidateOutput& out) {
    const auto& cfg = cuts.config();
    const RecBranches* ePtr = firstParticle(selection, "electron");
    const RecBranches* pPtr = firstParticle(selection, "proton");
    const auto gammaIt = selection.find("gamma");

    if (!ePtr || !pPtr || gammaIt == selection.end() || gammaIt->second.size() < 2) return;

    const RecBranches& e = *ePtr;
    const RecBranches& p = *pPtr;
    const RecBranches& g1 = *gammaIt->second[0];
    const RecBranches& g2 = *gammaIt->second[1];

    const TLorentzVector beam = Kinematics::beam(cfg.beamEnergy);
    const TLorentzVector target = Kinematics::target();
    const TLorentzVector lvE = Kinematics::particle(e);
    const TLorentzVector lvP = Kinematics::particle(p);
    const Kinematics::DIS dis = Kinematics::dis(lvE, cfg.beamEnergy);

    const TLorentzVector lvG1 = Kinematics::particle(g1);
    const TLorentzVector lvG2 = Kinematics::particle(g2);
    const TLorentzVector lvPi0 = lvG1 + lvG2;

    const TLorentzVector missing = Kinematics::missingSystem(cfg.beamEnergy, {lvE, lvP, lvPi0});
    const TLorentzVector epX = Kinematics::missingSystem(cfg.beamEnergy, {lvE, lvP});
    const TLorentzVector epi0X = Kinematics::missingSystem(cfg.beamEnergy, {lvE, lvPi0});

    out.eppi0_eIdx = e.particleIdx;
    out.eppi0_pIdx = p.particleIdx;
    out.eppi0_g1Idx = g1.particleIdx;
    out.eppi0_g2Idx = g2.particleIdx;
    out.eppi0_eDet = e.det;
    out.eppi0_pDet = p.det;
    out.eppi0_g1Det = g1.det;
    out.eppi0_g2Det = g2.det;
    out.eppi0_eSector = e.sector;
    out.eppi0_pSector = p.sector;
    out.eppi0_g1Sector = g1.sector;
    out.eppi0_g2Sector = g2.sector;
    const CutDecision fiducial = cuts.evaluateFiducial(e);
    const CutDecision fiducialP = cuts.evaluateFiducial(p);
    const CutDecision fiducialG1 = cuts.evaluateFiducial(g1);
    const CutDecision fiducialG2 = cuts.evaluateFiducial(g2);
    out.eppi0_electronPassFiducial = fiducial.pass;
    out.eppi0_protonPassFiducial = fiducialP.pass;
    out.eppi0_gamma1PassFiducial = fiducialG1.pass;
    out.eppi0_gamma2PassFiducial = fiducialG2.pass;
    CutDecision combinedFiducial = fiducialP;
    combinedFiducial.merge(fiducialG1);
    combinedFiducial.merge(fiducialG2);
    combinedFiducial.merge(fiducial);
    out.eppi0_passFiducial = combinedFiducial.pass;
    out.eppi0_passSamplingFraction = cuts.evaluateSamplingFraction(e).pass;

    out.y = dis.y;
    out.W = dis.W;
    out.t = -1.0 * (target - lvP).M2();
    out.t_pi0 = -1.0 * (beam - lvE - lvPi0).M2();
    out.trentoPhi = Kinematics::trentoPhi(beam, lvE, lvP);
    out.pi0_p = lvPi0.P();
    out.pi0_theta = lvPi0.Theta();
    out.pi0_phi = lvPi0.Phi();
    out.pi0_deltaPhi = Kinematics::deltaPhi(lvPi0.Phi(), epX.Phi());
    out.pi0_thetaX = Kinematics::angle(lvPi0, epX);
    out.m_gg = lvPi0.M();
    out.m2_miss = missing.M2();
    out.m2_epX = epX.M2();
    out.m2_epi0X = epi0X.M2();
    out.m_eggX = Kinematics::massIfTimelike(epi0X);
    out.E_miss = missing.E();
    out.pT_miss = missing.Pt();
    out.theta_e_g1 = Kinematics::angle(lvE, lvG1);
    out.theta_e_g2 = Kinematics::angle(lvE, lvG2);
    out.theta_g1_g2 = Kinematics::angle(lvG1, lvG2);
    const CutDecision exclusivity = cuts.evaluateLooseExclusivity({
        out.E_miss,
        out.theta_e_g1 * 180.0 / kPi,
        out.theta_e_g2 * 180.0 / kPi,
        out.theta_g1_g2 * 180.0 / kPi,
        out.pi0_thetaX * 180.0 / kPi
    });
    out.eppi0_passExclusivity = exclusivity.pass;
    out.eppi0_evaluatedCuts = joinCsv(exclusivity.evaluated);
    out.eppi0_failedCuts = exclusivity.failedCsv();
}

void buildCandidateOutput(const EventRows& rows,
                          const Selection& selection,
                          const Cuts& cuts,
                          CandidateOutput& out) {
    fillGenericCandidate(rows, selection, cuts.config(), out);
    if (supportsEppi0Logic(cuts.config())) {
        runEppi0Logic(selection, cuts, out);
    }
}

bool processEvent(const EventRows& rows,
                  const Cuts& cuts,
                  CandidateOutput& out,
                  ProcessingStats& stats) {
    const auto& cfg = cuts.config();
    const bool runEppi0 = supportsEppi0Logic(cfg);

    CandidateChoiceKey bestKey;
    CandidateOutput best;
    bool found = false;
    Selection selection;
    std::vector<std::string> taggedCuts;
    std::vector<std::string> taggedFailures;

    const auto alreadySelected = [&](const RecBranches* candidate) {
        for (const auto& [_, particles] : selection) {
            for (const auto* selected : particles) {
                if (selected == candidate) return true;
            }
        }
        return false;
    };

    const auto selectedContext = [&]() {
        std::map<std::string, const RecBranches*> context;
        for (const auto& [role, particles] : selection) {
            if (!particles.empty()) context[role] = particles.front();
        }
        return context;
    };

    std::function<void(size_t)> visitRole;
    visitRole = [&](size_t roleIndex) {
        if (roleIndex >= cfg.channel.particles.size()) {
            double compositeDistance = 0.0;
            CutDecision compositeDecision;
            if (!evaluateCompositeRank(selection, cfg, compositeDistance, compositeDecision)) {
                ++stats.compositeFailures;
                stats.addFailures(compositeDecision);
                return;
            }

            CandidateOutput candidate;
            buildCandidateOutput(rows, selection, cuts, candidate);
            std::vector<std::string> candidateTaggedCuts = taggedCuts;
            std::vector<std::string> candidateTaggedFailures = taggedFailures;
            appendUnique(candidateTaggedCuts, compositeDecision.tagged);
            appendUnique(candidateTaggedFailures, compositeDecision.taggedFailed);
            appendUnique(candidateTaggedCuts, splitCsv(candidate.eppi0_evaluatedCuts));
            appendUnique(candidateTaggedFailures, splitCsv(candidate.eppi0_failedCuts));
            candidate.eppi0_evaluatedCuts = joinCsv(candidateTaggedCuts);
            candidate.eppi0_failedCuts = joinCsv(candidateTaggedFailures);

            CandidateChoiceKey key;
            if (!makeCandidateChoiceKey(candidate, cfg, compositeDistance, key)) return;
            if (!found || candidateChoiceLess(key, bestKey)) {
                bestKey = std::move(key);
                best = candidate;
                found = true;
            }
            return;
        }

        const ParticleRoleSpec& role = cfg.channel.particles[roleIndex];
        std::vector<const RecBranches*> chosen;

        std::function<void(size_t)> chooseParticle;
        chooseParticle = [&](size_t start) {
            if (chosen.size() == static_cast<size_t>(role.count)) {
                selection[role.role] = chosen;
                visitRole(roleIndex + 1);
                selection.erase(role.role);
                return;
            }

            for (size_t i = start; i < rows.recs.size(); ++i) {
                const RecBranches* candidate = &rows.recs[i];
                if (candidate->pid != role.pid || alreadySelected(candidate)) continue;

                const auto context = selectedContext();
                const CutDecision decision = cuts.evaluateParticle(*candidate, role, rows.recs, context);
                if (!decision.pass) {
                    stats.addFailures(decision);
                    continue;
                }

                const size_t taggedCutCount = taggedCuts.size();
                const size_t taggedFailureCount = taggedFailures.size();
                appendUnique(taggedCuts, decision.tagged);
                appendUnique(taggedFailures, decision.taggedFailed);
                chosen.push_back(candidate);
                chooseParticle(i + 1);
                chosen.pop_back();
                taggedCuts.resize(taggedCutCount);
                taggedFailures.resize(taggedFailureCount);
            }
        };

        chooseParticle(0);
    };

    visitRole(0);

    if (!found) {
        ++stats.eventsWithoutSavedCandidate;
        return false;
    }
    if (runEppi0 && !best.eppi0_passExclusivity && !cfg.saveFailedCandidates) {
        ++stats.exclusivityFailures;
        addCsvFailures(best.eppi0_failedCuts, stats);
        ++stats.eventsWithoutSavedCandidate;
        return false;
    }
    out = best;
    return true;
}

}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: post_process <post_config.json> <input.root> [progress_rows]\n";
        return 1;
    }

    long long progressEvery = 1000000;
    if (argc >= 4) {
        progressEvery = std::stoll(argv[3]);
        if (progressEvery < 0) progressEvery = 0;
    }

    const PostCutConfig cfg = PostCutConfig::fromFile(argv[1]);
    const Cuts cuts(cfg);

    TFile input(argv[2], "READ");
    if (input.IsZombie()) {
        std::cerr << "[ERROR] Could not open input ROOT file: " << argv[2] << "\n";
        return 1;
    }

    std::string inputTreeName = TreeNames::rParticles;
    auto* inTree = dynamic_cast<TTree*>(input.Get(TreeNames::rParticles));
    if (!inTree) {
        inputTreeName = TreeNames::legacyRParticles;
        inTree = dynamic_cast<TTree*>(input.Get(TreeNames::legacyRParticles));
    }
    if (!inTree) {
        inputTreeName = TreeNames::legacyEvents;
        inTree = dynamic_cast<TTree*>(input.Get(TreeNames::legacyEvents));
    }
    if (!inTree) {
        std::cerr << "[ERROR] Could not find rParticles or a legacy particle tree\n";
        return 1;
    }
    if (inputTreeName != TreeNames::rParticles) {
        std::cerr << "[WARN] Using legacy particle tree " << inputTreeName
                  << "; reconvert to obtain rParticles\n";
    }
    const Long64_t nEntries = inTree->GetEntries();
    std::string inputEventTreeName = TreeNames::rEvents;
    auto* inEventTree = dynamic_cast<TTree*>(input.Get(TreeNames::rEvents));
    if (!inEventTree) {
        inputEventTreeName = TreeNames::legacyREvents;
        inEventTree = dynamic_cast<TTree*>(input.Get(TreeNames::legacyREvents));
    }
    ReconstructedEventReader reconstructedEventReader(inEventTree);

    const char* outputTreeName = cfg.outputMode == "matchedRows"
        ? TreeNames::rParticles
        : TreeNames::sEvents;

    std::cout << "[INFO] Config file : " << argv[1] << "\n"
              << "[INFO] Input file  : " << argv[2] << "\n"
              << "[INFO] Input tree  : " << inputTreeName << "\n"
              << "[INFO] Event tree  : "
              << (reconstructedEventReader.available() ? inputEventTreeName : "unavailable")
              << "\n"
              << "[INFO] Input rows  : " << nEntries << "\n"
              << "[INFO] Output file : " << cfg.outputFile << "\n"
              << "[INFO] Output tree : " << outputTreeName << "\n"
              << "[INFO] Output mode : " << cfg.outputMode << "\n"
              << "[INFO] Beam energy : " << cfg.beamEnergy << " GeV\n"
              << "[INFO] Torus       : " << cfg.torus << "\n"
              << "[INFO] Progress    : "
              << (progressEvery > 0 ? std::to_string(progressEvery) + " rows" : "disabled")
              << "\n";
    printChannelSummary(cfg);

    EventBranches* event = nullptr;
    RecBranches* rec = nullptr;
    GenBranches* gen = nullptr;
    inTree->SetBranchAddress("event", &event);
    inTree->SetBranchAddress("rec", &rec);
    const bool hasGenBranch = inTree->GetBranch("gen") != nullptr;
    if (hasGenBranch) inTree->SetBranchAddress("gen", &gen);

    TFile output(cfg.outputFile.c_str(), "RECREATE");
    TTree outTree(outputTreeName, outputTreeName);

    if (cfg.outputMode == "matchedRows") {
        if (cfg.channel.particles.size() != 1) {
            std::cerr << "[ERROR] outputMode=matchedRows requires exactly one particle role\n";
            return 1;
        }
        if (!hasGenBranch) {
            std::cerr << "[ERROR] outputMode=matchedRows requires an input gen branch\n";
            return 1;
        }

        EventBranches outEvent;
        RecBranches outRec;
        GenBranches outGen;
        outTree.Branch("event", &outEvent);
        outTree.Branch("rec", &outRec);
        outTree.Branch("gen", &outGen);

        ProcessingStats stats;
        long long nInputRows = 0;
        long long nWritten = 0;
        long long lastProgressRow = 0;
        const Clock::time_point startTime = Clock::now();
        const ParticleRoleSpec& role = cfg.channel.particles.front();

        for (Long64_t i = 0; i < nEntries; ++i) {
            inTree->GetEntry(i);
            ++nInputRows;
            if (progressEvery > 0 && nInputRows - lastProgressRow >= progressEvery) {
                printProgress(nInputRows, nEntries, nInputRows, nWritten, stats, startTime);
                lastProgressRow = nInputRows;
            }

            if (!event || !rec || !gen || rec->pid == -999) continue;
            if (rec->pid != role.pid) continue;

            const std::vector<RecBranches> eventParticles{*rec};
            const std::map<std::string, const RecBranches*> selected;
            const CutDecision decision = cuts.evaluateParticle(*rec, role, eventParticles, selected);
            if (!decision.pass) {
                stats.addFailures(decision);
                continue;
            }

            outEvent = *event;
            outRec = *rec;
            outGen = *gen;
            outTree.Fill();
            ++nWritten;
        }

        if (progressEvery > 0 && nInputRows != lastProgressRow) {
            printProgress(nInputRows, nEntries, nInputRows, nWritten, stats, startTime);
        }

        output.Write();
        output.Close();

        const double elapsed = std::chrono::duration<double>(Clock::now() - startTime).count();
        const double savedFraction = 100.0 * fraction(nWritten, nInputRows);
        std::cout << "[DONE]\n"
                  << "  Input rows       : " << nInputRows << "\n"
                  << "  Rows saved       : " << nWritten << "\n"
                  << "  Selection yield  : " << std::fixed << std::setprecision(2)
                  << savedFraction << "%\n"
                  << "  Elapsed time     : " << std::fixed << std::setprecision(1)
                  << elapsed << " s\n"
                  << "  Output file      : " << cfg.outputFile << "\n";

        if (!stats.cutFailures.empty()) {
            std::cout << "  Cut rejection counts:\n";
            for (const auto& [name, count] : stats.cutFailures) {
                std::cout << "    " << name << ": " << count << "\n";
            }
        }
        return 0;
    }

    if (cfg.outputMode != "candidates") {
        std::cerr << "[ERROR] Unsupported outputMode: " << cfg.outputMode << "\n";
        return 1;
    }

    CandidateOutput out;
    out.registerBranches(outTree, supportsEppi0Logic(cfg));
    SelectedRoleBranches selectedRoleBranches(cfg);
    selectedRoleBranches.registerBranches(outTree);
    PidMultiplicityBranches pidMultiplicityBranches(cfg);
    pidMultiplicityBranches.registerBranches(outTree);
    TTree selectedParticleTree(TreeNames::sParticles, TreeNames::sParticles);
    SelectedParticleOutput selectedParticleOutput;
    selectedParticleOutput.registerBranches(selectedParticleTree);

    EventRows rows;
    ProcessingStats stats;
    bool haveRows = false;
    long long nInputRows = 0;
    long long nEvents = 0;
    long long nWritten = 0;
    long long lastProgressRow = 0;
    const Clock::time_point startTime = Clock::now();

    const auto flushEvent = [&]() {
        if (!haveRows || rows.recs.empty()) return;
        ++nEvents;
        CandidateOutput candidate;
        if (processEvent(rows, cuts, candidate, stats)) {
            reconstructedEventReader.copyFor(rows.event, candidate);
            out = candidate;
            selectedRoleBranches.fill(out);
            pidMultiplicityBranches.fill(out);
            outTree.Fill();
            selectedParticleOutput.fillRows(out, selectedParticleTree);
            ++nWritten;
        }
    };

    for (Long64_t i = 0; i < nEntries; ++i) {
        inTree->GetEntry(i);
        ++nInputRows;
        if (progressEvery > 0 && nInputRows - lastProgressRow >= progressEvery) {
            printProgress(nInputRows, nEntries, nEvents, nWritten, stats, startTime);
            lastProgressRow = nInputRows;
        }

        if (!event || !rec || rec->pid == -999) continue;

        const bool sourceAware = event->sourceFileId != INVALID_SOURCE_ID &&
                                 rows.event.sourceFileId != INVALID_SOURCE_ID;
        const bool newEvent = haveRows &&
            (sourceAware
                ? (event->sourceFileId != rows.event.sourceFileId ||
                   event->sourceEventIndex != rows.event.sourceEventIndex)
                : (event->runNum != rows.event.runNum ||
                   event->eventNum != rows.event.eventNum));
        if (newEvent) {
            flushEvent();
            rows.clear();
        }

        if (!haveRows || rows.recs.empty()) rows.event = *event;
        rows.recs.push_back(*rec);
        haveRows = true;
    }

    flushEvent();
    if (progressEvery > 0 && nInputRows != lastProgressRow) {
        printProgress(nInputRows, nEntries, nEvents, nWritten, stats, startTime);
    }

    output.Write();
    output.Close();

    const double elapsed = std::chrono::duration<double>(Clock::now() - startTime).count();
    const double savedFraction = 100.0 * fraction(nWritten, nEvents);

    std::cout << "[DONE]\n"
              << "  Input rows       : " << nInputRows << "\n"
              << "  Events processed : " << nEvents << "\n"
              << "  Candidates saved : " << nWritten << "\n"
              << "  Selection yield  : " << std::fixed << std::setprecision(2)
              << savedFraction << "%\n"
              << "  Events rejected  : " << stats.eventsWithoutSavedCandidate << "\n"
              << "  Composite rejects: " << stats.compositeFailures << "\n"
              << "  Exclusivity rejects: " << stats.exclusivityFailures << "\n"
              << "  Elapsed time     : " << std::fixed << std::setprecision(1)
              << elapsed << " s\n"
              << "  Output file      : " << cfg.outputFile << "\n";

    if (!stats.cutFailures.empty()) {
        std::cout << "  Cut rejection counts:\n";
        for (const auto& [name, count] : stats.cutFailures) {
            std::cout << "    " << name << ": " << count << "\n";
        }
    }

    return 0;
}
