#include <algorithm>
#include <cmath>
#include <functional>
#include <map>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include "TFile.h"
#include "TLorentzVector.h"
#include "TTree.h"
#include "TVector2.h"
#include "TVector3.h"

#include "Cuts.h"
#include "PhysicalConstants.h"
#include "ROOTBranches.h"

namespace {

constexpr double kPi = 3.14159265358979323846;
constexpr double kTargetMass = 0.9382720813;

using Selection = std::map<std::string, std::vector<const RecBranches*>>;

struct EventRows {
    EventBranches event;
    std::vector<RecBranches> recs;

    void clear() {
        event.reset();
        recs.clear();
    }
};

struct CandidateOutput {
    int runNum = -999;
    int eventNum = -999;
    int helicity = -999;
    int passTopology = 0;
    std::vector<std::string> selectedRoles;
    std::vector<int> selectedIdx;
    std::vector<int> selectedPid;
    std::vector<int> selectedDet;
    std::vector<double> selectedP;
    std::vector<double> selectedTheta;
    std::vector<double> selectedPhi;

    double charge = NAN;

    int eppi0_eIdx = -999;
    int eppi0_pIdx = -999;
    int eppi0_g1Idx = -999;
    int eppi0_g2Idx = -999;
    int eppi0_eDet = -999;
    int eppi0_pDet = -999;
    int eppi0_g1Det = -999;
    int eppi0_g2Det = -999;
    int eppi0_passFiducial = 0;
    int eppi0_passSamplingFraction = 0;
    int eppi0_passExclusivity = 0;
    std::string eppi0_failedCuts;
    double Q2 = NAN;
    double nu = NAN;
    double xB = NAN;
    double y = NAN;
    double W = NAN;
    double t = NAN;
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
        tree.Branch("runNum", &runNum, "runNum/I");
        tree.Branch("eventNum", &eventNum, "eventNum/I");
        tree.Branch("helicity", &helicity, "helicity/I");
        tree.Branch("charge", &charge, "charge/D");
        tree.Branch("passTopology", &passTopology, "passTopology/I");
        tree.Branch("selectedRoles", &selectedRoles);
        tree.Branch("selectedIdx", &selectedIdx);
        tree.Branch("selectedPid", &selectedPid);
        tree.Branch("selectedDet", &selectedDet);
        tree.Branch("selectedP", &selectedP);
        tree.Branch("selectedTheta", &selectedTheta);
        tree.Branch("selectedPhi", &selectedPhi);
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
        tree.Branch("passFiducial", &eppi0_passFiducial, "passFiducial/I");
        tree.Branch("passSamplingFraction", &eppi0_passSamplingFraction, "passSamplingFraction/I");
        tree.Branch("passExclusivity", &eppi0_passExclusivity, "passExclusivity/I");
        tree.Branch("failedCuts", &eppi0_failedCuts);
        tree.Branch("Q2", &Q2, "Q2/D");
        tree.Branch("nu", &nu, "nu/D");
        tree.Branch("xB", &xB, "xB/D");
        tree.Branch("y", &y, "y/D");
        tree.Branch("W", &W, "W/D");
        tree.Branch("t", &t, "t/D");
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

double massForPid(int pid) {
    switch (std::abs(pid)) {
        case 11: return M_ELECTRON;
        case 22: return 0.0;
        case 111: return M_PI0;
        case 2212: return M_PROTON;
        default: return 0.0;
    }
}

TLorentzVector particleLV(const RecBranches& p, double mass) {
    TLorentzVector lv;
    lv.SetPxPyPzE(p.px, p.py, p.pz, std::sqrt(mass * mass + p.p * p.p));
    return lv;
}

double computeTrentoPhi(const TLorentzVector& beam,
                        const TLorentzVector& electron,
                        const TLorentzVector& hadron) {
    const TVector3 q = beam.Vect() - electron.Vect();
    const TVector3 nLepton = beam.Vect().Cross(electron.Vect()).Unit();
    const TVector3 nHadron = hadron.Vect().Cross(q).Unit();
    const double cosPhi = nLepton.Dot(nHadron);
    const double sinPhi = q.Unit().Dot(nLepton.Cross(nHadron));
    return std::atan2(sinPhi, cosPhi);
}

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
            out.selectedP.push_back(particle->p);
            out.selectedTheta.push_back(particle->theta);
            out.selectedPhi.push_back(particle->phi);
        }
    }
}

bool evaluateCompositeRank(const Selection& selection,
                           const PostCutConfig& cfg,
                           double& rank) {
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

        const TLorentzVector lvLeft = particleLV(*left, massForPid(left->pid));
        const TLorentzVector lvRight = particleLV(*right, massForPid(right->pid));
        const double mass = (lvLeft + lvRight).M();
        const double delta = std::abs(mass - composite.mass);
        if (std::isfinite(composite.window) && delta > composite.window) return false;
        rank += delta;
        usedComposite = true;
    }

    if (!usedComposite) rank = 0.0;
    return true;
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
    out.runNum = rows.event.runNum;
    out.eventNum = rows.event.eventNum;
    out.helicity = rows.event.helicity;
    out.charge = rows.event.charge;
    out.passTopology = 1;
    fillSelectedParticleBranches(selection, cfg, out);
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

    const TLorentzVector beam(0, 0, cfg.beamEnergy, cfg.beamEnergy);
    const TLorentzVector target(0, 0, 0, kTargetMass);
    const TLorentzVector lvE = particleLV(e, M_ELECTRON);
    const TLorentzVector lvP = particleLV(p, M_PROTON);

    TLorentzVector lvG1, lvG2;
    lvG1.SetPxPyPzE(g1.px, g1.py, g1.pz, g1.p);
    lvG2.SetPxPyPzE(g2.px, g2.py, g2.pz, g2.p);
    const TLorentzVector lvPi0 = lvG1 + lvG2;

    const TLorentzVector q = beam - lvE;
    const TLorentzVector missing = beam + target - lvE - lvP - lvPi0;
    const TLorentzVector epX = beam + target - lvE - lvP;
    const TLorentzVector epi0X = beam + target - lvE - lvPi0;

    out.eppi0_eIdx = e.particleIdx;
    out.eppi0_pIdx = p.particleIdx;
    out.eppi0_g1Idx = g1.particleIdx;
    out.eppi0_g2Idx = g2.particleIdx;
    out.eppi0_eDet = e.det;
    out.eppi0_pDet = p.det;
    out.eppi0_g1Det = g1.det;
    out.eppi0_g2Det = g2.det;
    out.eppi0_passFiducial = 1;
    out.eppi0_passSamplingFraction = 1;

    out.Q2 = -q.M2();
    out.nu = cfg.beamEnergy - lvE.E();
    out.xB = out.Q2 / (2.0 * kTargetMass * out.nu);
    out.y = out.nu / cfg.beamEnergy;
    out.W = std::sqrt(std::max(0.0, (target + q).M2()));
    out.t = -1.0 * (target - lvP).M2();
    out.trentoPhi = computeTrentoPhi(beam, lvE, lvP);
    out.pi0_p = lvPi0.P();
    out.pi0_theta = lvPi0.Theta();
    out.pi0_phi = lvPi0.Phi();
    out.pi0_deltaPhi = TVector2::Phi_mpi_pi(lvPi0.Phi() - epX.Phi());
    out.pi0_thetaX = lvPi0.Vect().Angle(epX.Vect());
    out.m_gg = lvPi0.M();
    out.m2_miss = missing.M2();
    out.m2_epX = epX.M2();
    out.m2_epi0X = epi0X.M2();
    out.m_eggX = out.m2_epi0X >= 0.0 ? std::sqrt(out.m2_epi0X) : NAN;
    out.E_miss = missing.E();
    out.pT_miss = missing.Pt();
    out.theta_e_g1 = lvE.Vect().Angle(lvG1.Vect());
    out.theta_e_g2 = lvE.Vect().Angle(lvG2.Vect());
    out.theta_g1_g2 = lvG1.Vect().Angle(lvG2.Vect());
    const CutDecision exclusivity = cuts.evaluateLooseExclusivity({
        out.E_miss,
        out.theta_e_g1 * 180.0 / kPi,
        out.theta_e_g2 * 180.0 / kPi,
        out.theta_g1_g2 * 180.0 / kPi,
        out.pi0_thetaX * 180.0 / kPi
    });
    out.eppi0_passExclusivity = exclusivity.pass;
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

bool processEvent(const EventRows& rows, const Cuts& cuts, CandidateOutput& out) {
    const auto& cfg = cuts.config();
    const bool runEppi0 = supportsEppi0Logic(cfg);

    double bestRank = std::numeric_limits<double>::max();
    CandidateOutput best;
    bool found = false;
    Selection selection;

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
            double rank = 0.0;
            if (!evaluateCompositeRank(selection, cfg, rank)) return;

            CandidateOutput candidate;
            buildCandidateOutput(rows, selection, cuts, candidate);
            if (runEppi0 && !candidate.eppi0_passExclusivity && !cfg.saveFailedCandidates) {
                return;
            }
            if (!found || rank < bestRank) {
                bestRank = rank;
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
                if (!cuts.evaluateParticle(*candidate, role, context).pass) continue;

                chosen.push_back(candidate);
                chooseParticle(i + 1);
                chosen.pop_back();
            }
        };

        chooseParticle(0);
    };

    visitRole(0);

    if (!found) return false;
    out = best;
    return true;
}

}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: apply_cuts <post_config.json> <input.root>\n";
        return 1;
    }

    const PostCutConfig cfg = PostCutConfig::fromFile(argv[1]);
    const Cuts cuts(cfg);

    TFile input(argv[2], "READ");
    if (input.IsZombie()) {
        std::cerr << "[ERROR] Could not open input ROOT file: " << argv[2] << "\n";
        return 1;
    }

    auto* inTree = dynamic_cast<TTree*>(input.Get(cfg.inputTree.c_str()));
    if (!inTree) {
        std::cerr << "[ERROR] Could not find input tree: " << cfg.inputTree << "\n";
        return 1;
    }

    EventBranches* event = nullptr;
    RecBranches* rec = nullptr;
    inTree->SetBranchAddress("event", &event);
    inTree->SetBranchAddress("rec", &rec);

    TFile output(cfg.outputFile.c_str(), "RECREATE");
    TTree outTree(cfg.outputTree.c_str(), cfg.outputTree.c_str());
    CandidateOutput out;
    out.registerBranches(outTree, supportsEppi0Logic(cfg));

    EventRows rows;
    bool haveRows = false;
    long long nInputRows = 0;
    long long nEvents = 0;
    long long nWritten = 0;

    const auto flushEvent = [&]() {
        if (!haveRows || rows.recs.empty()) return;
        ++nEvents;
        CandidateOutput candidate;
        if (processEvent(rows, cuts, candidate)) {
            out = candidate;
            outTree.Fill();
            ++nWritten;
        }
    };

    const Long64_t nEntries = inTree->GetEntries();
    for (Long64_t i = 0; i < nEntries; ++i) {
        inTree->GetEntry(i);
        ++nInputRows;
        if (!event || !rec || rec->pid == -999) continue;

        const bool newEvent = haveRows &&
                              (event->runNum != rows.event.runNum ||
                               event->eventNum != rows.event.eventNum);
        if (newEvent) {
            flushEvent();
            rows.clear();
        }

        if (!haveRows || rows.recs.empty()) rows.event = *event;
        rows.recs.push_back(*rec);
        haveRows = true;
    }

    flushEvent();

    output.Write();
    output.Close();

    std::cout << "[DONE]\n"
              << "  Input rows       : " << nInputRows << "\n"
              << "  Events processed : " << nEvents << "\n"
              << "  Candidates saved : " << nWritten << "\n"
              << "  Output file      : " << cfg.outputFile << "\n";

    return 0;
}
