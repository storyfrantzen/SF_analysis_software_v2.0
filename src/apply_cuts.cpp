#include <algorithm>
#include <cmath>
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

struct EventRows {
    EventBranches event;
    std::vector<RecBranches> recs;

    void clear() {
        event.reset();
        recs.clear();
    }
};

struct OutputRow {
    int runNum = -999;
    int eventNum = -999;
    int helicity = -999;
    int eIdx = -999;
    int pIdx = -999;
    int g1Idx = -999;
    int g2Idx = -999;
    int eDet = -999;
    int pDet = -999;
    int g1Det = -999;
    int g2Det = -999;
    int passTopology = 0;
    int passFiducial = 0;
    int passSamplingFraction = 0;
    int passExclusivity = 0;

    double charge = NAN;
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

    void reset() { *this = OutputRow{}; }

    void makeBranches(TTree& tree) {
        tree.Branch("runNum", &runNum, "runNum/I");
        tree.Branch("eventNum", &eventNum, "eventNum/I");
        tree.Branch("helicity", &helicity, "helicity/I");
        tree.Branch("charge", &charge, "charge/D");
        tree.Branch("eIdx", &eIdx, "eIdx/I");
        tree.Branch("pIdx", &pIdx, "pIdx/I");
        tree.Branch("g1Idx", &g1Idx, "g1Idx/I");
        tree.Branch("g2Idx", &g2Idx, "g2Idx/I");
        tree.Branch("eDet", &eDet, "eDet/I");
        tree.Branch("pDet", &pDet, "pDet/I");
        tree.Branch("g1Det", &g1Det, "g1Det/I");
        tree.Branch("g2Det", &g2Det, "g2Det/I");
        tree.Branch("passTopology", &passTopology, "passTopology/I");
        tree.Branch("passFiducial", &passFiducial, "passFiducial/I");
        tree.Branch("passSamplingFraction", &passSamplingFraction, "passSamplingFraction/I");
        tree.Branch("passExclusivity", &passExclusivity, "passExclusivity/I");
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
};

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

void fillCandidate(const EventRows& rows,
                   const RecBranches& e,
                   const RecBranches& p,
                   const RecBranches& g1,
                   const RecBranches& g2,
                   const Cuts& cuts,
                   OutputRow& out) {
    const auto& cfg = cuts.config();
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

    out.reset();
    out.runNum = rows.event.runNum;
    out.eventNum = rows.event.eventNum;
    out.helicity = rows.event.helicity;
    out.charge = rows.event.charge;
    out.eIdx = e.particleIdx;
    out.pIdx = p.particleIdx;
    out.g1Idx = g1.particleIdx;
    out.g2Idx = g2.particleIdx;
    out.eDet = e.det;
    out.pDet = p.det;
    out.g1Det = g1.det;
    out.g2Det = g2.det;
    out.passTopology = 1;
    out.passFiducial = 1;
    out.passSamplingFraction = 1;

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
    out.passExclusivity = cuts.passesLooseExclusivity(out.E_miss,
                                                      out.theta_e_g1 * 180.0 / kPi,
                                                      out.theta_e_g2 * 180.0 / kPi,
                                                      out.theta_g1_g2 * 180.0 / kPi,
                                                      out.pi0_thetaX * 180.0 / kPi);
}

bool processEvent(const EventRows& rows, const Cuts& cuts, OutputRow& out) {
    std::vector<const RecBranches*> electrons;
    std::vector<const RecBranches*> protons;
    std::vector<const RecBranches*> photons;

    for (const auto& rec : rows.recs) {
        if (rec.pid == 11) electrons.push_back(&rec);
        else if (rec.pid == 2212) protons.push_back(&rec);
        else if (rec.pid == 22) photons.push_back(&rec);
    }

    double bestDeltaM = std::numeric_limits<double>::max();
    OutputRow best;
    bool found = false;

    for (const auto* e : electrons) {
        if (!cuts.passesElectronPreselection(*e)) continue;
        for (const auto* p : protons) {
            if (!cuts.passesProtonPreselection(*p, e->vz)) continue;
            for (size_t i = 0; i < photons.size(); ++i) {
                if (!cuts.passesPhotonPreselection(*photons[i], e->sector)) continue;
                for (size_t j = i + 1; j < photons.size(); ++j) {
                    if (!cuts.passesPhotonPreselection(*photons[j], e->sector)) continue;

                    OutputRow candidate;
                    fillCandidate(rows, *e, *p, *photons[i], *photons[j], cuts, candidate);
                    const double deltaM = std::abs(candidate.m_gg - M_PI0);
                    if (deltaM > cuts.config().pi0MassWindow) continue;
                    if (deltaM < bestDeltaM) {
                        bestDeltaM = deltaM;
                        best = candidate;
                        found = true;
                    }
                }
            }
        }
    }

    if (!found) return false;
    if (!best.passExclusivity && !cuts.config().saveFailedCandidates) return false;
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
    OutputRow out;
    out.makeBranches(outTree);

    EventRows rows;
    bool haveRows = false;
    long long nInputRows = 0;
    long long nEvents = 0;
    long long nWritten = 0;

    const auto flushEvent = [&]() {
        if (!haveRows || rows.recs.empty()) return;
        ++nEvents;
        OutputRow candidate;
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
