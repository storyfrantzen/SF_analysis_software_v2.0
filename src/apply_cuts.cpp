#include <algorithm>
#include <chrono>
#include <cmath>
#include <functional>
#include <iomanip>
#include <map>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>
#include <vector>

#include "TFile.h"
#include "TLorentzVector.h"
#include "TTree.h"

#include "Cuts.h"
#include "Kinematics.h"
#include "ROOTBranches.h"

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
    std::cout << "\n";
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
    std::vector<double> selectedP;
    std::vector<double> selectedTheta;
    std::vector<double> selectedPhi;

    double charge = NAN;

    int electronIdx = -999;
    int electronDet = -999;
    int electronSector = -999;
    double electronP = NAN;
    double electronTheta = NAN;
    double electronPhi = NAN;
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
        tree.Branch("selectedP", &selectedP);
        tree.Branch("selectedTheta", &selectedTheta);
        tree.Branch("selectedPhi", &selectedPhi);
        tree.Branch("electronIdx", &electronIdx, "electronIdx/I");
        tree.Branch("electronDet", &electronDet, "electronDet/I");
        tree.Branch("electronSector", &electronSector, "electronSector/I");
        tree.Branch("electronP", &electronP, "electronP/D");
        tree.Branch("electronTheta", &electronTheta, "electronTheta/D");
        tree.Branch("electronPhi", &electronPhi, "electronPhi/D");
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
        tree.Branch("passFiducial", &eppi0_passFiducial, "passFiducial/I");
        tree.Branch("passSamplingFraction", &eppi0_passSamplingFraction, "passSamplingFraction/I");
        tree.Branch("passExclusivity", &eppi0_passExclusivity, "passExclusivity/I");
        tree.Branch("failedCuts", &eppi0_failedCuts);
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

void fillElectronBranches(const Selection& selection,
                          CandidateOutput& out) {
    if (const RecBranches* electron = firstParticle(selection, "electron")) {
        out.electronIdx = electron->particleIdx;
        out.electronDet = electron->det;
        out.electronSector = electron->sector;
        out.electronP = electron->p;
        out.electronTheta = electron->theta;
        out.electronPhi = electron->phi;
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
                           std::string& failedCut) {
    rank = 0.0;
    bool usedComposite = false;
    failedCut.clear();

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
        if (std::isfinite(composite.window) && delta > composite.window) {
            failedCut = composite.role + ".mass_window";
            return false;
        }
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
    out.sourceFileId = rows.event.sourceFileId;
    out.sourceEventIndex = rows.event.sourceEventIndex;
    out.runNum = rows.event.runNum;
    out.eventNum = rows.event.eventNum;
    out.helicity = rows.event.helicity;
    out.charge = rows.event.charge;
    out.passTopology = 1;
    fillSelectedParticleBranches(selection, cfg, out);
    fillElectronBranches(selection, out);
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
    const CutDecision fiducial = cuts.evaluateFiducial(e);
    CutDecision fiducialP = cuts.evaluateFiducial(p);
    CutDecision fiducialG1 = cuts.evaluateFiducial(g1);
    CutDecision fiducialG2 = cuts.evaluateFiducial(g2);
    fiducialP.merge(fiducialG1);
    fiducialP.merge(fiducialG2);
    fiducialP.merge(fiducial);
    out.eppi0_passFiducial = fiducialP.pass;
    out.eppi0_passSamplingFraction = cuts.evaluateSamplingFraction(e).pass;

    out.y = dis.y;
    out.W = dis.W;
    out.t = -1.0 * (target - lvP).M2();
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
            std::string failedCompositeCut;
            if (!evaluateCompositeRank(selection, cfg, rank, failedCompositeCut)) {
                ++stats.compositeFailures;
                ++stats.cutFailures[failedCompositeCut.empty() ? "composite" : failedCompositeCut];
                return;
            }

            CandidateOutput candidate;
            buildCandidateOutput(rows, selection, cuts, candidate);
            if (runEppi0 && !candidate.eppi0_passExclusivity && !cfg.saveFailedCandidates) {
                ++stats.exclusivityFailures;
                addCsvFailures(candidate.eppi0_failedCuts, stats);
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
                const CutDecision decision = cuts.evaluateParticle(*candidate, role, rows.recs, context);
                if (!decision.pass) {
                    stats.addFailures(decision);
                    continue;
                }

                chosen.push_back(candidate);
                chooseParticle(i + 1);
                chosen.pop_back();
            }
        };

        chooseParticle(0);
    };

    visitRole(0);

    if (!found) {
        ++stats.eventsWithoutSavedCandidate;
        return false;
    }
    out = best;
    return true;
}

}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: apply_cuts <post_config.json> <input.root> [progress_rows]\n";
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

    auto* inTree = dynamic_cast<TTree*>(input.Get(cfg.inputTree.c_str()));
    if (!inTree) {
        std::cerr << "[ERROR] Could not find input tree: " << cfg.inputTree << "\n";
        return 1;
    }
    const Long64_t nEntries = inTree->GetEntries();

    std::cout << "[INFO] Config file : " << argv[1] << "\n"
              << "[INFO] Input file  : " << argv[2] << "\n"
              << "[INFO] Input tree  : " << cfg.inputTree << "\n"
              << "[INFO] Input rows  : " << nEntries << "\n"
              << "[INFO] Output file : " << cfg.outputFile << "\n"
              << "[INFO] Output tree : " << cfg.outputTree << "\n"
              << "[INFO] Beam energy : " << cfg.beamEnergy << " GeV\n"
              << "[INFO] Torus       : " << cfg.torus << "\n"
              << "[INFO] Progress    : "
              << (progressEvery > 0 ? std::to_string(progressEvery) + " rows" : "disabled")
              << "\n";
    printChannelSummary(cfg);

    EventBranches* event = nullptr;
    RecBranches* rec = nullptr;
    inTree->SetBranchAddress("event", &event);
    inTree->SetBranchAddress("rec", &rec);

    TFile output(cfg.outputFile.c_str(), "RECREATE");
    TTree outTree(cfg.outputTree.c_str(), cfg.outputTree.c_str());
    CandidateOutput out;
    out.registerBranches(outTree, supportsEppi0Logic(cfg));

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
            out = candidate;
            outTree.Fill();
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
