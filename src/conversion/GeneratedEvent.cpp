#include "GeneratedEvent.h"

#include <vector>

#include "TTree.h"
#include "TLorentzVector.h"

#include "clas12reader.h"

#include "Kinematics.h"

namespace {

void fillKinematics(const TLorentzVector& particle,
                    double& p,
                    double& theta,
                    double& phi) {
    p = particle.P();
    theta = particle.Theta();
    phi = particle.Phi();
}

}  // namespace

void GeneratedEventBranches::reset() {
    sourceFileId = INVALID_SOURCE_ID;
    sourceEventIndex = INVALID_SOURCE_ID;
    runNum = -999;
    eventNum = -999;
    topologyValid = false;
    radiative = false;
    weight = 1.0;
    Q2 = nu = xB = y = W = minusT = trentoPhi = NAN;
    electronP = electronTheta = electronPhi = NAN;
    protonP = protonTheta = protonPhi = NAN;
    gamma1P = gamma1Theta = gamma1Phi = NAN;
    gamma2P = gamma2Theta = gamma2Phi = NAN;
    pi0P = pi0Theta = pi0Phi = NAN;
}

void GeneratedEventBranches::registerBranches(TTree& tree) {
    tree.Branch("sourceFileId", &sourceFileId, "sourceFileId/l");
    tree.Branch("sourceEventIndex", &sourceEventIndex, "sourceEventIndex/l");
    tree.Branch("runNum", &runNum, "runNum/I");
    tree.Branch("eventNum", &eventNum, "eventNum/I");
    tree.Branch("topologyValid", &topologyValid, "topologyValid/O");
    tree.Branch("radiative", &radiative, "radiative/O");
    tree.Branch("weight", &weight, "weight/D");
    tree.Branch("Q2", &Q2, "Q2/D");
    tree.Branch("nu", &nu, "nu/D");
    tree.Branch("xB", &xB, "xB/D");
    tree.Branch("y", &y, "y/D");
    tree.Branch("W", &W, "W/D");
    tree.Branch("minusT", &minusT, "minusT/D");
    tree.Branch("trentoPhi", &trentoPhi, "trentoPhi/D");
    tree.Branch("electronP", &electronP, "electronP/D");
    tree.Branch("electronTheta", &electronTheta, "electronTheta/D");
    tree.Branch("electronPhi", &electronPhi, "electronPhi/D");
    tree.Branch("protonP", &protonP, "protonP/D");
    tree.Branch("protonTheta", &protonTheta, "protonTheta/D");
    tree.Branch("protonPhi", &protonPhi, "protonPhi/D");
    tree.Branch("gamma1P", &gamma1P, "gamma1P/D");
    tree.Branch("gamma1Theta", &gamma1Theta, "gamma1Theta/D");
    tree.Branch("gamma1Phi", &gamma1Phi, "gamma1Phi/D");
    tree.Branch("gamma2P", &gamma2P, "gamma2P/D");
    tree.Branch("gamma2Theta", &gamma2Theta, "gamma2Theta/D");
    tree.Branch("gamma2Phi", &gamma2Phi, "gamma2Phi/D");
    tree.Branch("pi0P", &pi0P, "pi0P/D");
    tree.Branch("pi0Theta", &pi0Theta, "pi0Theta/D");
    tree.Branch("pi0Phi", &pi0Phi, "pi0Phi/D");
}

void GeneratedEventBranches::fill(clas12::mcparticle* particles,
                                  int runNumber,
                                  int eventNumber,
                                  std::uint64_t fileId,
                                  std::uint64_t eventIndex,
                                  double beamEnergy) {
    reset();
    sourceFileId = fileId;
    sourceEventIndex = eventIndex;
    runNum = runNumber;
    eventNum = eventNumber;
    if (!particles) return;

    TLorentzVector electron;
    TLorentzVector proton;
    TLorentzVector pi0;
    std::vector<TLorentzVector> photons;
    bool haveElectron = false;
    bool haveProton = false;
    bool havePi0 = false;

    for (int index = 0; index < particles->getRows(); ++index) {
        const int pid = particles->getPid(index);
        const TLorentzVector particle = Kinematics::particle(
            particles->getPx(index),
            particles->getPy(index),
            particles->getPz(index),
            Kinematics::massForPid(pid));

        if (pid == 11 && !haveElectron) {
            electron = particle;
            haveElectron = true;
        } else if (pid == 2212 && !haveProton) {
            proton = particle;
            haveProton = true;
        } else if (pid == 111 && !havePi0) {
            pi0 = particle;
            havePi0 = true;
        } else if (pid == 22) {
            photons.push_back(particle);
        }
    }

    radiative = havePi0;
    const bool haveMeson = radiative ? !photons.empty() : photons.size() >= 2;
    topologyValid = haveElectron && haveProton && haveMeson;
    if (!topologyValid) return;

    if (!radiative) pi0 = photons[0] + photons[1];
    fillKinematics(electron, electronP, electronTheta, electronPhi);
    fillKinematics(proton, protonP, protonTheta, protonPhi);
    if (!photons.empty()) fillKinematics(photons[0], gamma1P, gamma1Theta, gamma1Phi);
    if (photons.size() > 1) fillKinematics(photons[1], gamma2P, gamma2Theta, gamma2Phi);
    fillKinematics(pi0, pi0P, pi0Theta, pi0Phi);
    const Kinematics::DIS dis = Kinematics::dis(electron, beamEnergy);
    Q2 = dis.Q2;
    nu = dis.nu;
    xB = dis.xB;
    y = dis.y;
    W = dis.W;
    minusT = -1.0 * (Kinematics::target() - proton).M2();
    trentoPhi = Kinematics::trentoPhi(Kinematics::beam(beamEnergy), electron, proton);
}
