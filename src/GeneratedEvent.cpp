#include "GeneratedEvent.h"

#include <vector>

#include "TTree.h"
#include "TLorentzVector.h"

#include "clas12reader.h"

#include "Kinematics.h"

void GeneratedEventBranches::reset() {
    sourceFileId = INVALID_SOURCE_ID;
    sourceEventIndex = INVALID_SOURCE_ID;
    runNum = -999;
    eventNum = -999;
    topologyValid = false;
    radiative = false;
    weight = 1.0;
    Q2 = nu = xB = y = W = minusT = trentoPhi = NAN;
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
    const Kinematics::DIS dis = Kinematics::dis(electron, beamEnergy);
    Q2 = dis.Q2;
    nu = dis.nu;
    xB = dis.xB;
    y = dis.y;
    W = dis.W;
    minusT = -1.0 * (Kinematics::target() - proton).M2();
    trentoPhi = Kinematics::trentoPhi(Kinematics::beam(beamEnergy), electron, proton);
}
