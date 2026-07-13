#pragma once

#include <cstdint>
#include <cmath>

#include "ROOTBranches.h"

class TTree;

namespace clas12 {
class mcparticle;
}

struct GeneratedEventBranches {
    std::uint64_t sourceFileId = INVALID_SOURCE_ID;
    std::uint64_t sourceEventIndex = INVALID_SOURCE_ID;
    int runNum = -999;
    int eventNum = -999;
    bool topologyValid = false;
    bool radiative = false;

    double weight = 1.0;
    double Q2 = NAN;
    double nu = NAN;
    double xB = NAN;
    double y = NAN;
    double W = NAN;
    double minusT = NAN;
    double trentoPhi = NAN;
    double electronP = NAN;
    double electronTheta = NAN;
    double electronPhi = NAN;
    double protonP = NAN;
    double protonTheta = NAN;
    double protonPhi = NAN;
    double gamma1P = NAN;
    double gamma1Theta = NAN;
    double gamma1Phi = NAN;
    double gamma2P = NAN;
    double gamma2Theta = NAN;
    double gamma2Phi = NAN;
    double pi0P = NAN;
    double pi0Theta = NAN;
    double pi0Phi = NAN;

    void reset();
    void registerBranches(TTree& tree);
    void fill(clas12::mcparticle* particles,
              int runNumber,
              int eventNumber,
              std::uint64_t fileId,
              std::uint64_t eventIndex,
              double beamEnergy);
};
