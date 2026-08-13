#include <cmath>
#include <iostream>
#include <map>
#include <string>
#include <vector>

#include "Cuts.h"

namespace {
bool evaluateVz(double vz, double min, double max, const std::string& mode = "require") {
    PostCutConfig cfg;
    ParticleRoleSpec electron;
    electron.role = "electron";
    electron.pid = 11;

    PrimitiveCutSpec vertex;
    vertex.name = "electron.vertex";
    vertex.op = "vzRange";
    vertex.mode = mode;
    vertex.min = min;
    vertex.max = max;
    electron.cuts.push_back(vertex);

    RecBranches particle;
    particle.pid = 11;
    particle.vz = vz;

    const Cuts cuts(cfg);
    const auto decision = cuts.evaluateParticle(particle, electron, {particle}, {});
    return decision.pass;
}

bool evaluateMinCalEnergy(const RecBranches& particle, double min) {
    PostCutConfig cfg;
    ParticleRoleSpec gamma;
    gamma.role = "gamma";
    gamma.pid = 22;

    PrimitiveCutSpec calEnergy;
    calEnergy.name = "gamma.cal_energy";
    calEnergy.op = "minCalEnergy";
    calEnergy.min = min;
    gamma.cuts.push_back(calEnergy);

    const Cuts cuts(cfg);
    const auto decision = cuts.evaluateParticle(particle, gamma, {particle}, {});
    return decision.pass;
}
}

int main() {
    if (!evaluateVz(-18.0, -18.0, 10.0)) {
        std::cerr << "vzRange must include its lower boundary\n";
        return 1;
    }
    if (!evaluateVz(10.0, -18.0, 10.0)) {
        std::cerr << "vzRange must include its upper boundary\n";
        return 1;
    }
    if (evaluateVz(-18.01, -18.0, 10.0) || evaluateVz(10.01, -18.0, 10.0)) {
        std::cerr << "vzRange accepted a value outside the configured window\n";
        return 1;
    }
    if (evaluateVz(NAN, -18.0, 10.0)) {
        std::cerr << "vzRange accepted a non-finite vertex\n";
        return 1;
    }
    if (!evaluateVz(100.0, -18.0, 10.0, "tag")) {
        std::cerr << "tag-mode vzRange must not reject the candidate\n";
        return 1;
    }

    RecBranches ftPhoton;
    ftPhoton.pid = 22;
    ftPhoton.det = 0;
    ftPhoton.E_FTCAL = 0.15;
    if (!evaluateMinCalEnergy(ftPhoton, 0.15)) {
        std::cerr << "minCalEnergy must include its FT energy boundary\n";
        return 1;
    }
    ftPhoton.E_FTCAL = 0.149;
    ftPhoton.E_PCAL = ftPhoton.E_ECIN = ftPhoton.E_ECOUT = 1.0;
    if (evaluateMinCalEnergy(ftPhoton, 0.15)) {
        std::cerr << "minCalEnergy used FD calorimeter energy for an FT photon\n";
        return 1;
    }
    ftPhoton.E_FTCAL = NAN;
    if (evaluateMinCalEnergy(ftPhoton, 0.15)) {
        std::cerr << "minCalEnergy accepted a non-finite FT deposit\n";
        return 1;
    }

    RecBranches fdPhoton;
    fdPhoton.pid = 22;
    fdPhoton.det = 1;
    fdPhoton.E_FTCAL = 1.0;
    fdPhoton.E_PCAL = 0.05;
    fdPhoton.E_ECIN = 0.06;
    fdPhoton.E_ECOUT = 0.04;
    if (!evaluateMinCalEnergy(fdPhoton, 0.15)) {
        std::cerr << "minCalEnergy no longer accepts the FD energy boundary\n";
        return 1;
    }
    fdPhoton.E_ECOUT = 0.039;
    if (evaluateMinCalEnergy(fdPhoton, 0.15)) {
        std::cerr << "minCalEnergy used FT energy for an FD photon\n";
        return 1;
    }

    fdPhoton.det = 2;
    fdPhoton.E_PCAL = fdPhoton.E_ECIN = fdPhoton.E_ECOUT = 1.0;
    if (evaluateMinCalEnergy(fdPhoton, 0.15)) {
        std::cerr << "minCalEnergy accepted a detector without supported calorimetry\n";
        return 1;
    }
    return 0;
}
