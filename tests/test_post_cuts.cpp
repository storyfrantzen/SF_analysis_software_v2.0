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
    return 0;
}
