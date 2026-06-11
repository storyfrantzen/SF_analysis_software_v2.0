#include "Cuts.h"

#include <algorithm>
#include <cmath>

#include "PhysicalConstants.h"
#include "TLorentzVector.h"

bool passesFinalState(const Config& cfg, clas12::clas12reader& c12) {
    if (cfg.finalState.empty()) return true;

    for (const auto& s : cfg.finalState) {
        const int n = static_cast<int>(c12.getByID(s.pid).size());
        if (s.exact && n != s.count) return false;
        if (!s.exact && n < s.count) return false;
    }
    return true;
}

bool passesDISSkim(const Config& cfg, clas12::clas12reader& c12) {
    if (!cfg.enableSkim) return true;

    auto electrons = c12.getByID(11);
    if (electrons.empty()) return false;

    auto* e = electrons[0];
    TLorentzVector lv_e(e->par()->getPx(), e->par()->getPy(), e->par()->getPz(),
                        std::sqrt(e->getP() * e->getP() + M_ELECTRON * M_ELECTRON));

    TLorentzVector lv_beam(0, 0, cfg.beamEnergy, cfg.beamEnergy);
    TLorentzVector q = lv_beam - lv_e;

    const double Q2 = -q.M2();
    const double nu = cfg.beamEnergy - lv_e.E();
    const double y = nu / cfg.beamEnergy;
    const double W = std::sqrt(std::max(0.0, (TLorentzVector(0, 0, 0, M_PROTON) + q).M2()));

    return (Q2 >= cfg.Q2_min && W >= cfg.W_min && y <= cfg.y_max);
}
