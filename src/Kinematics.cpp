#include "Kinematics.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>

#include "TVector3.h"
#include "TVector2.h"

#include "PhysicalConstants.h"

namespace Kinematics {

double massForPid(int pid) {
    switch (std::abs(pid)) {
        case 11: return M_ELECTRON;
        case 22: return 0.0;
        case 111: return M_PI0;
        case 2212: return M_PROTON;
        default: return 0.0;
    }
}

TLorentzVector beam(double beamEnergy) {
    return TLorentzVector(0, 0, beamEnergy, beamEnergy);
}

TLorentzVector target() {
    return TLorentzVector(0, 0, 0, M_PROTON);
}

TLorentzVector particle(double px, double py, double pz, double mass) {
    const double p2 = px * px + py * py + pz * pz;
    TLorentzVector lv;
    lv.SetPxPyPzE(px, py, pz, std::sqrt(mass * mass + p2));
    return lv;
}

TLorentzVector particle(const RecBranches& p) {
    return particle(p.px, p.py, p.pz, massForPid(p.pid));
}

TLorentzVector missingSystem(double beamEnergy, std::initializer_list<TLorentzVector> observed) {
    TLorentzVector missing = beam(beamEnergy) + target();
    for (const auto& particle : observed) missing -= particle;
    return missing;
}

DIS dis(const TLorentzVector& electron, double beamEnergy) {
    DIS vars;
    const TLorentzVector lvBeam = beam(beamEnergy);
    const TLorentzVector q = lvBeam - electron;
    const TLorentzVector lvTarget = target();

    vars.Q2 = -q.M2();
    vars.nu = beamEnergy - electron.E();
    if (std::isfinite(vars.nu) && vars.nu != 0.0) {
        vars.xB = vars.Q2 / (2.0 * M_PROTON * vars.nu);
    }
    if (beamEnergy != 0.0) vars.y = vars.nu / beamEnergy;
    vars.W = std::sqrt(std::max(0.0, (lvTarget + q).M2()));
    return vars;
}

double massIfTimelike(const TLorentzVector& v) {
    return v.M2() >= 0.0 ? std::sqrt(v.M2()) : NAN;
}

double angle(const TLorentzVector& a, const TLorentzVector& b) {
    return a.Vect().Angle(b.Vect());
}

double deltaPhi(double phi, double referencePhi) {
    return TVector2::Phi_mpi_pi(phi - referencePhi);
}

double trentoPhi(const TLorentzVector& beam, const TLorentzVector& electron, const TLorentzVector& hadron) {
    const TVector3 q = beam.Vect() - electron.Vect();
    const TVector3 nLepton = beam.Vect().Cross(electron.Vect()).Unit();
    const TVector3 nHadron = hadron.Vect().Cross(q).Unit();
    const double cosPhi = nLepton.Dot(nHadron);
    const double sinPhi = q.Unit().Dot(nLepton.Cross(nHadron));
    return std::atan2(sinPhi, cosPhi);
}

} // namespace Kinematics
