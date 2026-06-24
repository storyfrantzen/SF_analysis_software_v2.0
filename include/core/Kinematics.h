#pragma once

#include <cmath>
#include <initializer_list>

#include "TLorentzVector.h"

#include "ROOTBranches.h"

namespace Kinematics {

struct DIS {
    double Q2 = NAN;
    double nu = NAN;
    double xB = NAN;
    double y = NAN;
    double W = NAN;
};

double massForPid(int pid);

TLorentzVector beam(double beamEnergy);
TLorentzVector target();
TLorentzVector particle(double px, double py, double pz, double mass);
TLorentzVector particle(const RecBranches& p);
TLorentzVector missingSystem(double beamEnergy, std::initializer_list<TLorentzVector> observed);

DIS dis(const TLorentzVector& electron, double beamEnergy);
double massIfTimelike(const TLorentzVector& v);
double angle(const TLorentzVector& a, const TLorentzVector& b);
double deltaPhi(double phi, double referencePhi);
double trentoPhi(const TLorentzVector& beam, const TLorentzVector& electron, const TLorentzVector& hadron);

} // namespace Kinematics
