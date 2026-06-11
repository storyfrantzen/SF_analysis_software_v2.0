#include "ROOTBranches.h"
#include "TVector3.h"
using namespace clas12;

int getDetector(int status) {
    const int abs_s = std::abs(status);
    if (abs_s >= 1000 && abs_s < 2000) return 0;  // FT
    if (abs_s >= 2000 && abs_s < 4000) return 1;  // FD
    if (abs_s >= 4000 && abs_s < 5000) return 2;  // CD
    return -999;
}

// ─── EventBranches ───────────────────────────────────────────────────────────────

void EventBranches::reset() {
    runNum = -999;
    eventNum = -999;
    helicity = -999;
    charge = NAN;
}

void EventBranches::fill(clas12::clas12reader& c12) {
    reset();

    runNum   = c12.runconfig()->getRun();
    eventNum = c12.runconfig()->getEvent();
    helicity = c12.event()->getHelicity();
    charge   = c12.event()->getBeamCharge();
}

// ─── RecBranches ─────────────────────────────────────────────────────────────────

void RecBranches::reset() {
    runNum = -999;
    eventNum = -999;
    particleIdx = -999;

    pid = -999;
    charge = -999;
    status = -999;
    det = -999;
    sector = -999;

    p = NAN;
    px = NAN;
    py = NAN;
    pz = NAN;
    theta = NAN;
    phi = NAN;
    beta = NAN;
    chi2pid = NAN;

    vx = NAN;
    vy = NAN;
    vz = NAN;
    time = NAN;

    xFT = yFT = NAN;
    xDC1 = yDC1 = xDC2 = yDC2 = xDC3 = yDC3 = NAN;
    uPCAL = vPCAL = wPCAL = E_PCAL = NAN;
    uECIN = vECIN = wECIN = E_ECIN = NAN;
    uECOUT = vECOUT = wECOUT = E_ECOUT = NAN;
    edge_cvt1 = edge_cvt3 = edge_cvt5 = edge_cvt7 = edge_cvt12 = NAN;
    theta_cvt = phi_cvt = NAN;
}

void RecBranches::fill(clas12::region_particle* rec, int rn, int en, int idx) {
    reset();

    runNum      = rn;
    eventNum    = en;
    particleIdx = idx;

    pid     = rec->getPid();
    charge  = rec->par()->getCharge();
    status  = rec->par()->getStatus();
    det     = getDetector(status);
    sector  = rec->getSector();

    p       = safeGet(rec->getP());
    px      = safeGet(rec->par()->getPx());
    py      = safeGet(rec->par()->getPy());
    pz      = safeGet(rec->par()->getPz());
    theta   = safeGet(rec->getTheta());
    phi     = safeGet(rec->getPhi());
    beta    = safeGet(rec->par()->getBeta());
    chi2pid = safeGet(rec->par()->getChi2Pid());

    vx      = safeGet(rec->par()->getVx());
    vy      = safeGet(rec->par()->getVy());
    vz      = safeGet(rec->par()->getVz());
    time    = safeGet(rec->getTime());

    // ── FT ───────────────────────────────────────────────
    if (det == 0) {
        xFT = safeGet(rec->ft(FTCAL)->getX());
        yFT = safeGet(rec->ft(FTCAL)->getY());
    }

    // ── FD ───────────────────────────────────────────────
    if (det == 1) {
        xDC1 = safeGet(rec->traj(DC,  6)->getX());
        yDC1 = safeGet(rec->traj(DC,  6)->getY());
        xDC2 = safeGet(rec->traj(DC, 18)->getX());
        yDC2 = safeGet(rec->traj(DC, 18)->getY());
        xDC3 = safeGet(rec->traj(DC, 36)->getX());
        yDC3 = safeGet(rec->traj(DC, 36)->getY());

        uPCAL  = safeGet(rec->cal(1)->getLu());
        vPCAL  = safeGet(rec->cal(1)->getLv());
        wPCAL  = safeGet(rec->cal(1)->getLw());
        E_PCAL = safeGet(rec->cal(1)->getEnergy());

        uECIN  = safeGet(rec->cal(4)->getLu());
        vECIN  = safeGet(rec->cal(4)->getLv());
        wECIN  = safeGet(rec->cal(4)->getLw());
        E_ECIN = safeGet(rec->cal(4)->getEnergy());

        uECOUT  = safeGet(rec->cal(7)->getLu());
        vECOUT  = safeGet(rec->cal(7)->getLv());
        wECOUT  = safeGet(rec->cal(7)->getLw());
        E_ECOUT = safeGet(rec->cal(7)->getEnergy());
    }

    // ── CD ───────────────────────────────────────────────
    if (det == 2) {
        edge_cvt1  = safeGet(rec->traj(CVT,  1)->getEdge());
        edge_cvt3  = safeGet(rec->traj(CVT,  3)->getEdge());
        edge_cvt5  = safeGet(rec->traj(CVT,  5)->getEdge());
        edge_cvt7  = safeGet(rec->traj(CVT,  7)->getEdge());
        edge_cvt12 = safeGet(rec->traj(CVT, 12)->getEdge());

        double x = safeGet(rec->traj(CVT, 1)->getX());
        double y = safeGet(rec->traj(CVT, 1)->getY());
        double z = safeGet(rec->traj(CVT, 1)->getZ());
        double r = (!std::isnan(x) && !std::isnan(y) && !std::isnan(z)) ? std::sqrt(x*x + y*y + z*z) : NAN;

        theta_cvt = (!std::isnan(r) && r > 0) ? std::acos(z / r) : NAN;
        phi_cvt   = (!std::isnan(x) && !std::isnan(y)) ? std::atan2(y, x) : NAN;
    }
}

void RecBranches::fill(clas12::region_particle* rec, int rn, int en, int idx,
                   double pCorr, double thetaCorr, double phiCorr) {
    fill(rec, rn, en, idx);
    p     = pCorr;
    theta = thetaCorr;
    phi   = phiCorr;
    px    = p * std::sin(theta) * std::cos(phi);
    py    = p * std::sin(theta) * std::sin(phi);
    pz    = p * std::cos(theta);
}

// ─── GenBranches ─────────────────────────────────────────────────────────────────

void GenBranches::reset() {
    runNum = -999;
    eventNum = -999;
    particleIdx = -999;

    pid = -999;
    p = NAN;
    theta = NAN;
    phi = NAN;
}

void GenBranches::fill(clas12::mcparticle* mc, int rn, int en, int idx) {
    reset();

    runNum      = rn;
    eventNum    = en;
    particleIdx = idx;

    pid = mc->getPid(idx);
    TVector3 v(mc->getPx(idx), mc->getPy(idx), mc->getPz(idx));
    p     = v.Mag();
    theta = v.Theta();
    phi   = v.Phi();
}

ClassImp(EventBranches);
ClassImp(RecBranches);
ClassImp(GenBranches);
