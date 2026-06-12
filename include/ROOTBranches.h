#pragma once
#include <cmath>
#include <string>
#include "TObject.h"

#ifndef __CLING__
namespace clas12 {
    class clas12reader;
    class region_particle;
    class mcparticle;
}
#endif

// ─── Helpers ────────────────────────────────────────────────────────────────

int getDetector(int status);  // FT=0, FD=1, CD=2

template <typename T>
inline double safeGet(T expr) {
    double val = static_cast<double>(expr);
    return std::isnan(val) ? NAN : val;
}

// ─── EventBranches ───────────────────────────────────────────────────────────────
// One row per event.

struct EventBranches : public TObject {
    int    runNum   = -999;
    int    eventNum = -999;
    int    helicity = -999;
    double charge   = NAN;

    void reset();

    #ifndef __CLING__
    void fill(clas12::clas12reader& c12);
    #endif
    ClassDef(EventBranches, 1);
};

// ─── RecBranches ─────────────────────────────────────────────────────────────────
// One row per reconstructed particle per event.
// particleIdx ties rows back to the event; detector-specific fields are NAN
// when the particle isn't in that detector.

struct RecBranches : public TObject {
    // ── Identity ──────────────────────────────────────────
    int    runNum      = -999;
    int    eventNum    = -999;
    int    particleIdx = -999;  // position in clas12reader particle list

    // ── Particle bank ─────────────────────────────────────
    int    pid    = -999;
    int    charge = -999;
    int    status = -999;
    int    det    = -999;   // 0=FT, 1=FD, 2=CD
    int    sector = -999;

    double p      = NAN;
    double px     = NAN;
    double py     = NAN;
    double pz     = NAN;
    double theta  = NAN;
    double phi    = NAN;
    double beta   = NAN;
    double chi2pid = NAN;

    double vx     = NAN;
    double vy     = NAN;
    double vz     = NAN;
    double time   = NAN;

    // ── FT (det == 0 only, else NAN) ──────────────────────
    double xFT    = NAN;
    double yFT    = NAN;

    // ── FD DC trajectory (det == 1 only, else NAN) ────────
    double xDC1   = NAN,  yDC1  = NAN;
    double xDC2   = NAN,  yDC2  = NAN;
    double xDC3   = NAN,  yDC3  = NAN;
    double edgeDC1 = NAN, edgeDC2 = NAN, edgeDC3 = NAN;

    // ── FD calorimeter (det == 1 only, else NAN) ──────────
    double xPCAL  = NAN,  yPCAL  = NAN;
    double uPCAL  = NAN,  vPCAL  = NAN,  wPCAL  = NAN,  E_PCAL  = NAN;
    double uECIN  = NAN,  vECIN  = NAN,  wECIN  = NAN,  E_ECIN  = NAN;
    double uECOUT = NAN,  vECOUT = NAN,  wECOUT = NAN,  E_ECOUT = NAN;

    // ── CD CVT (det == 2 only, else NAN) ──────────────────
    double edge_cvt1  = NAN;
    double edge_cvt3  = NAN;
    double edge_cvt5  = NAN;
    double edge_cvt7  = NAN;
    double edge_cvt12 = NAN;
    double theta_cvt  = NAN;
    double phi_cvt    = NAN;

    void reset();

    #ifndef __CLING__
    void fill(clas12::region_particle* rec, int runNum, int eventNum, int idx);
    void fill(clas12::region_particle* rec, int runNum, int eventNum, int idx,
              double pCorr, double thetaCorr, double phiCorr);
    #endif

    ClassDef(RecBranches, 2);
};

// ─── GenBranches ─────────────────────────────────────────────────────────────────
// One row per MC particle per event.
// Join to RecBranches on (runNum, eventNum, particleIdx) after matching.

struct GenBranches : public TObject {
    int    runNum      = -999;
    int    eventNum    = -999;
    int    particleIdx = -999;  // position in MC particle list

    int    pid   = -999;
    double p     = NAN;
    double theta = NAN;
    double phi   = NAN;

    void reset();

    #ifndef __CLING__
    void fill(clas12::mcparticle* mc, int runNum, int eventNum, int idx);
    #endif
    ClassDef(GenBranches, 1);
};
