#pragma once

#include <TDatabasePDG.h>

inline const TDatabasePDG* db = TDatabasePDG::Instance();

inline const double M_ELECTRON = db->GetParticle(11)->Mass();    // GeV
inline const double M_PROTON   = db->GetParticle(2212)->Mass();  // GeV
inline const double M_PI0      = db->GetParticle(111)->Mass();   // GeV