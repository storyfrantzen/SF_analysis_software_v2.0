#pragma once

#include "Config.h"

#ifndef __CLING__
#include "clas12reader.h"

bool passesFinalState(const Config& cfg, clas12::clas12reader& c12);
bool passesDISSkim(const Config& cfg, clas12::clas12reader& c12);
#endif
