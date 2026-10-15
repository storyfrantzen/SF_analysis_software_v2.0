#pragma once

#include <cstddef>

enum class GeneratedTopology {
    invalid,
    nonradiative,
    hardRadiative,
};

constexpr GeneratedTopology classifyGeneratedTopology(bool haveElectron,
                                                       bool haveProton,
                                                       bool havePi0,
                                                       std::size_t photonCount) {
    if (!haveElectron || !haveProton) return GeneratedTopology::invalid;

    // With idecay=1, EXCLUGEN retains a status-0 pi0 parent and appends its
    // two decay photons.  Full-RC hard events insert the radiated photon
    // before that pair, so two photons are non-hard and three are hard.  Do
    // not infer the event state from the presence of the retained parent.
    if (photonCount == 2) return GeneratedTopology::nonradiative;
    if (photonCount == 3) return GeneratedTopology::hardRadiative;

    // Legacy AAO radiative records contain an undecayed pi0 and one photon.
    if (havePi0 && photonCount == 1) {
        return GeneratedTopology::hardRadiative;
    }
    return GeneratedTopology::invalid;
}

// Precondition: photonCount is two or three for a valid EXCLUGEN event.
constexpr std::size_t pi0DecayPhotonOffset(std::size_t photonCount) {
    return photonCount == 3 ? 1 : 0;
}
