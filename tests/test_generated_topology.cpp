#include <iostream>

#include "GeneratedTopology.h"

namespace {

bool expect(GeneratedTopology actual,
            GeneratedTopology expected,
            const char* description) {
    if (actual == expected) return true;
    std::cerr << "unexpected generated topology for " << description << '\n';
    return false;
}

}  // namespace

int main() {
    bool valid = true;
    valid &= expect(
        classifyGeneratedTopology(true, true, false, 2),
        GeneratedTopology::nonradiative,
        "electron + proton + two photons"
    );
    valid &= expect(
        classifyGeneratedTopology(true, true, true, 2),
        GeneratedTopology::nonradiative,
        "EXCLUGEN non-hard event retaining its parent pi0"
    );
    valid &= expect(
        classifyGeneratedTopology(true, true, true, 1),
        GeneratedTopology::hardRadiative,
        "legacy AAO pi0 plus one radiated photon"
    );
    valid &= expect(
        classifyGeneratedTopology(true, true, true, 3),
        GeneratedTopology::hardRadiative,
        "EXCLUGEN hard event with parent pi0 and three photons"
    );
    valid &= expect(
        classifyGeneratedTopology(true, true, false, 3),
        GeneratedTopology::hardRadiative,
        "EXCLUGEN hard event after dropping the documentation parent"
    );
    valid &= expect(
        classifyGeneratedTopology(true, true, false, 1),
        GeneratedTopology::invalid,
        "one photon without a pi0"
    );
    valid &= expect(
        classifyGeneratedTopology(false, true, true, 1),
        GeneratedTopology::invalid,
        "missing electron"
    );
    valid &= expect(
        classifyGeneratedTopology(true, false, false, 2),
        GeneratedTopology::invalid,
        "missing proton"
    );
    valid &= expect(
        classifyGeneratedTopology(true, true, true, 4),
        GeneratedTopology::invalid,
        "unexpected fourth photon"
    );
    if (pi0DecayPhotonOffset(2) != 0 || pi0DecayPhotonOffset(3) != 1) {
        std::cerr << "EXCLUGEN decay-photon offset does not skip the hard photon\n";
        valid = false;
    }
    return valid ? 0 : 1;
}
