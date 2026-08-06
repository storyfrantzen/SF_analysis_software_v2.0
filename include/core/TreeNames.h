#pragma once

namespace TreeNames {

inline constexpr const char* rEvents = "rEvents";
inline constexpr const char* rParticles = "rParticles";
inline constexpr const char* gEvents = "gEvents";
inline constexpr const char* sEvents = "sEvents";
inline constexpr const char* sParticles = "sParticles";

// Read-only compatibility names. New files always use the names above.
inline constexpr const char* legacyREvents = "ReconstructedEvents";
inline constexpr const char* legacyRParticles = "ReconstructedParticles";
inline constexpr const char* legacyGEvents = "GeneratedEvents";
inline constexpr const char* legacySEvents = "SelectedEvents";
inline constexpr const char* legacySParticles = "SelectedParticles";
inline constexpr const char* legacyEvents = "Events";

}  // namespace TreeNames
