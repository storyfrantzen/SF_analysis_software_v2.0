#pragma once

#include <memory>

#include "Config.h"

class QualityAssurance {
public:
    explicit QualityAssurance(const QADBConfig& config);
    ~QualityAssurance();

    QualityAssurance(const QualityAssurance&) = delete;
    QualityAssurance& operator=(const QualityAssurance&) = delete;

    bool pass(int runNum, int eventNum);
    double accumulatedCharge() const;
    bool enabled() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
