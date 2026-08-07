#pragma once

#include <memory>
#include <vector>

#include "Config.h"

struct RunChargeRecord {
    int runNum = -999;
    double accumulatedChargeNC = 0.0;
    long long totalEvents = 0;
    long long passedQADBEvents = 0;
    long long failedQADBEvents = 0;
};

class QualityAssurance {
public:
    explicit QualityAssurance(const QADBConfig& config);
    ~QualityAssurance();

    QualityAssurance(const QualityAssurance&) = delete;
    QualityAssurance& operator=(const QualityAssurance&) = delete;

    bool pass(int runNum, int eventNum);
    double accumulatedCharge() const;
    std::vector<RunChargeRecord> runChargeRecords() const;
    bool enabled() const;

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};
