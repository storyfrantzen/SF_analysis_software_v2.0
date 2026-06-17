#include "QualityAssurance.h"

#include <stdexcept>
#include <utility>

#ifdef HAVE_QADB
#include "QADB.h"
#endif

class QualityAssurance::Impl {
public:
    explicit Impl(QADBConfig settings) : config(std::move(settings)) {
#ifdef HAVE_QADB
        if (config.enabled) {
            qadb = std::make_unique<QA::QADB>(config.database.c_str());
            for (const auto& defect : config.rejectDefects) {
                qadb->CheckForDefect(defect.c_str());
            }
            for (const int run : config.allowMiscRuns) {
                qadb->AllowMiscBit(run);
            }
        }
#else
        if (config.enabled) {
            throw std::runtime_error(
                "QADB was enabled in the config, but hipo2root was built without QADB support. "
                "Load the qadb module and rebuild."
            );
        }
#endif
    }

    bool pass(int runNum, int eventNum) {
        if (!config.enabled || runNum == 11) return true;
#ifdef HAVE_QADB
        if (!qadb->Pass(runNum, eventNum)) return false;
        qadb->AccumulateCharge();
#else
        (void)eventNum;
#endif
        return true;
    }

    double accumulatedCharge() const {
#ifdef HAVE_QADB
        return qadb ? qadb->GetAccumulatedCharge() : 0.0;
#else
        return 0.0;
#endif
    }

    QADBConfig config;
#ifdef HAVE_QADB
    std::unique_ptr<QA::QADB> qadb;
#endif
};

QualityAssurance::QualityAssurance(const QADBConfig& config)
    : impl_(std::make_unique<Impl>(config)) {}

QualityAssurance::~QualityAssurance() = default;

bool QualityAssurance::pass(int runNum, int eventNum) {
    return impl_->pass(runNum, eventNum);
}

double QualityAssurance::accumulatedCharge() const {
    return impl_->accumulatedCharge();
}

bool QualityAssurance::enabled() const {
    return impl_->config.enabled;
}
