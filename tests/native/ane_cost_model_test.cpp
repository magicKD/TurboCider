#include "../../native/backends/ane_cost_model.hpp"
#include <iostream>

using namespace tc::ane;
namespace {
void check(bool value,const char *why) {if(!value)throw std::runtime_error(why);}
bool close(double a,double b) {return std::abs(a-b)<1e-12;}
}
int main() {
    try {
        check(close(calibration_layer_seconds(.017,.041),.008),"one/four fixed-cost cancellation failed");
        for(auto spans:{std::pair{0.,1.},std::pair{1.,1.},std::pair{2.,1.},std::pair{double(NAN),1.}}) {
            bool rejected=false;
            try {(void)calibration_layer_seconds(spans.first,spans.second);}catch(const std::invalid_argument&) {rejected=true;}
            check(rejected,"invalid batch spans accepted");
        }
        CalibrationPoint p{.4,.008,.0032,.0096},q{.8,.004,.0064,.0072};
        auto model=ChannelCostModel::fit(p,q,0);
        check(bool(model),"valid binding model rejected");
        check(close(*model->predict(.6),.0084),"binding prediction lost coupled bandwidth");
        auto swapped=ChannelCostModel::fit(q,p,0);
        check(swapped && close(*swapped->predict(.6),*model->predict(.6)),"share order changed fit");
        check(!model->predict(.2) && !model->predict(.9) && !model->predict(NAN),"unmeasured extrapolation allowed");
        auto allow=[](int){return true;};
        auto choice=model->select(10240,512,.012,allow);
        check(choice.ane_channels==8192 && close(choice.predicted_seconds,.0072) && choice.requires_candidate_trial,
              "aligned best prediction/trial gate mismatch");
        auto bounded=model->select(10240,512,.012,[](int c){return c<=6144;});
        check(bounded.ane_channels==6144 && close(bounded.predicted_seconds,.0084),"memory ceiling did not constrain share");
        auto disabled=model->select(10240,512,.0075,allow);
        check(disabled.ane_channels==0 && close(disabled.predicted_seconds,.0075),"minimum gain used extrapolated GPU intercept");
        check(model->select(10240,512,.012,[](int){return false;}).ane_channels==0,"empty memory set did not use GPU");
        auto broad=model->select(10240,512,.012,allow,.05,.1);
        check(broad.ane_channels==7168 && broad.predicted_seconds<=1.1*.0072,"near-optimal smallest share failed");
        for(int full:{0,513,512}) {
            bool rejected=false;
            try {model->select(full,512,.012,allow);}catch(const std::invalid_argument&) {rejected=true;}
            check(rejected,"invalid full width/channel unit accepted");
        }
        // In a non-binding observation, max(G,A) does NOT identify uG/uA.
        // The fitted polygon retains uncertainty instead of inventing zero
        // bandwidth fractions and a spuriously low interior prediction.
        auto nonbinding=ChannelCostModel::fit({.4,.008,.0032,.008},{.8,.004,.0064,.0064},0);
        check(nonbinding && close(*nonbinding->predict(.6),.0072),"non-binding uncertainty envelope lost");
        auto flat=ChannelCostModel::fit({.4,.004,.004,.004},{.8,.004,.004,.004},0);
        check(flat && flat->select(10240,512,.01,allow).ane_channels==4096,"flat optimum did not choose least share");
        for(auto bad:{CalibrationPoint{.4,0,.003,.009},CalibrationPoint{.4,.008,NAN,.009},
            CalibrationPoint{1,.008,.003,.009},CalibrationPoint{.4,.008,.003,.001},
            CalibrationPoint{.4,.008,.003,.05}})
            check(!ChannelCostModel::fit(bad,q),"invalid or incomplete timing point accepted");
        check(!ChannelCostModel::fit(p,p),"duplicate share accepted");
        check(!ChannelCostModel::fit(p,q,NAN) && !ChannelCostModel::fit(p,q,.2),"invalid fit tolerance accepted");
        // Equivalent scaling from seconds to a different time magnitude must
        // not change the share. API still documents seconds, not mixed units.
        auto scaled=ChannelCostModel::fit({.4,8,3.2,9.6},{.8,4,6.4,7.2},0);
        check(scaled && scaled->select(10240,512,12,allow).ane_channels==choice.ane_channels,"scale instability in fit");
        std::cout<<"PASS shared channel cost model: independent one/four spans, coupled bandwidth/uncertainty, measured GPU baseline, 512 alignment, memory ceiling, smallest near-optimal share, trial gate and invalid evidence\n";
    } catch(const std::exception &e) {std::cerr<<e.what()<<'\n';return 1;}
}
