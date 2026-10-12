#include "../../native/backends/ane_row_window.hpp"
#include <cassert>
#include <iostream>
#include <vector>

int main() {
    using namespace tc::ane;
    for(int rows:{1056,4128})for(RowPlacement placement:{RowPlacement::Suffix,RowPlacement::ImagePrefix,RowPlacement::ImageTail}) {
        RowPolicy policy{placement,placement==RowPlacement::Suffix?0:32};
        auto w=plan_row_window(rows,352,policy);
        assert(w.first>=0 && w.end()<=rows && w.gpu_before()+w.count+w.gpu_after()==rows);
        if(placement==RowPlacement::ImagePrefix)assert(w.first==0);
        else assert(w.end()==rows-policy.protected_suffix_rows);
        std::vector<int> gpu,ane,joined;
        for(int i=0;i<rows;++i)(i>=w.first&&i<w.end()?ane:gpu).push_back(i*7+13);
        joined.insert(joined.end(),gpu.begin(),gpu.begin()+w.gpu_before());
        joined.insert(joined.end(),ane.begin(),ane.end());
        joined.insert(joined.end(),gpu.begin()+w.gpu_before(),gpu.end());
        for(int i=0;i<rows;++i)assert(joined[i]==i*7+13);
        if(policy.protected_suffix_rows)for(int i=rows-32;i<rows;++i)assert(i<w.first || i>=w.end());
    }
    for(auto policy:{RowPolicy{},RowPolicy{RowPlacement::ImagePrefix,0},RowPolicy{RowPlacement::ImageTail,0}}) {
        auto w=plan_row_window(352,352,policy);assert(w.first==0&&w.end()==352);
    }
    for(auto test:{std::vector<int>{0,1,0,0},{10,0,0,0},{10,11,0,0},{10,1,0,1},
        {10,1,1,10},{10,2,2,9},{10,1,1,-1},{10,1,99,0}}) {
        bool caught=false;try{(void)plan_row_window(test[0],test[1],{RowPlacement(test[2]),test[3]});}
        catch(const std::invalid_argument&){caught=true;}assert(caught);
    }
    std::cout<<"PASS row windows: prefix/suffix/interior, exact source order, protected captions, empty GPU and invalid bounds\n";
}
