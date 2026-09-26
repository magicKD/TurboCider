#include "../../native/models/qwen21/sequence.hpp"
#include "../../native/core/common.hpp"
#include <iostream>

int main() {
    using namespace tc::qwen21;
    try {
        auto text = make_sequence_geometry(3, 2, 3, {});
        tc::require(text.prefix_length == 3 && text.positions.size() == 9 && text.segments.size() == 2, "text-only layout");
        tc::require(text.positions[3] == std::array<float,3>{3,-1,-2}, "target grid coordinates");
        auto edit = make_sequence_geometry(5, 2, 2, {{3,1,2}, {2,3,2}});
        tc::require(edit.prefix_length == 14 && edit.positions.size() == 18 && edit.segments.size() == 5, "interleaved reference layout");
        tc::require(edit.positions[2] == std::array<float,3>{2,-1.5f,-.5f}, "reference parity offset");
        tc::require(edit.positions[5] == std::array<float,3>{5,-1,-1.5f}, "second reference frame position");
        tc::require(edit.positions[11] == std::array<float,3>{8,8,8}, "text positions after references");
        tc::require(edit.positions[14] == std::array<float,3>{11,-1,-1}, "target frame after references");
        auto ten = make_sequence_geometry(1, 1, 1, std::vector<ReferenceGeometry>(10,{1,1,0}));
        tc::require(ten.prefix_length == 11 && ten.segments.size() == 12, "ten reference boundary");
        int rejected = 0;
        for (auto refs : {std::vector<ReferenceGeometry>(11,{1,1,0}),
                         std::vector<ReferenceGeometry>{{1,1,3},{1,1,2}},
                         std::vector<ReferenceGeometry>{{1,1,-1}},
                         std::vector<ReferenceGeometry>{{1,1,6}},
                         std::vector<ReferenceGeometry>{{0,1,0}},
                         std::vector<ReferenceGeometry>{{2147483647,2,0}}}) {
            try { make_sequence_geometry(5,2,2,refs); }
            catch (const std::exception &) { ++rejected; }
        }
        tc::require(rejected == 6, "invalid reference geometry accepted");
        std::cout << "PASS: Qwen21 sequence placement, half-grid parity, ten references and invalid inputs\n";
        return 0;
    } catch (const std::exception &e) {
        std::cerr << e.what() << '\n';
        return 1;
    }
}
