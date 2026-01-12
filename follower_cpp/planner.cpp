// cppimport
#include "planner.h"

PYBIND11_MODULE(planner, m) {
    py::class_<planner>(m, "planner")
            .def(py::init<std::vector<std::vector<int>>, bool, bool, bool>())
            .def("set_abs_start", &planner::set_abs_start)
            .def("update_path", &planner::update_path)
            .def("update_focal_paths", &planner::update_focal_paths)
            .def("get_path", &planner::get_path)
            .def("get_focal_paths", &planner::get_focal_paths)
            .def("get_next_node", &planner::get_next_node)
            .def("precompute_penalty_matrix", &planner::precompute_penalty_matrix)
            .def("set_penalties", &planner::set_penalties)
            .def("update_occupations", &planner::update_occupations)
            .def("update_dist_mat", &planner::update_dist_mat)
            .def("get_dist_mat", &planner::get_dist_mat)
            .def("set_dynamic_cost", &planner::set_dynamic_cost);
}

<%
cfg['extra_compile_args'] = ['-std=c++17']
setup_pybind11(cfg)
%>