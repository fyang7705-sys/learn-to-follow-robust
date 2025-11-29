#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/stl_bind.h>
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <vector>
#include <queue>
#include <cmath>
#include <set>
#include <map>
#include <list>
#include <iostream>
#include <random>
#include <chrono>
#define INF 1000000000
namespace py = pybind11;

struct Node {
    Node(int _i = INF, int _j = INF, float _g = INF, float _h = 0, float _focal_value = INF) : i(_i), j(_j), g(_g), h(_h), f(_g+_h), focal_value(_focal_value)
    {
        visited = false;
        parent = {INF, INF}; 
        now_id = -1;
    }
    int i;
    int j;
    float g;
    float h;
    float f;
    float focal_value;
    bool visited;
    int now_id;
    std::pair<int, int> parent;
    bool operator<(const Node& other) const
    {
        return this->f < other.f or (std::abs(this->f - other.f) < 1e-5 and this->g < other.g);
    }
    bool operator>(const Node& other) const
    {
        return this->f > other.f or (std::abs(this->f - other.f) < 1e-5 and this->g > other.g);
    }
    bool operator==(const Node& other) const
    {
        return this->i == other.i and this->j == other.j;
    }
    bool operator==(const std::pair<int, int> &other) const
    {
        return this->i == other.first and this->j == other.second;
    }
};

struct CompareFocal {
    bool operator()(const Node& a, const Node& b) const {
        if (std::abs(a.focal_value - b.focal_value) > 1e-5)
            return a.focal_value > b.focal_value; // Min-heap behavior for focal_value
        return a.f > b.f; // Tie-break: prefer lower f
    }
};

class planner
{
    std::pair<int, int> start;
    std::pair<int, int> goal;
    std::pair<int, int> abs_offset;
    std::priority_queue<Node, std::vector<Node>, std::greater<Node>> OPEN;
    std::vector<std::vector<int>> grid;
    std::vector<std::vector<float>> num_occupations;
    std::vector<std::vector<float>> penalties;
    std::vector<std::vector<float>> h_values;
    std::vector<std::vector<Node>> nodes;
    std::list<std::list<std::pair<int, int>>> focal_paths;
    bool use_static_cost;
    bool use_dynamic_cost;
    bool reset_dynamic_cost;
    std::mt19937 rng;
    std::uniform_real_distribution<float> dist_w;
    std::uniform_real_distribution<float> dist_g_weight;
    float w;
    float g_weight;
    int current_id;
    std::vector<std::vector<float>> diversity_map;
    inline float h(std::pair<int, int> n)
    {
        //return abs(n.first - goal.first) + abs(n.second - goal.second);
        return h_values[n.first][n.second];
    }

    inline void reset_node(Node & node)
    {
        if (node.now_id != current_id)
        {
            node.now_id = current_id;
            node.visited = false;
            node.parent = {INF, INF};
            node.g = INF;
            node.h = 0;
            node.f = INF;
            node.focal_value = INF;
        }
    }
    std::vector<std::pair<int,int>> get_neighbors(std::pair<int, int> node)
    {
        std::vector<std::pair<int,int>> neighbors;
        std::vector<std::pair<int,int>> deltas = {{0,1},{1,0},{-1,0},{0,-1}};
        for(auto d:deltas)
        {
            std::pair<int,int> n(node.first + d.first, node.second + d.second);
            if(grid[n.first][n.second] == 0)
                neighbors.push_back(n);
        }
        return neighbors;
    }


    void compute_shortest_path()
    {
        OPEN = std::priority_queue<Node, std::vector<Node>, std::greater<Node>>();
        // py::print("giao0");
        reset_node(nodes[start.first][start.second]);
        // py::print("giao1");
        OPEN.push(Node(start.first, start.second, 0, h(start)));
        Node current;
        while(!OPEN.empty() and !(current == goal))
        {
            current = OPEN.top();
            OPEN.pop();
            reset_node(nodes[current.i][current.j]);
            if(nodes[current.i][current.j].g < current.g)
                continue;
            // py::print("current", current.i, current.j);
            for(auto n: get_neighbors({current.i, current.j})) {
                float cost(1);
                reset_node(nodes[n.first][n.second]);
                if(use_static_cost)
                    cost = penalties[n.first][n.second];
                if(use_dynamic_cost)
                    cost += num_occupations[n.first][n.second];
                if(nodes[n.first][n.second].g > current.g + cost)
                {
                    // py::print("neighbor", n.first, n.second, nodes[n.first][n.second].g);
                    OPEN.push(Node(n.first, n.second, current.g + cost, h(n)));
                    nodes[n.first][n.second].g = current.g + cost;
                    nodes[n.first][n.second].parent = {current.i, current.j};
                }
            }

        }
        // py::print("giao2");
    }

    float frechet_distance(const std::list<std::pair<int,int>>& p_list, const std::list<std::pair<int,int>>& q_list)
    {
        int n = p_list.size();
        int m = q_list.size();
        if (n == 0 || m == 0) return 1e9f;
        std::vector<std::pair<int,int>> p(p_list.begin(), p_list.end());
        std::vector<std::pair<int,int>> q(q_list.begin(), q_list.end());
        // dp[i][j]
        std::vector<std::vector<float>> dp(n, std::vector<float>(m, 0.0));

        auto dist = [&](int i, int j) {
            float dx = p[i].first  - q[j].first;
            float dy = p[i].second - q[j].second;
            return std::sqrt(dx*dx + dy*dy);
        };

        dp[0][0] = dist(0,0);

        for (int i = 1; i < n; i++)
            dp[i][0] = std::max(dp[i-1][0], dist(i, 0));

        for (int j = 1; j < m; j++)
            dp[0][j] = std::max(dp[0][j-1], dist(0, j));

        for (int i = 1; i < n; i++) {
            for (int j = 1; j < m; j++) {
                float d = dist(i, j);
                float prev = std::min({ dp[i-1][j], dp[i][j-1], dp[i-1][j-1] });
                dp[i][j] = std::max(d, prev);
            }
        }
        return dp[n-1][m-1];
    }

    bool is_similar_to_existing(const std::list<std::pair<int,int>>& new_path, float threshold)
    {
        for (const auto& p : focal_paths) {
            float d = frechet_distance(new_path, p);
            if (d < threshold)
                return true;   // 过于相似，拒绝
        }
        return false;
    }

    float dir_consistency(Node & node, std::pair<int, int> & goal) {
        // 如果 parent 未初始化（INF），返回 0
        if (node.parent.first == INF && node.parent.second == INF)
            return 0.0f;

        // 当前节点相对父节点的方向
        int dir_i = node.i - node.parent.first;
        int dir_j = node.j - node.parent.second;

        // 理想方向（x/y轴正负极）
        int goal_dir_i = 0, goal_dir_j = 0;
        if (node.i != goal.first)
            goal_dir_i = (node.i < goal.first) ? 1 : -1;
        if (node.j != goal.second)
            goal_dir_j = (node.j < goal.second) ? 1 : -1;

        return (dir_i == goal_dir_i && dir_j == goal_dir_j) ? 1.0f : 0.0f;
    }

    float calculate_focal_value(Node & node, std::pair<int, int> & goal) {
        int grid_width =  grid.front().size();
        int grid_height = grid.size();
        float consistency = dir_consistency(node, goal);
        int max_g = grid_width + grid_height;
        float normalized_g = (max_g != 0) ? (node.g / max_g) : 0.0f;
        return 1.0f - ((1 - g_weight) * consistency + g_weight * (1.0f - normalized_g));
    }

    float calculate_diverse_focal_value(Node & node, std::pair<int, int> & goal)
    {
        int grid_width =  grid.front().size();
        int grid_height = grid.size();
        // py::print("giao0");
        float diversity = calculate_diversity_score(node);
        int max_g = grid_width + grid_height;
        float normalized_g = (max_g != 0) ? (node.g / max_g) : 0.0f;
        // py::print("giao1");
        // py::print("normalized_g", normalized_g, "g_weight", g_weight);
        // py::print("diversity", diversity, "weight", 1 - g_weight);
        return g_weight * normalized_g + (1 - g_weight) * diversity;
    }

    float calculate_diversity_score(Node & node)
    {
        int grid_width =  grid.front().size();
        int grid_height = grid.size();
        if (focal_paths.empty()) return 0.0f;
        float min_dist_to_existing = INF;
        for (const auto& path : focal_paths)
        {
            for (const auto& existing : path)
            {                
                float dist = std::hypot(existing.first - node.i, existing.second - node.j);
                if (dist < min_dist_to_existing) min_dist_to_existing = dist;
            }        
        }
        // py::print("mid_dist", min_dist_to_existing);
        // py::print("node", node.i, node.j);
        // min_dist_to_existing = diversity_map[node.i][node.j];
        // py::print("mid_dist2", min_dist_to_existing);
        if (min_dist_to_existing >= 99990.0f) min_dist_to_existing = 0.0f;
        
        float max_possible_dist = std::hypot(grid_width, grid_height);
        return (max_possible_dist != 0) ? (min_dist_to_existing / max_possible_dist) : 0.0f;
    }

    void update_diversity_map() 
    {
        int rows = grid.size();
        int cols = grid[0].size();
        
        // 1. 初始化：全图设为 INF
        for(int i = 0; i < rows; ++i) {
            std::fill(diversity_map[i].begin(), diversity_map[i].end(), INF);
        }
        
        // 2. 标记旧路径点为 0.0
        if (focal_paths.empty()) return;
        
        for (const auto& path : focal_paths) {
            for (const auto& p : path) {
                int r = p.first + abs_offset.first;
                int c = p.second + abs_offset.second;
                if (r >= 0 && r < rows && c >= 0 && c < cols) {
                    diversity_map[r][c] = 0.0f;
                }
            }
        }

        // 预定义代价：直向 1.0，斜向 1.414
        const float d1 = 1.0f;
        const float d2 = 1.41421f;

        // 3. 第一遍扫描 (Pass 1): 左上 -> 右下
        // 检查方向：左(L), 左上(TL), 上(T), 右上(TR)
        for(int i = 0; i < rows; ++i) {
            for(int j = 0; j < cols; ++j) {
                if (diversity_map[i][j] == 0.0f) continue;

                float min_val = INF;
                
                if (i > 0)          min_val = std::min(min_val, diversity_map[i-1][j] + d1); // 上
                if (j > 0)          min_val = std::min(min_val, diversity_map[i][j-1] + d1); // 左
                if (i > 0 && j > 0) min_val = std::min(min_val, diversity_map[i-1][j-1] + d2); // 左上
                if (i > 0 && j < cols - 1) min_val = std::min(min_val, diversity_map[i-1][j+1] + d2); // 右上

                diversity_map[i][j] = min_val;
            }
        }

        // 4. 第二遍扫描 (Pass 2): 右下 -> 左上
        // 检查方向：右(R), 右下(BR), 下(B), 左下(BL)
        for(int i = rows - 1; i >= 0; --i) {
            for(int j = cols - 1; j >= 0; --j) {
                if (diversity_map[i][j] == 0.0f) continue;

                float min_val = diversity_map[i][j];

                if (i < rows - 1)   min_val = std::min(min_val, diversity_map[i+1][j] + d1); // 下
                if (j < cols - 1)   min_val = std::min(min_val, diversity_map[i][j+1] + d1); // 右
                if (i < rows - 1 && j < cols - 1) min_val = std::min(min_val, diversity_map[i+1][j+1] + d2); // 右下
                if (i < rows - 1 && j > 0) min_val = std::min(min_val, diversity_map[i+1][j-1] + d2); // 左下

                diversity_map[i][j] = min_val;
            }
        }
    }
    std::list<std::pair<int, int>> compute_focal_path_once(float optimal_cost)
    {
        reset();
        if (optimal_cost >= INF) {
            return {};
        }

        float cost_limit = optimal_cost * w;

        std::priority_queue<Node, std::vector<Node>, CompareFocal> FOCAL;

        Node &start_node = nodes[start.first][start.second];
        
        reset_node(start_node); // 更新 ID，重置状态
        start_node.i = start.first;
        start_node.j = start.second;
        start_node.g = 0;       // 起点代价为 0
        start_node.h = h(start);
        start_node.f = start_node.g + start_node.h;
        // start_node.focal_value = calculate_diverse_focal_value(start_node, goal);
        start_node.focal_value = calculate_focal_value(start_node, goal);
        
        FOCAL.push(Node(start.first, start.second, 0, h(start), start_node.focal_value));

        while (!FOCAL.empty())
        {
            Node current = FOCAL.top();
            FOCAL.pop();

            if (nodes[current.i][current.j].visited) continue;
            nodes[current.i][current.j].visited = true;

            if (current.i == goal.first && current.j == goal.second) {
                return get_path();
            }

            for (auto npos : get_neighbors({current.i, current.j}))
            {
                Node &neighbor_ref = nodes[npos.first][npos.second];
                reset_node(neighbor_ref);

                if (neighbor_ref.visited) continue;

                float cost = 1;
                if (use_static_cost) cost = penalties[npos.first][npos.second];
                if (use_dynamic_cost) cost += num_occupations[npos.first][npos.second];

                float new_g = current.g + cost;
                float new_h = h(npos);
                float new_f = new_g + new_h;

                if (new_f > cost_limit) continue;

                if (new_g < neighbor_ref.g)
                {
                    neighbor_ref.i = npos.first;
                    neighbor_ref.j = npos.second;
                    neighbor_ref.g = new_g;
                    neighbor_ref.h = new_h;
                    neighbor_ref.f = new_f;
                    neighbor_ref.parent = {current.i, current.j};
                    
                    // neighbor_ref.focal_value = calculate_diverse_focal_value(neighbor_ref, goal);
                    neighbor_ref.focal_value = calculate_focal_value(neighbor_ref, goal);
                    FOCAL.push(Node(npos.first, npos.second, new_g, new_h, neighbor_ref.focal_value));
                }
            }
        }
        return {};
    }

    void compute_focal_paths(int candidate_num = 3, int max_tries = 5, float w_min = 1.0, float w_max = 3.0)
    {
        int total_tries = 0;
        int path_found = 0;
        float frechet_threshold = 3.0;

        using param_t = std::uniform_real_distribution<float>::param_type;
        dist_w.param(param_t(w_min, w_max));
        focal_paths.clear();
        reset();
        compute_shortest_path();
        float c_opt = INF;
        if(nodes[goal.first][goal.second].g < INF)  c_opt = nodes[goal.first][goal.second].g;
        while(total_tries < max_tries && path_found < candidate_num)
        {
            w = dist_w(rng);
            g_weight = dist_g_weight(rng);
            // if(!focal_paths.empty()) update_diversity_map();

            auto path = compute_focal_path_once(c_opt);
            if(!path.empty())
            {
                // focal_paths.push_back(path);
                // path_found++;
                if (!is_similar_to_existing(path, frechet_threshold))
                {
                    focal_paths.push_back(path);
                    path_found++;
                }
            }
            total_tries++;
        }
    }
    float get_avg_distance(int si, int sj)
    {
        std::queue<std::pair<int, int>> fringe;
        fringe.emplace(si, sj);
        auto result = std::vector<std::vector<int>>(grid.size(), std::vector<int>(grid.front().size(), -1));
        result[si][sj] = 0;
        std::vector<std::pair<int, int>> moves = {{0,1},{1,0},{-1,0},{0,-1}};
        while(!fringe.empty())
        {
            auto pos = fringe.front();
            fringe.pop();
            for(const auto& move: moves)
            {
                int new_i(pos.first + move.first), new_j(pos.second + move.second);
                if(grid[new_i][new_j] == 0 && result[new_i][new_j] < 0)
                {
                    result[new_i][new_j] = result[pos.first][pos.second] + 1;
                    fringe.emplace(new_i, new_j);
                }
            }
        }
        float avg_dist(0), total_nodes(0);
        for(size_t i = 0; i < grid.size(); i++)
            for(size_t j = 0; j < grid[0].size(); j++)
                if(result[i][j] > 0)
                {
                    avg_dist += result[i][j];
                    total_nodes++;
                }
        return avg_dist/total_nodes;
    }

    void update_h_values(std::pair<int, int> g)
    {
        std::priority_queue<Node, std::vector<Node>, std::greater<Node>> open;
        h_values = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), INF));
        h_values[g.first][g.second] = 0;
        open.push(Node(g.first, g.second, 0, 0));
        while(!open.empty())
        {
            Node current = open.top();
            open.pop();
            for(auto n: get_neighbors({current.i, current.j})) {
                float cost(1);
                if(use_static_cost)
                    cost = penalties[n.first][n.second];
                if(h_values[n.first][n.second] > current.g + cost)
                {
                    open.push(Node(n.first, n.second, current.g + cost, 0));
                    h_values[n.first][n.second] = current.g + cost;
                }
            }
        }
    }

    void reset()
    {
        current_id++;
    }

public:
    planner(std::vector<std::vector<int>> _grid={}, float _use_static_cost=1.0, float _use_dynamic_cost=1.0, bool _reset_dynamic_cost=true):
    grid(_grid), use_static_cost(_use_static_cost), use_dynamic_cost(_use_dynamic_cost), reset_dynamic_cost(_reset_dynamic_cost)
    {
        abs_offset = {0, 0};
        goal = {0,0};
        start = {0, 0};
        current_id = 0;
        nodes = std::vector<std::vector<Node>>(grid.size(), std::vector<Node>(grid.front().size(), Node()));
        num_occupations = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        penalties = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 1));
        rng.seed(std::chrono::steady_clock::now().time_since_epoch().count());
        dist_g_weight = std::uniform_real_distribution<float>(0, 1);
        diversity_map = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
    }

    std::vector<std::vector<float>> get_num_occupied_matrix()
    {
        return num_occupations;
    }

    std::vector<std::vector<float>> precompute_penalty_matrix(int obs_radius)
    {
        penalties = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        float max_avg_dist(0);
        for(size_t i = obs_radius; i < grid.size() - obs_radius; i++)
            for(size_t j = obs_radius; j < grid.front().size() - obs_radius; j++)
                if(grid[i][j] == 0)
                {
                    penalties[i][j] = get_avg_distance(i, j);
                    max_avg_dist = std::fmax(max_avg_dist, penalties[i][j]);
                }
        for(size_t i = obs_radius; i < grid.size() - obs_radius; i++)
            for(size_t j = obs_radius; j < grid.front().size() - obs_radius; j++)
                if(grid[i][j] == 0)
                    penalties[i][j] = max_avg_dist / penalties[i][j];
        return penalties;
    }

    void set_penalties(std::vector<std::vector<float>> _penalties)
    {
        penalties = std::move(_penalties);
    }

    void update_occupied_cells(const std::list<std::pair<int, int>>& _occupied_cells, std::pair<int, int> cur_goal)
    {
        if(reset_dynamic_cost)
            if(goal.first != cur_goal.first || goal.second != cur_goal.second)
                num_occupations = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        for(auto o:_occupied_cells)
            num_occupations[o.first][o.second] += 1.0;
    }

    void update_occupations(py::array_t<double> array, std::pair<int, int> cur_pos, std::pair<int, int> cur_goal)
    {
        cur_goal = {cur_goal.first + abs_offset.first, cur_goal.second + abs_offset.second};
        if(reset_dynamic_cost)
            if(goal.first != cur_goal.first || goal.second != cur_goal.second)
                num_occupations = std::vector<std::vector<float>>(grid.size(), std::vector<float>(grid.front().size(), 0));
        py::buffer_info buf = array.request();
        std::list<std::pair<int, int>> occupied_cells;
        double *ptr = (double *) buf.ptr;
        cur_pos = {cur_pos.first + abs_offset.first, cur_pos.second + abs_offset.second};
        for(size_t i = 0; i < static_cast<size_t>(buf.shape[0]); i++)
            for(size_t j = 0; j < static_cast<size_t>(buf.shape[1]); j++)
                if(ptr[i*buf.shape[1] + j] != 0)
                    occupied_cells.push_back({cur_pos.first + i, cur_pos.second + j});
        for(auto o:occupied_cells)
            num_occupations[o.first][o.second]+= 1.0;
    }

    void update_path(std::pair<int, int> s, std::pair<int, int> g)
    {
        s = {s.first + abs_offset.first, s.second + abs_offset.second};
        g = {g.first + abs_offset.first, g.second + abs_offset.second};
        start = s;
        if(goal != g)
            update_h_values(g);
        goal = g;
        reset();
        compute_shortest_path();
    }

    void update_focal_paths(std::pair<int, int> s, std::pair<int, int> g)
    {
        s = {s.first + abs_offset.first, s.second + abs_offset.second};
        g = {g.first + abs_offset.first, g.second + abs_offset.second};
        start = s;
        if(goal != g)
            update_h_values(g);
        goal = g;
        compute_focal_paths();
    }

    std::list<std::list<std::pair<int, int>>> get_focal_paths()
    {
        return focal_paths;
    }

    std::list<std::pair<int, int>> get_path()
    {
        std::list<std::pair<int, int>> path;
        std::pair<int, int> next_node(INF,INF);
        if(nodes[goal.first][goal.second].g < INF)
            next_node = goal;
        if(next_node.first < INF and (next_node.first != start.first or next_node.second != start.second))
        {
            while (nodes[next_node.first][next_node.second].parent != start) {
                path.push_back(next_node);
                next_node = nodes[next_node.first][next_node.second].parent;
            }
            path.push_back(next_node);
            path.push_back(start);
            path.reverse();
        }
        for(auto it = path.begin(); it != path.end(); it++)
        {
            it->first -= abs_offset.first;
            it->second -= abs_offset.second;
        }
        return path;
    }
    std::pair<std::pair<int, int>, std::pair<int, int>> get_next_node()
    {
        std::pair<int, int> next_node(INF, INF);
        if(nodes[goal.first][goal.second].g < INF)
            next_node = goal;
        if(next_node.first < INF and (next_node.first != start.first or next_node.second != start.second))
            while (nodes[next_node.first][next_node.second].parent != start)
                next_node = nodes[next_node.first][next_node.second].parent;
        if(next_node == start)
            next_node = {INF, INF};
        if(next_node.first < INF)
            return {{start.first - abs_offset.first, start.second - abs_offset.second},
                    {next_node.first - abs_offset.first, next_node.second - abs_offset.second}};
        return {{INF, INF}, {INF, INF}};
    }
    void set_abs_start(std::pair<int, int> offset)
    {
        abs_offset = offset;
    }
};
