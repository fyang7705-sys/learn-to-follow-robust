from pogema import GridConfig

# noinspection PyUnresolvedReferences
import cppimport.import_hook
# noinspection PyUnresolvedReferences
from follower_cpp.planner import planner

from pydantic import BaseModel

try:
    from typing import Literal
except ImportError:
    from typing_extensions import Literal
import numpy as np

class PlannerConfig(BaseModel):
    use_static_cost: bool = True
    use_dynamic_cost: bool = True
    reset_dynamic_cost: bool = True
    planwindow: int = 1
    path_planner: Literal['diversefocal', 'focal', 'astar'] = "astar"

class PathPlanner:
    def __init__(self, cfg: PlannerConfig):
        self.planner = None
        self.obstacles = None
        self.starts = None
        self.cfg = cfg
        self.results = None
        self.obs_radius = None
        self.num_agents = None
        
    def add_grid_obstacles(self, obstacles, starts):
        self.obstacles = obstacles
        self.starts = starts
        self.planner = None
        self.results = None
    def set_planner(self):
        self.planner = [planner(self.obstacles, self.cfg.use_static_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost) for _ in range(self.num_agents)]
        for i, p in enumerate(self.planner):
            p.set_abs_start(self.starts[i])
        if self.cfg.use_static_cost:
            pen_calc = planner(self.obstacles, self.cfg.use_static_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost)
            penalties = pen_calc.precompute_penalty_matrix(self.obs_radius)
            for p in self.planner:
                p.set_penalties(penalties)
    def set_dynamic_cost(self, cost_map, observations):
        if self.num_agents is None:
            self.num_agents = len(cost_map)
        if self.obs_radius is None:
            self.obs_radius = len(cost_map[0]) // 2
        if self.planner is None:
            self.set_planner()
        for k, p in enumerate(self.planner):
            p.set_dynamic_cost(cost_map, (observations[k]['xy'][0] - self.obs_radius, observations[k]['xy'][1] - self.obs_radius))

    def update(self, obs):
        self.num_agents = len(obs)
        self.obs_radius = len(obs[0]['obstacles']) // 2
        if self.planner is None:
            self.set_planner()

        if self.results is None:
            self.results = [None] * self.num_agents
        for k in range(self.num_agents):
            if obs[k]['xy'] == obs[k]['target_xy']:
                continue
            obs[k]['agents'][self.obs_radius][self.obs_radius] = 1
            self.planner[k].update_path(obs[k]['xy'], obs[k]['target_xy'])
            self.results[k] = self.planner[k].get_path()
    def update_dist_mat(self, obs):
        self.num_agents = len(obs)
        self.obs_radius = len(obs[0]['obstacles']) // 2
        if self.planner is None:
            self.set_planner()
        for k in range(self.num_agents):
            self.planner[k].update_dist_mat(obs[k]['xy'], obs[k]['target_xy'])

    def get_path(self):
        return self.results

    def get_dist_mat(self):
        return [p.get_dist_mat(self.obs_radius) for p in self.planner]

class Planner:
    def __init__(self, cfg: PlannerConfig):
        self._cfg = cfg
        self._agent = None

    def update(self, observations):
        return self._agent.update(observations)

    def get_path(self):
        return self._agent.get_path()

    def reset_states(self, ):
        self._agent = PathPlanner(self._cfg)
