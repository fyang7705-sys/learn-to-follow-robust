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
from sample_factory.utils.utils import log

class PlannerConfig(BaseModel):
    use_static_cost: bool = True
    use_dynamic_cost: bool = True
    use_dist_mat: bool = False
    reset_dynamic_cost: bool = True
    planwindow: int = 1
    path_planner: Literal['diversefocal', 'focal', 'astar'] = "astar"

class Planner:
    def __init__(self, cfg: PlannerConfig):
        self.planner = None
        self.obstacles = None
        self.starts = None
        self.cfg = cfg
        self.results = None
        self.dist_mats = None
        self.replan_window = cfg.planwindow
        self.replan_counts = None

    def add_grid_obstacles(self, obstacles, starts):
        self.obstacles = obstacles
        self.starts = starts
        self.planner = None
        self.results = None
        self.replan_counts = np.zeros((len(self.starts),))
        
    def update(self, obs, use_dist_mat=False):
        num_agents = len(obs)
        obs_radius = len(obs[0]['obstacles']) // 2
        if self.planner is None:
            self.planner = [planner(self.obstacles, self.cfg.use_static_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost) for _ in range(num_agents)]
            for i, p in enumerate(self.planner):
                p.set_abs_start(self.starts[i])
            if self.cfg.use_static_cost:
                pen_calc = planner(self.obstacles, self.cfg.use_static_cost, self.cfg.use_dynamic_cost, self.cfg.reset_dynamic_cost)
                penalties = pen_calc.precompute_penalty_matrix(obs_radius)
                for p in self.planner:
                    p.set_penalties(penalties)
        if self.results is None:
            self.results = [None] * num_agents
        if use_dist_mat and self.dist_mats is None:
            self.dist_mats = [None] * num_agents
        for k in range(num_agents):
            self.replan_counts[k] += 1
            if obs[k]['xy'] == obs[k]['target_xy']:
                continue
            obs[k]['agents'][obs_radius][obs_radius] = 0
            self.planner[k].update_occupations(obs[k]['agents'], (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius), obs[k]['target_xy'])
            obs[k]['agents'][obs_radius][obs_radius] = 1
            if self.if_replan(k, obs[k], use_dist_mat):
                if use_dist_mat:
                    self.planner[k].update_dist_mat(obs[k]['xy'], obs[k]['target_xy'])
                    self.dist_mats[k] = self.planner[k].get_dist_mat(obs_radius)
                elif self.cfg.path_planner == "astar":
                    self.planner[k].update_path(obs[k]['xy'], obs[k]['target_xy'])
                    self.results[k] = self.planner[k].get_path()
                else: # diversefocal or focal
                    self.planner[k].update_focal_paths(obs[k]['xy'], obs[k]['target_xy'])
                    self.results[k] = self.planner[k].get_focal_paths()

    def if_replan(self, agent_index, agent_obs, use_dist_mat=False):
        if self.replan_counts[agent_index] % self.replan_window == 0:
            return True
        if use_dist_mat:
            cur_goal = agent_obs['target_xy']
        elif self.cfg.path_planner == "astar":
            cur_goal = list(self.results[agent_index][-1]) if self.results[agent_index] else agent_obs['target_xy']
        else: # diversefocal or focal
            cur_goal = list(self.results[agent_index][0][-1]) if self.results[agent_index][0] else agent_obs['target_xy']
        if agent_obs['xy'] == cur_goal:
            return True
        return False
    
    def get_path(self):
        return self.results
    def get_dist_mat(self):
        return self.dist_mats

class ResettablePlanner:
    def __init__(self, cfg: PlannerConfig):
        self._cfg = cfg
        self._agent = None

    def update(self, observations, use_dist_mat=False):
        return self._agent.update(observations, use_dist_mat)

    def get_path(self):
        return self._agent.get_path()
    def get_dist_mat(self):
        return self._agent.get_dist_mat()
    def reset_states(self):
        self._agent = Planner(self._cfg)
