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

class Planner:
    def __init__(self, cfg: PlannerConfig):
        self.planner = None
        self.obstacles = None
        self.starts = None
        self.cfg = cfg
        self.results = None
        self.replan_window = cfg.planwindow
        self.replan_counts = None

    def add_grid_obstacles(self, obstacles, starts):
        self.obstacles = obstacles
        self.starts = starts
        self.planner = None
        self.results = None
        self.replan_counts = np.zeros((len(self.starts),))
        
    def update(self, obs):
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
        for k in range(num_agents):
            if obs[k]['xy'] == obs[k]['target_xy']:
                continue
            obs[k]['agents'][obs_radius][obs_radius] = 0
            self.planner[k].update_occupations(obs[k]['agents'], (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius), obs[k]['target_xy'])
            obs[k]['agents'][obs_radius][obs_radius] = 1
            if self.if_replan(k, obs[k]):
                if self.cfg.path_planner == "astar":
                    self.planner[k].update_path(obs[k]['xy'], obs[k]['target_xy'])
                    self.results[k] = self.planner[k].get_path()
                else:
                    self.planner[k].update_focal_paths(obs[k]['xy'], obs[k]['target_xy'])
                    self.results[k] = self.planner[k].get_focal_paths()
            self.replan_counts[k] += 1

    def if_replan(self, agent_index, agent_obs):
        if self.replan_counts[agent_index] % self.replan_window == 0:
            return True
        if self.cfg.path_planner == "astar":
            cur_goal = list(self.results[agent_index][-1]) if self.results[agent_index] else agent_obs['target_xy']
        else:
            cur_goal = list(self.results[agent_index][0][-1]) if self.results[agent_index][0] else agent_obs['target_xy']
        if agent_obs['xy'] == cur_goal:
            return True
        return False
    
    def get_path(self):
        return self.results


class ResettablePlanner:
    def __init__(self, cfg: PlannerConfig):
        self._cfg = cfg
        self._agent = None

    def update(self, observations):
        return self._agent.update(observations)

    def get_path(self):
        return self._agent.get_path()

    def reset_states(self, ):
        self._agent = Planner(self._cfg)
