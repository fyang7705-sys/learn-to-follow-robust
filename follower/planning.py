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

class Planner:
    def __init__(self, cfg: PlannerConfig):
        self.planner = None
        self.obstacles = None
        self.starts = None
        self.cfg = cfg
        self.results = None
        self.replan_window = 5
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

        for k in range(num_agents):
            if obs[k]['xy'] == obs[k]['target_xy']:
                continue
            obs[k]['agents'][obs_radius][obs_radius] = 0
            self.planner[k].update_occupations(obs[k]['agents'], (obs[k]['xy'][0] - obs_radius, obs[k]['xy'][1] - obs_radius), obs[k]['target_xy'])
            obs[k]['agents'][obs_radius][obs_radius] = 1
            if self.if_replan(k, obs[k]):
                self.planner[k].update_focal_paths(obs[k]['xy'], obs[k]['target_xy'])
                # self.planner[k].update_path(obs[k]['xy'], obs[k]['target_xy'])
            self.replan_counts[k] += 1

    def if_replan(self, agent_index, agent_obs):
        if self.replan_counts[agent_index] % self.replan_window == 0:
            return True
        cur_goal = list(self.results[agent_index][0][-1]) if self.results[agent_index][0] else obs[agent_index]['target_xy']
        # print("cur_goal", cur_goal)
        # print("agent_obs['xy']", agent_obs['xy'])
        if agent_obs['xy'] == cur_goal:
            return True
        return False

    def get_path(self):
        self.results = []
        for idx in range(len(self.planner)):
            self.results.append(self.planner[idx].get_focal_paths())
            # results.append(self.planner[idx].get_path())
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
