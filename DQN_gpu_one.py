from __future__ import annotations
import numpy as np
import random
import time
import math
import collections
from collections import namedtuple, deque
from copy import deepcopy
from dataclasses import dataclass, field
from typing import List, Tuple
import matplotlib.pyplot as plt
from scipy.signal import convolve2d

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# ──────────────────────────────  constants  ──────────────────────────────
SIZE = 17  # board is 17×17
CENTER_START = 4  # inclusive index of central 9×9 block (rows/cols 4‑12)
CENTER_END = 12  # inclusive

EMPTY = 0
OBSTACLE = -1
BOMB = -3  # static bombs placed by map‑maker or NPC
# We **do not** place the players as negative numbers into the grid itself – the
# log writer does that only when serialising a snapshot so that grid state is
# always pure except for bombs / obstacles / coins.

# Direction encoding: 0↑, 1↓, 2←, 3→ (same as dev‑kit)
DIRS: Tuple[Tuple[int, int], ...] = ((-1, 0), (1, 0), (0, -1), (0, 1), (0, 0))

# Gold parameters (tweak if the official rules change)
MAX_GOLD_VALUES = 15
MAX_GOLD_CELLS = 3
MAX_BOMB_VALUES = 22  # max value of a bomb (not used in this game)
MIN_BOMB_VALUES = 9  # min value of a bomb (not used in this game)
INITIAL_GOLD_CELLS = 50  # random coins sprinkled at start
# CENTER_PER_TURN = 1  # coins that appear each turn in the 9×9 centre
NPC_CYCLE = 20  # every 3 turns an NPC visits
BOMB_CYCLE = 50 # every 50 turns a bomb refreshes

# ────────────────────────────────  helpers  ───────────────────────────────

def within_board(r: int, c: int) -> bool:
    return 0 <= r < SIZE and 0 <= c < SIZE


def add_dir(pos: Tuple[int, int], dir_idx: int) -> Tuple[int, int]:
    dr, dc = DIRS[dir_idx]
    return pos[0] + dr, pos[1] + dc

# ───────────────────────────────  dataclasses  ───────────────────────────────
@dataclass
class PlayerState:
    id: int
    position: List[int]  # [row, col]
    gold: int = 0
    actions: List[int] = field(default_factory=list)
    cost: int = 0  # micro‑seconds used for MoveDecision

    def to_dict(self):
        return {
            "id": self.id,
            "position": self.position[:],
            "gold": self.gold,
            "actions": self.actions[:],
            "cost": self.cost,
        }


# ────────────────────────────  abstract Player API  ───────────────────────────

class AbstractPlayer:
    """Same interface as the official dev-kit."""

    def MoveDecision(
        self, grid: List[List[int]], gold: int, gold_2: int
    ) -> List[int]:
        raise NotImplementedError


class RandomPlayer():
    """Fallback bot - moves randomly."""

    def MoveDecision(self, grid, gold, gold_2):
        return [4,4,4]
        # return [random.randint(0, 4) for _ in range(3)]


# ───────────────────────────────  main Game class  ───────────────────────────────

class Game:
    def __init__(self, p1, p2, seed: int | None = None):
        self.rng = random.Random(seed)
        self.players: List[PlayerState] = [
            PlayerState(id=1, position=[0, 0]),
            PlayerState(id=2, position=[SIZE - 1, SIZE - 1]),
        ]
        self.player_objs = [p1, p2]
        self.total_gold = 0  # total gold on the board, used for debugging
        # initialise board
        self.grid = np.zeros((SIZE, SIZE), dtype=int) 
        self.grid_bomb = np.zeros((SIZE, SIZE), dtype=int)  # bomb grid
        self._place_static_obstacles()
        self.grid[0,0]=-2
        self.grid[SIZE-1,SIZE-1]=-2
        #self._spawn_center_gold()
        #self._spawn_random_bomb(INITIAL_GOLD_CELLS)

        self.log: List[dict] = []
        self._round = 0

    def reset(self, p1, p2, seed: int | None = None):
        self.rng = random.Random(seed)
        self.players: List[PlayerState] = [
            PlayerState(id=1, position=[0, 0]),
            PlayerState(id=2, position=[SIZE - 1, SIZE - 1]),
        ]
        self.player_objs = [p1, p2]
        self.total_gold = 0  # total gold on the board, used for debugging
        # initialise board
        self.grid = np.zeros((SIZE, SIZE), dtype=int) 
        self.grid_bomb = np.zeros((SIZE, SIZE), dtype=int)  # bomb grid
        self._place_static_obstacles()
        self.grid[0,0]=-2
        self.grid[SIZE-1,SIZE-1]=-2
        #self._spawn_center_gold()
        #self._spawn_random_bomb(INITIAL_GOLD_CELLS)

        self.log: List[dict] = []
        self._round = 0

    # ───────────────────────────────  setup helpers  ───────────────────────────────

    def _place_static_obstacles(self):
        # Minimalistic map: ring of walls + four bombs in corners just like sample log
        self.grid=np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                            [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0], 
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], 
                            [0, 0, 0, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0], 
                            [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0], 
                            [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0], 
                            [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [0, 0, 0, -1, -1, -1, -1, 0, 0, 0, -1, -1, -1, -1, 0, 0, 0], 
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                            [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0], 
                            [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0], 
                            [0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0, 0, -1, 0, 0, 0], 
                            [0, 0, 0, -1, -1, -1, -1, 0, -1, 0, -1, -1, -1, -1, 0, 0, 0], 
                            [0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0], 
                            [0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0], 
                            [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])

    # def _spawn_random_gold(self, count: int):
    #     placed = 0
    #     while placed < count:
    #         r = self.rng.randrange(SIZE)
    #         c = self.rng.randrange(SIZE)
    #         if [r, c] not in [p.position for p in self.players]:
    #             self.grid[r][c] += self.rng.choice(GOLD_VALUES)
    #             placed += 1

    # ───────────────────────────────  public API  ───────────────────────────────

    def winner(self) -> int | None:
        g1, g2 = self.players[0].gold, self.players[1].gold
        return 1 if g1 > g2 else 2 if g2 > g1 else None
    def play(self, max_rounds: int = 900):
        self.total_gold = 0
        for _ in range(max_rounds):
            self._single_round()
        winner = self.winner()
        if winner is not None:
            print(f"Total gold: {self.total_gold} p1_vs_p2: {self.players[0].gold} vs {self.players[1].gold}")
            percent= 1.0 * self.players[winner-1].gold / self.total_gold * 100 if self.total_gold > 0 else 0
            print(f"Player {winner} wins, get {percent}% gold!")
            return winner,percent
        else:
            print("It's a draw!")
            return None
    # def save_log(self, path: str | Path):
    #     Path(path).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in self.log), encoding="utf-8")

    # ───────────────────────────────  core mechanics  ───────────────────────────────

    def _single_round(self):
        #print(self.grid)
        # ── spawn gold and NPC drops ──────────────────────
        if self._round % 3 == 0:
            self._spawn_center_gold()
        if self._round % (3*NPC_CYCLE) == 0:
            self._npc_visit()
        if self._round % (3*BOMB_CYCLE) == 0:
            self._spawn_random_bomb()
        # print('Round:', self._round, 'gold_in_map:', self.grid[np.where(self.grid > 0)].sum(), 'total_gold:', self.total_gold)
        # print(np.array(self._serialise_grid(self.players[0])))
        # ── ask both AIs for decisions ────────────────────
        for idx, p in enumerate(self.players):
            t0 = time.perf_counter_ns()
            actions = self.player_objs[idx].MoveDecision(deepcopy(self._serialise_grid(p)), p.gold, self.players[1 - idx].gold)[:1]
            cost = (time.perf_counter_ns() - t0) // 1000  # micro‑seconds
            p.actions = list(actions)
            p.cost = int(cost)
        # ── execute moves in increasing time cost order ───
        for p in sorted(self.players, key=lambda pl: pl.cost):
            old_gold = p.gold
            self._execute_actions(p)
            new_gold = p.gold
            p.step_gold = new_gold-old_gold
        self._round += 1
        # new_grid = self._serialise_grid(self.players[0])
        if self._round<900:
            done = False
        else:
            done = True
        return self.players,self._serialise_grid(self.players[0]),self._serialise_grid(self.players[1]),done

    def get_state(self):
        return self.players,self._serialise_grid(self.players[0]),self._serialise_grid(self.players[1])
    # ───────────────────────────────  sub‑helpers  ───────────────────────────────

    def _execute_actions(self, p: PlayerState):
        # print('Player:', p.id, 'actions:', p.actions, 'gold:', p.gold, 'total_gold:', self.total_gold, 'cost:', p.cost)
        for dir_idx in p.actions:
            if dir_idx == 4:
                continue
            # print('action:', dir_idx)
            r_raw, c_raw = p.position
            r, c = add_dir(tuple(p.position), dir_idx)
            self.grid[r_raw,c_raw]=EMPTY
            #self.grid_bomb[r_raw,c_raw]=EMPTY
            # clip invalid move
            if not within_board(r, c) or self.grid[r][c] == OBSTACLE or self.grid[r,c] == -2:
                self.grid[r_raw,c_raw]=-2
                self.grid_bomb[r_raw,c_raw]=EMPTY
                continue
            old_grid = self.grid_bomb.copy()
            # bombs – lose 10 % gold, bomb removed
            if self.grid_bomb[r,c] == BOMB:
                lost = np.ceil(p.gold * 0.10)
                p.gold -= lost
                self.grid_bomb[r,c]=EMPTY
                # print('player ',p.id, 'hit a bomb, lost', lost, 'gold, now has', p.gold, 'gold')
                # print(old_grid)
                # print(r,c)
            # coins
            if self.grid[r,c] > 0:
                p.gold += self.grid[r,c]
                self.grid[r,c] = EMPTY
            # finally move
            self.grid[r,c] = -2
            p.position[:] = [r, c]
            # print(np.array(self._serialise_grid(self.players[0])))
            #if self.players[0].gold + self.players[1].gold + self.grid[np.where(self.grid > 0)].sum() != self.total_gold:
                #print(f"Warning: total gold mismatch! In round {self._round} action {dir_idx}, {self.players[0].gold} + {self.grid[np.where(self.grid > 0)].sum()} != {self.total_gold}")
            # else:
            # print(f"gold: {p.gold} gold_in_map: {self.grid[np.where(self.grid > 0)].sum()} total_gold: {self.total_gold}")

    def _spawn_center_gold(self):
        for _ in range(self.rng.randrange(MAX_GOLD_CELLS+1)):
            r = self.rng.randrange(CENTER_START, CENTER_END+1)
            c = self.rng.randrange(CENTER_START, CENTER_END+1)
            if self.grid[r,c] >=0:
                if self.grid_bomb[r,c] == BOMB:
                    continue
                #print('loc: ',(r,c), 'old_gold: ',self.grid[r,c])
                add_gold = self.rng.randrange(1,MAX_GOLD_VALUES+1)
                self.grid[r,c] += add_gold
                self.total_gold += add_gold
    def _spawn_random_bomb(self):
        self.grid_bomb = np.zeros((SIZE, SIZE), dtype=int)  # reset bomb grid
        r, c = np.where(self.grid == EMPTY)
        empty_cells = list(zip(r, c))
        for _ in range(self.rng.randrange(MIN_BOMB_VALUES, MAX_BOMB_VALUES+1)):
            # 从空格中随机选一个
            if empty_cells:
                r, c = self.rng.choice(empty_cells)
                if self.grid[r,c]==0:
                    self.grid_bomb[r,c] = BOMB
                    empty_cells.remove((r, c))

    def _npc_visit(self):
        # simple straight path across middle row
        raw_gold = np.array([[0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3], 
                             [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0], 
                             [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0], 
                             [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0], 
                             [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0], 
                             [0, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 0], 
                             [3, 3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 3, 3], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 3, 3, 3, 3, 3, 3, 3, 0, 0, 0, 0, 0], 
                             [0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0, 3, 0, 0, 0, 0, 0]])
        bomb_pos = np.where(self.grid_bomb == BOMB)
        player_pos = [p.position for p in self.players]
        for p in player_pos:
            raw_gold[p[0],p[1]] = EMPTY
        # remove bombs from the grid
        for r, c in zip(bomb_pos[0], bomb_pos[1]):
            raw_gold[r,c] = EMPTY
        self.grid += raw_gold
        gold=np.sum(raw_gold)
        self.total_gold += gold
        # print(f"NPC visit: spawned {gold} gold, total gold now: {self.total_gold}")

    # ───────────────────────────────  serialisation  ───────────────────────────────

    def _serialise_grid(self, player) -> List[List[int]]:
        g = deepcopy((self.grid+self.grid_bomb).tolist())
        r, c = player.position
        g[r][c]=-9
        return g

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    def __init__(self, n_s, n_a):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_s, 128)
        self.layer2 = nn.Linear(128, 128)
        self.layer3 = nn.Linear(128, 64)
      #  self.layer4 = nn.Linear(192, 128)
        # self.layer5 = nn.Linear(300, 300)
        # self.layer6 = nn.Linear(300, 150)
        self.layer4 = nn.Linear(64, n_a)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
      #  x = F.relu(self.layer4(x))
        # x = F.relu(self.layer5(x))
        # x = F.relu(self.layer6(x))
        return self.layer4(x)

def encode_state(player_pos, original_map, gold, Z):
    # 创建地图的深拷贝以避免修改原始地图
    new_map = [row.copy() for row in original_map]
    
    # 确保地图是17x17
    assert len(new_map) == 17 and all(len(row) == 17 for row in new_map), "地图尺寸必须为17x17"
    
    # 确保玩家坐标在地图范围内
    x, y = player_pos
    assert 0 <= x < 17 and 0 <= y < 17, "玩家坐标超出地图范围"
    
    # 1. 将玩家位置设为0
    new_map[x][y] = 0
    
    # 处理地图中的特殊值
    for i in range(17):
        for j in range(17):
            value = new_map[i][j]
            if value == -2 or value == -1:
                # 2. 将-1和-2设为负无穷
                new_map[i][j] = 0
            if value == -3:
                # percentage = np.ceil(gold * Z)
                new_map[i][j] = 0
    
    return new_map

def valid_action(pos,grid:list[list[int]]):
    x,y = pos
    valid_actions = [True,True,True,True,True]
    grid[x][y]=0
    move_map = {0:(-1,0),1:(1,0),2:(0,-1),3:(0,1),4:(0,0)}
    for i in range(5):
        nx,ny = x+move_map[i][0],y+move_map[i][1]
        if nx < 0 or nx > 16 or ny < 0 or ny > 16:
            valid_actions[i] = False
        else:
            value = grid[nx][ny]
            if value == -1 or value == -2 or value == -3:
                valid_actions[i] = False
    return valid_actions

def decode_action():
    moves = []
    # v = np.zeros(125,dtype=int)
    for i in range(4):
        moves.append(i)
    return moves

def preprocess_state(pos,grid: list[list[int]], gold: int, gold_2: int):
    new_grid = encode_state(pos, grid, gold, 0.1)
    # den_grid = den_calc(new_grid,3)
    len_g = 17
    state = np.zeros(3+len_g**2)
    state[0],state[1] = pos
    state[2] = gold
    # state[2],state[3] = gold,gold_2
    for i in range(len_g):
        state[len_g*i+3:len_g*(i+1)+3] = new_grid[i]
    state = normalize_state(state)
    s_ten = torch.from_numpy(state)
    return s_ten

def clip_reward(reward):
    return np.clip(reward/10, -1.0, 1.0)

def normalize_state(state):
    new_state = np.zeros([len(state)])
    pos = state[0:2]
    gold = state[2:3]
    map_data = state[3:]
    mean = map_data.mean()
    std = map_data.std() + 1e-6
    new_state[0:2] = pos/17
    new_state[2:3] = gold/5000
    new_state[3:] = (map_data-mean)/std
    return new_state

def valid_move(x, y, grid,dim):
        if not (0 <= x < dim and 0 <= y < dim):
            return False
        if grid[x][y] == -1 or grid[x][y] == -2 or grid[x][y] == -3:
            return False
        return True

def bfs(start_pos, grid):
        queue = collections.deque([(start_pos, [])])
        vted = {start_pos}
        gold_paths = {}
        lg=len(grid)

        while queue:
            (x, y), path = queue.popleft()

            if grid[x][y] >= 1:
                gold_paths[(x, y)] = path

            
            moves = [(0, -1), (0, 1), (-1, 0), (1, 0)]
            move_map = { (0, -1): 2, (0, 1): 3, (-1, 0): 0, (1, 0): 1 } 

            for m in moves:
                dx, dy = m
                nx, ny = x + dx, y + dy
                if valid_move(nx, ny, grid,lg) and (nx, ny) not in vted:
                    
                    vted.add((nx, ny))
                    new_path = move_map[(dx, dy)] 
                    queue.append(((nx, ny), path+[new_path]))
    
        return gold_paths

def random_dual_range():
    if random.random() < 0.5:  # 50%概率选择负区间
        return random.uniform(-1, -0.5)
    else:
        return random.uniform(0.5, 1.0)

def potential(pos,grid):
    paths = bfs(pos,grid)
    Vp = 0
    # new_grid = encode_state(pos, grid, 0, 0)
    # den_grid = den_calc(pos,new_grid,5)
    for (gx, gy), path in paths.items():
        lp = len(path)
        Vp += grid[gx][gy]/lp
    Vp += random_dual_range()/100
    return Vp

def den_calc(pos,grid,ker_dim):
    grid = np.array(grid)
    # if ker_dim == 3:
    #     kernel = np.array([[0.11,0.25,0.11],
    #                        [0.25,1,0.25],
    #                        [0.11,0.25,0.11]])
    # elif ker_dim == 5:
    #     kernel = np.array([[0.06,0.09,0.11,0.09,0.06],
    #                        [0.09,0.17,0.25,0.17,0.09],
    #                        [0.11,0.25,1.00,0.25,0.11],
    #                        [0.09,0.17,0.25,0.17,0.09],
    #                        [0.06,0.09,0.11,0.09,0.06]])
    kernel = np.ones((ker_dim,ker_dim))/ker_dim**2
    den_grid = convolve2d(grid, kernel, mode='same', boundary='symm')
    return den_grid

class RewardNormal:
    def __init__(self, clip=None):
        self.clip = clip  # 可选裁剪
        self.mean = 0
        self.std = 1
        self.count = 0

    def update(self, reward):
        self.count += 1
        new_mean = self.mean + (reward - self.mean) / self.count
        new_std = self.std + ((reward - self.mean) * (reward - new_mean) - self.std) / self.count
        self.mean, self.std = new_mean, max(new_std, 1e-6)  # 避免除零

    def normalize(self, reward):
        if self.count < 10:  # 初始阶段不归一化
            return reward
        normalized = (reward - self.mean) / self.std
        return np.clip(normalized, -self.clip, self.clip) if self.clip else normalized

class Player:
    def __init__(self):
        self.gold = 0
        self.step_gold = 0
        self.pos = None
        self.op_pos = None
        self.len_g = 17
        self.len_memory = 10000
        self.memory = deque(maxlen=self.len_memory)
        self.batch_size = 128
        self.gamma = 0.99
        self.num_actions = 3
        self.eps_start = 1.0
        self.eps_end = 0.05
        self.eps_decay = 10000
        self.tau = 0.005
        self.lr = 1e-4
        self.model = DQN(3+self.len_g**2,5).to(device)
        # for layer in self.model.modules():
        #     if isinstance(layer, nn.Linear):
        #         nn.init.kaiming_normal_(layer.weight, mode='fan_in', nonlinearity='relu')
        self.target_model = DQN(3+self.len_g**2,5).to(device)
        self.target_model.load_state_dict(self.model.state_dict())
        self.optimizer = optim.AdamW(self.model.parameters(),lr = self.lr)
        self.steps_done = 0

    def find_pos(self, grid: list[list[int]]):
        for i in range(self.len_g):
            for j in range(self.len_g):
                if grid[i][j] == -9:
                    self.pos = (i, j) 

                elif grid[i][j] == -2:
                    self.op_pos = (i, j)
        return self.pos

    def select_action(self,state,pos,grid):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * \
            math.exp(-1. * self.steps_done / self.eps_decay)
        self.steps_done += 1
        valid_actions = torch.BoolTensor(valid_action(pos,grid)).to(device)
        if sample > eps_threshold:
            with torch.no_grad():
                state = state.to(torch.float32)
                state = state.unsqueeze(0).to(device)
                q_values = self.model(state)
                q_values[~valid_actions.unsqueeze(0)] = -float('inf')
                return q_values.argmax().item()
        else:
            while True:
                i = random.randint(0, 3)
                valid = valid_actions[i]
                if valid:
                    return i
                
        
    def MoveDecision(self, grid: list[list[int]], gold: int, gold_2: int) -> list[int]:
        self.pos = self.find_pos(grid)
        s_ten = preprocess_state(self.pos, grid, gold, gold_2)
        n_move = self.select_action(s_ten,self.pos, grid)
        return [n_move]


    def remember(self, players, grid, action,reward, next_players, next_grid, done):
        self.memory.append((players, grid, action,reward,next_players, next_grid, done))

    def replay(self):
        batch = random.sample(self.memory, self.batch_size)
        players, grid,actions, rewards, next_players, next_grid, dones = zip(*batch)
        states = torch.tensor(np.array([preprocess_state(p[0].position, g, p[0].gold, p[1].gold) for p, g in zip(players, grid)])).to(device)
        # code_actions = dict(zip(decode_action().values(), decode_action().keys()))
        # for i in range(5):
        #     for j in range(5):
        #         for k in range(5):
        #             code_actions[(i,j,k)] = i
        actions = torch.tensor(np.array([a[0] for a in actions])).to(device)
        rewards = torch.tensor(rewards).to(device)
        next_states = torch.tensor(np.array([preprocess_state(p[0].position, g, p[0].gold, p[1].gold) for p, g in zip(next_players, next_grid)])).to(device)
        dones = torch.tensor(dones).to(device)
        dones = dones.long() 
        # states = torch.FloatTensor(np.array(states))
        # states = torch.FloatTensor(np.array(states))
        # actions = torch.LongTensor(actions)
        # rewards = torch.FloatTensor(rewards)
        # next_states = torch.FloatTensor(np.array(next_states))
        # dones = torch.FloatTensor(dones)
        states = states.to(torch.float32)
        next_states = next_states.to(torch.float32)
        current_q = self.model(states).gather(1, actions.unsqueeze(1))
        next_q = self.target_model(next_states).max(1)[0].detach()
        target_q = rewards + (1 - dones) * self.gamma * next_q
        # print(target_q)

        current_q = current_q.to(torch.float32)
        target_q = target_q.to(torch.float32)
        criterion = nn.SmoothL1Loss()
        loss = criterion(current_q.squeeze(), target_q)
        # loss = nn.MSELoss()(current_q.squeeze(), target_q)
        self.optimizer.zero_grad()
        loss = loss.to(torch.float32)   
        loss.backward()
        torch.nn.utils.clip_grad_value_(self.model.parameters(), 100)
        self.optimizer.step()
        self.update_target()
        return loss.detach().cpu().numpy()

    def update_target(self):
        target_model_dict = self.target_model.state_dict()
        model_dict = self.model.state_dict()
        for key in self.model.state_dict():
            target_model_dict[key] = model_dict[key]*self.tau+target_model_dict[key]*(1-self.tau)
        self.target_model.load_state_dict(target_model_dict)

def plot_loss(loss1,loss2,show_result=False):
    plt.figure(1)
    if show_result:
        plt.title('Result')
    else:
        plt.clf()
        plt.title('Training...')
    plt.xlabel('Episode')
    plt.ylabel('Loss')
    plt.plot(range(1,len(loss1)+1),loss1,label = '1')
    plt.plot(range(1,len(loss2)+1),loss2,label = '2')
    plt.legend()
    # Take 100 episode averages and plot them too
    plt.savefig('loss.png')
    plt.show()

def train():
    # p = Player()
    rp = RandomPlayer()
    # env = Game(p,rp,300)
    agent1 = Player()
    # agent2 = Player()
    print("Model device:", next(agent1.model.parameters()).device)
    env1 = Game(agent1,rp,1)
    # env2 = Game(agent2,rp,1)
    # env = Game(agent1,agent2,1)
    # env = Game(p,rp,300)
    # env2 = Game(agent2,rp,300)
    if torch.cuda.is_available() or torch.backends.mps.is_available():
        num_episodes = 50
    else:
        num_episodes = 50
    ll1 = []
    ll2 = []
    rn1 = RewardNormal()
    rn2 = RewardNormal()
    memory_count = 0
    for e in range(num_episodes):
        print("Episode: ", e)
        env1.reset(agent1,rp,1)
        # env2.reset(agent2,rp,1)
        # processed_state = preprocess_state(players[0].position,grid,players[0].gold,players[1].gold)
        done = False
        while not done:
            # action1 = agent1.MoveDecision(grid,players[0].gold,players[1].gold)
            # action2 = agent2.MoveDecision(grid,players[1].gold,players[0].gold)
            # players,grid1,grid2,done = env._single_round()
            # action1 = players[0].actions
            # action2 = players[1].actions
            # reward1 = players[0].step_gold
            # reward2 = players[1].step_gold
            players,grid,rg = env1.get_state()
            Vp_old = potential(tuple(players[0].position),grid)
            players1,grid1,rg,done = env1._single_round()
            # players2,grid2,rg,done = env2._single_round()
            pos1 = players1[0].position
            # pos2 = players2[0].position
            Vp_new = potential(tuple(pos1),grid1)
            # new_grid1 = encode_state(pos1, grid1, players1[0].gold, 0.1)
            # new_grid2 = encode_state(pos2, grid2, players2[0].gold, 0.1)
            # new_grid1[pos1[0]][pos1[1]] = players1[0].step_gold
            # new_grid2[pos2[0]][pos2[1]] = players2[0].step_gold
            # den_grid1 = den_calc(new_grid1,5)
            # den_grid2 = den_calc(new_grid2,5)
            # s1 = preprocess_state(pos1,grid1, players1[0].gold, players1[1].gold)
            # s2 = preprocess_state(pos2,grid2, players2[0].gold, players2[1].gold)
            action1 = players1[0].actions
            # action2 = players2[0].actions
            reward1 = (Vp_new-Vp_old)/100
            # if abs(reward1) < 1e-4:
            #     reward1 = -1e-4
            # reward1 = den_grid1[pos1[0]][pos1[1]]
            # reward2 = den_grid2[pos2[0]][pos2[1]]
            # rn1.update(reward1)
            # rn2.update(reward2)
            # reward1 = rn1.normalize(reward1)
            # reward2 = rn2.normalize(reward2)
            # reward1 = clip_reward(reward1)
            # reward2 = clip_reward(reward2)
            # map_data1 = s1[4:]
            # mean = map_data1.mean()
            # std = map_data1.std() + 1e-6
            # reward1 = (players1[0].step_gold-mean)/std
            # map_data2 = s2[4:]
            # mean = map_data2.mean()
            # std = map_data2.std() + 1e-6
            # reward2 = (players2[0].step_gold-mean)/std
            # reward1 = clip_reward(players1[0].step_gold)
            # reward2 = clip_reward(players1[1].step_gold)
            # print(reward1,reward2)

            # next_players1,next_grid1,next_rg,done= env1._single_round()
            # next_players2,next_grid2,next_rg,done= env2._single_round()
            # next_players,next_grid1,next_grid2,done= env._single_round()
            # Convert actions to moves (simplified)
            # next_state, rewards, done = env.step([moves1, moves2])
            # next_processed_state = preprocess_state(next_state)
            agent1.remember(players, grid, action1, reward1, players1,grid1, done)
            memory_count += 1
            # agent2.remember(players2, grid2, action2, reward2, next_players2,next_grid2, done)
            # agent1.remember(players, grid1, action1, reward1, next_players,next_grid1, done)
            # agent2.remember(players[::-1], grid2, action2, reward2, next_players[::-1],next_grid2, done)
            if memory_count >= agent1.len_memory:
                l1 = agent1.replay()
            # l2 = agent2.replay()
                if done:
                # print(reward1,reward2)
                    print(l1)
                    print(reward1)
                    ll1.append(l1)
                # ll2.append(l2)
        # if e % 5 == 0:
        #     agent1.update_target()
        # agent2.update_target()
    
    plot_loss(ll1,ll2,False)

    return agent1

def plot_q(golds1,golds2):
    plt.figure(1) 
    plt.clf()
    plt.xlabel('Frame')
    plt.ylabel('Gold')
    plt.plot(range(1,len(golds1)+1),golds1,label = '1')
    plt.plot(range(1,len(golds2)+1),golds2,label = '2')
    plt.legend()
    # Take 100 episode averages and plot them too
    plt.savefig('gold.png')
    plt.show()

def main():
    # 检查是否有可用的 GPU
    print("GPU available:", torch.cuda.is_available())
    # 查看当前默认设备（第一个 GPU 或 CPU）
    print("Current device:", torch.cuda.current_device())
    # 查看 GPU 名称（如果有）
    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))
    p1 = train()
    torch.save(p1.model.state_dict(), 'model_1.pth')
    # torch.save(p2.model.state_dict(), 'model_2.pth')
    rp = RandomPlayer()
    # p1 = Player()
    # p2 = Player()
    # p1.model.load_state_dict(torch.load('model_1.pth'))
    # p2.model.load_state_dict(torch.load('model_2.pth'))
    ps = [p1,p1]
    for seed in range(100,101):
        golds = []
        for p in ps:
            g = []
            game = Game(p, rp, seed)
            for _ in range(900):
                players,grid,rg,done = game._single_round()
                g.append(players[0].gold)
            golds.append(g)
        plot_q(golds[0],golds[1])
        # game = Game(p1, rp, seed)
        # print(game.play(max_rounds=900))#raw
    #    plist1.append(p)
        # game = Game(p1, rp, seed)
        # print(game.play(max_rounds=900))#raw
        # plist.append(p)
    # print(np.average(plist))
    
if __name__ == '__main__':
    main()

