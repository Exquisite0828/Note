```python
"""
Env 2D
@author: huiming zhou
"""


class Env:
    def __init__(self):
        self.x_range = 51  # size of background
        self.y_range = 31
        self.motions = [(-1, 0), (-1, 1), (0, 1), (1, 1),
                        (1, 0), (1, -1), (0, -1), (-1, -1)]
        self.obs = self.obs_map()

    def update_obs(self, obs):
        self.obs = obs

    def obs_map(self):
        """
        Initialize obstacles' positions
        :return: map of obstacles
        """

        x = self.x_range
        y = self.y_range
        obs = set()

        for i in range(x):
            obs.add((i, 0)) #下边界：（0,0）到（50,0）

        for i in range(x):
            obs.add((i, y - 1)) #上边界：（0,30）到（50,30）

        for i in range(y):
            obs.add((0, i)) #左边界：（0,0）到（0,30）
        for i in range(y):
            obs.add((x - 1, i)) #右边界：（50,0）（50,30）

        for i in range(10, 21):
            obs.add((i, 15))     #水平障碍物1：从（10，15）到（20，15）的水平线

        for i in range(15):
            obs.add((20, i))     #垂直障碍物1：（20,0）到（20，14）的垂线

        for i in range(15, 30):
            obs.add((30, i))     #垂直障碍物2：（30,15）到（30,29）的垂线

        for i in range(16):
            obs.add((40, i))    #从(40,0)到(40,15)的垂直线

        return obs
```