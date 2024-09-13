import gymnasium as gym
from gymnasium import spaces
import numpy as np


class TrafficIntersectionEnv(gym.Env):
    def __init__(self):
        super(TrafficIntersectionEnv, self).__init__()

        # переключение светофоров (0: NS зелёный, EW красный, 1: NS красный, EW зелёный)
        self.action_space = spaces.Discrete(2)

        # Состояния: количество машин на каждом направлении (NS, EW)
        self.observation_space = spaces.Box(low=0, high=100, shape=(2,), dtype=np.int32)

        # Изначальное количество машин
        self.cars_ns = 0
        self.cars_ew = 0

        # Состояние светофора (True - NS зелёный, False - EW зелёный)
        self.light_ns_green = True

        # Время для моделирования
        self.time_step = 0
        self.max_time_steps = 1000

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        np.random.seed(seed)

        # Инициализация количества машин
        self.cars_ns = np.random.randint(0, 20)
        self.cars_ew = np.random.randint(0, 20)
        self.light_ns_green = True
        self.time_step = 0

        return np.array([self.cars_ns, self.cars_ew], dtype=np.int32), {}

    def step(self, action):
        if action == 0:
            self.light_ns_green = True
        else:
            self.light_ns_green = False

        # Уменьшаем количество машин в зависимости от состояния светофора
        if self.light_ns_green:
            cars_passed = min(self.cars_ns, 5)
            self.cars_ns -= cars_passed
        else:
            cars_passed = min(self.cars_ew, 5)
            self.cars_ew -= cars_passed

        # Добавляем случайное количество новых машин
        self.cars_ns += np.random.randint(0, 20)
        self.cars_ew += np.random.randint(0, 20)

        # Награда: минимизируем количество машин на перекрестке
        reward = - (self.cars_ns + self.cars_ew)

        # Переход к следующему шагу
        self.time_step += 1
        terminated = self.time_step >= self.max_time_steps  # Эпизод завершен после определенного количества шагов
        truncated = False

        return np.array([self.cars_ns, self.cars_ew], dtype=np.int32), reward, terminated, truncated, {}

    def render(self, mode='human'):
        print(f"NS cars: {self.cars_ns}, EW cars: {self.cars_ew}, NS light green: {self.light_ns_green}")
