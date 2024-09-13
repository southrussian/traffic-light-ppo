from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env
from env import TrafficIntersectionEnv

# Проверим, что среда корректно настроена
env = TrafficIntersectionEnv()
check_env(env, warn=True)

model = PPO("MlpPolicy", env, verbose=1)

model.learn(total_timesteps=10000)

obs, _ = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    done = terminated or truncated
    env.render()
    if done:
        obs = env.reset()
