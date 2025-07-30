import time
import gymnasium as gym
import push_env 
from stable_baselines3 import TD3, SAC, PPO


gym.register_envs(push_env)

ALGO_NAME = "TD3"  # TD3 or "PPO", "SAC"
ALGO_MAP = {
    "PPO": PPO,
    "TD3": TD3,
    "SAC": SAC,
}
REWARD_TYPE = "Manhattan"  # Change to "Dense", "Manhattan", "Time_penalty", or "Custom"
Agent_ID = "TD3_Manhattan_agent"  # Change to the appropriate agent ID
ENV_ID = f'FetchPush{REWARD_TYPE.capitalize()}-v0'


env = gym.make(ENV_ID, render_mode='human')


model = ALGO_MAP[ALGO_NAME].load(f"{Agent_ID}")  


obs, info = env.reset()

while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)

  
    # print(f"Action: {action}, Reward: {reward:.2f}, Success: {info.get('is_success')}")
    print(f"Reward: {reward:.2f}, Success: {info.get('is_success')}")
   
    if info.get('is_success') == 1.0:
        print("\n Task completed successfully! Stopping inference.")
        break

    
    if terminated or truncated:
        print("\n Episode ended without success. Resetting...")
        obs, info = env.reset()

    time.sleep(0.1)

env.close()
# import time
# import gymnasium as gym
# from gymnasium.wrappers import RecordVideo
# import push_env
# from stable_baselines3 import TD3, SAC, PPO

# # Register custom envs
# gym.register_envs(push_env)

# # === Configuration ===
# ALGO_NAME = "TD3"  # Options: "PPO", "TD3", "SAC"
# REWARD_TYPE = "Custom"  # Options: "Dense", "Manhattan", "Time_penalty", "Custom"
# Agent_ID = "TD3_Custom_agent"

# # Video output folder — use raw string to avoid \U issues on Windows
# video_folder = r"C:\Users\Sara Shaban\Downloads\internship_rl_project_starter_code\internship_rl_project\Videos"

# # === Load model and environment ===
# ENV_ID = f'FetchPush{REWARD_TYPE.capitalize()}-v0'
# env = gym.make(ENV_ID, render_mode="rgb_array")
# env = RecordVideo(env, video_folder=video_folder, episode_trigger=lambda e: True)

# # Load the trained model
# ALGO_MAP = {"PPO": PPO, "TD3": TD3, "SAC": SAC}
# model = ALGO_MAP[ALGO_NAME].load(f"{Agent_ID}")

# # === Run and record inference ===
# obs, info = env.reset()
# while True:
#     action, _ = model.predict(obs, deterministic=True)
#     obs, reward, terminated, truncated, info = env.step(action)

#     print(f"Reward: {reward:.2f}, Success: {info.get('is_success')}")

#     if info.get('is_success') == 1.0:
#         print("\n Task completed successfully! Stopping inference.")
#         break

#     if terminated or truncated:
#         print("\n Episode ended without success. Resetting...")
#         obs, info = env.reset()

#     time.sleep(0.1)

# env.close()
# print(f"\n Video saved in: {video_folder}")
