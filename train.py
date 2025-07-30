import push_env  # Ensure this is available
import time
import gymnasium as gym
# TODO: Try PPO as well as other RL algorithms from SB3.
from stable_baselines3 import PPO, SAC, TD3
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.noise import NormalActionNoise
import numpy as np

# Register the environment
gym.register_envs(push_env)


ALGO = "TD3"  # Change to "SAC" or "TD3" or "PPO"
# Define environment ID
# ENV_ID = 'FetchPushDense-v0'
# Choose the reward function: "Dense", "Manhattan", "Time_penalty", or "Custom"
REWARD_TYPE = "Custom"  
ENV_ID = f'FetchPush{REWARD_TYPE.capitalize()}-v0'


# Create and wrap the environment
def make_env():
    # env = gym.make(ENV_ID, render_mode="human")
    env = gym.make(ENV_ID)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    return env

# Use DummyVecEnv for compatibility with SB3
vec_env = DummyVecEnv([make_env])
vec_env = VecMonitor(vec_env)


n_actions = vec_env.action_space.shape[0]
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))
# --------------------------
# Model setup based on ALGO
# --------------------------
if ALGO == "PPO":
    model = PPO(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        n_steps=2048,
        batch_size=64,
        gae_lambda=0.95,
        gamma=0.99,
        n_epochs=10,
        learning_rate=3e-4,
        clip_range=0.2,
        tensorboard_log=f"./logs/{ALGO}_{REWARD_TYPE}/"
    )

elif ALGO == "SAC":
    model = SAC(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        batch_size=256,
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        train_freq=1,
        gradient_steps=1,
        tensorboard_log=f"./logs/{ALGO}_{REWARD_TYPE}/"
    )

elif ALGO == "TD3":
    model = TD3(
        policy="MultiInputPolicy",
        env=vec_env,
        verbose=1,
        action_noise=action_noise,
        # batch_size=100,
        batch_size=128,
        buffer_size=500_000, 
        # learning_rate=1e-3,
        learning_rate=0.0004,
        # gamma=0.99,
        gamma=0.98,
        tau=0.003,
        train_freq=(1, "episode"),
        gradient_steps=1,
        # tensorboard_log="./td3_custom_tensorboard/"
        tensorboard_log=f"./logs/{ALGO}_{REWARD_TYPE}/"

    )

else:
    raise ValueError(f"Unsupported algorithm: {ALGO}")

# Train the model
model.learn(total_timesteps=500_000 , tb_log_name="TD3_Custom_Exp2")  # Adjust as needed.

# Save the trained model
model.save(f"{ALGO.lower()}_{REWARD_TYPE}_agent")

# Evaluate the trained model
print("\nEvaluating trained agent...")
env = gym.make(ENV_ID, render_mode='human')
obs, info = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = env.step(action)
    time.sleep(0.1)  # Slow it down a bit for visualization
    if terminated or truncated:
        break

env.close()
