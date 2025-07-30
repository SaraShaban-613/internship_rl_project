import push_env  # Ensure this module is in your PYTHONPATH or current directory
import gymnasium as gym
import time

# Register custom environments from push_env
gym.register_envs(push_env)

# Create the environment
env = gym.make('FetchPushDense-v0', render_mode='human')

# Print the action and observation space
print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

# Reset the environment
obs, info = env.reset()
print("\nInitial observation:")
print(obs)

# Run a few random actions
for step in range(10):
    print(f"\nStep {step + 1}")
    
    # Sample a random action
    action = env.action_space.sample()
    print("Random action:", action)
    
    # Apply the action
    obs, reward, terminated, truncated, info = env.step(action)
    
    # Print the new observation and reward
    print("Observation:", obs)
    print("Reward:", reward)
    
    # Render the environment (requires render_mode='human' during make)
    time.sleep(0.5)  # Slow it down a bit for visualization

    if terminated or truncated:
        print("Episode finished, resetting environment.")
        obs, info = env.reset()

# Close the environment
env.close()
