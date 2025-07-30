from gymnasium.envs.registration import register

def register_robotics_envs():
    """Register all environment ID's to Gymnasium."""

    # for reward_type in ["sparse", "dense"]:
    #     suffix = "Dense" if reward_type == "dense" else ""
    #     kwargs = {
    #         "reward_type": reward_type,
    #     }
    
    reward_types = ["sparse", "dense", "manhattan", "time_penalty", "custom"]

    for reward_type in reward_types:
        suffix = reward_type.capitalize() if reward_type != "sparse" else ""
        kwargs = {
            "reward_type": reward_type,
        }

        register(
            id=f"FetchPush{suffix}-v0",
            entry_point="push_env.envs.push:MujocoFetchPushEnv",
            kwargs=kwargs,
            max_episode_steps=50,
        )

register_robotics_envs()
