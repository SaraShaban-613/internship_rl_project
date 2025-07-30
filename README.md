| Algorithm | Reward Type   | ep\_rew\_mean | success\_rate | actor\_loss | critic\_loss | learning\_rate |
| --------- | ------------- | ------------- | ------------- | ----------- | ------------ | -------------- |
| PPO       | Custom        | 2.63          | 0.03          | N/A         | N/A          | 0.0003         |
| SAC       | Custom        | -4.02         | 0.06          | -4.44       | 9.83         | 0.0003         |
| TD3       | Custom        | 8.96          | 0.13          | -0.922      | 1.13         | 0.001          |
| TD3       | Dense         | -9.13         | 0.06          | 1.1         | 0.00305      | 0.001          |
| SAC       | Dense         | -12.1         | 0.06          | -13.8       | 0.648        | 0.0003         |
| TD3       | Manhattan     | -11.2         | 0.09          | 1.38        | 0.00982      | 0.001          |
| SAC       | Manhattan     | -13.0         | 0.02          | -32.1       | 28.9         | 0.0003         |
| TD3       | Time\_penalty | -9.01         | 0.11          | 1.04        | 0.00197      | 0.001          |
| SAC       | Time\_penalty | -9.26         | 0.09          | 12.0        | 0.557        | 0.0003         |

| Algorithm   | Reward       |   ep_rew_mean |   success_rate |   value_loss |   actor_or_policy_loss |
|:------------|:-------------|--------------:|---------------:|-------------:|-----------------------:|
| PPO         | Custom       |          2.63 |           0.03 |      2.61    |              -0.00333  |
| SAC         | Custom       |         -4.02 |           0.06 |      9.83    |              -4.44     |
| TD3         | Custom       |          8.96 |           0.13 |      1.13    |              -0.922    |
| PPO         | Dense        |         -7.98 |           0.09 |      0.455   |              -0.00173  |
| SAC         | Dense        |        -12.1  |           0.06 |      0.648   |             -13.8      |
| TD3         | Dense        |         -9.13 |           0.06 |      0.00305 |               1.1      |
| PPO         | Manhattan    |        -10.9  |           0.06 |      0.675   |              -0.00516  |
| SAC         | Manhattan    |        -13    |           0.02 |     28.9     |             -32.1      |
| TD3         | Manhattan    |        -11.2  |           0.09 |      0.00982 |               1.38     |
| PPO         | Time_penalty |         -9.08 |           0.06 |      0.487   |              -0.000769 |
| SAC         | Time_penalty |         -9.26 |           0.09 |      0.557   |              12        |
| TD3         | Time_penalty |         -9.01 |           0.11 |      0.00197 |               1.04     |



| Algorithm | Reward Type   | Success Rate  | Episode Reward | Actor Loss  | Critic Loss  | Time (s) |
|-----------|---------------|---------------|----------------|-------------|--------------|----------|
| PPO       | Custom        | 0.03          | 2.63           | —           | —            | 2874     |
| TD3       | Custom        | 0.13          | 8.96           | -0.92       | 1.13         | 287      |
| SAC       | Custom        | 0.06          | -4.02          | -4.44       | 9.83         | 2874     |
| PPO       | Dense         | 0.09          | -7.98          | —           | —            | 391      |
| TD3       | Dense         | 0.06          | -9.13          | 1.10        | 0.003        | 349      |
| SAC       | Dense         | 0.06          | -12.1          | -13.8       | 0.648        | 2084     |
| PPO       | Manhattan     | 0.06          | -10.9          | —           | —            | 445      |
| TD3       | Manhattan     | 0.09          | -11.2          | 1.38        | 0.0098       | 277      |
| SAC       | Manhattan     | 0.02          | -13.0          | -32.1       | 28.9         | 2372     |
| PPO       | Time_penalty  | 0.06          | -9.08          | —           | —            | 271      |
| TD3       | Time_penalty  | 0.11          | -9.01          | 1.04        | 0.0019       | 276      |
| SAC       | Time_penalty  | 0.09          | -9.26          | 12.0        | 0.557        | 2392     |


![Alt Text](C:\Users\Sara Shaban\Downloads\internship_rl_project_starter_code\internship_rl_project\image.png)

Performance Analysis:
    Best success rate:

        TD3 + Custom → 13%

        TD3 + Time_penalty → 11%

    Worst success rate:

        SAC + Manhattan → 2%

    Best ep_rew_mean:

        TD3 + Custom → 8.96

    Worst value_loss:

        SAC + Manhattan → 28.9 → Poor stability

Edits 
![Alt Text](C:\Users\Sara Shaban\Downloads\internship_rl_project_starter_code\internship_rl_project\TD3+Custom.png)
Experiments Breakdown

1. TD3 + Custom (Base)

    Hyperparameters:

    learning_rate = 0.001

    batch_size = 100

    gamma = 0.99

    buffer_size = 1_000_000

    Results:

    success_rate = 0.13 ✅

    ep_rew_mean = 8.96 ✅

    actor_loss = -0.922

    critic_loss = 1.13

    time_elapsed = 287s

2. TD3 + Custom (Exp 1)

    Changes:

    learning_rate = 0.0003

    Results:

    success_rate = 0.08 ⬇️

    ep_rew_mean = 2.85 ⬇️

    actor_loss = 0.401

    critic_loss = 1.13

3. TD3 + Custom (Exp 2)

    Changes:

    learning_rate = 0.0001

    Results:

    success_rate = 0.05 ⬇️

    ep_rew_mean = -1.83 ⬇️

    actor_loss = 0.382

    critic_loss = 1.25

4. TD3 + Custom (Exp 3)

    Changes:

    learning_rate = 0.001 (same as base, used different batch_size)

    Results:

    success_rate = 0.04 ⬇️

    ep_rew_mean = -4.79 ⬇️

    actor_loss = 0.702

    critic_loss = 1.21

5. TD3 + Custom (Exp 4)

    Changes:

    learning_rate = 0.0002

    Results:

    success_rate = 0.07 ⬇️

    ep_rew_mean = 1.45

    actor_loss = 0.142

    critic_loss = 0.987

6. TD3 + Custom (Exp 5)

    Changes:

    learning_rate = 0.0001

    Results:

    success_rate = 0.09

    ep_rew_mean = 5.58

    actor_loss = 0.257

    critic_loss = 1.69



venv310\Scripts\activate 
cd internship_rl_project 