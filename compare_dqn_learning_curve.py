import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from sb3_contrib import QRDQN
from rocket_env import SimpleRocketEnv
import pandas as pd

variants = {
    "DQN": DQN,                 # DQN standar
    "Double DQN": DQN,          # SB3 sudah mengandung mekanisme Double DQN
    "Dueling DQN": QRDQN        # Dueling DQN dari sb3_contrib
}

learning_curves = {}
results = []
timesteps = 15000

for name, model_class in variants.items():
    print(f"\n=== Training {name} ===")
    env = SimpleRocketEnv(render_mode=None)

    model = model_class(
        "MlpPolicy", env,
        verbose=0,
        learning_rate=1e-3,
        buffer_size=10000,
        batch_size=64,
        learning_starts=1000,
        gamma=0.99,
        tau=1.0
    )

    # Simpan reward per episode selama training
    episode_rewards = []
    obs, _ = env.reset()
    ep_reward = 0
    for step in range(timesteps):
        action, _ = model.predict(obs, deterministic=False)
        obs, reward, term, trunc, _ = env.step(action)
        ep_reward += reward
        model.learn(total_timesteps=1, reset_num_timesteps=False)
        if term or trunc:
            episode_rewards.append(ep_reward)
            ep_reward = 0
            obs, _ = env.reset()

    learning_curves[name] = episode_rewards

    # Evaluasi akhir setelah training
    total_rewards = []
    for ep in range(10):
        obs, _ = env.reset()
        total_r = 0
        while True:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, term, trunc, _ = env.step(action)
            total_r += reward
            if term or trunc:
                break
        total_rewards.append(total_r)

    env.close()
    results.append({
        "Varian": name,
        "Rata-rata Reward": np.mean(total_rewards),
        "Reward Maksimum": np.max(total_rewards),
        "Reward Minimum": np.min(total_rewards),
        "Standar Deviasi": np.std(total_rewards)
    })
    print(f"{name}: Mean={np.mean(total_rewards):.2f}, Max={np.max(total_rewards):.2f}")

# === Plot Learning Curves ===
plt.figure(figsize=(8,5))
for name, rewards in learning_curves.items():
    plt.plot(rewards, label=name, linewidth=1.5)
plt.xlabel("Episode")
plt.ylabel("Total Reward")
plt.title("Learning Curves - DQN vs Double DQN vs Dueling DQN")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("learning_curves_dqn_variants.png", dpi=200)
plt.show()

# === Ringkasan Statistik ===
df = pd.DataFrame(results)
print("\n=== Statistik Hasil Eksperimen ===")
print(df.to_string(index=False))
df.to_csv("compare_dqn_learning_curve_results.csv", index=False)
print("\nFile disimpan sebagai learning_curves_dqn_variants.png dan compare_dqn_learning_curve_results.csv")
