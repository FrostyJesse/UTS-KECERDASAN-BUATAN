import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import DQN
from sb3_contrib import QRDQN  # untuk Dueling dan varian lain
from rocket_env import SimpleRocketEnv
import pandas as pd

variants = {
    "DQN": DQN,                 # DQN standar
    "Double DQN": DQN,          # SB3 sudah double DQN secara default
    "Dueling DQN": QRDQN        # gunakan QRDQN dari sb3_contrib
}

results = {}
details = []

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
    model.learn(total_timesteps=15000)

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
    results[name] = np.mean(total_rewards)
    details.append({
        "Varian": name,
        "Rata-rata Reward": np.mean(total_rewards),
        "Reward Maksimum": np.max(total_rewards),
        "Reward Minimum": np.min(total_rewards),
        "Standar Deviasi": np.std(total_rewards)
    })
    print(f"{name}: Mean={np.mean(total_rewards):.2f}, Max={np.max(total_rewards):.2f}")

plt.figure(figsize=(7,4))
plt.bar(results.keys(), results.values(), color=['gray','black','dimgray'])
plt.ylabel("Rata-rata Reward")
plt.title("Perbandingan Varian DQN (SB3 & QRDQN)")
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
plt.savefig("compare_dqn_variants_bw.png", dpi=200)
plt.show()

df = pd.DataFrame(details)
print("\n=== Statistik Hasil Eksperimen ===")
print(df.to_string(index=False))
df.to_csv("compare_dqn_results.csv", index=False)
print("\nFile disimpan sebagai compare_dqn_results.csv dan compare_dqn_variants_bw.png")
