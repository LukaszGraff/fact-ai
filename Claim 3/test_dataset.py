import pickle
from pathlib import Path

# Quick validation
filepath = Path("data_terminate/RandomMOMDP-v1/RandomMOMDP-v1_50000_expert_uniform_49.pkl")

with open(filepath, "rb") as f:
    data = pickle.load(f)

print(f"✅ {len(data)} trajectories loaded")
print(f"Trajectory #0 length: {len(data[0]['observations'])} steps")
print(f"Shapes: obs={data[0]['observations'].shape}, acts={data[0]['actions'].shape}, rews={data[0]['raw_rewards'].shape}")
print(f"Objectives: {data[0]['raw_rewards'].shape[1]} ✓")
print(f"Terminals: {data[0]['dones'].any()} ✓")

print("\n🎯 READY FOR TRAINING!")

import pickle
from pathlib import Path
import numpy as np


def print_trajectory_details():
    filepath = Path("data_terminate/RandomMOMDP-v1/RandomMOMDP-v1_50000_expert_uniform_49.pkl")

    with open(filepath, "rb") as f:
        trajectories = pickle.load(f)

    print("🔍 PRINTING FIRST 3 TRAJECTORIES\n" + "=" * 80)

    for traj_idx, traj in enumerate(trajectories[:3]):
        print(f"\n📊 TRAJECTORY #{traj_idx} (length: {len(traj['observations'])} steps)")
        print("-" * 50)

        obs = np.argmax(traj['observations'], axis=1)
        acts = traj['actions'][:, 0].astype(int)  # (T,) actions 0-3
        rews = traj['raw_rewards']  # (T, 3) reward vectors
        dones = traj['dones']  # (T,) terminals
        obs_int = np.argmax(traj['observations'], axis=1)
        print(obs_int[:20])

        # Print first 20 steps + terminal
        for t in range(min(20, len(obs))):
            state = obs[t]
            action = acts[t]
            r1, r2, r3 = rews[t]
            is_terminal = dones[t]

            marker = "🏁 TERMINAL" if is_terminal else f"step {t}"
            print(f"  {marker:>8}: S={state:2d} → A={action} → R=[{r1:.1f},{r2:.1f},{r3:.1f}]")

        # Show if trajectory ended early
        if len(obs) <= 20:
            print(f"  ✅ Episode ended naturally (goal reached)")
        else:
            print(f"  ⏱️  Hit horizon limit (continues...)")

        # Summary stats
        total_r1 = rews[:, 0].sum()
        total_r2 = rews[:, 1].sum()
        total_r3 = rews[:, 2].sum()
        print(f"  📈 Totals: Obj1={total_r1:.1f}, Obj2={total_r2:.1f}, Obj3={total_r3:.1f}")
        print()


if __name__ == "__main__":
    print_trajectory_details()

import pickle
from pathlib import Path
import numpy as np

# Load your dataset
filepath = Path("data_terminate/RandomMOMDP-v1/RandomMOMDP-v1_50000_expert_uniform_49.pkl")
with open(filepath, "rb") as f:
    trajectories = pickle.load(f)

# Get ALL trajectory lengths
lengths = [len(traj['observations']) for traj in trajectories]

print("🎯 TRAJECTORY LENGTH ANALYSIS")
print("=" * 50)
print(f"Total trajectories: {len(trajectories)}")
print(f"Total steps: {sum(lengths):,}")
print(f"Average length: {np.mean(lengths):.1f}")
print()
print(f"MIN length: {min(lengths)} steps")
print(f"MAX length: {max(lengths)} steps")
print()
print("FIRST 10 trajectory lengths:")
for i, length in enumerate(lengths[:10]):
    print(f"  Traj #{i}: {length:3d} steps")
print()
print("Are ALL 100 steps?", all(l == 100 for l in lengths))
print("Any < 100 steps?", any(l < 100 for l in lengths))
print("Any terminals before 100?", any(len(t['observations']) < 100 and t['dones'][-1] for t in trajectories))
