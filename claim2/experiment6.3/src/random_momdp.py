import numpy as np

class RandomMOMDP:
    def __init__(self, seed: int, n_states=50, n_actions=4, n_obj=3, gamma=0.95, horizon=200):
        self.rng = np.random.default_rng(seed)
        self.n_states = n_states
        self.n_actions = n_actions
        self.n_obj = n_obj
        self.gamma = gamma
        self.horizon = horizon

        self.init_state = 0

        print(
            f"[random_momdp] Init with seed={seed}, states={n_states}, actions={n_actions},"
            f" objectives={n_obj}, gamma={gamma}, horizon={horizon}"
        )

        # pick 3 goal states among 1..49
        self.goal_states = self.rng.choice(np.arange(1, n_states), size=n_obj, replace=False)
        print(f"[random_momdp] Goal states: {self.goal_states.tolist()}")

        # transitions: for each (s,a), pick 4 next states and probs
        self.next_state_support = np.zeros((n_states, n_actions, 4), dtype=np.int64)
        self.next_state_prob = np.zeros((n_states, n_actions, 4), dtype=np.float64)

        for s in range(n_states):
            for a in range(n_actions):
                ns = self.rng.choice(n_states, size=4, replace=False)
                p = self.rng.dirichlet(np.ones(4))
                self.next_state_support[s, a] = ns
                self.next_state_prob[s, a] = p

        # reward is one-hot only when landing in a goal state (episodic-style sparse rewards)
        self.reward_on_state = np.zeros((n_states, n_obj), dtype=np.float64)
        for i, gs in enumerate(self.goal_states):
            self.reward_on_state[gs, i] = 1.0
        print("[random_momdp] Environment tensors initialized")

    def step(self, s: int, a: int):
        ns_support = self.next_state_support[s, a]
        p = self.next_state_prob[s, a]
        ns = self.rng.choice(ns_support, p=p)
        r = self.reward_on_state[ns].copy()
        done = False
        return ns, r, done

    def reset(self):
        return self.init_state
