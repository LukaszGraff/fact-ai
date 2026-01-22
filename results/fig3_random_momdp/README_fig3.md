Reproduce Figure 3 (RandomMOMDP)

Run (two-step):
python -m scripts.train_fairdice_mu_star --seeds 0,1,2,3,4
python -m scripts.run_fairdice_fixed_perturbation --seeds 0,1,2,3,4

Outputs:
- results/fig3_random_momdp/fig3_random_momdp.png
- results/fig3_random_momdp/fig3_random_momdp_results.npz
- results/fig3_random_momdp/fig3_random_momdp_meta.json

CLI flags

1) Dataset generation:
python -m scripts.generate_random_momdp_dataset [flags]

Flags:
- --seed (int) default 0
- --num_traj (int) default 100
- --max_steps (int) default 200
- --optimality (float) default 0.5
- --out_dir (str) default ./data/random_momdp/

2) Train mu* (FairDICE-NSW):
python -m scripts.train_fairdice_mu_star [flags]

Flags:
- --seeds (str) default "0"
- --betas (str) default "0.001,0.01,0.1,1.0"
- --train_steps (int) default 100000
- --batch_size (int) default 256
- --hidden_dim (int) default 768
- --num_layers (int) default 3
- --divergence (str) default "SOFT_CHI"
- --max_steps (int) default 200
- --out_dir (str) default "./results/fig3_random_momdp/"
- --log_interval (int) default 10000
- --eval_interval (int) default 200
- --eval_episodes (int) default 50
- --debug_mu (flag) default off
- --mu_init_noise_std (float) default 0.0

3) Fixed-mu perturbations + plot:
python -m scripts.run_fairdice_fixed_perturbation [flags]

Flags:
- --seeds (str) default "0"
- --betas (str) default "0.001,0.01,0.1,1.0"
- --sigmas (str) default "0.001,0.01,0.1,1.0"
- --train_steps (int) default 100000
- --batch_size (int) default 256
- --hidden_dim (int) default 768
- --num_layers (int) default 3
- --divergence (str) default "SOFT_CHI"
- --max_steps (int) default 200
- --max_eval_steps (int) default 200
- --eval_episodes (int) default 50
- --out_dir (str) default "./results/fig3_random_momdp/"
- --mu_path (str) default "./results/fig3_random_momdp/mu_star.npz"
- --log_interval (int) default 10000
- --eval_interval (int) default 200
