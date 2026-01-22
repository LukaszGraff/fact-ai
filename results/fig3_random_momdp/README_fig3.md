Reproduce Figure 3 (RandomMOMDP)

Run (two-step):
python -m scripts.train_fairdice_mu_star --seeds 0,1,2,3,4
python -m scripts.run_fairdice_fixed_perturbation --seeds 0,1,2,3,4

Outputs:
- results/fig3_random_momdp/fig3_random_momdp.png
- results/fig3_random_momdp/fig3_random_momdp_results.npz
- results/fig3_random_momdp/fig3_random_momdp_meta.json
