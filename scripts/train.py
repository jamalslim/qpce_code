"""
scripts/train.py -- THE training script.  There is one, and this is it.

    python scripts/train.py                          # defaults: L=4
    python scripts/train.py --blocks 6 --maxiter 800 # deeper, longer
    python scripts/train.py --walls fixed --dither   # the published Eq. 3

One ansatz  : the published interferometer with the germ re-uploaded per
              block (qpce.quantum_features).
One loss    : the unbiased energy distance on the generated images
              (qpce.energy).  No weights, no bandwidth, no schedule.
One readout : Y_k = m_k + s_k <Z_k>.  No copula, no PIT, no recalibration.

Everything reported afterwards -- per-pixel W1 and chi2, rank correlations,
the tail ladder, the strip test, the effect size and its null controls -- is
computed on fresh germs against the held-out test set and is never
optimised.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
import numpy as np

from qpce import (QPCEConfig, QPCE, load_dataset, make_train_test,
                    COUPLER_GRAPHS)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="run")
    ap.add_argument("--data", default=None, help="path to the .npy dataset")
    ap.add_argument("--qubits", type=int, default=8)
    ap.add_argument("--loss", default="energy",
                    choices=["energy", "mmd", "mmd3", "mmd1"],
                    help="kernel for the ONE loss. 'energy' IS MMD "
                         "with the distance kernel (E = 2 MMD^2, "
                         "verified 1.9e-14), no bandwidth; 'mmd3' is "
                         "the measured-better trimmed Gaussian "
                         "mixture {1/2,1,2} x sigma_median")
    ap.add_argument("--coupler-graph", default="ring_skip2",
                    choices=sorted(COUPLER_GRAPHS),
                    help="coupler topology. 'ring_skip2' is the published "
                         "C8(1,2): 16 edges, chi'=4, but NOT a subgraph of any "
                         "IBM device, so it must be routed -- measured 24.5 "
                         "two-qubit layers per block instead of 4. The others "
                         "are hardware-native and compile with ZERO SWAPs: "
                         "grid2x4/ladder (10 edges, chi'=3, square lattice), "
                         "ring8 (8, chi'=2, square), path8 (7, chi'=2, "
                         "heavy-hex/Heron).")
    ap.add_argument("--blocks", type=int, default=4,
                    help="L: interferometer blocks AND chaos order per germ")
    ap.add_argument("--walls", default="trainable",
                    choices=["fixed", "trainable"],
                    help="'fixed' = published Ry(pi/2), preserves the IQP form; "
                         "measured cost of fixing them, together with --dither: "
                         "mean |rho_S| 0.363 instead of 0.601")
    ap.add_argument("--shared-germ", type=int, default=0, metavar="K",
                    help="add K latents each fed to every qubit with a trainable "
                         "per-qubit amplitude. Costs n single-qubit rotations "
                         "per block and ZERO two-qubit gates. The CLIC data's "
                         "rank-correlation matrix has one eigenvalue carrying "
                         "64 %% of the dependence (a global shower-depth "
                         "factor); without this wire every latent is private "
                         "and all correlation must come from the coupler "
                         "graph, so it decays with graph distance.")
    ap.add_argument("--dither", action="store_true",
                    help="restore the published final private dither and its "
                         "|w| >= w_min floor (ablation: conditionally "
                         "independent per-pixel noise, forces lambda_L = 0)")
    ap.add_argument("--batch", type=int, default=1500)
    ap.add_argument("--data-batch", type=int, default=1500)
    ap.add_argument("--validate-every", type=int, default=25,
                    help="evaluate the loss on FRESH germs every N iterations. "
                         "The training curve uses a FROZEN germ batch (common "
                         "random numbers) and keeps falling long after the "
                         "model stops improving, because L-BFGS ends up "
                         "minimising that batch's sampling noise. The "
                         "fresh-germ number is the one that measures the model.")
    ap.add_argument("--patience", type=int, default=8,
                    help="stop after this many validations with no fresh-germ "
                         "improvement (0 disables early stopping)")
    ap.add_argument("--n-valid", type=int, default=None,
                    help="fresh germs per validation (default: --batch)")
    ap.add_argument("--maxiter", type=int, default=600)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--n-eval", type=int, default=20000)
    ap.add_argument("--no-eval", action="store_true")
    args = ap.parse_args()

    cfg = QPCEConfig(output_dir=args.out, n_qubits=args.qubits,
                      n_blocks=args.blocks, walls=args.walls, loss=args.loss,
                      coupler_graph=args.coupler_graph,
                      shared_germ=args.shared_germ,
                      dither=args.dither, batch=args.batch,
                      data_batch=args.data_batch, maxiter=args.maxiter,
                      seed=args.seed, n_eval=args.n_eval)
    if args.data:
        cfg.data_path = args.data
    out = cfg.output_path()

    X_all, src = load_dataset(cfg.data_path, cfg.n_total_samples, cfg.seed)
    d = make_train_test(X_all, cfg.test_size, cfg.seed)
    Y_train, Y_test = d["Y_train"], d["Y_test"]
    print(f"dataset {src} {X_all.shape} | train {Y_train.shape} "
          f"test {Y_test.shape}")
    print(f"model   {json.dumps(cfg.summary())}")
    print(f"coupler {cfg.coupler_graph}: {len(cfg.edges)} edges, "
          f"chi' = {cfg.chromatic_index}  ->  "
          f"{cfg.n_blocks * cfg.chromatic_index} two-qubit layers if native "
          f"(zero SWAPs), {cfg.n_blocks * len(cfg.edges)} RZZ pulses")

    model = QPCE(cfg).fit(Y_train, verbose=True,
                           validate_every=args.validate_every,
                           patience=(args.patience if args.patience > 0 else 10**9),
                           early_stop=args.patience > 0,
                           n_valid=args.n_valid)
    model.save(out / "params.npz")

    report = {"dataset": src, "config": cfg.summary(),
              "parameters": model.n_parameters(),
              "training": model.train_report}

    if not args.no_eval:
        ev, Y_gen = model.evaluate(Y_test)
        # floor from the larger training split; model scored on HELD-OUT test
        cert = model.certify(Y_train, Y_gen, verbose=True, Y_score=Y_test)
        report["evaluation"] = ev
        report["certification"] = cert
        np.save(out / "generated.npy", Y_gen[:5000])
        rc = ev["rank_correlation"]
        print(f"\n  energy distance vs test   {ev['energy_distance_vs_test']:.5f}")
        print(f"  permutation p (joint GoF) {ev['permutation_p_value']:.3f}")
        print(f"  W1 per pixel              {ev['W1_per_pixel_mean']:.5f}")
        print(f"  chi2/ndf mean             {ev['chi2_ndf_mean']:.2f}")
        print(f"  mean |rho_S|  {rc['mean_abs_rho_model']:.4f}   "
              f"(data {rc['mean_abs_rho_data']:.4f})")

    with open(out / "report.json", "w") as f:
        json.dump(report, f, indent=2, default=float)
    print(f"\nwrote {out}/params.npz, {out}/report.json")


if __name__ == "__main__":
    main()
