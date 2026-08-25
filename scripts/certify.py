"""
scripts/certify.py -- certification with its null controls.

    python scripts/certify.py --load run/params.npz

Prints the model's effect size on the same table as (a) a separable
product-state device, (b) the model under monotone per-pixel distortion, and
(c) a classical Gaussian copula.  If the model is not clearly separated from
all three, the protocol has not certified what it appears to certify.
"""
import sys, os, json, argparse
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))

from qpce import QPCEConfig, QPCE, load_dataset, make_train_test


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--load", default="run/params.npz")
    ap.add_argument("--data", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-gen", type=int, default=20000)
    args = ap.parse_args()

    cfg = QPCEConfig()
    if args.data:
        cfg.data_path = args.data
    X_all, _ = load_dataset(cfg.data_path, cfg.n_total_samples, cfg.seed)
    d = make_train_test(X_all, cfg.test_size, cfg.seed)

    model = QPCE.load(args.load, cfg)
    Y_gen = model.generate(args.n_gen)
    full = model.certify(d["Y_train"], Y_gen, verbose=False)

    m, c = full["model"], full["controls"]
    print(f"\nfloor F = {full['floor']['F']:.4e} +- {full['floor']['sd']:.1e}"
          f"   ({full['floor']['estimator']})\n")
    print(f"{'':40s} {'D':>9s}")
    print(f"{'model':40s} {m['D']:9.4f}")
    print(f"{'CONTROL  product-state (separable)':40s} "
          f"{c['product_state_surrogate']['D']:9.4f}")
    print(f"{'CONTROL  Gaussian copula':40s} "
          f"{c['classical_gaussian_copula']['D']:9.4f}")
    print(f"\n{'CONTROL  monotone distortion of <Z>':40s}"
          f"  (max rank-correlation deviation)")
    for k, v in c["monotone_depolarising"].items():
        print(f"{'   ' + k:40s} {v['max_rank_corr_deviation']:9.2e}")
    print(f"\nstrip test (couplers off): "
          f"{full['strip_test_max_offdiag_rank_corr']:.4f}")
    print(f"\n{full['interpretation']}")

    path = args.out or os.path.join(os.path.dirname(args.load),
                                    "certification.json")
    json.dump(full, open(path, "w"), indent=2, default=float)
    print(f"\nwrote {path}")


if __name__ == "__main__":
    main()
