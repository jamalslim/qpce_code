#!/usr/bin/env python3
"""Investigate the three advantage ingredients, one by one, staged.

Stages: 1a lightcone | 1b entropy+anticoncentration+noise-window
        2 sampling readout (sim + REAL ibm_fez p_raw) | 3 no-go + feed-forward
Results accumulate in outputs/advantage_investigation.json
"""
import json, os, sys, time
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "src"))
from qpce.quantum_features import InterferometricCircuit

RNG = np.random.default_rng(2026)
T0 = time.time()
STAGE = sys.argv[1] if len(sys.argv) > 1 else 'all'
JPATH = 'outputs/advantage_investigation.json'
OUT = json.load(open(JPATH)) if os.path.exists(JPATH) else {}
def save():
    OUT.setdefault('runtime_s', {})[STAGE] = round(time.time() - T0, 1)
    json.dump(OUT, open(JPATH, 'w'), indent=1)

# ---------------- helpers ----------------
def probs_and_signs(psi, n):
    B = psi.shape[0]
    P = np.abs(psi.reshape(B, -1)) ** 2
    idx = np.arange(2 ** n)
    signs = np.stack([1 - 2 * ((idx >> (n - 1 - q)) & 1) for q in range(n)])
    return P, signs

def zz_matrix(P, signs):
    Zs = P @ signs.T
    ZZ = np.einsum('bk,ik,jk->bij', P, signs, signs, optimize=True)
    return Zs, ZZ

def spearman_meanabs(Y):
    from scipy.stats import spearmanr
    r = spearmanr(Y).statistic
    iu = np.triu_indices(Y.shape[1], 1)
    return float(np.mean(np.abs(r[iu]))), r

def tail_coeffs(Y, q):
    from scipy.stats import rankdata
    N, n = Y.shape
    U = np.column_stack([rankdata(Y[:, i]) / (N + 1) for i in range(n)])
    L, Uu = [], []
    for i in range(n):
        for j in range(i + 1, n):
            L.append(np.mean((U[:, i] < q) & (U[:, j] < q)) / q)
            Uu.append(np.mean((U[:, i] > 1 - q) & (U[:, j] > 1 - q)) / q)
    return float(np.mean(L)), float(np.mean(Uu))

def load_deployed():
    z = np.load('outputs/checkpoint/params.npz', allow_pickle=True)
    edges = [tuple(e) for e in z['edges']]
    m = InterferometricCircuit(8, 6, edges, walls='trainable', dither=False,
                               shared_germ=1)
    for k in ('beta', 'phi', 'theta', 'w'):
        setattr(m, k, z[k].astype(float))
    m.a = z['a'].astype(float)
    return m, z['readout_m'].astype(float), z['readout_s'].astype(float)

n16 = 16
path_edges = [(i, i + 1) for i in range(15)]
grid_edges = []
for r in range(4):
    for c in range(4):
        q = 4 * r + c
        if c < 3: grid_edges.append((q, q + 1))
        if r < 3: grid_edges.append((q, q + 4))

def rand_model(edges, L, seed):
    m = InterferometricCircuit(n16, L, edges, seed=seed, walls='trainable',
                               dither=False, shared_germ=0)
    r = np.random.default_rng(seed)
    m.beta = 0.7 * r.standard_normal((L, n16))
    m.phi = 0.7 * r.standard_normal((L, len(edges)))
    m.theta = 0.7 * r.standard_normal((L, n16))
    return m

def dists(edges):
    import collections
    adj = collections.defaultdict(list)
    for i, j in edges: adj[i].append(j); adj[j].append(i)
    d = {0: 0}; Q = [0]
    while Q:
        u = Q.pop(0)
        for v in adj[u]:
            if v not in d: d[v] = d[u] + 1; Q.append(v)
    return d

# ================= stage 1a: light cone =================
if STAGE in ('1a', 'all'):
    print("== 1a: light cone, path vs grid ==")
    lightcone = {}
    for name, edges in (('path16', path_edges), ('grid4x4', grid_edges)):
        dmap = dists(edges)
        # one probe qubit per distinct distance
        probe = {}
        for q in range(1, n16): probe.setdefault(dmap[q], q)
        rows = {}
        for L in (1, 2, 4):
            m = rand_model(edges, L, 11)
            E = RNG.uniform(0, 1, (24, m.n_wires))
            z0 = m.expectations(E)
            per_d = {}
            for d, q in sorted(probe.items()):
                E2 = E.copy(); E2[:, q] = RNG.uniform(0, 1, 24)
                z1 = m.expectations(E2)
                per_d[int(d)] = float(np.abs(z1[:, 0] - z0[:, 0]).max())
            rows[L] = per_d
            print(f"  {name} L={L}:", {d: f"{v:.1e}" for d, v in per_d.items()})
        lightcone[name] = rows
    OUT['exp1_lightcone_maxDZ0_by_graphdist'] = lightcone
    save()

# ================= stage 1b: entropy + anticoncentration + noise window ====
if STAGE in ('1b', 'all'):
    print("== 1b: entanglement, anticoncentration, noise window ==")
    def halfcut_entropy(m, germs=4):
        E = RNG.uniform(0, 1, (germs, m.n_wires))
        psi = m.statevector(E).reshape(germs, 2 ** 8, 2 ** 8)
        S = []
        for b in range(germs):
            p = np.linalg.svd(psi[b], compute_uv=False) ** 2
            p = p[p > 1e-14]
            S.append(float(-(p * np.log2(p)).sum()))
        return float(np.mean(S))
    ent, anti = {}, {}
    for name, edges in (('path16', path_edges), ('grid4x4', grid_edges)):
        ent[name], anti[name] = {}, {}
        for L in (1, 2, 3, 4, 6):
            m = rand_model(edges, L, 21)
            ent[name][L] = halfcut_entropy(m)
            E = RNG.uniform(0, 1, (2, m.n_wires))
            P, _ = probs_and_signs(m.statevector(E), n16)
            anti[name][L] = float(np.mean((2 ** n16) * (P ** 2).sum(axis=1)))
        print(f"  {name} entropy:", {L: round(v, 2) for L, v in ent[name].items()})
        print(f"  {name} collision(PT=2):", {L: round(v, 2) for L, v in anti[name].items()})
    OUT['exp1_halfcut_entropy_bits'] = ent
    OUT['exp1_collision_ratio_PT2'] = anti
    OUT['exp1_noise_window'] = {}
    for tag, eps in (('measured_1.4e-3', 1.4e-3), ('budget_6.2e-3', 6.2e-3)):
        n_max = (np.log(10.0) / (2 * eps)) ** (2 / 3)
        OUT['exp1_noise_window'][tag] = {
            'n_max_at_depth_sqrt_n_F10pct': float(n_max),
            'depth_at_n_max': float(np.sqrt(n_max))}
    print("  noise window:", OUT['exp1_noise_window'])
    save()

# ================= stage 2: sampling readout =================
if STAGE in ('2', 'all'):
    print("== 2: sampling readout, sim + REAL ibm_fez ==")
    model, ro_m, ro_s = load_deployed()
    B = 12000
    E = RNG.uniform(0, 1, (B, model.n_wires))
    psi = model.statevector(E)
    P, signs = probs_and_signs(psi, 8)
    Zs, ZZ = zz_matrix(P, signs)
    iu = np.triu_indices(8, 1)
    C_quantum = (ZZ - np.einsum('bi,bj->bij', Zs, Zs)).mean(axis=0)
    C_mix = np.cov(Zs.T)
    np.fill_diagonal(C_quantum, 0.0)
    OUT['exp2_sim'] = {
        'mean_abs_intra_shot_connected_ZZ': float(np.abs(C_quantum[iu]).mean()),
        'max_abs_intra_shot_connected_ZZ': float(np.abs(C_quantum[iu]).max()),
        'mean_abs_mixture_cov': float(np.abs(C_mix[iu]).mean())}
    cum = np.cumsum(P, axis=1)
    pick = (RNG.uniform(0, 1, (B, 1)) > cum).sum(axis=1).clip(0, 255)
    z_ss = signs[:, pick].T
    jitter = 1e-9 * RNG.standard_normal((B, 8))
    OUT['exp2_sim']['single_shot_mean_abs_rhoS'] = spearman_meanabs(
        ro_m + ro_s * z_ss + jitter)[0]
    OUT['exp2_sim']['expectation_readout_mean_abs_rhoS'] = spearman_meanabs(
        ro_m + ro_s * Zs)[0]
    print("  sim intra-shot |connZZ| =",
          round(OUT['exp2_sim']['mean_abs_intra_shot_connected_ZZ'], 4),
          "| mixture-cov =", round(OUT['exp2_sim']['mean_abs_mixture_cov'], 4),
          "| ss rho =", round(OUT['exp2_sim']['single_shot_mean_abs_rhoS'], 4))

    OUT['exp2_data_quantization'] = {
        'continuous_mean_abs_rhoS': spearman_meanabs(X)[0], **quant}
    print("  data quantization rho:",
          {k: round(v['mean_abs_rhoS'], 3) for k, v in quant.items()},
          "cont:", round(OUT['exp2_data_quantization']['continuous_mean_abs_rhoS'], 3))
    save()

# ================= stage 3: no-go + feed-forward =================
if STAGE in ('3', 'all'):
    print("== 3: tail no-go on deployed model + feed-forward escape ==")
    model, ro_m, ro_s = load_deployed()
    BIG, CH = 200000, 20000
    qs = [0.05, 0.02, 0.01, 0.005, 0.002]
    delta = np.zeros((6, 8)); delta[-1, :] = 0.9
    ZA_l, ZB_l = [], []
    for k in range(0, BIG, CH):
        E0 = RNG.uniform(0, 1, (CH, model.n_wires))
        ZA_l.append(model.expectations(E0))
        th0 = model.theta.copy(); model.theta = th0 + delta
        ZB_l.append(model.expectations(E0))
        model.theta = th0
        print(f"   batch {k//CH+1}/{BIG//CH} done, {round(time.time()-T0)}s")
    ZA = np.concatenate(ZA_l); ZB = np.concatenate(ZB_l)
    Y_big = ro_m + ro_s * ZA
    nogo = {q: tail_coeffs(Y_big, q) for q in qs}
    OUT['exp3_nogo_deployed'] = {str(q): {'lambda_L': v[0], 'lambda_U': v[1]}
                                 for q, v in nogo.items()}
    print("  deployed lambda_L(q):", {q: round(v[0], 3) for q, v in nogo.items()})
    X = np.load('data/cal_shower_img_8q.npy')[:4000].astype(float)
    nd = {q: tail_coeffs(X[2600:], q) for q in (0.05, 0.02, 0.01)}
    OUT['exp3_data_tails'] = {str(q): {'lambda_L': v[0], 'lambda_U': v[1]}
                              for q, v in nd.items()}
    Bsel = (RNG.uniform(0, 1, BIG) < 0.15)[:, None]
    Y_ff = ro_m + ro_s * np.where(Bsel, ZB, ZA)
    ff = {q: tail_coeffs(Y_ff, q) for q in qs}
    OUT['exp3_feedforward_branch'] = {
        'branch_weight': 0.15, 'final_wall_offset': 0.9,
        'branch_mean_shift_per_cell': [float(x) for x in (ZB - ZA).mean(axis=0)],
        'tails': {str(q): {'lambda_L': v[0], 'lambda_U': v[1]}
                  for q, v in ff.items()},
        'mean_abs_rhoS': spearman_meanabs(Y_ff)[0]}
    print("  feed-forward lambda_L(q):", {q: round(v[0], 3) for q, v in ff.items()})
    print("  feed-forward mean|rho_S| =",
          round(OUT['exp3_feedforward_branch']['mean_abs_rhoS'], 3))
    save()

# ================= figures =================
if STAGE in ('fig', 'all'):
    import matplotlib; matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    ent = OUT['exp1_halfcut_entropy_bits']; anti = OUT['exp1_collision_ratio_PT2']
    nogo = OUT['exp3_nogo_deployed']; ff = OUT['exp3_feedforward_branch']['tails']
    nd = OUT['exp3_data_tails']
    fig, ax = plt.subplots(1, 3, figsize=(13.5, 3.6))
    for name, mk in (('path16', 'o-'), ('grid4x4', 's-')):
        Ls = sorted(ent[name], key=int)
        ax[0].plot([int(L) for L in Ls], [ent[name][L] for L in Ls], mk, label=name)
    ax[0].set_xlabel('blocks $L$'); ax[0].set_ylabel('half-cut entropy [bits]')
    ax[0].set_title('entanglement growth'); ax[0].legend(); ax[0].grid(alpha=0.3)
    for name, mk in (('path16', 'o-'), ('grid4x4', 's-')):
        Ls = sorted(anti[name], key=int)
        ax[1].plot([int(L) for L in Ls], [anti[name][L] for L in Ls], mk, label=name)
    ax[1].axhline(2.0, color='k', ls='--', lw=0.8, label='Porter-Thomas')
    ax[1].set_xlabel('blocks $L$'); ax[1].set_ylabel(r'$2^n\sum_z p(z)^2$')
    ax[1].set_title('anticoncentration (germ-conditioned)')
    ax[1].legend(); ax[1].grid(alpha=0.3)
    qs = sorted((float(q) for q in nogo), reverse=True)
    ax[2].plot(qs, [nogo[str(q)]['lambda_L'] for q in qs], 'o-',
               label=r'deployed $\bar\lambda_L(q)$')
    ax[2].plot(qs, [ff[str(q)]['lambda_L'] for q in qs], 's-',
               label=r'feed-forward $\bar\lambda_L(q)$')
    dq = sorted((float(q) for q in nd), reverse=True)
    ax[2].plot(dq, [nd[str(q)]['lambda_L'] for q in dq], 'k^--', label='Geant4')
    ax[2].set_xscale('log'); ax[2].invert_xaxis()
    ax[2].set_xlabel('quantile $q$'); ax[2].set_ylabel(r'$\bar\lambda_L(q)$')
    ax[2].set_title('tail no-go and its feed-forward escape')
    ax[2].legend(); ax[2].grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig('outputs/advantage_investigation.pdf')
    fig.savefig('outputs/advantage_investigation.png', dpi=150)
    print("figures saved")
