"""
Robustness check: confirm reported TFIM quantities are insensitive to the
exact value of the symmetry-breaking pinning field `pin` (introduced to
lift the near-degenerate Z2 doublet in the ordered phase, h<1 - see
lib.hamiltonians.build_tfim docstring), across several orders of magnitude,
both at the critical point and deep in the ordered phase.
"""
import sys, os, csv
sys.path.insert(0, os.path.dirname(__file__))
from lib import hamiltonians as H
from lib import entanglement as E
from lib import laplacian as L

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
PINS = [1e-8, 1e-7, 1e-6, 1e-5, 1e-4, 1e-3]
N = 8


def lambda1_at(h, pin):
    Hs = H.build_tfim(N, h=h, pin=pin)
    e0, psi = E.ground_state_sparse(Hs)
    I, _ = E.mutual_information_matrix(psi, N)
    sq = L.spectral_quantities(L.normalized_laplacian(I))
    return sq["lambda1"]


def main():
    rows = []
    for h in [0.5, 1.0]:
        for pin in PINS:
            l1 = lambda1_at(h, pin)
            rows.append(dict(h=h, pin=pin, lambda1=l1))
            print(f"h={h} pin={pin:.0e}: lambda1={l1:.8f}")
        vals = [r["lambda1"] for r in rows if r["h"] == h]
        print(f"h={h}: spread across pin values = {max(vals)-min(vals):.2e}")

    os.makedirs(DATA_DIR, exist_ok=True)
    out = os.path.join(DATA_DIR, "pin_sensitivity.csv")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["h", "pin", "lambda1"])
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
