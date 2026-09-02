import numpy as np
from scipy.special import sph_harm_y

def solid_harmonics_np(x, y, z, L):
    """Real regular solid harmonics R_l^m, Racah normalised.
    Returns dict (l, m) -> array;  m > 0 cosine-type, m < 0 sine-type."""
    r2 = x*x + y*y + z*z
    out = {}
    out[(0, 0)] = np.ones_like(x)
    for l in range(1, L+1):
        # diagonal step: (l-1,l-1) -> (l,l)
        c = np.sqrt((2*l-1)/(2*l)) * (np.sqrt(2.0) if l == 1 else 1.0)
        Cp = out[(l-1, l-1)]
        Sp = out[(l-1, -(l-1))] if l > 1 else np.zeros_like(x)
        out[(l,  l)] = c*(x*Cp - y*Sp)
        out[(l, -l)] = c*(y*Cp + x*Sp)
        # vertical step at fixed m
        for m in range(0, l):
            for sgn in ([1] if m == 0 else [1, -1]):
                mm = sgn*m
                a = (2*l-1)/np.sqrt(l*l - m*m)
                t = a*z*out[(l-1, mm)]
                if l-1 > m:
                    b = np.sqrt((l-1)**2 - m*m)/np.sqrt(l*l - m*m)
                    t = t - b*r2*out[(l-2, mm)]
                out[(l, mm)] = t
    return out

# --- check 1: closed forms -------------------------------------------------
rng = np.random.default_rng(0)
x, y, z = rng.normal(size=(3, 5))
R = solid_harmonics_np(x, y, z, 3)
s3 = np.sqrt(3.0)
ref = {
    (1, 0): z, (1, 1): x, (1, -1): y,
    (2, 0): z*z - 0.5*(x*x + y*y),
    (2, 1): s3*x*z, (2, -1): s3*y*z,
    (2, 2): 0.5*s3*(x*x - y*y), (2, -2): s3*x*y,
    (3, 0): 0.5*z*(2*z*z - 3*x*x - 3*y*y),
}
print("closed-form check:")
for k, v in ref.items():
    print(f"  {k}  max err {np.abs(R[k]-v).max():.2e}")

# --- check 2: against scipy for all l,m up to L ----------------------------
L = 12
n = 200
v = rng.normal(size=(3, n))
x, y, z = v
r = np.sqrt(x*x + y*y + z*z)
theta = np.arccos(z/r)          # polar
phi = np.arctan2(y, x)          # azimuthal
R = solid_harmonics_np(x, y, z, L)

worst = 0.0
for l in range(L+1):
    for m in range(0, l+1):
        Y = sph_harm_y(l, m, theta, phi)
        pref = np.sqrt(4*np.pi/(2*l+1)) * r**l
        if m == 0:
            got, exp = R[(l, 0)], pref*Y.real
            worst = max(worst, np.abs(got-exp).max()/max(np.abs(exp).max(), 1))
        else:
            f = pref*np.sqrt(2)*(-1)**m
            for key, comp in [((l, m), Y.real), ((l, -m), Y.imag)]:
                got, exp = R[key], f*comp
                worst = max(worst, np.abs(got-exp).max()/max(np.abs(exp).max(), 1))
print(f"\nscipy check up to L={L}: worst relative err {worst:.3e}")
