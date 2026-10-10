"""Diagnostics of a GRTeclyn/PBHVlasov checkpoint: per-level min/max of the state, an x-line through the centre
on the finest level, coordinate-speed bound of the particle push, and the spherical expansion Theta(r)."""
import sys, re, glob, os, numpy as np
chk = sys.argv[1]
names = ("chi h11 h12 h13 h22 h23 h33 K A11 A12 A13 A22 A23 A33 Theta Gamma1 Gamma2 Gamma3 lapse shift1 shift2 shift3 "
         "B1 B2 B3 rho_p S1 S2 S3 S11 S12 S13 S22 S23 S33 fref").split()
names4 = "chi K lapse rho_p".split()
I = {n: i for i, n in enumerate(names)}

def parse_header(path):
    txt = open(path).read().split("\n")
    ncomp = int(txt[2]); ngrow = int(txt[3])
    assert txt[4].startswith("("), txt[4]
    nb = int(txt[4][1:].split()[0])
    boxes = []
    for b in range(nb):
        lo, hi = re.findall(r"\(([-\d]+),([-\d]+),([-\d]+)\)", txt[5 + b])[:2]
        boxes.append((tuple(map(int, lo)), tuple(map(int, hi))))
    i = 5 + nb + 1
    nfab = int(txt[i]); i += 1
    fabs = []
    for b in range(nfab):
        m = re.match(r"FabOnDisk:\s+(\S+)\s+(\d+)", txt[i + b]); fabs.append((m.group(1), int(m.group(2))))
    i += nfab
    while not re.match(r"^\d+,\d+$", txt[i]): i += 1
    mins = [[float(v) for v in txt[i + 1 + b].rstrip(",").split(",")] for b in range(nb)]
    i += 1 + nb
    while not re.match(r"^\d+,\d+$", txt[i]): i += 1
    maxs = [[float(v) for v in txt[i + 1 + b].rstrip(",").split(",")] for b in range(nb)]
    return ncomp, ngrow, boxes, fabs, np.array(mins), np.array(maxs)

def read_fab(levdir, fname, offset, box, ncomp, ngrow):
    with open(os.path.join(levdir, fname), "rb") as f:
        f.seek(offset)
        head = f.readline().decode()
        m = re.search(r"\(8, \(([\d ]+)\)\)\)", head)
        order = m.group(1).split()
        big = order[0] == "1"
        lo, hi = box
        n = [hi[d] - lo[d] + 1 + 2 * ngrow for d in range(3)]
        npts = n[0] * n[1] * n[2]
        data = np.fromfile(f, dtype=(">f8" if big else "<f8"), count=npts * ncomp)
    return data.reshape(ncomp, n[2], n[1], n[0])  # [comp, k, j, i]

sel = "chi K lapse shift1 B1 Gamma1 A11 h11 h22 rho_p S1 S11".split()
levels = sorted(glob.glob(f"{chk}/Level_*"))
hdr0 = glob.glob(f"{chk}/Level_0/*_H")[0]
small = False
if os.path.exists(f"{chk}/Header") and open(f"{chk}/Header").readline().startswith("HyperCLaw"):
    # plotfile: the variable names are listed in the Header
    lines = open(f"{chk}/Header").read().split("\n")
    nv = int(lines[1]); names = lines[2:2 + nv]; I = {n: i for i, n in enumerate(names)}
    sel = [n for n in "chi K lapse rho_p h11 h22 A11 A22 shift1".split() if n in I]
    small = not all(n in I for n in "h11 h12 h13 h22 h23 h33 A11 A12 A13 A22 A23 A33 K chi".split())
    cols_plot = ["x"] + sel
finest = len(levels) - 1
for lev in range(len(levels)):
    hdr = glob.glob(f"{chk}/Level_{lev}/*_H")[0]
    ncomp, ngrow, boxes, fabs, mins, maxs = parse_header(hdr)
    print(f"L{lev} ({os.path.basename(hdr)}, {len(boxes)} boxes, ngrow={ngrow}): " +
          " ".join(f"{n}=[{mins[:, I[n]].min():.3g},{maxs[:, I[n]].max():.3g}]" for n in sel))
# finest level: boxes touching the centre corner
lev = finest
hdr = glob.glob(f"{chk}/Level_{lev}/*_H")[0]
ncomp, ngrow, boxes, fabs, mins, maxs = parse_header(hdr)
hi_all = max(b[1][0] for b in boxes); lo_all = min(b[0][0] for b in boxes)
L = float(sys.argv[2]) if len(sys.argv) > 2 else 6194.411298
ncell_dom = 64 * 2 ** lev
dx = L / ncell_dom
c = ncell_dom // 2  # centre corner index
# gather the x-line at j = k = c (cell just above the centre in y, z) for i in [c-24, c+24)
i0, i1 = c - 24, c + 24
line = np.full((ncomp, i1 - i0), np.nan)
for b, (box, fab) in enumerate(zip(boxes, fabs)):
    lo, hi = box
    if lo[1] <= c <= hi[1] and lo[2] <= c <= hi[2] and hi[0] >= i0 and lo[0] < i1:
        d = read_fab(f"{chk}/Level_{lev}", fab[0], fab[1], box, ncomp, ngrow)
        for i in range(max(lo[0], i0), min(hi[0], i1 - 1) + 1):
            line[:, i - i0] = d[:, c - lo[2] + ngrow, c - lo[1] + ngrow, i - lo[0] + ngrow]
x = (np.arange(i0, i1) + 0.5 - c) * dx
r = np.sqrt(x**2 + 2 * (0.5 * dx) ** 2)
print(f"\nfinest level {lev}, dx={dx:.4f}; x-line through the centre (y=z=+dx/2):")
cols = (cols_plot if "cols_plot" in globals() else "x chi lapse K shift1 Gamma1 h11 h22 A11 A22 rho_p S1".split())
print("  ".join(f"{cname:>10s}" for cname in cols))
for n in range(len(x)):
    vals = [x[n]] + [line[I[cname], n] for cname in cols[1:]]
    print("  ".join(f"{v:10.3g}" for v in vals))
if small: sys.exit(0)
# coordinate speed bound of the particle push: |beta| + lapse*sqrt(chi*lambda_max(h~^{-1}))
chi = line[I["chi"]]; al = line[I["lapse"]]
hmat = np.array([[line[I["h11"]], line[I["h12"]], line[I["h13"]]], [line[I["h12"]], line[I["h22"]], line[I["h23"]]],
                 [line[I["h13"]], line[I["h23"]], line[I["h33"]]]]).transpose(2, 0, 1)
lam = np.array([np.linalg.eigvalsh(np.linalg.inv(h)).max() if np.isfinite(h).all() else np.nan for h in hmat])
beta = np.sqrt(line[I["shift1"]] ** 2 + line[I["shift2"]] ** 2 + line[I["shift3"]] ** 2)
vmax = beta + al * np.sqrt(np.maximum(chi * lam, 0))
dt_f = 0.25 * dx
print(f"\nmax |beta| on line = {np.nanmax(beta):.4g}, max coordinate speed bound = {np.nanmax(vmax):.4g}, "
      f"=> max displacement per fine step = {np.nanmax(vmax) * dt_f / dx:.3f} cells")
# spherical expansion along the +x half line: R = r sqrt(h22/chi), s^r = sqrt(chi/h11), Theta = 2 R'/(R sqrt(h11/chi)) - 2 (K/3 + A22/h22)
sel_p = x > 0
rp = r[sel_p]; R = rp * np.sqrt(line[I["h22"], sel_p] / chi[sel_p])
dR = np.gradient(R, rp)
Theta = 2 * dR / (R * np.sqrt(line[I["h11"], sel_p] / chi[sel_p])) - 2 * (line[I["K"], sel_p] / 3 + line[I["A22"], sel_p] / line[I["h22"], sel_p])
print("\nspherical expansion on the +x half line (r, R_areal, Theta, M_MS=R/2(1-(dR/ds)^2+...) approx):")
# Misner-Sharp mass: M = R/2 (1 - gamma^{ij} d_i R d_j R + (K_theta R)^2 ... ) : for spherical: 1 - (dR/ds)^2 + (R K^th_th)^2
Kth = line[I["K"], sel_p] / 3 + line[I["A22"], sel_p] / line[I["h22"], sel_p]
dRds = dR / np.sqrt(line[I["h11"], sel_p] / chi[sel_p])
M_MS = 0.5 * R * (1 - dRds**2 + (R * Kth) ** 2)
for n in range(len(rp)):
    print(f"  r={rp[n]:8.3f}  R={R[n]:10.4g}  Theta={Theta[n]:10.4g}  M_MS={M_MS[n]:10.4g}  (M_H=183.71 -> {M_MS[n]/183.71:.4f})")
