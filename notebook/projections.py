"""
projections.py — Géométrie fisheye et reprojections pour la détection de personnes.

Conventions (identiques à OpenCV et à WoodScape) :
  repère caméra : x vers la droite, y vers le BAS, z le long de l'axe optique.
  repère « à niveau » (L) : même convention, mais y_L = verticale du monde (vers le bas)
  et z_L = axe optique projeté sur le plan horizontal. Toutes les représentations
  (perspective, multivues, cylindrique, ERP) sont construites dans ce repère L, de sorte
  que les verticales du monde restent verticales dans l'image.

Chaque représentation fournit :
  - .W, .H             taille de l'image cible
  - .pixel_to_dir(u,v) directions unitaires (repère L) pour des pixels cibles
  - .dir_to_pixel(d)   pixels cibles (u,v) + masque de validité pour des directions
Le principe est toujours la reprojection inverse (backward mapping) : on part de la
grille cible, on remonte au rayon, puis au pixel fisheye -> table de correspondance (LUT).
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field

import cv2
import numpy as np
from scipy.spatial.transform import Rotation


# --------------------------------------------------------------------------------------
# Modèles de caméra fisheye
# --------------------------------------------------------------------------------------
class _RadialFisheye:
    """Base commune : loi radiale rho(theta) monotone, inversée par table."""

    W: int
    H: int
    cx: float
    cy: float
    theta_lim: float

    def _radial(self, theta):  # rho en pixels (axe x)
        raise NotImplementedError

    def _build_inverse(self, theta_cap_deg: float):
        th = np.linspace(0.0, math.radians(theta_cap_deg), 40001)
        rho = self._radial(th)
        d = np.diff(rho)
        stop = int(np.argmax(d <= 0)) if np.any(d <= 0) else len(d)
        self.theta_lim = float(th[stop])          # domaine où la loi est injective
        self._th_tab, self._rho_tab = th[: stop + 1], rho[: stop + 1]

    @property
    def f0(self) -> float:
        """Résolution au centre (pixels par radian) = dérivée de rho en 0."""
        return float((self._radial(np.array([1e-4])) / 1e-4)[0])

    def project(self, d):
        """Directions (...,3) du repère caméra -> (u, v, valide)."""
        d = np.asarray(d, dtype=np.float64)
        x, y, z = d[..., 0], d[..., 1], d[..., 2]
        chi = np.hypot(x, y)
        theta = np.arctan2(chi, z)
        rho = self._radial(theta)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(chi > 1e-12, rho / chi, 0.0)
        u = self.cx + s * x
        v = self.cy + s * y * self._aspect
        valid = (theta <= self.theta_lim) & (u >= -0.5) & (u <= self.W - 0.5) \
            & (v >= -0.5) & (v <= self.H - 0.5)
        return u, v, valid

    def unproject(self, u, v):
        """Pixels fisheye -> directions unitaires (repère caméra)."""
        xp = np.asarray(u, dtype=np.float64) - self.cx
        yp = (np.asarray(v, dtype=np.float64) - self.cy) / self._aspect
        rho = np.hypot(xp, yp)
        theta = np.interp(rho, self._rho_tab, self._th_tab)
        with np.errstate(invalid="ignore", divide="ignore"):
            s = np.where(rho > 1e-12, np.sin(theta) / rho, 0.0)
        return np.stack([s * xp, s * yp, np.cos(theta)], axis=-1)


class WoodScapeFisheye(_RadialFisheye):
    """Modèle WoodScape : rho(theta) = k1 θ + k2 θ² + k3 θ³ + k4 θ⁴ (rho en pixels).

    Conforme à scripts/calibration/projection.py du dépôt valeoai/WoodScape :
      u = u' + cx_offset + W/2 - 0.5 ;  v = v' * aspect_ratio + cy_offset + H/2 - 0.5
    L'extrinsèque (quaternion x,y,z,w) donne la rotation caméra -> véhicule (ISO 8855 :
    x avant, y gauche, z haut).
    """

    def __init__(self, k, width, height, cx_offset=0.0, cy_offset=0.0, aspect_ratio=1.0,
                 R_cam2world=None, name="", theta_cap_deg=100.0):
        self.k = np.asarray(k, dtype=np.float64)
        self.W, self.H = int(width), int(height)
        self.cx = 0.5 * self.W - 0.5 + cx_offset
        self.cy = 0.5 * self.H - 0.5 + cy_offset
        self._aspect = float(aspect_ratio)
        self.R_cam2world = None if R_cam2world is None else np.asarray(R_cam2world, float)
        self.world_down = np.array([0.0, 0.0, -1.0])   # ISO 8855 : z vers le haut
        self.name = name
        self._build_inverse(theta_cap_deg)

    def _radial(self, theta):
        t = np.asarray(theta, dtype=np.float64)
        k1, k2, k3, k4 = self.k
        return t * (k1 + t * (k2 + t * (k3 + t * k4)))

    @classmethod
    def from_json(cls, path_or_dict):
        cfg = path_or_dict
        if not isinstance(cfg, dict):
            with open(cfg) as f:
                cfg = json.load(f)
        it = cfg["intrinsic"]
        R = Rotation.from_quat(cfg["extrinsic"]["quaternion"]).as_matrix()
        return cls([it["k1"], it["k2"], it["k3"], it["k4"]], it["width"], it["height"],
                   it.get("cx_offset", 0.0), it.get("cy_offset", 0.0),
                   it.get("aspect_ratio", 1.0), R_cam2world=R, name=cfg.get("name", ""))

    def key(self):
        """Clé de cache : deux images de même calibration partagent les mêmes LUT."""
        R = None if self.R_cam2world is None else np.round(self.R_cam2world, 6).tobytes()
        return (tuple(np.round(self.k, 6)), self.W, self.H, round(self.cx, 4),
                round(self.cy, 4), self._aspect, R)


class KannalaBrandtFisheye(_RadialFisheye):
    """Modèle OpenCV fisheye (Kannala-Brandt), pour votre propre banc :
    θ_d = θ (1 + k1 θ² + k2 θ⁴ + k3 θ⁶ + k4 θ⁸) ;  u = fx θ_d x/r + cx ;  v = fy θ_d y/r + cy.
    `R_cam2world` et `world_down` sont optionnels (sinon, pas de mise à niveau).
    """

    def __init__(self, K, D, width, height, R_cam2world=None, world_down=(0, 1, 0),
                 name="", theta_cap_deg=100.0):
        K = np.asarray(K, float)
        self.fx, self.fy = K[0, 0], K[1, 1]
        self.cx, self.cy = K[0, 2], K[1, 2]
        self.D = np.asarray(D, float).ravel()[:4]
        self.W, self.H = int(width), int(height)
        self._aspect = self.fy / self.fx
        self.R_cam2world = None if R_cam2world is None else np.asarray(R_cam2world, float)
        self.world_down = np.asarray(world_down, float)
        self.name = name
        self._build_inverse(theta_cap_deg)

    def _radial(self, theta):
        t = np.asarray(theta, dtype=np.float64)
        t2 = t * t
        k1, k2, k3, k4 = self.D
        return self.fx * t * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))

    def key(self):
        return (self.fx, self.fy, self.cx, self.cy, tuple(self.D), self.W, self.H)


class EquidistantFisheye(KannalaBrandtFisheye):
    """Calibration APPROXIMATIVE quand on n'a pas de calibration : modèle équidistant
    (r = f θ) dont le cercle image (centre, rayon) correspond au champ `fov_deg`."""

    def __init__(self, cx, cy, radius, fov_deg, width, height, name=""):
        f = radius / math.radians(fov_deg / 2)
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
        super().__init__(K, [0, 0, 0, 0], width, height, name=name, theta_cap_deg=fov_deg / 2)
        self.radius, self.fov_deg = radius, fov_deg


def estimate_image_circle(images, threshold=20):
    """Estime le cercle utile d'images fisheye (bord noir) : renvoie (cx, cy, rayon, fiable).

    On prend le maximum de plusieurs images (le bord noir ne bouge pas), on garde la plus
    grande zone claire, puis on ajuste un cercle sur son contour en ignorant les points
    collés au bord du cadre (cercle rogné). Sans bord noir visible : cercle inscrit, fiable=False.
    """
    g = np.max([cv2.cvtColor(im, cv2.COLOR_BGR2GRAY) for im in images], axis=0)
    H, W = g.shape
    m = (cv2.GaussianBlur(g, (5, 5), 0) > threshold).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if cnts:
        c = max(cnts, key=cv2.contourArea)[:, 0, :].astype(float)
        keep = (c[:, 0] > 3) & (c[:, 0] < W - 4) & (c[:, 1] > 3) & (c[:, 1] < H - 4)
        if keep.sum() > 100:
            x, y = c[keep, 0], c[keep, 1]
            A = np.stack([2 * x, 2 * y, np.ones_like(x)], 1)
            (a, b, k), *_ = np.linalg.lstsq(A, x * x + y * y, rcond=None)
            r = math.sqrt(k + a * a + b * b)
            if 0.2 * min(W, H) < r < 1.5 * max(W, H):
                return float(a), float(b), float(r), True
    return (W - 1) / 2, (H - 1) / 2, min(W, H) / 2, False


# --------------------------------------------------------------------------------------
# Repère à niveau
# --------------------------------------------------------------------------------------
def level_frame(cam, mounting: str = "auto", extra_pitch_deg: float = 0.0) -> np.ndarray:
    """Matrice M (3x3) telle que d_cam = M @ d_L.

    Le repère à niveau L a son axe y_L sur la verticale du monde (vers le bas) : c'est
    l'axe du cylindre, donc ce qui rend les personnes debout verticales dans l'image.
      - "horizontal" : caméra à l'horizontale (votre banc) -> L = repère caméra ;
      - "plafond"    : caméra qui regarde vers le bas (mât, plafond) -> y_L = axe optique,
                       z_L = direction du haut de l'image ;
      - "auto"       : utilise l'extrinsèque si la caméra en a une (WoodScape), sinon horizontal.
    """
    if mounting == "plafond":
        M = np.stack([[1.0, 0, 0], [0, 0, 1.0], [0, -1.0, 0]], axis=1)
    elif mounting == "horizontal" or getattr(cam, "R_cam2world", None) is None:
        M = np.eye(3)
    else:
        down = cam.R_cam2world.T @ cam.world_down          # verticale vers le bas, repère caméra
        yL = down / np.linalg.norm(down)
        z = np.array([0.0, 0.0, 1.0])
        zL = z - (z @ yL) * yL
        n = np.linalg.norm(zL)
        if n < 0.2:
            raise ValueError("Axe optique proche de la verticale : utiliser mounting='plafond'.")
        zL /= n
        M = np.stack([np.cross(yL, zL), yL, zL], axis=1)
    if extra_pitch_deg:
        M = M @ Rotation.from_euler("x", -extra_pitch_deg, degrees=True).as_matrix()
    return M


def tilt_deg(cam) -> float:
    """Inclinaison de l'axe optique sous l'horizontale (degrés, >0 = vers le bas)."""
    if getattr(cam, "R_cam2world", None) is None:
        return float("nan")
    down = cam.R_cam2world.T @ cam.world_down
    down /= np.linalg.norm(down)
    return float(np.degrees(np.arcsin(np.clip(down[2], -1, 1))))


# --------------------------------------------------------------------------------------
# Représentations cibles (toutes exprimées dans le repère à niveau L)
# --------------------------------------------------------------------------------------
def _grid(W, H):
    return np.meshgrid(np.arange(W, dtype=np.float64), np.arange(H, dtype=np.float64))


class Perspective:
    """Vue perspective virtuelle de focale f (px), champ horizontal hfov, tournée de
    yaw_deg autour de la verticale (yaw > 0 = vers la droite)."""

    kind = "perspective"

    def __init__(self, f, hfov_deg, vfov_deg, yaw_deg=0.0, pitch_deg=0.0):
        """Perspective virtuelle : focale f (px), champ hfov × vfov, orientée par lacet/tangage."""
        self.f = float(f)
        self.W = int(round(2 * f * math.tan(math.radians(hfov_deg) / 2)))
        self.H = int(round(2 * f * math.tan(math.radians(vfov_deg) / 2)))
        self.cx, self.cy = (self.W - 1) / 2, (self.H - 1) / 2
        # d_L = R d_local, avec R = Ry(lacet) · Rx(-tangage) : tangage > 0 = vers le bas
        self.R = Rotation.from_euler("YX", [yaw_deg, -pitch_deg], degrees=True).as_matrix()
        self.yaw, self.pitch = yaw_deg, pitch_deg

    def pixel_to_dir(self, u, v):
        x = (np.asarray(u, float) - self.cx) / self.f
        y = (np.asarray(v, float) - self.cy) / self.f
        d = np.stack([x, y, np.ones_like(x)], axis=-1)
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        return d @ self.R.T

    def dir_to_pixel(self, d):
        dl = np.asarray(d, float) @ self.R          # repère local de la vue
        z = dl[..., 2]
        with np.errstate(invalid="ignore", divide="ignore"):
            u = self.cx + self.f * dl[..., 0] / z
            v = self.cy + self.f * dl[..., 1] / z
        valid = (z > 1e-6) & (u >= -0.5) & (u <= self.W - 0.5) & (v >= -0.5) & (v <= self.H - 0.5)
        return u, v, valid


class Cylindrical:
    """Cylindre vertical déroulé : u = cx + f ψ, v = cy + f tan(ε) (ε > 0 vers le bas)."""

    kind = "cylindrique"

    def __init__(self, f, hfov_deg=180.0, up_deg=55.0, down_deg=55.0):
        self.f = float(f)
        self.W = int(round(f * math.radians(hfov_deg)))
        self.h_up, self.h_down = math.tan(math.radians(up_deg)), math.tan(math.radians(down_deg))
        self.H = int(round(f * (self.h_up + self.h_down)))
        self.cx = (self.W - 1) / 2
        self.cy = f * self.h_up - 0.5          # ligne d'horizon

    def pixel_to_dir(self, u, v):
        psi = (np.asarray(u, float) - self.cx) / self.f
        h = (np.asarray(v, float) - self.cy) / self.f
        d = np.stack([np.sin(psi), h, np.cos(psi)], axis=-1)
        return d / np.sqrt(1 + h * h)[..., None]

    def dir_to_pixel(self, d):
        d = np.asarray(d, float)
        r = np.hypot(d[..., 0], d[..., 2])
        psi = np.arctan2(d[..., 0], d[..., 2])
        with np.errstate(invalid="ignore", divide="ignore"):
            h = d[..., 1] / r
        u, v = self.cx + self.f * psi, self.cy + self.f * h
        valid = (r > 1e-6) & (u >= -0.5) & (u <= self.W - 0.5) & (v >= -0.5) & (v <= self.H - 0.5)
        return u, v, valid


class Equirectangular:
    """ERP : u = cx + k λ, v = cy + k φ avec φ = arcsin(d_y) (>0 vers le bas).
    Attention au signe : avec y vers le bas (OpenCV), écrire v = H(1/2 - φ/ΔΦ) avec
    φ = arcsin(d_y) donne une image À L'ENVERS. Ici le haut de l'image est bien le ciel."""

    kind = "erp"

    def __init__(self, k, hfov_deg=180.0, up_deg=80.0, down_deg=80.0):
        self.k = float(k)
        self.W = int(round(k * math.radians(hfov_deg)))
        self.H = int(round(k * math.radians(up_deg + down_deg)))
        self.cx = (self.W - 1) / 2
        self.cy = k * math.radians(up_deg) - 0.5        # ligne d'horizon

    def pixel_to_dir(self, u, v):
        lam = (np.asarray(u, float) - self.cx) / self.k
        phi = (np.asarray(v, float) - self.cy) / self.k
        return np.stack([np.cos(phi) * np.sin(lam), np.sin(phi), np.cos(phi) * np.cos(lam)], -1)

    def dir_to_pixel(self, d):
        d = np.asarray(d, float)
        lam = np.arctan2(d[..., 0], d[..., 2])
        phi = np.arcsin(np.clip(d[..., 1], -1, 1))
        u, v = self.cx + self.k * lam, self.cy + self.k * phi
        valid = (u >= -0.5) & (u <= self.W - 0.5) & (v >= -0.5) & (v <= self.H - 0.5)
        return u, v, valid


class FisheyeView:
    """L'image fisheye d'origine présentée comme une « représentation » du repère L :
    permet de ramener les boîtes de n'importe quelle représentation dans l'image brute."""

    kind = "fisheye"

    def __init__(self, cam, M):
        self.cam, self.M, self.W, self.H = cam, M, cam.W, cam.H

    def pixel_to_dir(self, u, v):
        return self.cam.unproject(u, v) @ self.M

    def dir_to_pixel(self, d):
        return self.cam.project(np.asarray(d, float) @ self.M.T)


# --------------------------------------------------------------------------------------
# Tables de correspondance (LUT) et remappage
# --------------------------------------------------------------------------------------
@dataclass
class Lut:
    map_x: np.ndarray          # float32 (H,W) : abscisse fisheye, -1 si hors champ
    map_y: np.ndarray
    valid: np.ndarray          # bool (H,W)
    xi: np.ndarray = field(repr=False, default=None)   # versions entières (plus proche voisin)
    yi: np.ndarray = field(repr=False, default=None)

    def remap(self, img, interp=cv2.INTER_LINEAR):
        return cv2.remap(img, self.map_x, self.map_y, interp,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def build_lut(cam, target, M_level: np.ndarray) -> Lut:
    """Reprojection inverse : pixel cible -> direction (repère L) -> repère caméra -> pixel fisheye."""
    uu, vv = _grid(target.W, target.H)
    d_L = target.pixel_to_dir(uu, vv)
    d_cam = d_L @ M_level.T
    u, v, valid = cam.project(d_cam)
    mx = np.where(valid, u, -1).astype(np.float32)
    my = np.where(valid, v, -1).astype(np.float32)
    xi = np.clip(np.rint(mx), 0, cam.W - 1).astype(np.int16)
    yi = np.clip(np.rint(my), 0, cam.H - 1).astype(np.int16)
    return Lut(mx, my, valid, xi, yi)


def identity_lut(cam) -> Lut:
    uu, vv = _grid(cam.W, cam.H)
    valid = np.ones((cam.H, cam.W), bool)
    return Lut(uu.astype(np.float32), vv.astype(np.float32), valid,
               uu.astype(np.int16), vv.astype(np.int16))


# --------------------------------------------------------------------------------------
# Jeu de représentations à comparer
# --------------------------------------------------------------------------------------
REPRESENTATIONS = ["0_brut", "1_perspective", "2_multivues", "3_cylindrique", "4_erp"]

# Réglages par montage. Angles en degrés ; pitch > 0 = vue tournée vers le bas.
PRESETS = {
    # caméra horizontale (votre banc) : panorama de 180° devant la caméra
    "horizontal": dict(persp=dict(hfov=110, vfov=90, pitch=0),
                       mv=dict(yaws=(-55, 0, 55), pitch=0, hfov=75, vfov=90),
                       cyl=dict(hfov=180, up=55, down=55),
                       erp=dict(hfov=180, up=80, down=80)),
    # caméra qui regarde vers le bas (FishEye8K) : panorama de 360° autour de la verticale
    "plafond": dict(persp=dict(hfov=120, vfov=120, pitch=90),
                    mv=dict(yaws=(0, 90, 180, 270), pitch=35, hfov=95, vfov=95),
                    cyl=dict(hfov=380, up=5, down=65),     # 380° : recouvrement à la jointure
                    erp=dict(hfov=380, up=5, down=85)),
}


def make_representations(cam, mounting="horizontal", scale=1.0, overrides=None) -> dict:
    """Construit les 5 représentations pour une caméra.

    Toutes ont la même résolution angulaire au centre : f = f0 de la caméra × scale.
    Renvoie {nom: {'views': [(target, lut), ...], 'M': M, 'fisheye': FisheyeView}}.
    """
    preset = PRESETS["plafond" if mounting == "plafond" else "horizontal"]
    p = {k: dict(v) for k, v in preset.items()}
    for k, v in (overrides or {}).items():
        p[k].update(v)
    f = cam.f0 * scale
    M = level_frame(cam, mounting)
    fe = FisheyeView(cam, M)
    reps = {"0_brut": {"views": [(fe, identity_lut(cam))]}}
    q = p["persp"]
    t = Perspective(f, q["hfov"], q["vfov"], pitch_deg=q["pitch"])
    reps["1_perspective"] = {"views": [(t, build_lut(cam, t, M))]}
    q = p["mv"]
    views = [Perspective(f, q["hfov"], q["vfov"], yaw_deg=y, pitch_deg=q["pitch"]) for y in q["yaws"]]
    reps["2_multivues"] = {"views": [(t, build_lut(cam, t, M)) for t in views]}
    q = p["cyl"]
    t = Cylindrical(f, q["hfov"], q["up"], q["down"])
    reps["3_cylindrique"] = {"views": [(t, build_lut(cam, t, M))]}
    q = p["erp"]
    t = Equirectangular(f, q["hfov"], q["up"], q["down"])
    reps["4_erp"] = {"views": [(t, build_lut(cam, t, M))]}
    for name, r in reps.items():
        r["M"], r["fisheye"] = M, fe
        # doublons possibles : plusieurs vues, ou panorama qui se recouvre au-delà de 360°
        r["dedup"] = len(r["views"]) > 1 or (name in ("3_cylindrique", "4_erp")
                                             and p["cyl" if name == "3_cylindrique" else "erp"]["hfov"] > 360)
    return reps


# --------------------------------------------------------------------------------------
# Passage de boîtes entre représentations (fusion multivues)
# --------------------------------------------------------------------------------------
def box_outline(box, n=16):
    """Points régulièrement espacés sur le contour d'une boîte (x1,y1,x2,y2)."""
    x1, y1, x2, y2 = box
    t = np.linspace(0, 1, n, endpoint=False)
    top = np.stack([x1 + t * (x2 - x1), np.full_like(t, y1)], 1)
    right = np.stack([np.full_like(t, x2), y1 + t * (y2 - y1)], 1)
    bot = np.stack([x2 - t * (x2 - x1), np.full_like(t, y2)], 1)
    left = np.stack([np.full_like(t, x1), y2 - t * (y2 - y1)], 1)
    return np.concatenate([top, right, bot, left])


def transfer_box(box, src, dst):
    """Boîte dans la représentation src -> boîte englobante dans dst (via les rayons)."""
    pts = box_outline(box)
    d = src.pixel_to_dir(pts[:, 0], pts[:, 1])
    u, v, ok = dst.dir_to_pixel(d)
    if ok.sum() < 3:
        return None
    u, v = u[ok], v[ok]
    return np.array([max(u.min(), 0), max(v.min(), 0),
                     min(u.max(), dst.W - 1), min(v.max(), dst.H - 1)])


def box_iou(a, b):
    """IoU entre deux ensembles de boîtes (N,4) et (M,4)."""
    a, b = np.asarray(a, float).reshape(-1, 4), np.asarray(b, float).reshape(-1, 4)
    ix1 = np.maximum(a[:, None, 0], b[None, :, 0]); iy1 = np.maximum(a[:, None, 1], b[None, :, 1])
    ix2 = np.minimum(a[:, None, 2], b[None, :, 2]); iy2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(ix2 - ix1, 0, None) * np.clip(iy2 - iy1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]); ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.maximum(aa[:, None] + ab[None, :] - inter, 1e-9)


def nms(boxes, scores, iou_thr=0.5):
    boxes, scores = np.asarray(boxes, float).reshape(-1, 4), np.asarray(scores, float)
    order, keep = np.argsort(-scores), []
    while order.size:
        i = order[0]
        keep.append(i)
        if order.size == 1:
            break
        ious = box_iou(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious < iou_thr]
    return np.array(keep, dtype=int)


def fuse_multiview(dets_per_view, views, dst, iou_thr=0.5):
    """Ramène les détections de chaque vue dans la représentation dst (ex. cylindrique),
    puis supprime les doublons entre vues par NMS. dets_per_view : [(boxes, scores), ...]."""
    B, S = [], []
    for (boxes, scores), (target, _) in zip(dets_per_view, views):
        for b, s in zip(boxes, scores):
            tb = transfer_box(b, target, dst)
            if tb is not None and tb[2] > tb[0] and tb[3] > tb[1]:
                B.append(tb); S.append(s)
    if not B:
        return np.zeros((0, 4)), np.zeros(0)
    B, S = np.array(B), np.array(S)
    k = nms(B, S, iou_thr)
    return B[k], S[k]


# --------------------------------------------------------------------------------------
# Contrôles géométriques
# --------------------------------------------------------------------------------------
def check_roundtrip(cam, n=20000, seed=0):
    """Erreur (px) de l'aller-retour pixel -> direction -> pixel sur l'image fisheye."""
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(0, cam.W - 1, n), rng.uniform(0, cam.H - 1, n)
    d = cam.unproject(u, v)
    th = np.arccos(np.clip(d[:, 2], -1, 1))
    m = th < cam.theta_lim * 0.999
    u2, v2, _ = cam.project(d[m])
    return float(np.max(np.hypot(u2 - u[m], v2 - v[m])))


def check_target_roundtrip(target, n=20000, seed=0):
    rng = np.random.default_rng(seed)
    u, v = rng.uniform(0, target.W - 1, n), rng.uniform(0, target.H - 1, n)
    u2, v2, _ = target.dir_to_pixel(target.pixel_to_dir(u, v))
    du = u2 - u
    scale = getattr(target, "f", None) if target.kind == "cylindrique" else getattr(target, "k", None)
    if target.kind in ("cylindrique", "erp") and scale:          # panorama > 360° : même rayon, 2 colonnes
        period = 2 * math.pi * scale
        du = np.minimum(np.abs(du), np.abs(np.abs(du) - period))
    return float(np.max(np.hypot(du, v2 - v)))


def check_vertical_line(target, azimuth_deg=60.0, elev_range_deg=(-40, 40)):
    """Une verticale du monde (azimut fixe) doit rester une colonne : renvoie l'écart max en u."""
    e = np.radians(np.linspace(*elev_range_deg, 50))
    a = math.radians(azimuth_deg)
    d = np.stack([np.cos(e) * math.sin(a), np.sin(e), np.cos(e) * math.cos(a)], 1)
    u, _, ok = target.dir_to_pixel(d)
    return float(np.ptp(u[ok])) if ok.any() else float("nan")


def check_upright(target):
    """Au centre de la vue, un point légèrement plus haut dans le monde doit apparaître plus
    haut dans l'image (v plus petit). Renvoie None si la vue regarde à la verticale (nadir)."""
    d0 = target.pixel_to_dir(np.array([(target.W - 1) / 2]), np.array([(target.H - 1) / 2]))[0]
    if abs(d0[1]) > 0.95:
        return None
    up = d0 - np.array([0.0, 0.05, 0.0]); up /= np.linalg.norm(up)
    _, v0, _ = target.dir_to_pixel(d0[None]); _, v1, _ = target.dir_to_pixel(up[None])
    return bool(v1[0] < v0[0])
