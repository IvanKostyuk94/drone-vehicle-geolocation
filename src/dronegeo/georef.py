import os

import cv2
import numpy as np

from dronegeo.config import load_config

cfg = load_config()


def get_global_point() -> list[tuple]:
    """SIFT keypoints and descriptors of all master frames (half size).

    Returns a list of (keypoints, descriptors, intersection letter).
    """
    detector = cv2.SIFT_create(nfeatures=4000)
    ref_base = cfg["paths"]["master_frames"]
    refs = []
    for ref_path in sorted(ref_base.glob("*.png")):
        ref = cv2.imread(str(ref_path))
        ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
        ref = cv2.resize(
            ref, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA
        )
        ref_keys, ref_finger = detector.detectAndCompute(ref, None)
        refs.append((ref_keys, ref_finger, ref_path.stem))
    return refs


def find_best_match(
    img_path: str, glob_points: list[tuple]
) -> tuple[str | None, int, int]:
    """Intersection of one frame by matching against all master frames.

    glob_points: output of get_global_point.
    Returns (best letter, inliers of the best, inliers of the second best).
    """
    detector = cv2.SIFT_create(nfeatures=4000)
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img_keys, img_finger = detector.detectAndCompute(img, None)
    max_val = 0
    second_val = 0
    best = None
    for ref_keys, ref_finger, letter in glob_points:
        sim = cv2.BFMatcher().knnMatch(img_finger, ref_finger, k=2)
        new_sim = []
        for p1, p2 in sim:
            if p1.distance < 0.75 * p2.distance:
                new_sim.append(p1)
        if len(new_sim) < 4:
            continue
        photo_pos = np.zeros((len(new_sim), 2), dtype=np.float32)
        ref_pos = np.zeros((len(new_sim), 2), dtype=np.float32)
        for i, p in enumerate(new_sim):
            photo_pos[i, :] = img_keys[p.queryIdx].pt
            ref_pos[i, :] = ref_keys[p.trainIdx].pt
        mapping, mask = cv2.findHomography(photo_pos, ref_pos, cv2.RANSAC, 5.0)
        if mask is None:
            continue
        inliers = mask.sum()
        if inliers > max_val:
            second_val = max_val
            max_val = inliers
            best = letter
        elif inliers > second_val:
            second_val = inliers
    return (best, int(max_val), int(second_val))


def frame_to_master(
    img_path: str, master_path: str
) -> tuple[np.ndarray | None, int]:
    """Homography from frame pixels to master-frame pixels (full size).

    Returns (H [3, 3], number of RANSAC inliers), or (None, 0) if fewer than
    4 matches are found.
    """
    detector = cv2.SIFT_create(nfeatures=4000)
    img = cv2.imread(str(img_path))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    img_keys, img_finger = detector.detectAndCompute(img, None)
    ref = cv2.imread(str(master_path))
    ref = cv2.cvtColor(ref, cv2.COLOR_BGR2GRAY)
    ref = cv2.resize(ref, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
    ref_keys, ref_finger = detector.detectAndCompute(ref, None)
    sim = cv2.BFMatcher().knnMatch(img_finger, ref_finger, k=2)
    new_sim = []
    for p1, p2 in sim:
        if p1.distance < 0.75 * p2.distance:
            new_sim.append(p1)
    if len(new_sim) < 4:
        return None, 0
    photo_pos = np.zeros((len(new_sim), 2), dtype=np.float32)
    ref_pos = np.zeros((len(new_sim), 2), dtype=np.float32)
    for i, p in enumerate(new_sim):
        photo_pos[i, :] = img_keys[p.queryIdx].pt
        ref_pos[i, :] = ref_keys[p.trainIdx].pt
    mapping, mask = cv2.findHomography(photo_pos, ref_pos, cv2.RANSAC, 5.0)
    S = np.diag((0.5, 0.5, 1))
    S_inv = np.diag((2, 2, 1))
    return S_inv @ mapping @ S, int(mask.sum())


def read_master_homography(path: str) -> np.ndarray:
    """Homography master frame -> orthophoto cut-out from {ID}.txt.

    Returns H [3, 3].
    """
    with open(path) as f:
        elements = f.readline()
    elements = elements.split(",")
    trans = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            trans[i, j] = float(elements[i * 3 + j])
    return trans


def read_ortho_params(
    orthophotos_dir: str, intersection_id: str, cutout_original_size: int
) -> dict:
    """Georeferencing of one orthophoto cut-out.

    Returns dict(lon, lat, dlon, dlat): the top-left corner of the cut-out
    and the size of one cut-out pixel, in degrees.
    """
    img_path = os.path.join(orthophotos_dir, f"{intersection_id}.png")
    width = cv2.imread(img_path).shape[1]
    conversion = cutout_original_size / width
    params_path = os.path.join(orthophotos_dir, "ortho_parameters.txt")
    coord = {"lon": 0, "lat": 0, "dlon": 0, "dlat": 0}
    with open(params_path) as f:
        for key in coord.keys():
            coord[key] = float(f.readline())
    centers_path = os.path.join(
        orthophotos_dir, f"{intersection_id}_center.txt"
    )
    with open(centers_path) as f:
        lonc = float(f.readline())
        latc = float(f.readline())
    coord["lon"] = (
        coord["lon"] + (lonc - cutout_original_size / 2) * coord["dlon"]
    )
    coord["lat"] = (
        coord["lat"] + (latc - cutout_original_size / 2) * coord["dlat"]
    )
    coord["dlon"] *= conversion
    coord["dlat"] *= conversion
    return coord


def frame_to_lonlat(
    points_xy: np.ndarray,
    H_m_from_f: np.ndarray,
    H_o_from_m: np.ndarray,
    params: dict,
) -> np.ndarray:
    """Frame pixels -> longitude and latitude.

    points_xy: [N, 2] frame pixels (x, y); H_m_from_f: frame -> master;
    H_o_from_m: master -> cut-out; params: from read_ortho_params.
    Returns [N, 2] (longitude, latitude).
    """
    full_trans = H_o_from_m @ H_m_from_f
    points_xy = np.concatenate(
        [points_xy, np.ones(points_xy.shape[0]).reshape(-1, 1)], axis=1
    )
    points_in_o = points_xy @ full_trans.T
    points_in_o = points_in_o[:, :2] / points_in_o[:, 2:3]
    coord_in_o_lon = params["lon"] + points_in_o[:, 0] * params["dlon"]
    coord_in_o_lat = params["lat"] + points_in_o[:, 1] * params["dlat"]
    full_coord = np.concatenate(
        [coord_in_o_lon.reshape(-1, 1), coord_in_o_lat.reshape(-1, 1)], axis=1
    )
    return full_coord


def homography_ok(
    H: np.ndarray,
    n_inliers: int,
    min_inliers: int,
    scale_range: list[float],
) -> bool:
    """True if H has at least min_inliers inliers and its Jacobian
    determinant at the frame centre lies inside scale_range."""
    if n_inliers < min_inliers:
        return False
    x = 1920
    y = 1080
    a = H[0, 0] * x + H[0, 1] * y + H[0, 2]
    b = H[1, 0] * x + H[1, 1] * y + H[1, 2]
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    du_dx = (H[0, 0] * w - a * H[2, 0]) / w**2
    du_dy = (H[0, 1] * w - a * H[2, 1]) / w**2
    dv_dx = (H[1, 0] * w - b * H[2, 0]) / w**2
    dv_dy = (H[1, 1] * w - b * H[2, 1]) / w**2
    det_J = du_dx * dv_dy - du_dy * dv_dx
    return scale_range[0] <= det_J <= scale_range[1]
