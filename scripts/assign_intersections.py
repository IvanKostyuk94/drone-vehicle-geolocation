"""Match every frame to its intersection and write matches.csv."""

import pandas as pd

from dronegeo.config import load_config
from dronegeo.georef import find_best_match, get_global_point

cfg = load_config()


def find_best_matches() -> None:
    """Write image, best letter and inlier counts of every frame to
    paths.matches_csv."""
    refs = get_global_point()
    img_base = cfg["paths"]["images"]
    best_matches = []
    for i, img_path in enumerate(sorted(img_base.glob("*.jpg"))):
        best_matches.append([img_path.name])
        best_matches[i].extend(find_best_match(str(img_path), refs))
    pd.DataFrame(
        best_matches, columns=["image", "best", "max matches", "second best"]
    ).to_csv(str(cfg["paths"]["matches_csv"]), index=False)


if __name__ == "__main__":
    find_best_matches()
