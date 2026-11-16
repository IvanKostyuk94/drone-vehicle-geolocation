# Data (not committed)

`python scripts/download_data.py` fills this folder:

```
data/
├── downloads/                      zip files (deleted after unpacking unless --keep-zips)
├── songdo_vision/test/             1,084 frames: images/, labels/, coco_annotations.json
└── songdo_traffic/
    ├── master_frames/              {ID}.png reference frame, {ID}.txt homography to the orthophoto
    ├── orthophotos/                {ID}.png 8000 x 8000 cut-out, {ID}_center.txt, ortho_parameters.txt
    └── segmentations/              {ID}.csv lane polygons in orthophoto pixels
```

Both datasets: CC BY 4.0, see the main README for citations.
