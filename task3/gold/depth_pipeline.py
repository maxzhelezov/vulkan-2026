#!/usr/bin/env python3
import argparse
import json
from dataclasses import dataclass
import pathlib as pt

import cv2
import matplotlib.colors as mcolors
import numpy as np
import logging 



def get_palette():
    """
    Ручная палитра "Abyss Thermal".
    near → яркий/тёплый,  far → тёмный/холодный.
    Перцептивно равномерная, читаемая на тёмном фоне.
    """
    # stops = [ 
    #     (0.00, (0.00, 0.00, 0.08)),   # near  – deep space black
    #     (0.10, (0.30, 0.00, 0.52)),   # violet
    #     (0.26, (0.00, 0.18, 0.82)),   # cobalt blue
    #     (0.44, (0.00, 0.72, 0.92)),   # electric cyan
    #     (0.62, (0.00, 0.96, 0.52)),   # aquamarine
    #     (0.80, (0.62, 1.00, 0.08)),   # acid lime
    #     (1.00, (1.00, 1.00, 1.00)),   # far   – solar white
    # ]
    # cmap = mcolors.LinearSegmentedColormap.from_list(
    #     "abyss_thermal", [(pos, rgb) for pos, rgb in stops]
    # )

    stops = list(reversed(["#1a1a1a", "#4e342e", "#6e1e05", "#bc4200", "#f57c00", "#f1b000"]))
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "october", stops, gamma=0.7
    )


    t = np.linspace(0, 1, 256)
    rgb = (cmap(t)[:, :3] * 255).astype(np.uint8)
    return rgb[:, ::-1].copy()  # RGB → BGR (OpenCV)


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int


@dataclass(frozen=True)
class BBox3D:
    min_x: float; max_x: float
    min_y: float; max_y: float
    min_z: float; max_z: float


@dataclass
class PersonConfig:
    person_dir: pt.Path
    frames_dir: pt.Path
    intrinsics: Intrinsics
    bbox3d:     BBox3D
    name:      str = ""   # Person123 и т.п.

    def __post_init__(self):
        if not self.name:
            self.name = self.person_dir.name


def parse_json(config_path, height, width):
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
        p = data.get("params", data)
        i = p["intrinsics"]
        b = p["bounding_box"]
        return (
            Intrinsics(
                fx=float(i["fx"]), fy=float(i["fy"]),
                cx=float(i["cx"]), cy=float(i["cy"]),
                height=height, width=width
            ),
            BBox3D(
                min_x=float(b["min_x"]), max_x=float(b["max_x"]),
                min_y=float(b["min_y"]), max_y=float(b["max_y"]),
                min_z=float(b["min_z"]), max_z=float(b["max_z"]),
            ),
        )
    except Exception as e:
        logging.warning(f"{config_path.name}: {e}")
        return None


def load_person_config(person_dir):
    frames_dir = person_dir.joinpath("frames")

    if not frames_dir.is_dir():
        return None

    # get one random png from frames dir
    img = next(frames_dir.rglob("*.png"))
    height, width, _ = cv2.imread(img).shape

    params_path = person_dir.joinpath("params.json")

    result = parse_json(params_path, height, width)

    if result is None:
        return None
    
    intrinsics, bbox3d = result
    return PersonConfig(
        person_dir=person_dir,
        frames_dir=frames_dir,
        intrinsics=intrinsics,
        bbox3d=bbox3d,
    )


def discover_people(dataset_path):
    configs = []
    if not dataset_path.exists():
        logging.error(f"Path doesnt exist: {dataset_path}")
        exit()
    for frames_path in sorted(dataset_path.rglob("frames")):
        if not frames_path.is_dir():
            continue
        person_dir = frames_path.parent
        cfg = load_person_config(person_dir)
        if cfg is not None:
            n = len(list(frames_path.glob("*.png")))
            logging.info(f"Found {person_dir.name} with {n} frames")
            configs.append(cfg)
    return configs


def project_bbox3d(bbox: BBox3D, K: Intrinsics):
    us, vs = [], []
    for X in (bbox.min_x, bbox.max_x):
        for Y in (bbox.min_y, bbox.max_y):
            for Z in (bbox.min_z, bbox.max_z):
                if Z <= 0:
                    continue
                us.append(K.fx * X / Z + K.cx)
                vs.append(K.fy * Y / Z + K.cy)

    if not us:
        return 0, 0, K.width, K.height

    u1 = max(0,        int(np.floor(min(us))))
    v1 = max(0,        int(np.floor(min(vs))))
    u2 = min(K.width,  int(np.ceil(max(us))))
    v2 = min(K.height, int(np.ceil(max(vs))))

    return u1, v1, u2, v2


def remove_noise(crop_mm: np.ndarray, bbox: BBox3D) -> np.ndarray:
    d = crop_mm.astype(np.float32)

    hist, bins = np.histogram(d, bins=100) # 2700 -> 
    d[d > bins[len(bins) - 4]] = 0

    lower_part = d[d.shape[0] * 3 // 4:] # second stronger path for feet at 3/4 of frame height
    hist, bins = np.histogram(lower_part, bins=100) # 2700 -> 
    lower_part[lower_part > bins[len(bins) - 20]] = 0

    return d


def colorize(depth_mm: np.ndarray, lut: np.ndarray) -> np.ndarray:
    """
    Диапазон нормируется per-crop: ближнее=index 0, дальнее=index 255.
    """
    valid = depth_mm > 0
    out = np.ones((*depth_mm.shape, 3), dtype=np.uint8) * 255
    if not valid.any():
        return out

    d_min = float(depth_mm[valid].min())
    d_rng = float(depth_mm[valid].max()) - d_min
    if d_rng < 1e-3:
        d_rng = 1.0

    norm = np.zeros_like(depth_mm)
    norm[valid] = (depth_mm[valid] - d_min) / d_rng
    idx = (norm * 255.0).clip(0, 255).astype(np.uint8)
    out[valid] = lut[idx[valid]]
    return out



def make_tile(depth_raw, cfg, lut):
    """
    uint16 depth map + PersonConfig → BGR tile.
    Возвращает None если crop пустой или весь фон.
    """
    u1, v1, u2, v2 = project_bbox3d(cfg.bbox3d, cfg.intrinsics)
    crop = depth_raw.astype(np.float32)[v1:v2, u1:u2]

    if crop.size == 0 or float(crop.max()) == 0:
        return None

    denoised = remove_noise(crop, cfg.bbox3d)

    if float(denoised.max()) == 0:
        return None

    return colorize(denoised, lut)


def assemble_grid(tiles, labels, n_cols, bg_color = (255, 255, 255), gap = 8):
    """
    Раскладывает тайлы в MxN без масштабирования.
    Каждая ячейка = размер наибольшего тайла; меньшие центрируются.
    Подписывает имя человека снизу каждого тайла.
    """
    if not tiles:
        return np.zeros((200, 200, 3), dtype=np.uint8)

    n = len(tiles)
    if n_cols <= 0:
        n_cols = max(1, int(np.ceil(np.sqrt(n))))
    n_rows = int(np.ceil(n / n_cols))

    cell_h = max(t.shape[0] for t in tiles)
    cell_w = max(t.shape[1] for t in tiles)

    label_h = 18
    total_h = n_rows * (cell_h + gap + label_h) + gap
    total_w = n_cols * (cell_w + gap) + gap

    canvas = np.full((total_h, total_w, 3), bg_color, dtype=np.uint8)

    for i, (tile, lbl) in enumerate(zip(tiles, labels)):
        row, col = divmod(i, n_cols)
        y0 = gap + row * (cell_h + gap + label_h)
        x0 = gap + col * (cell_w + gap)
        th, tw = tile.shape[:2]
        dy = (cell_h - th) // 2
        dx = (cell_w - tw) // 2
        canvas[y0 + dy: y0 + dy + th, x0 + dx: x0 + dx + tw] = tile

        lbl_y = y0 + cell_h + label_h - 4
        lbl_x = x0 + cell_w // 2 - len(lbl) * 8 # why 8 I DUNNO?
        cv2.putText(canvas, lbl, (lbl_x, lbl_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 0, 0), 1, cv2.LINE_AA)

    return canvas


def draw_hud(img, frame_idx, n_people, lut):
    """Рисует полупрозрачный HUD: счётчик кадра + легенда глубины."""
    out = img.copy()
    h, w = out.shape[:2]

    # Полупрозрачная шапка
    header = out.copy()
    cv2.rectangle(header, (0, 0), (w, 34), (0, 0, 0), -1)
    cv2.addWeighted(header, 0.60, out, 0.40, 0, out)

    text = f"frame [{frame_idx:05d}]   {n_people} person{'s' if n_people != 1 else ''}"
    cv2.putText(out, text, (12, 23),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (220, 220, 220), 1, cv2.LINE_AA)

    # Легенда-градиент (правый нижний угол)
    bar_w = min(220, w - 32)
    bar_h = 10
    bx = w - bar_w - 14
    by = h - 32
    if bar_w > 20 and bx >= 0 and by > 40:
        idx = np.linspace(0, 255, bar_w).astype(np.uint8)
        strip = np.tile(idx, (bar_h, 1))
        out[by: by + bar_h, bx: bx + bar_w] = lut[strip]
        cv2.rectangle(out, (bx, by), (bx + bar_w, by + bar_h), (80, 80, 80), 1)
        cv2.putText(out, "near", (bx, by + bar_h + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)
        cv2.putText(out, "far",  (bx + bar_w - 16, by + bar_h + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.32, (140, 140, 140), 1, cv2.LINE_AA)
    return out


def iter_frames(configs):
    """
    Iterates by frame for all ppl
    If frames for specific person ended -> None
    Yield: (global_idx, [(cfg, depth_uint16), ...])
    """
    frame_maps = []
    for cfg in configs:
        fmap = sorted(cfg.frames_dir.glob("*.png"))
        frame_maps.append(fmap)

    max_global_idx = max(map(len,frame_maps))

    for global_idx in range(max_global_idx):
        group = []
        for cfg, fmap in zip(configs, frame_maps):
            if global_idx >= len(fmap):
                group.append(None)
            depth = cv2.imread(str(fmap[global_idx]), cv2.IMREAD_ANYDEPTH)
            if depth is None:
                group.append(None)
            if depth.dtype != np.uint16:
                group.append(None)
            group.append((cfg, depth))
        if group:
            yield global_idx, group

def process_frame(global_idx, frames, grid_cols, lut):
    tiles, labels = [], []
    for cfg, depth_raw in frames:
        tile = make_tile(depth_raw, cfg, lut)
        if tile is not None:
            tiles.append(tile)
            labels.append(cfg.name)
    if not tiles:
        return None, None, global_idx
    grid = assemble_grid(tiles, labels, n_cols=grid_cols)
    grid = draw_hud(grid, global_idx, len(tiles), lut)
    return grid


def run_pipeline(dataset, output_dir, grid_cols, max_frames):
    lut = get_palette()
    output_dir.mkdir(parents=True, exist_ok=True)


    logging.info("Scanning dataset ...")
    configs = discover_people(dataset)
    if len(configs) == 0:
        logging.error("Nothing found")
        exit()

    logging.info(f"Found {len(configs)} people")

    logging.info(f"Gonna write here: {output_dir.resolve()}")

    saved = 0

    frame_iter = iter_frames(configs)
    if max_frames > 0:
        import itertools
        frame_iter = itertools.islice(frame_iter, max_frames)


    for (global_idx, frames) in frame_iter:
        grid = process_frame(global_idx, frames, grid_cols, lut)
        if grid is not None:
            frame_path = output_dir.joinpath(f"frame_{global_idx:05d}.png")
            cv2.imwrite(str(frame_path), grid)
            saved += 1
            if global_idx % 20 == 0:
                h, w = grid.shape[:2]
                logging.info(f"Proccesed frame_{global_idx:05d}.png")

    logging.info(f"Done. Saved {saved} frames to {output_dir.resolve()}")


def main():
    parser = argparse.ArgumentParser(
        description="Depth Map Pipeline — Gold (Python)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
            Exmaples:
            python depth_gold.py --dataset /data/Part1
            python depth_gold.py --dataset /data/Part1 /data/Part2 --output frames/ --rows 3
        """,
    )
    parser.add_argument("--dataset", help="Path to the dataset", type=pt.Path)
    parser.add_argument("--output", default=pt.Path("frames"), type=pt.Path, help="Path to output frames")
    parser.add_argument("--cols", type=int, default=0, help="Max cols in one frame (default 0 = best square fit with sqrt(ppl))")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to generate (default 0 == ALL)")
    args = parser.parse_args()

    if not args.dataset:
        parser.error("Specify path to the dataset via --dataset")

    run_pipeline(
        dataset=args.dataset,
        output_dir=args.output,
        grid_cols=args.cols,
        max_frames=args.max_frames
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", handlers=[logging.StreamHandler()])
    main()