import argparse
import pickle

import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation as R


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=0, required=False)
    parser.add_argument("--srcs", type=str, default="0,1,2,3", required=False)
    parser.add_argument("--no-pred", action="store_true")
    args = parser.parse_args()

    with open("data/carla/carla_infos_val.pkl", "rb") as f:
        data_infos = pickle.load(f)

    data_info = data_infos[args.id]

    scene_id = data_info["scene_id"]

    points = np.fromfile(f"data/carla/velodyne/{scene_id}", dtype=np.float32)
    points = points.reshape(-1, 4)

    src_indices = np.fromfile(
        f"data/carla/velodyne_src_indices/{scene_id}", dtype=np.int64
    )

    used_srcs = list(map(int, args.srcs.split(",")))

    points = points[:, :3]
    filtered_points = []
    for src in used_srcs:
        filtered_points.append(points[src_indices == src])
    filtered_points = np.concatenate(filtered_points, axis=0)
    points = filtered_points

    pc = o3d.geometry.PointCloud()
    pc.points = o3d.utility.Vector3dVector(points)

    bboxes = []
    for bbox, srcs in zip(data_info["annos"]["bboxes_3d"], data_info["annos"]["srcs"]):
        ok = False
        for used_src in used_srcs:
            if used_srcs in srcs:
                ok = True
        if not ok:
            continue

        bbox = np.asarray(bbox)
        bottom_center = bbox[:3]
        extent = bbox[3:6]
        yaw = bbox[6]

        center = bottom_center.copy()
        center[2] += extent[2] / 2

        r = R.from_euler("z", yaw, degrees=False)

        bbox = o3d.geometry.OrientedBoundingBox(
            center=center,
            R=r.as_matrix(),
            extent=extent,
        )
        bbox.color = [0, 1, 0]
        bboxes.append(bbox)

    with open("results.pkl", "rb") as f:
        results = pickle.load(f)

    result = results[args.id]
    boxes_3d = result["boxes_3d"].tensor.tolist()
    scores_3d = result["scores_3d"].tolist()

    if not args.no_pred:
        for bbox, score in zip(boxes_3d, scores_3d):
            # filter
            # if score < 0.3:
            #     continue

            bbox = np.asarray(bbox)
            bottom_center = bbox[:3]
            extent = bbox[3:6]
            yaw = bbox[6]

            center = bottom_center.copy()
            center[2] += extent[2] / 2

            r = R.from_euler("z", yaw, degrees=False)

            bbox = o3d.geometry.OrientedBoundingBox(
                center=center,
                R=r.as_matrix(),
                extent=extent,
            )
            bbox.color = [1, 0, 0]
            bboxes.append(bbox)

    o3d.visualization.draw([pc] + bboxes)


if __name__ == "__main__":
    main()
