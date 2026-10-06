"""Export computed trajectories and independent chemical observations to the web UI."""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np

BOHR_ANGSTROM = 0.529177210903


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf8")


def episodes(frames, key):
    result, begin = [], None
    for i in range(len(frames) + 1):
        current = i < len(frames) and frames[i][key]
        if current and begin is None:
            begin = i
        if not current and begin is not None:
            result.append(
                {
                    "start_fs": frames[begin]["time_fs"],
                    "end_fs": frames[i - 1]["time_fs"],
                    "duration_fs": frames[i - 1]["time_fs"] - frames[begin]["time_fs"],
                    "includes_initial_frame": begin == 0,
                }
            )
            begin = None
    return result


def analyze(report):
    """Add a stricter read-only check; it never changes positions, forces or energies."""
    frames = report["frames"]
    last = [None] * len(frames[0]["hydrogen_ids"])
    changes = []
    for frame in frames:
        bonds = np.array(frame["oh_mayer_indices"])
        oxygen = np.array(frame["oxygen_ids"])
        dominant = bonds.argmax(1)
        sorted_bonds = np.sort(bonds, axis=1)
        clear_bond = (sorted_bonds[:, -1] > 0.25) & (
            sorted_bonds[:, -1] - sorted_bonds[:, -2] > 0.05
        )
        geometric = np.array([np.where(oxygen == o)[0][0] for o in frame["owners"]])
        compact = bool(np.min(frame["oh_distances_angstrom"], axis=1).max() < 1.35)
        frame["bond_supported_ionic_pair"] = bool(
            frame["charge_supported_ionic_pair"]
            and compact
            and clear_bond.all()
            and (dominant == geometric).all()
        )
        for h, o, certain in zip(frame["hydrogen_ids"], oxygen[dominant], clear_bond):
            hi = frame["hydrogen_ids"].index(h)
            if not certain:
                continue
            if last[hi] is not None and last[hi] != int(o):
                changes.append(
                    {
                        "time_fs": frame["time_fs"],
                        "hydrogen_id": h,
                        "from_oxygen": last[hi],
                        "to_oxygen": int(o),
                    }
                )
            last[hi] = int(o)
    ion_episodes = episodes(frames, "bond_supported_ionic_pair")
    persistent = [r for r in ion_episodes if r["duration_fs"] >= 2.0 - 1e-10]
    tracked = 6 if report["scenario"] == "shared_proton" else 1
    h = frames[0]["hydrogen_ids"].index(tracked)
    difference = np.array(
        [f["oh_mayer_indices"][h][0] - f["oh_mayer_indices"][h][1] for f in frames]
    )
    crossings = []
    for i in np.where(difference[1:] * difference[:-1] < 0)[0]:
        fraction = abs(difference[i]) / (abs(difference[i]) + abs(difference[i + 1]))
        t = frames[i]["time_fs"] + fraction * (frames[i + 1]["time_fs"] - frames[i]["time_fs"])
        crossings.append({"time_fs": float(t), "hydrogen_id": tracked})
    return {
        "electronic_bond_owner_changes": changes,
        "bond_supported_ionic_episodes": ion_episodes,
        "persistent_bond_supported_ionic_episodes_2fs": persistent,
        "tracked_mayer_weight_crossings": crossings,
        "classification": "Read-only, basis-dependent diagnostic: compact OH, unambiguous geometry, opposing fragment Mulliken charges beyond 0.25 e, and dominant Mayer owner bond >0.25 with margin >0.05; persistence 2 fs. Not a universal ion definition or free-energy measurement.",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="outputs/water_clusters_v04")
    parser.add_argument("--output", default="web/data")
    args = parser.parse_args()
    root, output = Path(args.input), Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    one = json.loads((root / "benchmark_threads_1.json").read_text())
    four = json.loads((root / "benchmark_threads_4.json").read_text())
    performance = []
    for row in one["rows"]:
        match = next((r for r in four["rows"] if r["waters"] == row["waters"]), None)
        t1 = row.get("median_native_step_seconds")
        t4 = match.get("median_native_step_seconds") if match else None
        performance.append(
            {
                "waters": row["waters"],
                "one_thread_seconds": t1,
                "four_thread_seconds": t4,
                "speedup": t1 / t4 if t1 and t4 else None,
                "density_fill_fraction": row.get("density_block_fill_fraction"),
            }
        )
    metadata = {
        "neutral_300k": (
            "常温初态 · {n} 个水分子",
            "平动与转动动量已去除，初始动能按 300 K 设置；随后自由 NVE 演化。",
        ),
        "hot_10000k": (
            "高能随机初态 · {n} 个水分子",
            "随机动量，初始动能按 10000 K 尺度设置；未指定断键原子或方向，不是该温度的平衡液态水。",
        ),
        "prepared_ion_pair": (
            "离子几何初态 · 复合控制",
            "人为准备 1 / 3 配位的初始几何，再用统一量子规律自由演化；不代表中性水自发电离。",
        ),
        "oriented_ion_pair": (
            "分离离子初态 · 自由演化",
            "人为准备分离的离子结构，总电荷为零；先核验电子键与片段电荷，再观察自由演化。不是中性水自发电离。",
        ),
        "shared_proton": (
            "共享质子 · H₅O₂⁺",
            "初始含一个多余质子，总电荷 +1；观察核运动与电子键权重的协同变化。",
        ),
    }
    runs, summaries = [], []
    for path in sorted(root.glob("*/report.json")):
        report = json.loads(path.read_text())
        if not report["frames"]:
            failed = {k: v for k, v in report.items() if k != "frames"}
            failed["source_report"] = path.as_posix()
            summaries.append(failed)
            continue
        derived = analyze(report)
        trajectory = path.parent / "trajectory.npz"
        with np.load(trajectory) as raw:
            xyz = raw["positions_bohr"] * BOHR_ANGSTROM
            time = raw["time_fs"]
            energy = raw["energy_balance_ev"]
            numbers = raw["numbers"]
            total_momentum = raw["momenta_au"].sum(axis=1)
            angular_momentum = np.cross(raw["positions_bohr"], raw["momenta_au"]).sum(axis=1)
        assert len(xyz) == len(report["frames"]) == len(time)
        assert np.isfinite(xyz).all() and np.isfinite(energy).all()
        assert all(abs(f["time_fs"] - t) < 1e-9 for f, t in zip(report["frames"], time))
        identifier = path.parent.name
        scenario = report["scenario"]
        count = int((numbers == 8).sum())
        title, description = metadata[scenario]
        title = title.format(n=count)
        title += f" · Δt {report['dt_fs']:g} fs"
        if scenario == "hot_10000k":
            title += f" · seed {report['seed']}"
        if scenario == "shared_proton":
            finding = f"电子键权重交叉 {len(derived['tracked_mayer_weight_crossings'])} 次。初态已经质子化，不能据此判定水自电离。"
        elif derived["persistent_bond_supported_ionic_episodes_2fs"]:
            finding = "检出持续至少 2 fs、具有几何与电子支持的离子候选；仍需物理精度与统计验证。"
        elif scenario in ("prepared_ion_pair", "oriented_ion_pair"):
            finding = "初始离子几何发生中性配位重排；本轮没有确认持续 2 fs 的电子支持离子对。"
        else:
            finding = "本条短轨迹未检出持续 2 fs 的电子支持离子对；这不能推断反应速率为零。"
        raw_file, data_file = f"{identifier}.npz", f"{identifier}.json"
        shutil.copyfile(trajectory, output / raw_file)
        write_json(
            output / data_file,
            {
                "positions_angstrom": xyz.tolist(),
                "time_fs": time.tolist(),
                "energy_balance_ev": energy.tolist(),
                "numbers": numbers.tolist(),
                "dt_fs": report["dt_fs"],
                "charge": 1 if scenario == "shared_proton" else 0,
                "frames": report["frames"],
            },
        )
        runs.append(
            {
                "id": identifier,
                "scenario": scenario,
                "title": title,
                "description": description,
                "finding": finding,
                "tracked_hydrogen_id": 6 if scenario == "shared_proton" else 1,
                "data_file": data_file,
                "raw_file": raw_file,
                "scope": "经典核 · 有限水簇 · HF/STO-3G · 不代表液态水自由能或反应速率",
            }
        )
        summary = {k: v for k, v in report.items() if k != "frames"}
        summary.update(derived)
        summary["max_total_momentum_change_au"] = float(
            np.linalg.norm(total_momentum - total_momentum[0], axis=1).max()
        )
        summary["max_total_angular_momentum_change_au"] = float(
            np.linalg.norm(angular_momentum - angular_momentum[0], axis=1).max()
        )
        summary["trajectory_sha256"] = hashlib.sha256(trajectory.read_bytes()).hexdigest()
        summary["source_report"] = path.as_posix()
        summaries.append(summary)
    manifest = {"performance": performance, "runs": runs, "environment": one["environment"]}
    write_json(output / "manifest.json", manifest)
    summary = {
        "performance": performance,
        "trajectories": summaries,
        "scope": "Internal measurements only; no other solver benchmark, large-system speedup or neutral-water autoionization claim.",
        "source_hashes": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in (
                Path(__file__),
                Path("examples/experiment_water_clusters.py"),
                Path("src/unichem/quantum/gaussian.py"),
                Path("src/unichem/runtime/molecular.py"),
            )
        },
    }
    checks = []
    for scenario in ("prepared_ion_pair", "shared_proton", "oriented_ion_pair"):
        coarse = root / f"{scenario}_n_2_seed_11_dt_0.1/trajectory.npz"
        fine = root / f"{scenario}_n_2_seed_11_dt_0.05/trajectory.npz"
        if not coarse.exists() or not fine.exists():
            continue
        with np.load(coarse) as c, np.load(fine) as f:
            n = min(len(c["time_fs"]), (len(f["time_fs"]) + 1) // 2)
            assert np.allclose(c["time_fs"][:n], f["time_fs"][: 2 * n : 2], atol=1e-8)
            ec, ef = (
                max(abs(c["energy_balance_ev"][:n])),
                max(abs(f["energy_balance_ev"][: 2 * n - 1])),
            )
            checks.append(
                {
                    "scenario": scenario,
                    "common_duration_fs": float(c["time_fs"][n - 1]),
                    "coarse_max_energy_error_ev": float(ec),
                    "fine_max_energy_error_ev": float(ef),
                    "energy_error_reduction": float(ec / ef),
                    "max_cartesian_component_difference_angstrom": float(
                        np.max(abs(c["positions_bohr"][:n] - f["positions_bohr"][: 2 * n : 2]))
                        * BOHR_ANGSTROM
                    ),
                }
            )
    summary["time_step_checks"] = checks
    write_json(root / "summary.json", summary)
    print(json.dumps({"exported_runs": len(runs), "performance": performance}, indent=2))


if __name__ == "__main__":
    main()
