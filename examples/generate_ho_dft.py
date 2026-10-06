"""Actual isolated H/O reference calculations with PySCF (Linux/macOS).

Energy/forces are PBE0/aug-cc-pVDZ, explicit charge/spin/field. Geometry names
are provenance only; models receive atomic numbers, coordinates and fields.
No model-generated or synthetic energies enter this reference dataset.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
from pyscf import __version__, dft, gto, lib

BOHR = 0.529177210903


def water_pair(distance, proton):
    return [8, 8, 1, 1, 1, 1], np.array(
        [
            [-distance / 2, 0, 0],
            [distance / 2, 0, 0],
            [-distance / 2 + proton, 0, 0],
            [-distance / 2 - 0.239, 0.927, 0],
            [distance / 2 + 0.586, 0.757, 0],
            [distance / 2 + 0.586, -0.757, 0],
        ]
    )


def geometries():
    jobs = []

    def add(group, z, r, q=0, field=(0.0, 0.0, 0.0)):
        jobs.append(
            {
                "group": group,
                "numbers": z,
                "positions_angstrom": np.asarray(r).tolist(),
                "net_charge": q,
                "spin_excess": 0,
                "external_field_au": list(field),
            }
        )

    water = np.array([[0.0, 0.0, 0.0], [0.958, 0.0, 0.0], [-0.24, 0.927, 0.0]])
    for scale in [0.94, 1.0, 1.06, 1.15]:
        add("water", [8, 1, 1], water * scale)
    for length in [0.8, 1.2, 1.6, 2.0]:
        r = water.copy()
        r[1] = [length, 0, 0]
        add("stretched_oh", [8, 1, 1], r)
    for length in [0.92, 0.98, 1.05, 1.15]:
        r = [[0.0, 0.0, 0.0]] + [
            [length * np.cos(t), length * np.sin(t), 0.25] for t in np.arange(3) * 2 * np.pi / 3
        ]
        add("hydronium", [8, 1, 1, 1], r, 1)
    for length in [0.85, 0.97, 1.15, 1.45]:
        add("hydroxide", [8, 1], [[0, 0, 0], [length, 0, 0]], -1)
    for oo in [2.4, 2.6, 2.9]:
        for shift in [-0.15, 0, 0.15]:
            z, r = water_pair(oo, oo / 2 + shift)
            add("zundel", [8, 8, 1, 1, 1, 1, 1], np.vstack((r, [-oo / 2 - 0.35, -0.85, 0.0])), 1)
    for scale in [0.97, 1.03]:
        centre = [[0, 0, 0]] + [
            [0.96 * np.cos(t), 0.96 * np.sin(t), 0.22] for t in np.arange(3) * 2 * np.pi / 3
        ]
        z = [8, 1, 1, 1]
        r = list(centre)
        for t in np.arange(3) * 2 * np.pi / 3:
            axis = np.array([np.cos(t), np.sin(t), 0.0])
            tangent = np.array([-np.sin(t), np.cos(t), 0.0])
            o = 2.6 * axis
            r.extend([o, o + 0.55 * axis + 0.76 * tangent, o + 0.55 * axis - 0.76 * tangent])
            z.extend([8, 1, 1])
        add("eigen", z, np.asarray(r) * scale, 1)
    for oo in [2.5, 2.8]:
        for proton in [0.9, 1.1, 1.3, 1.5, 1.7]:
            z, r = water_pair(oo, proton)
            add("neutral_pair_scan", z, r)
    # Finite electric fields, explicit origin and vector; no field guessed from
    # a directory name. Field force derivatives are checked independently.
    for strength in [-0.002, 0.002]:
        add("external_field_water", [8, 1, 1], water, 0, (strength, 0.0, 0.0))
        z, r = water_pair(2.6, 1.3)
        add("external_field_pair", z, r, 0, (strength, 0.0, 0.0))
    for job in jobs:
        # Use a common origin at the centre of a 28.8-bohr vacuum box. Charged
        # field energies depend on origin; this shift is recorded and labeled.
        job["positions_angstrom"] = (np.asarray(job["positions_angstrom"]) + 14.4 * BOHR).tolist()
    return jobs


def calculate(job, *, grid_level=2):
    molecule = gto.M(
        atom=list(zip(job["numbers"], job["positions_angstrom"])),
        unit="Angstrom",
        charge=job["net_charge"],
        spin=job["spin_excess"],
        basis="aug-cc-pvdz",
        verbose=0,
    )
    mf = dft.RKS(molecule)
    mf.xc = "pbe0"
    mf.grids.level = grid_level
    mf.conv_tol = 1e-10
    mf.max_cycle = 120
    field = np.array(job["external_field_au"])
    if np.any(field):
        h0 = mf.get_hcore()
        dipole = molecule.intor_symmetric("int1e_r", comp=3)
        h = h0 + np.einsum("x,xij->ij", field, dipole)
        mf.get_hcore = lambda mol=None: h
        nuclear = molecule.energy_nuc() - np.einsum(
            "i,ix,x->", molecule.atom_charges(), molecule.atom_coords(), field
        )
        mf.energy_nuc = lambda: nuclear
    energy = float(mf.kernel())
    if not mf.converged:
        raise RuntimeError("Reference SCF unconverged; no label written")
    gradient = mf.nuc_grad_method()
    gradient.grid_response = True
    if np.any(field):
        old_generator = gradient.hcore_generator
        # PySCF exposes <bra|r_y nabla_x|ket>; transpose AO indices for
        # the corresponding bra derivative and contract the operator axis.
        irp = molecule.intor("int1e_irp", comp=9).reshape(
            3, 3, molecule.nao_nr(), molecule.nao_nr()
        )
        derivative = np.einsum("yxij,y->xji", irp, field)
        slices = molecule.aoslice_by_atom()

        def generator(mol=None):
            original = old_generator(mol)

            def atom_derivative(i):
                result = original(i).copy()
                p0, p1 = slices[i, 2:]
                result[:, p0:p1, :] -= derivative[:, p0:p1, :]
                result[:, :, p0:p1] -= derivative[:, p0:p1, :].transpose(0, 2, 1)
                return result

            return atom_derivative

        gradient.hcore_generator = generator
        old_nuclear = gradient.grad_nuc

        def nuclear_gradient(mol=None, atmlst=None):
            result = old_nuclear(mol, atmlst)
            charges = molecule.atom_charges() if atmlst is None else molecule.atom_charges()[atmlst]
            return result - charges[:, None] * field

        gradient.grad_nuc = nuclear_gradient
    force = -gradient.kernel()
    dm = mf.make_rdm1()
    ncore = sum(z == 8 for z in job["numbers"])
    c = mf.mo_coeff[:, :ncore]
    valence_dm = dm - 2 * c @ c.T
    return molecule, mf, energy, force, dm, valence_dm


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/generated_pbe0")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--check-field", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--jobs", help="Unlabeled schema-1 experiment design manifest")
    args = parser.parse_args()
    lib.num_threads(args.threads)
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    if args.jobs:
        manifest = json.loads(Path(args.jobs).read_text(encoding="utf8"))
        if manifest["schema_version"] != 1 or manifest["status"] != "unlabeled_geometry_design":
            raise ValueError("Current unlabeled experiment design required")
        jobs = [
            {
                "group": j["family"],
                "numbers": j["atomic_numbers"],
                "positions_angstrom": j["positions_angstrom"],
                "net_charge": j["net_charge"],
                "spin_excess": j["spin_excess"],
                "external_field_au": j["external_field_au"],
                "design_split": j["split"],
                "design_coordinates": j["coordinates"],
                "design_roles": j["roles"],
            }
            for j in manifest["jobs"]
        ]
    else:
        jobs = geometries()
    jobs = jobs[: args.limit]
    previous = root / "jobs.json"
    if previous.exists() and json.loads(previous.read_text(encoding="utf8")) != jobs:
        raise ValueError("Output directory contains a different job plan; choose a new directory")
    (root / "jobs.json").write_text(json.dumps(jobs, indent=2))
    for index, job in enumerate(jobs):
        path = root / f"{index:04d}.npz"
        if (
            path.exists()
            and path.with_name(path.stem + ".density.npz").exists()
            and not args.overwrite
        ):
            continue
        started = time.perf_counter()
        try:
            mol, mf, energy, force, dm, valence = calculate(job)
            metadata = {
                **job,
                "index": index,
                "method": "PBE0/aug-cc-pVDZ",
                "pyscf_version": __version__,
                "units": "atomic",
                "boundary": "isolated",
                "scf_converged": True,
                "scf_tolerance": 1e-10,
                "integration_grid_level": 2,
                "density_kind": "valence after subtraction of lowest oxygen 1s MOs",
                "energy_kind": "all-electron total energy including nuclear field energy",
                "forces_kind": "analytic total-energy derivative",
                "seconds": time.perf_counter() - started,
            }
            if args.check_field and np.any(job["external_field_au"]):
                eps = 2e-4  # bohr
                component = int(np.argmax(abs(np.asarray(job["external_field_au"]))))
                plus = json.loads(json.dumps(job))
                minus = json.loads(json.dumps(job))
                plus["positions_angstrom"][1][component] += eps * BOHR
                minus["positions_angstrom"][1][component] -= eps * BOHR
                fd = -(calculate(plus)[2] - calculate(minus)[2]) / (2 * eps)
                metadata["field_force_fd_component"] = component
                metadata["field_force_fd_error"] = float(abs(fd - force[1, component]))
                if metadata["field_force_fd_error"] > 2e-5:
                    raise RuntimeError(
                        f"Field force derivative validation failed: fd={fd}, force={force[1, component]}, error={metadata['field_force_fd_error']}"
                    )
            metadata["grid_response"] = True
            # Matched regular-grid valence density, in electron/bohr^3. This
            # supervises density directly without comparing unlike AO gauges.
            shape = (48, 48, 48)
            spacing = 0.6
            axis = (np.arange(48) + 0.5) * spacing
            points = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), -1).reshape(-1, 3)
            density = []
            for block in np.array_split(points, 28):
                ao = mol.eval_gto("GTOval_sph", block)
                density.append(np.einsum("gi,ij,gj->g", ao, valence, ao))
            density = np.concatenate(density)
            metadata["density_grid_shape"] = list(shape)
            metadata["density_grid_spacing_bohr"] = spacing
            metadata["density_integral_grid"] = float(density.sum() * spacing**3)
            # Actual AO reference matrices, not interchangeable with model AO
            # coefficients. Sample density when importing a matched model grid.
            np.savez_compressed(
                path,
                numbers=np.array(job["numbers"]),
                positions=mol.atom_coords(),
                energy=np.array(energy),
                forces=force,
                dipole_au=np.asarray(mf.dip_moment(unit="AU", verbose=0)),
                ao_density_matrix=dm,
                valence_ao_density_matrix=valence,
                ao_overlap=mol.intor_symmetric("int1e_ovlp"),
                metadata=np.asarray(json.dumps(metadata)),
            )
            # Keep density beside the reference matrix to make the convention
            # explicit and avoid serializing arbitrary Python objects.
            np.savez_compressed(
                root / f"{index:04d}.density.npz",
                density=density,
                grid_shape=np.array(shape),
                spacing=np.array(spacing),
            )
            failure = root / f"{index:04d}.failure.json"
            if failure.exists():
                failure.unlink()
            print(
                json.dumps(
                    {
                        "index": index,
                        "group": job["group"],
                        "energy": energy,
                        "seconds": metadata["seconds"],
                    }
                ),
                flush=True,
            )
        except Exception as exc:
            (root / f"{index:04d}.failure.json").write_text(
                json.dumps({"job": job, "error": str(exc)}, indent=2)
            )
            raise


if __name__ == "__main__":
    main()
