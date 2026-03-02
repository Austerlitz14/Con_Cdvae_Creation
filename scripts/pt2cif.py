import argparse
from pathlib import Path

import torch
from pymatgen.core.lattice import Lattice
from pymatgen.core.structure import Structure


def load_pt(path: Path):
    try:
        return torch.load(path, map_location=torch.device("cpu"), weights_only=False)
    except TypeError:
        return torch.load(path, map_location=torch.device("cpu"))


def main():
    parser = argparse.ArgumentParser(description="Convert generated .pt file to CIF files")
    parser.add_argument("--input", required=True, help="Path to eval_gen_*.pt file")
    parser.add_argument(
        "--outdir",
        default=None,
        help="Output directory for CIF files (default: <pt_dir>/ciffile)",
    )
    args = parser.parse_args()

    pt_path = Path(args.input).resolve()
    if not pt_path.exists():
        raise FileNotFoundError(f"Input file not found: {pt_path}")

    outdir = Path(args.outdir).resolve() if args.outdir else pt_path.parent / "ciffile"
    outdir.mkdir(parents=True, exist_ok=True)

    data = load_pt(pt_path)
    lengths = data["lengths"].cpu().numpy().tolist()
    angles = data["angles"].cpu().numpy().tolist()
    num_atoms = data["num_atoms"].cpu().tolist()
    frac_coords = data["frac_coords"].cpu().numpy().tolist()
    atom_types = data["atom_types"].cpu().tolist()

    total = 0
    for i in range(len(num_atoms)):
        atom_offset = 0
        for a in range(len(num_atoms[i])):
            atom_num = int(num_atoms[i][a])
            length = lengths[i][a]
            angle = angles[i][a]
            atom_type = atom_types[i][atom_offset: atom_offset + atom_num]
            frac_coord = frac_coords[i][atom_offset: atom_offset + atom_num]

            lattice = Lattice.from_parameters(
                a=length[0], b=length[1], c=length[2],
                alpha=angle[0], beta=angle[1], gamma=angle[2]
            )
            structure = Structure(lattice, atom_type, frac_coord, to_unit_cell=True)

            file_name = f"{pt_path.stem}__{total}.cif"
            structure.to(filename=str(outdir / file_name))

            atom_offset += atom_num
            total += 1

    print(f"Saved {total} CIF files to: {outdir}")


if __name__ == "__main__":
    main()
