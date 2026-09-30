"""INRIA-style Gaussian splat PLY files, as Brush writes them.

A splat is a structured numpy array with one field per PLY property (x, y, z,
f_dc_*, f_rest_*, opacity, scale_*, rot_*), parsed by name because Brush
does not promise a property order. Opacity is a logit and scales are logs,
as stored.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

_PLY_TYPES = {
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
    "uchar": "u1", "uint8": "u1", "char": "i1", "int8": "i1",
    "ushort": "u2", "uint16": "u2", "short": "i2", "int16": "i2",
    "uint": "u4", "uint32": "u4", "int": "i4", "int32": "i4",
}
_NAMES = {v: k for k, v in reversed(_PLY_TYPES.items())}  # f4 -> float, ...


def read_ply(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        if f.readline().strip() != b"ply":
            raise ValueError(f"{path} is not a PLY file")
        fields, count, fmt = [], 0, ""
        in_vertex = False
        while (line := f.readline().decode("ascii").strip()) != "end_header":
            parts = line.split()
            if parts[0] == "format":
                fmt = parts[1]
            elif parts[0] == "element":
                in_vertex = parts[1] == "vertex"
                if in_vertex:
                    count = int(parts[2])
            elif parts[0] == "property" and in_vertex:
                if parts[1] == "list":
                    raise ValueError(f"{path}: list properties are not supported")
                fields.append((parts[2], _PLY_TYPES[parts[1]]))
        if fmt != "binary_little_endian":
            raise ValueError(f"{path}: only binary_little_endian PLY is supported, not {fmt}")
        dtype = np.dtype([(name, "<" + t) for name, t in fields])
        return np.fromfile(f, dtype=dtype, count=count)


def write_ply(path: Path, splat: np.ndarray) -> None:
    header = ["ply", "format binary_little_endian 1.0", f"element vertex {len(splat)}"]
    for name in splat.dtype.names:
        header.append(f"property {_NAMES[splat.dtype[name].str.lstrip('<>|=')]} {name}")
    header.append("end_header")
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "wb") as f:
        f.write(("\n".join(header) + "\n").encode("ascii"))
        splat.astype(splat.dtype.newbyteorder("<")).tofile(f)
    tmp.replace(path)


def positions(splat: np.ndarray) -> np.ndarray:
    return np.stack([splat["x"], splat["y"], splat["z"]], axis=1).astype(np.float64)


def opacities(splat: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-splat["opacity"].astype(np.float64)))


def scales(splat: np.ndarray) -> np.ndarray:
    """World-space standard deviations along the three local axes."""
    return np.exp(np.stack([splat[f"scale_{i}"] for i in range(3)], axis=1).astype(np.float64))


def rotations(splat: np.ndarray) -> np.ndarray:
    """Unit quaternions (w, x, y, z)."""
    q = np.stack([splat[f"rot_{i}"] for i in range(4)], axis=1).astype(np.float64)
    return q / np.linalg.norm(q, axis=1, keepdims=True).clip(1e-12)


def sh_degree(splat: np.ndarray) -> int:
    n_rest = sum(1 for n in splat.dtype.names if n.startswith("f_rest_"))
    return {0: 0, 9: 1, 24: 2, 45: 3}[n_rest]
