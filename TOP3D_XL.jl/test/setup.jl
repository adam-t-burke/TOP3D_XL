import Pkg

Pkg.activate(".")

packages = [
    "LinearAlgebra",
    "SparseArrays",
    "Printf",
    "Statistics",
    "NIfTI",
    "FileIO",
    "GeometryBasics",
    "ImageFiltering",
]

Pkg.add(packages)
