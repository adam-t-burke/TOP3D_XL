module TOP3D_XL

using LinearAlgebra
using SparseArrays
using Printf
using Statistics
using Makie

include("Utils.jl")
include("FEA.jl")
include("Solvers.jl")
include("Optimization.jl")
include("IO.jl")
include("Visualization.jl")

# Export main function
export TOP3D_XL_main

function TOP3D_XL_main()
    # Main entry point for the TOP3D_XL project
    println("TOP3D_XL Julia version is running!")
end

end # module
