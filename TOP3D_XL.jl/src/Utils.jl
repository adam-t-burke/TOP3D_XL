module Utils

using SparseArrays
using LinearAlgebra

export CartesianMesh, solving_mission_partition, common_recover_halfe_nod_mat, common_include_adjacent_elements, accumarray

mutable struct CartesianMesh
    resX::Int
    resY::Int
    resZ::Int
    spanWidth::Int
    eleSize::Vector{Float64}
    numElements::Int
    numNodes::Int
    numDOFs::Int
    eNodMat::Matrix{Int}
    numNod2ElesVec::Vector{Int}
    freeDOFs::Union{BitVector, Nothing}
    fixedDOFs::Union{BitVector, Nothing}
    Ke::Matrix{Float64}
    eleModulus::Vector{Float64}
    Ks::Union{Array{Float64, 3}, Nothing} # Can be 3D array
    storingState::Int
    diagK::Union{Vector{Float64}, Nothing}
    eleMapBack::Vector{Int}
    eleMapForward::Vector{Int}
    nodMapBack::Vector{Int}
    nodMapForward::Vector{Int}
    solidNodeMapCoarser2Finer::Vector{Int}
    intermediateNumNodes::Int
    nodesOnBoundary::Vector{Int}
    boundaryNodeCoords::Vector{Float64} # Assuming coordinates are floats
    elementsOnBoundary::Vector{Int}
    boundaryEleFaces::Vector{Int} # Assuming integer indices
    elementUpwardMap::Matrix{Int}
    multiGridOperatorRI::Matrix{Float64}
    transferMat::Matrix{Int}
    transferMatCoeffi::Vector{Float64}

    # Constructor
    function CartesianMesh()
        new(0, 0, 0, 0, Float64[], 0, 0, 0, Matrix{Int}(undef, 0, 0), Vector{Int}(undef, 0),
            nothing, nothing, Matrix{Float64}(undef, 0, 0), Float64[], nothing, 0, nothing,
            Int[], Int[], Int[], Int[], Int[], 0, Int[], Float64[], Int[], Int[],
            Matrix{Int}(undef, 0, 0), Matrix{Float64}(undef, 0, 0), Matrix{Int}(undef, 0, 0), Float64[])
    end
end

function accumarray(subs, val, sz)
    A = zeros(eltype(val), sz...)
    for i in 1:length(val)
        A[subs[i]] += val[i]
    end
    return A
end

function solving_mission_partition(totalSize, blockSize)
    numBlocks = ceil(Int, totalSize / blockSize)
    blockIndex = ones(Int, numBlocks, 2)
    if numBlocks > 1
        blockIndex[1:numBlocks-1, 2] = (1:numBlocks-1) * blockSize
        blockIndex[2:numBlocks, 1] = blockIndex[2:numBlocks, 1] + blockIndex[1:numBlocks-1, 2]
    end
    blockIndex[numBlocks, 2] = totalSize
    return blockIndex
end

function common_recover_halfe_nod_mat(eNodMatHalf)
    if size(eNodMatHalf, 2) != 4
        return Matrix{Int}(undef, 0, 0)
    end
    numEles = size(eNodMatHalf, 1)
    eNodMat = zeros(Int32, numEles, 8)
    eNodMat[:, [3, 4, 7, 8]] = eNodMatHalf
    eNodMat[:, [2, 1, 6, 5]] = eNodMatHalf .+ 1
    return eNodMat
end

function common_include_adjacent_elements(iEleList, mesh)
    iEleListMapBack = mesh.eleMapBack[iEleList]

    resX = mesh.resX
    resY = mesh.resY
    resZ = mesh.resZ

    # The MATLAB code has a missing function Common_NodalizeDesignDomain, 
    # so we are translating the logic from the `else` block.
    numSeed = [resX - 1, resY - 1, resZ - 1]
    nx, ny, nz = numSeed[1], numSeed[2], numSeed[3]
    dd = [1 1 1; resX resY resZ]
    xSeed = dd[1, 1]:(dd[2, 1] - dd[1, 1])/nx:dd[2, 1]
    ySeed = dd[2, 2]:(dd[1, 2] - dd[2, 2])/ny:dd[1, 2]
    zSeed = dd[1, 3]:(dd[2, 3] - dd[1, 3])/nz:dd[2, 3]

    tmp = repeat(reshape(repeat(xSeed, inner = ny + 1), (nx + 1) * (ny + 1)), inner = nz + 1)
    eleX = tmp[iEleListMapBack]

    tmp = repeat(reshape(repeat(ySeed, outer = nx + 1), (nx + 1) * (ny + 1)), inner = nz + 1)
    eleY = tmp[iEleListMapBack]

    tmp = reshape(repeat(zSeed, inner = (nx + 1) * (ny + 1)), (nx + 1) * (ny + 1) * (nz + 1))
    eleZ = tmp[iEleListMapBack]

    tmpX = hcat(eleX .- 1, eleX .- 1, eleX .- 1, eleX, eleX, eleX, eleX .+ 1, eleX .+ 1, eleX .+ 1)
    tmpX = hcat(tmpX, tmpX, tmpX)
    tmpX = vec(tmpX)

    tmpY = hcat(eleY .+ 1, eleY, eleY .- 1, eleY .+ 1, eleY, eleY .- 1, eleY .+ 1, eleY, eleY .- 1)
    tmpY = hcat(tmpY, tmpY, tmpY)
    tmpY = vec(tmpY)

    tmpZ = repeat(eleZ, inner = 9)
    tmpZ = vcat(tmpZ .- 1, tmpZ, tmpZ .+ 1)
    tmpZ = vec(tmpZ)

    invalid_indices = findall(x -> x < 1 || x > resX, tmpX)
    invalid_indices = vcat(invalid_indices, findall(y -> y < 1 || y > resY, tmpY))
    invalid_indices = vcat(invalid_indices, findall(z -> z < 1 || z > resZ, tmpZ))
    unique!(invalid_indices)

    deleteat!(tmpX, invalid_indices)
    deleteat!(tmpY, invalid_indices)
    deleteat!(tmpZ, invalid_indices)

    oEleListMapBack = resX * resY * (tmpZ .- 1) + resY * (tmpX .- 1) + resY .- tmpY .+ 1
    oEleList = mesh.eleMapForward[oEleListMapBack]
    oEleList = oEleList[oEleList .> 0]
    unique!(oEleList)

    return oEleList
end

end # module
