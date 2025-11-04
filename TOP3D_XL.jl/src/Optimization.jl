module Optimization

using LinearAlgebra
using ..Utils

export top_opti_set_passive_elements, top_opti_compute_unit_compliance, top_opti_setup_pde_filter_matrix_free, top_opti_conduct_pde_filtering_matrix_free, mat_times_vec_matrix_free_b

function top_opti_set_passive_elements(meshHierarchy, loadingCond, fixingCond, passiveElements, numLayerboundary, numLayerLoads, numLayerFixation)
    numLayerboundary = round(Int, numLayerboundary)
    numLayerLoads = round(Int, numLayerLoads)
    numLayerFixation = round(Int, numLayerFixation)
    existingPassiveElements = passiveElements

    passiveElementsOnBoundary = []
    if numLayerboundary > 0
        index = 1
        while index <= numLayerboundary
            if index == 1
                passiveElementsOnBoundary = meshHierarchy[1].elementsOnBoundary
            else
                passiveElementsOnBoundary = common_include_adjacent_elements(passiveElementsOnBoundary, meshHierarchy[1])
            end
            index += 1
        end
    end

    passiveElementsNearLoads = []
    passiveElementsNearFixation = []
    if numLayerLoads > 0 || numLayerFixation > 0
        # Relate Elements to Nodes
        allElements = zeros(Int, meshHierarchy[1].numElements)
        allElements[meshHierarchy[1].elementsOnBoundary] .= 1
        nodeStruct = [Int[] for _ in 1:meshHierarchy[1].numNodes]
        for ii in 1:meshHierarchy[1].numElements
            if allElements[ii] == 1
                iNodes = meshHierarchy[1].eNodMat[ii, :]
                for jj in 1:8
                    push!(nodeStruct[iNodes[jj]], ii)
                end
            end
        end

        # Extract Elements near Loads
        if numLayerLoads > 0
            passiveElementsNearLoads = []
            for ii in 1:length(loadingCond)
                iLoad = loadingCond[ii]
                loadedNodes = meshHierarchy[1].nodesOnBoundary[iLoad[:, 1]]
                allLoadedNodes = nodeStruct[loadedNodes]
                passiveElementsNearLoads = vcat(passiveElementsNearLoads, unique(vcat(allLoadedNodes...)))
            end
            index = 2
            while index <= numLayerLoads
                passiveElementsNearLoads = common_include_adjacent_elements(passiveElementsNearLoads, meshHierarchy[1])
                index += 1
            end
        end

        # Extract Elements near Fixation
        if numLayerFixation > 0
            fixedNodes = meshHierarchy[1].nodesOnBoundary[fixingCond[:, 1]]
            allFixedNodes = nodeStruct[fixedNodes]
            passiveElementsNearFixation = unique(vcat(allFixedNodes...))
            index = 2
            while index <= numLayerFixation
                passiveElementsNearFixation = common_include_adjacent_elements(passiveElementsNearFixation, meshHierarchy[1])
                index += 1
            end
        end

        passiveElements = unique(vcat(existingPassiveElements, passiveElementsOnBoundary, passiveElementsNearLoads, passiveElementsNearFixation))
    else
        passiveElements = unique(vcat(existingPassiveElements, passiveElementsOnBoundary))
    end

    return passiveElements
end

function common_include_adjacent_elements(iEleList, mesh)
    # To be implemented
    return iEleList
end

function top_opti_compute_unit_compliance(U, meshHierarchy, objWeightingList)
    blockSize = 1.0e7
    ceList = zeros(meshHierarchy[1].numElements, size(U, 2))
    for jj in 1:size(U, 2)
        ithU = U[:, jj]
        blockIndex = solving_mission_partition(meshHierarchy[1].numElements, blockSize)
        for ii in 1:size(blockIndex, 1)
            rangeIndex = blockIndex[ii, 1]:blockIndex[ii, 2]
            iReshapedU = zeros(length(rangeIndex), 24)
            iElesNodMat = meshHierarchy[1].eNodMat[rangeIndex, :]
            tmp = ithU[1:3:end]
            iReshapedU[:, 1:3:24] = tmp[iElesNodMat]
            tmp = ithU[2:3:end]
            iReshapedU[:, 2:3:24] = tmp[iElesNodMat]
            tmp = ithU[3:3:end]
            iReshapedU[:, 3:3:24] = tmp[iElesNodMat]
            ceList[rangeIndex, jj] = sum((iReshapedU * meshHierarchy[1].Ke) .* iReshapedU, dims=2)
        end
    end
    ceList = ceList * objWeightingList
    return ceList
end

function top_opti_setup_pde_filter_matrix_free(filterRadius, meshHierarchy)
    # Gaussian Points
    s = [-1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0, -1.0] / sqrt(3)
    t = [-1.0, -1.0, 1.0, 1.0, -1.0, -1.0, 1.0, 1.0] / sqrt(3)
    p = [-1.0, -1.0, -1.0, -1.0, 1.0, 1.0, 1.0, 1.0] / sqrt(3)
    w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]

    # Trilinear Shape Functions (N)
    N = zeros(length(s), 8)
    N[:, 1] = 0.125 * (1 .- s) .* (1 .- t) .* (1 .- p)
    N[:, 2] = 0.125 * (1 .+ s) .* (1 .- t) .* (1 .- p)
    N[:, 3] = 0.125 * (1 .+ s) .* (1 .+ t) .* (1 .- p)
    N[:, 4] = 0.125 * (1 .- s) .* (1 .+ t) .* (1 .- p)
    N[:, 5] = 0.125 * (1 .- s) .* (1 .- t) .* (1 .+ p)
    N[:, 6] = 0.125 * (1 .+ s) .* (1 .- t) .* (1 .+ p)
    N[:, 7] = 0.125 * (1 .+ s) .* (1 .+ t) .* (1 .+ p)
    N[:, 8] = 0.125 * (1 .- s) .* (1 .+ t) .* (1 .+ p)

    # dN
    dN1ds = -0.125 * (1 .- t) .* (1 .- p); dN2ds = 0.125 * (1 .- t) .* (1 .- p); dN3ds = 0.125 * (1 .+ t) .* (1 .- p); dN4ds = -0.125 * (1 .+ t) .* (1 .- p)
    dN5ds = -0.125 * (1 .- t) .* (1 .+ p); dN6ds = 0.125 * (1 .- t) .* (1 .+ p); dN7ds = 0.125 * (1 .+ t) .* (1 .+ p); dN8ds = -0.125 * (1 .+ t) .* (1 .+ p)
    dN1dt = -0.125 * (1 .- s) .* (1 .- p); dN2dt = -0.125 * (1 .+ s) .* (1 .- p); dN3dt = 0.125 * (1 .+ s) .* (1 .- p); dN4dt = 0.125 * (1 .- s) .* (1 .- p)
    dN5dt = -0.125 * (1 .- s) .* (1 .+ p); dN6dt = -0.125 * (1 .+ s) .* (1 .+ p); dN7dt = 0.125 * (1 .+ s) .* (1 .+ p); dN8dt = 0.125 * (1 .- s) .* (1 .+ p)
    dN1dp = -0.125 * (1 .- s) .* (1 .- t); dN2dp = -0.125 * (1 .+ s) .* (1 .- t); dN3dp = -0.125 * (1 .+ s) .* (1 .+ t); dN4dp = -0.125 * (1 .- s) .* (1 .+ t)
    dN5dp = 0.125 * (1 .- s) .* (1 .- t); dN6dp = 0.125 * (1 .+ s) .* (1 .- t); dN7dp = 0.125 * (1 .+ s) .* (1 .+ t); dN8dp = 0.125 * (1 .- s) .* (1 .+ t)
    dShape = zeros(3 * length(s), 8)
    dShape[1:3:end, :] = [dN1ds dN2ds dN3ds dN4ds dN5ds dN6ds dN7ds dN8ds]
    dShape[2:3:end, :] = [dN1dt dN2dt dN3dt dN4dt dN5dt dN6dt dN7dt dN8dt]
    dShape[3:3:end, :] = [dN1dp dN2dp dN3dp dN4dp dN5dp dN6dp dN7dp dN8dp]

    # Jacobian Matrix
    CellSize = 1
    detJ = CellSize^3 / 8 * ones(8)
    wgt = w .* detJ
    KEF0 = dShape' * dShape
    KEF1 = N' * diagm(wgt) * N
    iRmin = (filterRadius * meshHierarchy[1].eleSize[1]) / 2 / sqrt(3)
    iKEF = iRmin^2 * KEF0 + KEF1
    PDEkernal = iKEF

    # Diagonal Preconditioner
    diagPrecond = zeros(meshHierarchy[1].numNodes)
    numElements = meshHierarchy[1].numElements
    diagKe = diag(PDEkernal)
    blockIndex = solving_mission_partition(numElements, 1.0e7)
    for jj in 1:size(blockIndex, 1)
        rangeIndex = blockIndex[jj, 1]:blockIndex[jj, 2]
        jElesNodMat = meshHierarchy[1].eNodMat[rangeIndex, :]
        diagKeBlock = repeat(diagKe, 1, length(rangeIndex))
        jElesNodMat = vec(jElesNodMat)
        diagKeBlockSingleDOF = vec(diagKeBlock)
        diagPrecond .+= accumarray(jElesNodMat, diagKeBlockSingleDOF, (meshHierarchy[1].numNodes,))
    end
    diagPrecond = 1 ./ diagPrecond

    return PDEkernal, diagPrecond
end

function top_opti_conduct_pde_filtering_matrix_free(src, PDEkernal, diagPrecond, meshHierarchy, passiveElements=nothing)
    PDEkernal_ = PDEkernal

    # Element to Node
    if passiveElements !== nothing
        src[passiveElements] .= 0.0
    end
    tmpVal = zeros(meshHierarchy[1].numNodes)
    values = src[:] * (1/8)

    for jj in 1:8
        tmpVal .+= accumarray(meshHierarchy[1].eNodMat[:, jj], values, (meshHierarchy[1].numNodes,))
    end
    src = tmpVal

    # Solving on Node
    PtV(x) = diagPrecond .* x
    tar, _ = solving_pcg(mat_times_vec_matrix_free_b, PtV, src, tol_ / 10, maxIT_, [false, false])

    # Node to Element
    tmpVal = zeros(meshHierarchy[1].numElements)
    blockIndex = solving_mission_partition(meshHierarchy[1].numElements, 3.0e7)
    for jj in 1:size(blockIndex, 1)
        rangeIndex = blockIndex[jj, 1]:blockIndex[jj, 2]
        iElesNodMat = meshHierarchy[1].eNodMat[rangeIndex, :]
        tmpVal[rangeIndex] = sum(tar[iElesNodMat], dims=2)
    end
    tar = tmpVal * (1/8)

    return tar
end

function mat_times_vec_matrix_free_b(uVec, meshHierarchy, PDEkernal)
    blockSize = 3.0e7
    Y = zeros(meshHierarchy[1].numNodes)
    PDEfilterkernal = PDEkernal
    blockIndex = solving_mission_partition(meshHierarchy[1].numElements, blockSize)
    for jj in 1:size(blockIndex, 1)
        rangeIndex = blockIndex[jj, 1]:blockIndex[jj, 2]
        iElesNodMat = meshHierarchy[1].eNodMat[rangeIndex, :]
        subDisVec = uVec[iElesNodMat]
        subDisVec = subDisVec * PDEfilterkernal
        Y .+= accumarray(vec(iElesNodMat), vec(subDisVec), (meshHierarchy[1].numNodes,))
    end
    return Y
end

end # module