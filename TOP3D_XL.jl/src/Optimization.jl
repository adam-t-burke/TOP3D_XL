module Optimization

using LinearAlgebra
using SparseArrays
using Printf
using ..Utils
using ..Solvers
using ..FEA

export top_opti_setup_pde_filter_matrix_free, top_opti_conduct_pde_filtering_matrix_free, top_opti_compute_unit_compliance, top3d_xl_to

function mat_times_vec_matrix_free_B(uVec, mesh::CartesianMesh, PDEkernal)
    blockSize = 3.0e7
    Y = zeros(mesh.numNodes)
    blockIndex = solving_mission_partition(mesh.numElements, blockSize)

    for jj in 1:size(blockIndex, 1)
        rangeIndex = blockIndex[jj, 1]:blockIndex[jj, 2]
        iElesNodMat = mesh.eNodMat[rangeIndex, :]
        subDisVec = uVec[iElesNodMat]
        subDisVec = subDisVec * PDEkernal
        Y .+= accumarray(vec(iElesNodMat), vec(subDisVec), (mesh.numNodes,))
    end

    return Y
end

function top_opti_setup_pde_filter_matrix_free(filterRadius, mesh::CartesianMesh)
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
    dN1ds = -0.125 * (1 .- t) .* (1 .- p); dN2ds = 0.125 * (1 .- t) .* (1 .- p); dN3ds = 0.125 * (1 .+ t) .* (1 .- p);  dN4ds = -0.125 * (1 .+ t) .* (1 .- p)
    dN5ds = -0.125 * (1 .- t) .* (1 .+ p); dN6ds = 0.125 * (1 .- t) .* (1 .+ p); dN7ds = 0.125 * (1 .+ t) .* (1 .+ p);  dN8ds = -0.125 * (1 .+ t) .* (1 .+ p)
    dN1dt = -0.125 * (1 .- s) .* (1 .- p); dN2dt = -0.125 * (1 .+ s) .* (1 .- p); dN3dt = 0.125 * (1 .+ s) .* (1 .- p);  dN4dt = 0.125 * (1 .- s) .* (1 .- p)
    dN5dt = -0.125 * (1 .- s) .* (1 .+ p); dN6dt = -0.125 * (1 .+ s) .* (1 .+ p); dN7dt = 0.125 * (1 .+ s) .* (1 .+ p);  dN8dt = 0.125 * (1 .- s) .* (1 .+ p)
    dN1dp = -0.125 * (1 .- s) .* (1 .- t); dN2dp = -0.125 * (1 .+ s) .* (1 .- t); dN3dp = -0.125 * (1 .+ s) .* (1 .+ t); dN4dp = -0.125 * (1 .- s) .* (1 .+ t)
    dN5dp = 0.125 * (1 .- s) .* (1 .- t);  dN6dp = 0.125 * (1 .+ s) .* (1 .- t); dN7dp = 0.125 * (1 .+ s) .* (1 .+ t);  dN8dp = 0.125 * (1 .- s) .* (1 .+ t)
    dShape = zeros(3 * length(s), 8)
    dShape[1:3:end, :] = [dN1ds dN2ds dN3ds dN4ds dN5ds dN6ds dN7ds dN8ds]
    dShape[2:3:end, :] = [dN1dt dN2dt dN3dt dN4dt dN5dt dN6dt dN7dt dN8dt]
    dShape[3:3:end, :] = [dN1dp dN2dp dN3dp dN4dp dN5dp dN6dp dN7dp dN8dp]

    CellSize = 1.0
    detJ = (CellSize^3 / 8) * ones(8)
    wgt = w .* detJ
    KEF0 = dShape' * dShape
    KEF1 = N' * diagm(wgt) * N
    iRmin = (filterRadius * mesh.eleSize[1]) / 2 / sqrt(3)
    iKEF = iRmin^2 * KEF0 + KEF1
    PDEkernal = iKEF

    # Diagonal Preconditioner
    diagPrecond = zeros(mesh.numNodes)
    numElements = mesh.numElements
    diagKe = diag(PDEkernal)
    blockIndex = solving_mission_partition(numElements, 1.0e7)
    for jj in 1:size(blockIndex, 1)
        rangeIndex = blockIndex[jj, 1]:blockIndex[jj, 2]
        jElesNodMat = mesh.eNodMat[rangeIndex, :]
        diagKeBlock = repeat(diagKe, 1, length(rangeIndex))
        jElesNodMat = vec(jElesNodMat)
        diagKeBlockSingleDOF = vec(diagKeBlock)
        diagPrecond .+= accumarray(jElesNodMat, diagKeBlockSingleDOF, (mesh.numNodes,))
    end
    diagPrecond = 1.0 ./ diagPrecond

    return PDEkernal, diagPrecond
end

function top_opti_conduct_pde_filtering_matrix_free(src, PDEkernal, diagPrecond, mesh::CartesianMesh, tol, maxIT; passiveElements=nothing)
    PDEkernal_ = PDEkernal

    # Element to Node
    if passiveElements !== nothing
        src[passiveElements] .= 0.0
    end
    tmpVal = zeros(mesh.numNodes)
    values = src[:] * (1/8)

    for jj in 1:8
        tmpVal .+= accumarray(mesh.eNodMat[:, jj], values, (mesh.numNodes,))
    end
    src = tmpVal

    # Solving on Node
    PtV = x -> diagPrecond .* x
    tar, _ = solving_pcg(u -> mat_times_vec_matrix_free_B(u, mesh, PDEkernal_), PtV, src, tol / 10, maxIT, printP=[false, false])

    # Node to Element
    tmpVal = zeros(mesh.numElements)
    blockIndex = solving_mission_partition(mesh.numElements, 3.0e7)
    for jj in 1:size(blockIndex, 1)
        rangeIndex = blockIndex[jj, 1]:blockIndex[jj, 2]
        iElesNodMat = mesh.eNodMat[rangeIndex, :]
        tmpVal[rangeIndex] = sum(tar[iElesNodMat], dims=2)
    end
    tar = tmpVal * (1/8)

    return tar
end

function top_opti_compute_unit_compliance(U, mesh::CartesianMesh, objWeightingList)
    blockIndex = solving_mission_partition(mesh.numElements, 1.0e7)
    ceList = zeros(mesh.numElements, size(U, 2))

    for jj in 1:size(U, 2)
        ithU = U[:, jj]
        for ii in 1:size(blockIndex, 1)
            rangeIndex = blockIndex[ii, 1]:blockIndex[ii, 2]
            iReshapedU = zeros(length(rangeIndex), 24)
            iElesNodMat = mesh.eNodMat[rangeIndex, :]
            
            tmp = ithU[1:3:end]
            iReshapedU[:, 1:3:24] = tmp[iElesNodMat]
            tmp = ithU[2:3:end]
            iReshapedU[:, 2:3:24] = tmp[iElesNodMat]
            tmp = ithU[3:3:end]
            iReshapedU[:, 3:3:24] = tmp[iElesNodMat]

            ceList[rangeIndex, jj] = sum((iReshapedU * mesh.Ke) .* iReshapedU, dims=2)
        end
    end

    return ceList * objWeightingList
end

function top3d_xl_to(inputModel, V0, nLoop, rMin, tol, maxIT, outPath, coarsestResolutionControl, poissonRatio, cellSize, SIMPpenalty, modulusMin, modulus)
    tStartTotal = time()
    if !isdir(outPath)
        mkdir(outPath)
    end

    # 0. Modeling
    mesh, fixingCond, loadingCond, objWeightingList, passiveElements = create_voxel_fea_model(inputModel, coarsestResolutionControl)

    # 1. Pre. FEA
    # ... (FEA_ApplyBoundaryCondition and FEA_SetupVoxelBased to be translated)
    F = spzeros(mesh.numDOFs, length(loadingCond))
    # ...

    # 2. Setup PDE filter
    PDEkernal, diagPrecond = top_opti_setup_pde_filter_matrix_free(rMin, mesh)

    # 3. prepare optimizer
    numElements = mesh.numElements
    x = fill(V0, numElements)
    xPhys = copy(x)
    loop = 0
    change = 1.0
    sharpness = 1.0
    lssIts = []
    cHist = []
    volHist = []
    sharpHist = []
    consHist = []
    tHist = []

    # 4. Evaluate Compliance of Fully Solid Domain
    U = zeros(mesh.numDOFs, size(F, 2))
    mesh.eleModulus = fill(modulus, numElements)
    # ... (Solving_AssembleFEAstencil to be translated)
    # ... (Solving_PCG call)
    ceList = top_opti_compute_unit_compliance(U, mesh, objWeightingList)
    cSolid = mesh.eleModulus' * ceList
    @printf("Compliance of Fully Solid Domain: %16.6e\n", cSolid)

    SIMP(xPhys) = modulusMin .+ xPhys.^SIMPpenalty * (modulus - modulusMin)
    DeSIMP(xPhys) = SIMPpenalty * (modulus - modulusMin) * xPhys.^(SIMPpenalty - 1)

    # 5. optimization
    while loop < nLoop && change > 0.0001 && sharpness > 0.01
        perIteCost = time()
        loop += 1

        # 5.1 & 5.2 FEA, objective and sensitivity analysis
        mesh.eleModulus = SIMP(xPhys)
        # ... (Solving_AssembleFEAstencil call)
        # ... (Solving_PCG call)
        ceList = top_opti_compute_unit_compliance(U, mesh, objWeightingList)
        ceNorm = ceList / cSolid
        cObj = mesh.eleModulus' * ceNorm
        cDesign = cObj * cSolid
        V = sum(xPhys) / numElements
        dc = -DeSIMP(xPhys) .* ceNorm
        dv = ones(numElements)

        # 5.3 filtering/modification of sensitivity
        dc = top_opti_conduct_pde_filtering_matrix_free(x .* dc, PDEkernal, diagPrecond, mesh, tol, maxIT)

        # 5.4 solve the optimization probelm
        fval = mean(xPhys) - V0
        l1, l2 = 0, 1e9
        move = 0.2
        while (l2 - l1) / (l1 + l2) > 1e-6
            lmid = 0.5 * (l2 + l1)
            xnew = max.(0, max.(x .- move, min.(1, min.(x .+ move, x .* sqrt.(-dc ./ dv / lmid)))))
            xnew[passiveElements] .= 1.0
            gt = fval + sum(dv .* (xnew - x))
            if gt > 0
                l1 = lmid
            else
                l2 = lmid
            end
        end
        change = maximum(abs.(xnew - x))
        x = xnew

        # ... (density filtering)
        xPhys = xnew
        xPhys[passiveElements] .= 1.0
        sharpness = 4 * sum(xPhys .* (1 .- xPhys)) / numElements

        # ... (history and printing)
    end

    # ... (output and visualization)
end

end # module

