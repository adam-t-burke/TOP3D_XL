module Solvers

using LinearAlgebra
using SparseArrays
using Printf
using ..Utils

export solving_pcg, solving_K_by_U_matrix_free, solving_v_cycle, solving_building_mesh_hierarchy, solving_setup_ke_with_fixed_dofs, solving_apply_bc_on_ele_stiff_mat_b, solving_mission_partition, solving_assemble_fea_stencil, solving_operator4multi_grid_restriction_and_interpolation, solving_sub_ele_nod_mat

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
    blockIndex[1:numBlocks-1, 2] = (1:numBlocks-1) * blockSize
    blockIndex[2:numBlocks, 1] = blockIndex[2:numBlocks, 1] + blockIndex[1:numBlocks-1, 2]
    blockIndex[numBlocks, 2] = totalSize
    return blockIndex
end

function solving_pcg(AtX, PtV, b, tol, maxIT; printP=[false, false], y=zeros(size(b)))
    normB = norm(b)
    its = 0
    rVec1 = b - AtX(y)
    zVec = PtV(rVec1)
    pVec = zVec
    x1Val = zVec' * rVec1

    for its in 1:maxIT
        vVec = AtX(pVec)
        lambda = x1Val / (pVec' * vVec)
        y .+= lambda * pVec
        rVec1 .-= lambda * vVec
        resnorm = norm(rVec1) / normB

        if printP[1]
            @printf(" It.: %4i Res.: %16.6e\n", its, resnorm)
        end

        if resnorm < tol
            if printP[2]
                @printf("CG solver converged at iteration %5i to a solution with relative residual %16.6e\n", its, resnorm)
            end
            break
        end

        zVec = PtV(rVec1)
        x2Val = zVec' * rVec1
        pVec = zVec + (x2Val / x1Val) * pVec
        x1Val = x2Val
    end

    if its == maxIT
        @warn "Exceeded the maximum number of iterations"
        @printf("The iterative process stops at residual = %10.4f\n", resnorm)
    end

    return y, its
end

function solving_K_by_U_matrix_free(uVec, mesh::CartesianMesh, uniqueKesFixed, uniqueKesFree, mapUniqueKes; iLevel=1)
    blockSize = 1.0e7
    
    if iLevel == 1
        Ke = mesh.Ks
        uVec = reshape(uVec, 3, mesh.numNodes)'
        eleModulus = mesh.eleModulus
        Y = zeros(mesh.numNodes, 3)

        blockIndex = solving_mission_partition(mesh.numElements, blockSize)

        for jj in 1:size(blockIndex, 1)
            rangeIndex = blockIndex[jj, 1]:blockIndex[jj, 2]
            iElesNodMat = mesh.eNodMat[rangeIndex, :]
            iIntermediateModulus = eleModulus[rangeIndex]

            subDisVec = zeros(size(iElesNodMat, 1), 24)
            tmp = uVec[:, 1]
            subDisVec[:, 1:3:24] = tmp[iElesNodMat]
            tmp = uVec[:, 2]
            subDisVec[:, 2:3:24] = tmp[iElesNodMat]
            tmp = uVec[:, 3]
            subDisVec[:, 3:3:24] = tmp[iElesNodMat]

            iTarEles = mapUniqueKes[rangeIndex]
            eleWithFixedDOFs = findall(x -> x > 0, iTarEles)
            eleWithFixedDOFsLocal = iTarEles[eleWithFixedDOFs]
            numTarEles = length(eleWithFixedDOFs)
            subDisVecUnique = subDisVec[eleWithFixedDOFs, :]

            for kk in 1:numTarEles
                ss = eleWithFixedDOFsLocal[kk]
                K_free = reshape(uniqueKesFree[:, ss], 24, 24)
                K_fixed = reshape(uniqueKesFixed[:, ss], 24, 24)
                subDisVecUnique[kk, :] = subDisVecUnique[kk, :] * (K_free * iIntermediateModulus[eleWithFixedDOFs[kk]] + K_fixed)
            end

            subDisVec = subDisVec * Ke .* iIntermediateModulus
            subDisVec[eleWithFixedDOFs, :] = subDisVecUnique

            tmp = subDisVec[:, 1:3:24]
            Y[:, 1] .+= accumarray(vec(iElesNodMat), vec(tmp), (mesh.numNodes,))
            tmp = subDisVec[:, 2:3:24]
            Y[:, 2] .+= accumarray(vec(iElesNodMat), vec(tmp), (mesh.numNodes,))
            tmp = subDisVec[:, 3:3:24]
            Y[:, 3] .+= accumarray(vec(iElesNodMat), vec(tmp), (mesh.numNodes,))
        end

        return vec(Y')
    else
        # Handle other levels later
        return zeros(size(uVec))
    end
end

function solving_restrict_residual(rFiner, ii, meshHierarchy)
    rFiner = reshape(rFiner, 3, meshHierarchy[ii-1].numNodes)'
    rFiner1 = zeros(meshHierarchy[ii].intermediateNumNodes, 3)
    rFiner1[meshHierarchy[ii].solidNodeMapCoarser2Finer, :] = rFiner
    rFiner1 = rFiner1 ./ meshHierarchy[ii].transferMatCoeffi
    rCoaser = zeros(meshHierarchy[ii].numNodes, 3)

    for jj in 1:3
        tmp = rFiner1[:, jj]
        tmp = tmp[meshHierarchy[ii].transferMat]
        tmp = tmp' * meshHierarchy[ii].multiGridOperatorRI
        rCoaser[:, jj] = accumarray(vec(meshHierarchy[ii].eNodMat), vec(tmp), (meshHierarchy[ii].numNodes,))
    end

    return vec(rCoaser')
end

function solving_interpolation_deviation(xCoarser, ii, meshHierarchy)
    xCoarser = reshape(xCoarser, 3, meshHierarchy[ii].numNodes)'
    xFiner = zeros(meshHierarchy[ii].intermediateNumNodes, 3)
    transferMat = vec(meshHierarchy[ii].transferMat)

    for jj in 1:3
        tmp = xCoarser[:, jj]
        tmp = tmp[meshHierarchy[ii].eNodMat]
        tmp1 = meshHierarchy[ii].multiGridOperatorRI * tmp'
        xFiner[:, jj] = accumarray(transferMat, vec(tmp1), (meshHierarchy[ii].intermediateNumNodes,))
    end

    xFiner = xFiner ./ meshHierarchy[ii].transferMatCoeffi
    xFiner = xFiner[meshHierarchy[ii].solidNodeMapCoarser2Finer, :]
    return vec(xFiner')
end

function solving_v_cycle(r, meshHierarchy, weightFactorJacobi, typeVcycle, cholFac, cholPermut)
    numLevels = length(meshHierarchy)
    varVcycle = [ (x = Float64[], r = Float64[]) for _ in 1:numLevels ]
    varVcycle[1] = (x = Float64[], r = r)

    # Restriction
    for ii in 1:numLevels-1
        varVcycle[ii] = (x = weightFactorJacobi * (varVcycle[ii].r ./ meshHierarchy[ii].diagK), r = varVcycle[ii].r)
        if typeVcycle == "Adapted"
            varVcycle[ii+1] = (x = Float64[], r = solving_restrict_residual(varVcycle[ii].r, ii + 1, meshHierarchy))
        else # Standard
            d = varVcycle[ii].r - solving_K_by_U_matrix_free(varVcycle[ii].x, meshHierarchy[ii], uniqueKesFixed, uniqueKesFree, mapUniqueKes, iLevel=ii)
            varVcycle[ii+1] = (x = Float64[], r = solving_restrict_residual(d, ii + 1, meshHierarchy))
        end
    end

    varVcycle[end] = (x = cholPermut * (cholFac' \ (cholFac \ (cholPermut' * varVcycle[end].r))), r = varVcycle[end].r)

    # Interpolation
    for ii in numLevels-1:-1:1
        varVcycle[ii] = (x = varVcycle[ii].x + solving_interpolation_deviation(varVcycle[ii+1].x, ii + 1, meshHierarchy), r = varVcycle[ii].r)
        if typeVcycle == "Adapted"
            varVcycle[ii] = (x = varVcycle[ii].x + weightFactorJacobi * varVcycle[ii].r ./ meshHierarchy[ii].diagK, r = varVcycle[ii].r)
        else # Standard
            varVcycle[ii] = (x = varVcycle[ii].x + weightFactorJacobi * (varVcycle[ii].r - solving_K_by_U_matrix_free(varVcycle[ii].x, meshHierarchy[ii], uniqueKesFixed, uniqueKesFree, mapUniqueKes, iLevel=ii)) ./ meshHierarchy[ii].diagK, r = varVcycle[ii].r)
        end
    end

    return varVcycle[1].x
end

function solving_assemble_fea_stencil(meshHierarchy, numLevels, cholFac, cholPermut, isThisEle2ndLevelIncludingFixedDOFsOn1stLevel, uniqueKesFixed, uniqueKesFree, sonElesWithFixedDOFs, mapUniqueKes)
    # Compute 'Ks' on Coarser Levels
    reOrdering = [1, 9, 17, 2, 10, 18, 3, 11, 19, 4, 12, 20, 5, 13, 21, 6, 14, 22, 7, 15, 23, 8, 16, 24]
    for ii in 2:numLevels
        spanWidth = meshHierarchy[ii].spanWidth
        interpolatingKe = solving_operator4multi_grid_restriction_and_interpolation("inDOF", spanWidth)
        eNodMat4Finer2Coarser = solving_sub_ele_nod_mat(spanWidth)
        rowIndice, colIndice, _ = findnz(ones(24, 24))
        eDofMat4Finer2Coarser = hcat(3 * eNodMat4Finer2Coarser .- 2, 3 * eNodMat4Finer2Coarser .- 1, 3 * eNodMat4Finer2Coarser)
        eDofMat4Finer2Coarser = eDofMat4Finer2Coarser[:, reOrdering]
        iK = eDofMat4Finer2Coarser[:, rowIndice]'
        jK = eDofMat4Finer2Coarser[:, colIndice]'
        numProjectNodes = (spanWidth + 1)^3
        numProjectDOFs = numProjectNodes * 3
        meshHierarchy[ii].storingState = 1
        meshHierarchy[ii].Ke = meshHierarchy[ii-1].Ke * spanWidth
        numElements = meshHierarchy[ii].numElements
        diagK = zeros(meshHierarchy[ii].numNodes, 3)
        finerKes = zeros(24 * 24, spanWidth^3)
        elementUpwardMap = meshHierarchy[ii].elementUpwardMap

        # Compute Element Stiffness Matrices on Coarser Levels
        if ii == 2
            iKe = meshHierarchy[ii-1].Ke
            iKs = vec(iKe)
            eleModulus = meshHierarchy[1].eleModulus
            Ks = repeat(meshHierarchy[ii].Ke, 1, 1, numElements)
            # This part is parallelized in MATLAB, but we will do it sequentially for now
            for jj in 1:numElements
                sonEles = elementUpwardMap[jj, :]
                solidEles = findall(x -> x != 0, sonEles)
                sK = finerKes
                sK[:, solidEles] = iKs .* eleModulus[sonEles[solidEles]]
                idx = isThisEle2ndLevelIncludingFixedDOFsOn1stLevel[jj]
                if idx > 0
                    sonElesLocal = sonElesWithFixedDOFs[idx].arr
                    sonElesGlobal = sonEles[sonElesLocal]
                    sonElesIntermediate = mapAllElements2ElementsWithFixedDOFs[sonElesGlobal]
                    sK[:, sonElesLocal] = uniqueKeListFreeDOFs[:, sonElesIntermediate] .* eleModulus[sonElesGlobal] + uniqueKeListFixedDOFs[:, sonElesIntermediate]
                end
                tmpK = sparse(iK, jK, sK, numProjectDOFs, numProjectDOFs)
                tmpK = interpolatingKe' * tmpK * interpolatingKe
                Ks[:, :, jj] = Matrix(tmpK)
            end
        else
            KsPrevious = Ks
            Ks = repeat(meshHierarchy[ii].Ke, 1, 1, numElements)
            # This part is parallelized in MATLAB, but we will do it sequentially for now
            for jj in 1:numElements
                iFinerEles = elementUpwardMap[jj, :]
                solidEles = findall(x -> x != 0, iFinerEles)
                iFinerEles = iFinerEles[solidEles]
                sK = finerKes
                tarKes = KsPrevious[:, :, iFinerEles]
                for kk in 1:length(solidEles)
                    sK[:, solidEles[kk]] = vec(tarKes[:, :, kk])
                end
                tmpK = sparse(iK, jK, sK, numProjectDOFs, numProjectDOFs)
                tmpK = interpolatingKe' * tmpK * interpolatingKe
                Ks[:, :, jj] = Matrix(tmpK)
            end
        end
        meshHierarchy[ii].Ks = Ks

        # Initialize Jacobian Smoother on Coarser Levels
        if ii < numLevels
            blockIndex = solving_mission_partition(numElements, 1.0e7)
            for jj in 1:size(blockIndex, 1)
                rangeIndex = blockIndex[jj, 1]:blockIndex[jj, 2]
                jElesNodMat = meshHierarchy[ii].eNodMat[rangeIndex, :]
                jKs = Ks[:, :, rangeIndex]
                jKs = reshape(jKs, 24 * 24, length(rangeIndex))
                diagKeBlock = jKs[1:25:end, :]
                jElesNodMat = vec(jElesNodMat)
                diagKeBlockSingleDOF = vec(diagKeBlock[1:3:end, :])
                diagK[:, 1] .+= accumarray(jElesNodMat, diagKeBlockSingleDOF, (meshHierarchy[ii].numNodes,))
                diagKeBlockSingleDOF = vec(diagKeBlock[2:3:end, :])
                diagK[:, 2] .+= accumarray(jElesNodMat, diagKeBlockSingleDOF, (meshHierarchy[ii].numNodes,))
                diagKeBlockSingleDOF = vec(diagKeBlock[3:3:end, :])
                diagK[:, 3] .+= accumarray(jElesNodMat, diagKeBlockSingleDOF, (meshHierarchy[ii].numNodes,))
            end
            meshHierarchy[ii].diagK = vec(diagK')
        end
    end

    # Initialize Jacobian Smoother on Finest Level
    diagK = zeros(meshHierarchy[1].numNodes, 3)
    numElements = meshHierarchy[1].numElements
    diagKe = diag(meshHierarchy[1].Ke)
    eleModulus = meshHierarchy[1].eleModulus
    blockIndex = solving_mission_partition(numElements, 1.0e7)
    for jj in 1:size(blockIndex, 1)
        rangeIndex = blockIndex[jj, 1]:blockIndex[jj, 2]
        jElesNodMat = meshHierarchy[1].eNodMat[rangeIndex, :]
        jEleModulus = eleModulus[rangeIndex]
        diagKeBlock = diagKe .* jEleModulus'
        jTarEles = mapUniqueKes[rangeIndex]
        eleWithFixedDOFs = findall(x -> x > 0, jTarEles)
        eleWithFixedDOFsLocal = jTarEles[eleWithFixedDOFs]
        numTarEles = length(eleWithFixedDOFs)
        for kk in 1:numTarEles
            kKeFreeDOFs = reshape(uniqueKesFree[:, eleWithFixedDOFsLocal[kk]], 24, 24)
            kKeFixedDOFs = reshape(uniqueKesFixed[:, eleWithFixedDOFsLocal[kk]], 24, 24)
            diagKeBlock[:, eleWithFixedDOFs[kk]] = diag(kKeFreeDOFs) * jEleModulus[eleWithFixedDOFs[kk]] + diag(kKeFixedDOFs)
        end
        jElesNodMat = vec(jElesNodMat)
        diagKeBlockSingleDOF = vec(diagKeBlock[1:3:end, :])
        diagK[:, 1] .+= accumarray(jElesNodMat, diagKeBlockSingleDOF, (meshHierarchy[1].numNodes,))
        diagKeBlockSingleDOF = vec(diagKeBlock[2:3:end, :])
        diagK[:, 2] .+= accumarray(jElesNodMat, diagKeBlockSingleDOF, (meshHierarchy[1].numNodes,))
        diagKeBlockSingleDOF = vec(diagKeBlock[3:3:end, :])
        diagK[:, 3] .+= accumarray(jElesNodMat, diagKeBlockSingleDOF, (meshHierarchy[1].numNodes,))
    end
    meshHierarchy[1].diagK = vec(diagK')

    # Assemble & Factorize Stiffness Matrix on Coarsest Level
    rowIndice, colIndice, _ = findnz(ones(24, 24))
    sK = zeros(24^2, meshHierarchy[end].numElements)
    for ii in 1:meshHierarchy[end].numElements
        iKe = Ks[:, :, ii]
        sK[:, ii] = vec(iKe)
    end
    eNodMat = meshHierarchy[end].eNodMat
    eDofMat = hcat(3 * eNodMat .- 2, 3 * eNodMat .- 1, 3 * eNodMat)
    eDofMat = eDofMat[:, reOrdering]
    iK = eDofMat[:, rowIndice]
    jK = eDofMat[:, colIndice]
    KcoarsestLevel = sparse(iK, jK, sK')
    cholFac, cholPermut = chol(KcoarsestLevel, Val(true))

    return meshHierarchy, cholFac, cholPermut
end

function solving_operator4multi_grid_restriction_and_interpolation(opt, spanWidth)
    if spanWidth == 2
        return [
            0 0 0 1 0 0 0 0;
            0.5 0 0 0.5 0 0 0 0;
            1 0 0 0 0 0 0 0;
            0 0 0.5 0.5 0 0 0 0;
            0.25 0.25 0.25 0.25 0 0 0 0;
            0.5 0.5 0 0 0 0 0 0;
            0 0 1 0 0 0 0 0;
            0 0.5 0.5 0 0 0 0 0;
            0 1 0 0 0 0 0 0;
            0 0 0 0.5 0 0 0 0.5;
            0.25 0 0 0.25 0.25 0 0 0.25;
            0.5 0 0 0 0.5 0 0 0;
            0 0 0.25 0.25 0 0 0.25 0.25;
            0.125 0.125 0.125 0.125 0.125 0.125 0.125 0.125;
            0.25 0.25 0 0 0.25 0.25 0 0;
            0 0 0.5 0 0 0 0.5 0;
            0 0.25 0.25 0 0 0.25 0.25 0;
            0 0.5 0 0 0 0.5 0 0;
            0 0 0 0 0 0 0 1;
            0 0 0 0 0.5 0 0 0.5;
            0 0 0 0 1 0 0 0;
            0 0 0 0 0 0 0.5 0.5;
            0 0 0 0 0.25 0.25 0.25 0.25;
            0 0 0 0 0.5 0.5 0 0;
            0 0 0 0 0 0 1 0;
            0 0 0 0 0 0.5 0.5 0;
            0 0 0 0 0 1 0 0
        ]
    elseif spanWidth == 4
        return [
            0 0 0 1 0 0 0 0;
            0.25 0 0 0.75 0 0 0 0;
            0.50 0 0 0.50 0 0 0 0;
            0.75 0 0 0.25 0 0 0 0;
            1 0 0 0 0 0 0 0;
            0 0 0.2500 0.7500 0 0 0 0;
            0.1875 0.0625 0.1875 0.5625 0 0 0 0;
            0.3750 0.1250 0.1250 0.3750 0 0 0 0;
            0.5625 0.1875 0.0625 0.1875 0 0 0 0;
            0.7500 0.25 0 0 0 0 0 0;
            0 0 0.50 0.5 0 0 0 0;
            0.125 0.125 0.375 0.375 0 0 0 0;
            0.250 0.250 0.250 0.250 0 0 0 0;
            0.375 0.375 0.125 0.125 0 0 0 0;
            0.500 0.500 0 0 0 0 0 0;
            0 0 0.750 0.25 0 0 0 0;
            0.0625 0.1875 0.5625 0.1875 0 0 0 0;
            0.1250 0.3750 0.3750 0.1250 0 0 0 0;
            0.1875 0.5625 0.1875 0.0625 0 0 0 0;
            0.2500 0.7500 0 0 0 0 0 0;
            0 0 1 0 0 0 0 0;
            0 0.25 0.75 0 0 0 0 0;
            0 0.50 0.50 0 0 0 0 0;
            0 0.75 0.25 0 0 0 0 0;
            0 1 0 0 0 0 0 0;
            0 0 0 0.75 0 0 0 0.25;
            0.1875 0 0 0.5625 0.0625 0 0 0.1875;
            0.3750 0 0 0.3750 0.1250 0 0 0.1250;
            0.5625 0 0 0.1875 0.1875 0 0 0.0625;
            0.7500 0 0 0 0.25 0 0 0;
            0 0 0.1875 0.5625 0 0 0.0625 0.1875;
            0.140625 0.046875 0.140625 0.421875 0.046875 0.015625 0.046875 0.140625;
            0.281250 0.093750 0.093750 0.281250 0.093750 0.031250 0.031250 0.093750;
            0.421875 0.140625 0.046875 0.140625 0.140625 0.046875 0.015625 0.046875;
            0.562500 0.187500 0 0 0.1875 0.0625 0 0;
            0 0 0.375 0.375 0 0 0.125 0.125;
            0.09375 0.09375 0.28125 0.28125 0.03125 0.03125 0.09375 0.09375;
            0.18750 0.18750 0.18750 0.18750 0.06250 0.06250 0.06250 0.06250;
            0.28125 0.28125 0.09375 0.09375 0.09375 0.09375 0.03125 0.03125;
            0.37500 0.37500 0 0 0.125 0.125 0 0;
            0 0 0.5625 0.1875 0 0 0.1875 0.0625;
            0.046875 0.140625 0.421875 0.140625 0.015625 0.046875 0.140625 0.046875;
            0.093750 0.281250 0.281250 0.093750 0.031250 0.093750 0.093750 0.031250;
            0.140625 0.421875 0.140625 0.046875 0.046875 0.140625 0.046875 0.015625;
            0.187500 0.562500 0 0 0.0625 0.1875 0 0;
            0 0 0.75 0 0 0 0.25 0;
            0 0.1875 0.5625 0 0 0.0625 0.1875 0;
            0 0.3750 0.3750 0 0 0.1250 0.1250 0;
            0 0.5625 0.1875 0 0 0.1875 0.0625 0;
            0 0.75 0 0 0 0.25 0 0;
            0 0 0 0.5 0 0 0 0.5;
            0.125 0 0 0.375 0.125 0 0 0.375;
            0.250 0 0 0.250 0.250 0 0 0.250;
            0.375 0 0 0.125 0.375 0 0 0.125;
            0.5 0 0 0 0.5 0 0 0;
            0 0 0.125 0.375 0 0 0.125 0.375;
            0.09375 0.03125 0.09375 0.28125 0.09375 0.03125 0.09375 0.28125;
            0.18750 0.06250 0.06250 0.18750 0.18750 0.06250 0.06250 0.18750;
            0.28125 0.09375 0.03125 0.09375 0.28125 0.09375 0.03125 0.09375;
            0.37500 0.12500 0 0 0.375 0.125 0 0;
            0 0 0.25 0.25 0 0 0.25 0.25;
            0.0625 0.0625 0.1875 0.1875 0.0625 0.0625 0.1875 0.1875;
            0.1250 0.1250 0.1250 0.1250 0.1250 0.1250 0.1250 0.1250;
            0.1875 0.1875 0.0625 0.0625 0.1875 0.1875 0.0625 0.0625;
            0.2500 0.2500 0 0 0.25 0.2500 0 0;
            0 0 0.375 0.125 0 0 0.375 0.125;
            0.03125 0.09375 0.28125 0.09375 0.03125 0.09375 0.28125 0.09375;
            0.06250 0.18750 0.18750 0.06250 0.06250 0.18750 0.18750 0.06250;
            0.09375 0.28125 0.09375 0.03125 0.09375 0.28125 0.09375 0.03125;
            0.12500 0.37500 0 0 0.125 0.375 0 0;
            0 0 0.5 0 0 0 0.5 0;
            0 0.125 0.375 0 0 0.125 0.375 0;
            0 0.250 0.250 0 0 0.250 0.250 0;
            0 0.375 0.125 0 0 0.375 0.125 0;
            0 0.500 0 0 0 0.5 0 0;
            0 0 0 0.25 0 0 0 0.75;
            0.0625 0 0 0.1875 0.1875 0 0 0.5625;
            0.1250 0 0 0.1250 0.3750 0 0 0.3750;
            0.1875 0 0 0.0625 0.5625 0 0 0.1875;
            0.2500 0 0 0 0.75 0 0 0;
            0 0 0.0625 0.1875 0 0 0.1875 0.5625;
            0.046875 0.015625 0.046875 0.140625 0.140625 0.046875 0.140625 0.421875;
            0.093750 0.031250 0.031250 0.093750 0.281250 0.093750 0.093750 0.281250;
            0.140625 0.046875 0.015625 0.046875 0.421875 0.140625 0.046875 0.140625;
            0.187500 0.062500 0 0 0.5625 0.1875 0 0;
            0 0 0.125 0.125 0 0 0.375 0.375;
            0.03125 0.03125 0.09375 0.09375 0.09375 0.09375 0.28125 0.28125;
            0.06250 0.06250 0.06250 0.06250 0.18750 0.18750 0.18750 0.18750;
            0.09375 0.09375 0.03125 0.03125 0.28125 0.28125 0.09375 0.09375;
            0.12500 0.12500 0 0 0.37500 0.37500 0 0;
            0 0 0.1875 0.0625 0 0 0.5625 0.1875;
            0.015625 0.046875 0.140625 0.046875 0.046875 0.140625 0.421875 0.140625;
            0.031250 0.093750 0.093750 0.031250 0.093750 0.281250 0.281250 0.093750;
            0.046875 0.140625 0.046875 0.015625 0.140625 0.421875 0.140625 0.046875;
            0.062500 0.187500 0 0 0.187500 0.562500 0 0;
            0 0 0.25 0 0 0 0.75 0;
            0 0.0625 0.1875 0 0 0.1875 0.5625 0;
            0 0.1250 0.1250 0 0 0.3750 0.3750 0;
            0 0.1875 0.0625 0 0 0.5625 0.1875 0;
            0 0.2500 0 0 0 0.75 0 0;
            0 0 0 0 0 0 0 1;
            0 0 0 0 0.25 0 0 0.75;
            0 0 0 0 0.50 0 0 0.50;
            0 0 0 0 0.75 0 0 0.25;
            0 0 0 0 1 0 0 0;
            0 0 0 0 0 0 0.25 0.75;
            0 0 0 0 0.1875 0.0625 0.1875 0.5625;
            0 0 0 0 0.3750 0.1250 0.1250 0.3750;
            0 0 0 0 0.5625 0.1875 0.0625 0.1875;
            0 0 0 0 0.75 0.25 0 0;
            0 0 0 0 0 0 0.5 0.5;
            0 0 0 0 0.125 0.125 0.375 0.375;
            0 0 0 0 0.250 0.250 0.250 0.250;
            0 0 0 0 0.375 0.375 0.125 0.125;
            0 0 0 0 0.500 0.500 0 0;
            0 0 0 0 0 0 0.75 0.25;
            0 0 0 0 0.0625 0.1875 0.5625 0.1875;
            0 0 0 0 0.1250 0.3750 0.3750 0.1250;
            0 0 0 0 0.1875 0.5625 0.1875 0.0625;
            0 0 0 0 0.2500 0.7500 0 0;
            0 0 0 0 0 0 1 0;
            0 0 0 0 0 0.25 0.75 0;
            0 0 0 0 0 0.50 0.50 0;
            0 0 0 0 0 0.75 0.25 0;
            0 0 0 0 0 1 0 0
        ]
    else
        error("Wrong input of span width!")
    end
end

function solving_sub_ele_nod_mat(spanWidth)
    if spanWidth == 2
        return [
            2 5 4 1 11 14 13 10;
            3 6 5 2 12 15 14 11;
            6 9 8 5 15 18 17 14;
            5 8 7 4 14 17 16 13;
            11 14 13 10 20 23 22 19;
            12 15 14 11 21 24 23 20;
            15 18 17 14 24 27 26 23;
            14 17 16 13 23 26 25 22
        ]
    elseif spanWidth == 4
        return [
            2 7 6 1 27 32 31 26;
            3 8 7 2 28 33 32 27;
            4 9 8 3 29 34 33 28;
            5 10 9 4 30 35 34 29;
            7 12 11 6 32 37 36 31;
            8 13 12 7 33 38 37 32;
            9 14 13 8 34 39 38 33;
            10 15 14 9 35 40 39 34;
            12 17 16 11 37 42 41 36;
            13 18 17 12 38 43 42 37;
            14 19 18 13 39 44 43 38;
            15 20 19 14 40 45 44 39;
            17 22 21 16 42 47 46 41;
            18 23 22 17 43 48 47 42;
            19 24 23 18 44 49 48 43;
            20 25 24 19 45 50 49 44;
            27 32 31 26 52 57 56 51;
            28 33 32 27 53 58 57 52;
            29 34 33 28 54 59 58 53;
            30 35 34 29 55 60 59 54;
            32 37 36 31 57 62 61 56;
            33 38 37 32 58 63 62 57;
            34 39 38 33 59 64 63 58;
            35 40 39 34 60 65 64 59;
            37 42 41 36 62 67 66 61;
            38 43 42 37 63 68 67 62;
            39 44 43 38 64 69 68 63;
            40 45 44 39 65 70 69 64;
            42 47 46 41 67 72 71 66;
            43 48 47 42 68 73 72 67;
            44 49 48 43 69 74 73 68;
            45 50 49 44 70 75 74 69;
            52 57 56 51 77 82 81 76;
            53 58 57 52 78 83 82 77;
            54 59 58 53 79 84 83 78;
            55 60 59 54 80 85 84 79;
            57 62 61 56 82 87 86 81;
            58 63 62 57 83 88 87 82;
            59 64 63 58 84 89 88 83;
            60 65 64 59 85 90 89 84;
            62 67 66 61 87 92 91 86;
            63 68 67 62 88 93 92 87;
            64 69 68 63 89 94 93 88;
            65 70 69 64 90 95 94 89;
            67 72 71 66 92 97 96 91;
            68 73 72 67 93 98 97 92;
            69 74 73 68 94 99 98 93;
            70 75 74 69 95 100 99 94;
            77 82 81 76 102 107 106 101;
            78 83 82 77 103 108 107 102;
            79 84 83 78 104 109 108 103;
            80 85 84 79 105 110 109 104;
            82 87 86 81 107 112 111 106;
            83 88 87 82 108 113 112 107;
            84 89 88 83 109 114 113 108;
            85 90 89 84 110 115 114 109;
            87 92 91 86 112 117 116 111;
            88 93 92 87 113 118 117 112;
            89 94 93 88 114 119 118 113;
            90 95 94 89 115 120 119 114;
            92 97 96 91 117 122 121 116;
            93 98 97 92 118 123 122 117;
            94 99 98 93 119 124 123 118;
            95 100 99 94 120 125 124 119
        ]
    else
        error("Wrong input of span width!")
    end
end

function solving_building_mesh_hierarchy(meshHierarchy, numLevels, nonDyadic, eNodMatHalfTemp)
    if length(meshHierarchy) > 1
        return meshHierarchy, eNodMatHalfTemp
    end

    nodeVolume = reshape(1:(meshHierarchy[1].resX + 1) * (meshHierarchy[1].resY + 1) * (meshHierarchy[1].resZ + 1),
                         meshHierarchy[1].resY + 1, meshHierarchy[1].resX + 1, meshHierarchy[1].resZ + 1)

    if nonDyadic == 1 && numLevels >= 4
        numLevels -= 1
    else
        nonDyadic = 0
    end

    for ii in 2:numLevels
        if ii == 2 && nonDyadic == 1
            spanWidth = 4
        else
            spanWidth = 2
        end

        nx = Int(meshHierarchy[ii-1].resX / spanWidth)
        ny = Int(meshHierarchy[ii-1].resY / spanWidth)
        nz = Int(meshHierarchy[ii-1].resZ / spanWidth)

        push!(meshHierarchy, CartesianMesh())
        meshHierarchy[ii].resX = nx
        meshHierarchy[ii].resY = ny
        meshHierarchy[ii].resZ = nz
        meshHierarchy[ii].eleSize = meshHierarchy[ii-1].eleSize * spanWidth
        meshHierarchy[ii].spanWidth = spanWidth

        iEleVolume = reshape(meshHierarchy[ii-1].eleMapForward, spanWidth * ny, spanWidth * nx, spanWidth * nz)
        iEleVolumeTemp = reshape(1:spanWidth^3 * nx * ny * nz, spanWidth * ny, spanWidth * nx, spanWidth * nz)
        iFineNodVolumeTemp = reshape(1:(spanWidth * nx + 1) * (spanWidth * ny + 1) * (spanWidth * nz + 1),
                                     spanWidth * ny + 1, spanWidth * nx + 1, spanWidth * nz + 1)

        elementUpwardMap = zeros(Int32, nx * ny * nz, spanWidth^3)
        elementUpwardMapTemp = zeros(Int32, nx * ny * nz, spanWidth^3)
        transferMatTemp = zeros(Int32, (spanWidth + 1)^3, nx * ny * nz)

        for jj in 1:nz, kk in 1:nx, gg in 1:ny
            iFineEles = iEleVolume[(spanWidth * (gg - 1) + 1):spanWidth * gg, (spanWidth * (kk - 1) + 1):spanWidth * kk, (spanWidth * (jj - 1) + 1):spanWidth * jj]
            iFineElesTemp = iEleVolumeTemp[(spanWidth * (gg - 1) + 1):spanWidth * gg, (spanWidth * (kk - 1) + 1):spanWidth * kk, (spanWidth * (jj - 1) + 1):spanWidth * jj]
            iFineNodsTemp = iFineNodVolumeTemp[(spanWidth * (gg - 1) + 1):spanWidth * gg + 1, (spanWidth * (kk - 1) + 1):spanWidth * kk + 1, (spanWidth * (jj - 1) + 1):spanWidth * jj + 1]

            eleIndex = (jj - 1) * ny * nx + (kk - 1) * ny + gg
            elementUpwardMap[eleIndex, :] = vec(iFineEles)
            elementUpwardMapTemp[eleIndex, :] = vec(iFineElesTemp)
            transferMatTemp[:, eleIndex] = vec(iFineNodsTemp)
        end

        unemptyElements = findall(x -> x > 0, sum(elementUpwardMap, dims=2))
        elementUpwardMapTemp = elementUpwardMapTemp[unemptyElements, :]
        elementsIncVoidLastLevelGlobalOrdering = vec(elementUpwardMapTemp)
        nodesIncVoidLastLevelGlobalOrdering = eNodMatHalfTemp[elementsIncVoidLastLevelGlobalOrdering, :]
        nodesIncVoidLastLevelGlobalOrdering = common_recover_halfe_nod_mat(nodesIncVoidLastLevelGlobalOrdering)
        nodesIncVoidLastLevelGlobalOrdering = unique(nodesIncVoidLastLevelGlobalOrdering)
        meshHierarchy[ii].intermediateNumNodes = length(nodesIncVoidLastLevelGlobalOrdering)

        transferMatTemp = transferMatTemp[:, unemptyElements]
        temp = zeros(Int32, (spanWidth * nx + 1) * (spanWidth * ny + 1) * (spanWidth * nz + 1))
        temp[nodesIncVoidLastLevelGlobalOrdering] = 1:meshHierarchy[ii].intermediateNumNodes
        meshHierarchy[ii].transferMat = temp[transferMatTemp]

        meshHierarchy[ii].transferMatCoeffi = zeros(meshHierarchy[ii].intermediateNumNodes)
        for kk in 1:(spanWidth + 1)^3
            solidNodesLastLevel = meshHierarchy[ii].transferMat[kk, :]
            meshHierarchy[ii].transferMatCoeffi[solidNodesLastLevel] .+= 1
        end

        elementsLastLevelGlobalOrdering = meshHierarchy[ii-1].eleMapBack
        nodesLastLevelGlobalOrdering = eNodMatHalfTemp[elementsLastLevelGlobalOrdering, :]
        nodesLastLevelGlobalOrdering = common_recover_halfe_nod_mat(nodesLastLevelGlobalOrdering)
        nodesLastLevelGlobalOrdering = unique(nodesLastLevelGlobalOrdering)
        meshHierarchy[ii].solidNodeMapCoarser2Finer = indexin(nodesLastLevelGlobalOrdering, nodesIncVoidLastLevelGlobalOrdering)

        meshHierarchy[ii].eleMapForward = zeros(Int32, nx * ny * nz)
        meshHierarchy[ii].eleMapBack = unemptyElements
        meshHierarchy[ii].numElements = length(unemptyElements)
        meshHierarchy[ii].eleMapForward[unemptyElements] = 1:meshHierarchy[ii].numElements
        elementUpwardMap = elementUpwardMap[unemptyElements, :]
        meshHierarchy[ii].elementUpwardMap = elementUpwardMap

        nodenrs = reshape(1:(nx + 1) * (ny + 1) * (nz + 1), 1 + ny, 1 + nx, 1 + nz)
        eNodVec = reshape(nodenrs[1:end-1, 1:end-1, 1:end-1] .+ 1, nx * ny * nz, 1)
        eNodMat = repeat(eNodVec[meshHierarchy[ii].eleMapBack], 1, 8)
        eNodMatHalfTemp_new = repeat(eNodVec, 1, 8)
        tmp = [0, ny + 1, ny, -1, (ny + 1) * (nx + 1), (ny + 1) * (nx + 1) + ny + 1, (ny + 1) * (nx + 1) + ny, (ny + 1) * (nx + 1) - 1]
        for jj in 1:8
            eNodMat[:, jj] .+= tmp[jj]
            eNodMatHalfTemp_new[:, jj] .+= tmp[jj]
        end
        eNodMatHalfTemp = eNodMatHalfTemp_new[:, [3, 4, 7, 8]]

        meshHierarchy[ii].nodMapBack = unique(eNodMat)
        meshHierarchy[ii].numNodes = length(meshHierarchy[ii].nodMapBack)
        meshHierarchy[ii].numDOFs = meshHierarchy[ii].numNodes * 3
        meshHierarchy[ii].nodMapForward = zeros(Int32, (nx + 1) * (ny + 1) * (nz + 1))
        meshHierarchy[ii].nodMapForward[meshHierarchy[ii].nodMapBack] = 1:meshHierarchy[ii].numNodes

        for jj in 1:8
            eNodMat[:, jj] = meshHierarchy[ii].nodMapForward[eNodMat[:, jj]]
        end

        kk = (nonDyadic == 1) ? ii : ii - 1
        tmp = nodeVolume[1:2^kk:meshHierarchy[1].resY+1, 1:2^kk:meshHierarchy[1].resX+1, 1:2^kk:meshHierarchy[1].resZ+1]
        meshHierarchy[ii].nodMapBack = vec(tmp)[meshHierarchy[ii].nodMapBack]

        meshHierarchy[ii].multiGridOperatorRI = solving_operator4multi_grid_restriction_and_interpolation("inNODE", spanWidth)

        numElesAroundNode = zeros(Int32, meshHierarchy[ii].numNodes)
        for jj in 1:8
            iNodes = eNodMat[:, jj]
            numElesAroundNode[iNodes] .+= 1
        end
        meshHierarchy[ii].nodesOnBoundary = findall(x -> x < 8, numElesAroundNode)
        meshHierarchy[ii].eNodMat = eNodMat
    end

    @printf("Mesh Hierarchy...\n")
    @printf("             #Resolutions         #Elements   #DOFs\n")
    for ii in 1:length(meshHierarchy)
        @printf("...Level %i: %4i x %4i x %4i %11i %11i\n",
                ii, meshHierarchy[ii].resX, meshHierarchy[ii].resY, meshHierarchy[ii].resZ,
                meshHierarchy[ii].numElements, meshHierarchy[ii].numDOFs)
    end

    return meshHierarchy, eNodMatHalfTemp
end

function solving_setup_ke_with_fixed_dofs(meshHierarchy, fixingCond, modulus)
    allElements = zeros(Int32, meshHierarchy[1].numElements)
    allElements[meshHierarchy[1].elementsOnBoundary] .= 1
    allNodes = zeros(Int32, meshHierarchy[1].numNodes)
    allNodes[meshHierarchy[1].nodesOnBoundary] = 1:length(meshHierarchy[1].nodesOnBoundary)
    nodeBoundaryStruct = [ (arr = Int[]) for _ in 1:length(meshHierarchy[1].nodesOnBoundary) ]

    for ii in 1:meshHierarchy[1].numElements
        if allElements[ii] == 1
            iNodes = meshHierarchy[1].eNodMat[ii, :]
            for jj in 1:8
                jiNode = allNodes[iNodes[jj]]
                if jiNode > 0
                    push!(nodeBoundaryStruct[jiNode].arr, ii)
                end
            end
        end
    end

    _, fixedNodeIndices = intersect(meshHierarchy[1].nodesOnBoundary, fixingCond[:, 1])
    fixedNodesStruct = nodeBoundaryStruct[fixedNodeIndices]
    allElementsWithFixedDOFs = unique(vcat([s.arr for s in fixedNodesStruct]...))
    numElementsWithFixedDOFs = length(allElementsWithFixedDOFs)
    mapUniqueKes = zeros(Int32, meshHierarchy[1].numElements)
    mapUniqueKes[allElementsWithFixedDOFs] = 1:numElementsWithFixedDOFs
    KeCol = vec(meshHierarchy[1].Ke)
    uniqueKesFree = repeat(KeCol, 1, numElementsWithFixedDOFs)
    uniqueKesFixed = zeros(size(uniqueKesFree))

    for ii in 1:length(fixedNodesStruct)
        iFixation = fixingCond[ii, :]
        iNode = iFixation[1]
        iNodeFixationState = iFixation[2:end]
        iNodEles = fixedNodesStruct[ii].arr
        numElesOfThisNode = length(iNodEles)
        for jj in 1:numElesOfThisNode
            jEleNodes = meshHierarchy[1].eNodMat[iNodEles[jj], :]
            fixedNodeLocally = findall(x -> x == iNode, jEleNodes)
            jEleLocally = mapUniqueKes[iNodEles[jj]]
            fixedDOFsLocally = 3 * fixedNodeLocally .- [2 1 0]
            fixedDOFsLocally = fixedDOFsLocally[vec(iNodeFixationState) .== 1]
            uniqueKesFree[:, jEleLocally], uniqueKesFixed[:, jEleLocally] = 
                solving_apply_bc_on_ele_stiff_mat_b(uniqueKesFree[:, jEleLocally], uniqueKesFixed[:, jEleLocally], fixedDOFsLocally, numElesOfThisNode)
        end
    end

    uniqueKesFixed = uniqueKesFixed * modulus
    isThisEle2ndLevelIncludingFixedDOFsOn1stLevel = zeros(Int32, meshHierarchy[2].numElements)
    allElements = zeros(Int32, size(allElements))
    allElements[allElementsWithFixedDOFs] .= 1
    elementUpwardMap = meshHierarchy[2].elementUpwardMap

    # This part is parallelized in MATLAB, but we will do it sequentially for now
    for jj in 1:meshHierarchy[2].numElements
        sonEles = elementUpwardMap[jj, :]
        sonElesCompact = sonEles[sonEles .> 0]
        if sum(allElements[sonElesCompact]) > 0
            isThisEle2ndLevelIncludingFixedDOFsOn1stLevel[jj] = 1
        end
    end

    eles2ndFinestIncludingElesFinestWithFixedDOFs = findall(x -> x > 0, isThisEle2ndLevelIncludingFixedDOFsOn1stLevel)
    isThisEle2ndLevelIncludingFixedDOFsOn1stLevel[eles2ndFinestIncludingElesFinestWithFixedDOFs] = 1:length(eles2ndFinestIncludingElesFinestWithFixedDOFs)
    sonElesWithFixedDOFs = [ (arr = Int[]) for _ in 1:length(eles2ndFinestIncludingElesFinestWithFixedDOFs) ]

    for jj in 1:meshHierarchy[2].numElements
        idx = isThisEle2ndLevelIncludingFixedDOFsOn1stLevel[jj]
        if idx > 0
            sonEles = elementUpwardMap[jj, :]
            _, sonElesWithFixedDOFs[idx].arr = intersect(sonEles, allElementsWithFixedDOFs)
        end
    end

    return meshHierarchy, isThisEle2ndLevelIncludingFixedDOFsOn1stLevel, uniqueKesFixed, uniqueKesFree, sonElesWithFixedDOFs, mapUniqueKes
end

function solving_apply_bc_on_ele_stiff_mat_b(iKeFreeDOFs, iKeFixedDOFs, fixedDOFsLocally, nAdjEles)
    iKeFreeDOFs = reshape(iKeFreeDOFs, 24, 24)
    iKeFixedDOFs = reshape(iKeFixedDOFs, 24, 24)
    oKeFreeDOFs = copy(iKeFreeDOFs)
    oKeFixedDOFs = copy(iKeFixedDOFs)

    oKeFreeDOFs[fixedDOFsLocally, :] .= 0
    oKeFreeDOFs[:, fixedDOFsLocally] .= 0

    for ii in 1:length(fixedDOFsLocally)
        oKeFixedDOFs[fixedDOFsLocally[ii], fixedDOFsLocally[ii]] = 1 / nAdjEles
    end

    return vec(oKeFreeDOFs), vec(oKeFixedDOFs)
end

end # module
