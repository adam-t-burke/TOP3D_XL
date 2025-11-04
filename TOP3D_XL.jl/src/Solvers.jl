module Solvers

using LinearAlgebra
using SparseArrays
using Printf
using ..Utils

export solving_pcg, solving_K_by_U_matrix_free, solving_v_cycle, solving_building_mesh_hierarchy, solving_setup_ke_with_fixed_dofs, solving_apply_bc_on_ele_stiff_mat_b

function accumarray(subs, val, sz)
    A = zeros(eltype(val), sz...)
    for i in 1:length(val)
        A[subs[i]] += val[i]
    end
    return A
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
        # ... (to be implemented later)
        return zeros(1,1)
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
