module FEA

using LinearAlgebra
using SparseArrays
using ..Utils

export fea_voxel_based_element_stiffness_matrix, fea_voxel_based_discretization, create_voxel_fea_model, fea_apply_boundary_condition, fea_setup_voxel_based

function fea_voxel_based_element_stiffness_matrix(poisson_ratio, cell_size)
    nu = poisson_ratio
    C = [2/9, 1/18, 1/24, 1/36, 1/48, 5/72, 1/3, 1/6, 1/12]

    A11 = [-C[1] -C[3] -C[3]  C[2]  C[3]  C[3];
           -C[3] -C[1] -C[3] -C[3] -C[4] -C[5];
           -C[3] -C[3] -C[1] -C[3] -C[5] -C[4];
            C[2] -C[3] -C[3] -C[1]  C[3]  C[3];
            C[3] -C[4] -C[5]  C[3] -C[1] -C[3];
            C[3] -C[5] -C[4]  C[3] -C[3] -C[1]]

    B11 = [ C[7]   0     0     0   -C[8] -C[8];
              0    C[7]   0    C[8]   0     0;
              0     0    C[7]  C[8]   0     0;
              0    C[8]  C[8]  C[7]   0     0;
           -C[8]    0     0     0    C[7]   0;
           -C[8]    0     0     0     0    C[7]]

    A22 = [-C[1] -C[3]  C[3]  C[2]  C[3] -C[3];
           -C[3] -C[1]  C[3] -C[3] -C[4]  C[5];
            C[3]  C[3] -C[1]  C[3]  C[5] -C[4];
            C[2] -C[3]  C[3] -C[1]  C[3] -C[3];
            C[3] -C[4]  C[5]  C[3] -C[1]  C[3];
           -C[3]  C[5] -C[4] -C[3]  C[3] -C[1]]

    B22 = [ C[7]   0     0     0   -C[8]  C[8];
              0    C[7]   0    C[8]   0     0;
              0     0    C[7] -C[8]   0     0;
              0    C[8] -C[8]  C[7]   0     0;
           -C[8]    0     0     0    C[7]   0;
            C[8]    0     0     0     0    C[7]]

    A12 = [ C[6]  C[3]  C[5] -C[4] -C[3] -C[5];
            C[3]  C[6]  C[5]  C[3]  C[2]  C[3];
           -C[5] -C[5]  C[4] -C[5] -C[3] -C[4];
           -C[4]  C[3]  C[5]  C[6] -C[3] -C[5];
           -C[3]  C[2]  C[3] -C[3]  C[6]  C[5];
            C[5] -C[3] -C[4]  C[5] -C[5]  C[4]]

    B12 = [-C[9]   0   -C[9]   0    C[8]   0;
             0   -C[9] -C[9] -C[8]   0   -C[8];
            C[9]  C[9] -C[9]   0    C[8]   0;
             0   -C[8]   0   -C[9]   0    C[9];
            C[8]   0   -C[8]   0   -C[9] -C[9];
             0    C[8]   0   -C[9]  C[9] -C[9]]

    A13 = [-C[4] -C[5] -C[3]  C[6]  C[5]  C[3];
           -C[5] -C[4] -C[3] -C[5]  C[4] -C[5];
            C[3]  C[3]  C[2]  C[3]  C[5]  C[6];
            C[6] -C[5] -C[3] -C[4]  C[5]  C[3];
            C[5]  C[4] -C[5]  C[5] -C[4] -C[3];
           -C[3]  C[5]  C[6] -C[3]  C[3]  C[2]]

    B13 = [  0     0    C[8] -C[9] -C[9]   0;
             0     0    C[8]  C[9] -C[9]  C[9];
           -C[8] -C[8]   0     0   -C[9] -C[9];
           -C[9]  C[9]   0     0     0   -C[8];
           -C[9] -C[9]  C[9]   0     0    C[8];
             0   -C[9] -C[9]  C[8] -C[8]   0]

    A14 = [ C[2]  C[5]  C[5]  C[4] -C[5] -C[5];
            C[5]  C[2]  C[5]  C[5]  C[6]  C[3];
            C[5]  C[5]  C[2]  C[5]  C[3]  C[6];
            C[4]  C[5]  C[5]  C[2] -C[5] -C[5];
           -C[5]  C[6]  C[3] -C[5]  C[2]  C[5];
           -C[5]  C[3]  C[6] -C[5]  C[5]  C[2]]

    B14 = [-C[9]   0     0   -C[9]  C[9]  C[9];
             0   -C[9]   0   -C[9] -C[9]   0;
             0     0   -C[9] -C[9]   0   -C[9];
           -C[9] -C[9] -C[9] -C[9]   0     0;
            C[9] -C[9]   0     0   -C[9]   0;
            C[9]   0   -C[9]   0     0   -C[9]]

    A23 = [ C[2]  C[5] -C[5]  C[4] -C[5]  C[5];
            C[5]  C[2] -C[5]  C[5]  C[6] -C[3];
           -C[5] -C[5]  C[2] -C[5] -C[3]  C[6];
            C[4]  C[5] -C[5]  C[2] -C[5]  C[5];
           -C[5]  C[6] -C[3] -C[5]  C[2] -C[5];
            C[5] -C[3]  C[6]  C[5] -C[5]  C[2]]

    B23 = [-C[9]   0     0   -C[9]  C[9] -C[9];
             0   -C[9]   0   -C[9] -C[9]   0;
             0     0   -C[9]  C[9]   0   -C[9];
           -C[9] -C[9]  C[9] -C[9]   0     0;
            C[9] -C[9]   0     0   -C[9]   0;
           -C[9]   0   -C[9]   0     0   -C[9]]

    KE = 1 / ((1 + nu) * (2 * nu - 1)) * 
         ([A11 A12 A13 A14; 
           A12' A22 A23 A13'; 
           A13' A23' A22 A12'; 
           A14' A13 A12 A11] + 
           nu * [B11 B12 B13 B14; 
                 B12' B22 B23 B13'; 
                 B13' B23' B22 B12'; 
                 B14' B13 B12 B11])

    KE = KE * cell_size
    return KE
end

function fea_voxel_based_discretization(voxelizedVolume, coarsestResolutionControl)
    nely, nelx, nelz = size(voxelizedVolume)
    numVoxels = sum(voxelizedVolume)

    numLevels = 0
    while numVoxels >= coarsestResolutionControl
        numLevels += 1
        numVoxels = round(Int, numVoxels / 8)
    end
    numLevels = max(3, numLevels)

    adjustedNelx = ceil(Int, nelx / 2^numLevels) * 2^numLevels
    adjustedNely = ceil(Int, nely / 2^numLevels) * 2^numLevels
    adjustedNelz = ceil(Int, nelz / 2^numLevels) * 2^numLevels
    numLevels += 1

    if adjustedNelx > nelx
        voxelizedVolume = cat(voxelizedVolume, zeros(Bool, nely, adjustedNelx - nelx, nelz), dims=2)
    end
    if adjustedNely > nely
        voxelizedVolume = cat(voxelizedVolume, zeros(Bool, adjustedNely - nely, adjustedNelx, nelz), dims=1)
    end
    if adjustedNelz > nelz
        voxelizedVolume = cat(voxelizedVolume, zeros(Bool, adjustedNely, adjustedNelx, adjustedNelz - nelz), dims=3)
    end

    boundingBox = [0 0 0; adjustedNelx adjustedNely adjustedNelz]

    mesh = CartesianMesh()
    mesh.resX = adjustedNelx
    mesh.resY = adjustedNely
    mesh.resZ = adjustedNelz
    nx, ny, nz = mesh.resX, mesh.resY, mesh.resZ
    mesh.eleSize = (boundingBox[2,:] - boundingBox[1,:]) ./ [nx, ny, nz]

    voxelizedVolume = vec(voxelizedVolume)
    mesh.eleMapBack = findall(voxelizedVolume)
    mesh.numElements = length(mesh.eleMapBack)
    mesh.eleMapForward = zeros(Int, nx * ny * nz)
    mesh.eleMapForward[mesh.eleMapBack] = 1:mesh.numElements

    nodenrs = reshape(1:(nx + 1) * (ny + 1) * (nz + 1), 1 + ny, 1 + nx, 1 + nz)
    eNodVec = reshape(nodenrs[1:end-1, 1:end-1, 1:end-1] .+ 1, nx * ny * nz, 1)
    
    eNodMat = repeat(eNodVec[mesh.eleMapBack], 1, 8)
    tmp = [0, ny + 1, ny, -1, (ny + 1) * (nx + 1), (ny + 1) * (nx + 1) + ny + 1, (ny + 1) * (nx + 1) + ny, (ny + 1) * (nx + 1) - 1]
    for i in 1:8
        eNodMat[:, i] .+= tmp[i]
    end

    mesh.nodMapBack = unique(eNodMat)
    mesh.numNodes = length(mesh.nodMapBack)
    mesh.numDOFs = mesh.numNodes * 3
    mesh.nodMapForward = zeros(Int, (nx + 1) * (ny + 1) * (nz + 1))
    mesh.nodMapForward[mesh.nodMapBack] = 1:mesh.numNodes

    for i in 1:8
        eNodMat[:, i] = mesh.nodMapForward[eNodMat[:, i]]
    end
    mesh.eNodMat = eNodMat

    mesh.numNod2ElesVec = zeros(Int, mesh.numNodes)
    for j in 1:8
        iNodes = eNodMat[:, j]
        mesh.numNod2ElesVec[iNodes] .+= 1
    end

    mesh.nodesOnBoundary = findall(x -> x < 8, mesh.numNod2ElesVec)
    
    allNodes = zeros(Int, mesh.numNodes)
    allNodes[mesh.nodesOnBoundary] .= 1
    tmp = zeros(Int, mesh.numElements)
    for i in 1:8
        tmp .+= allNodes[eNodMat[:, i]]
    end
    mesh.elementsOnBoundary = findall(x -> x > 0, tmp)

    return mesh
end

function create_voxel_fea_model(inputModel::BitArray{3}, coarsestResolutionControl)
    nely, nelx, nelz = size(inputModel)
    if nelx < 3 || nely < 3 || nelz < 3
        error("Inappropriate Input Model!")
    end

    mesh = fea_voxel_based_discretization(inputModel, coarsestResolutionControl)

    # Apply Boundary Conditions
    nodeVolume4ApplyingBC = zeros(Bool, mesh.resY + 1, mesh.resX + 1, mesh.resZ + 1)
    nodeVolume4ApplyingBC[1:nely+1, 1, 1:nelz+1] .= true
    fixingCond_nodes = findall(nodeVolume4ApplyingBC)
    fixingCond_nodes = mesh.nodMapForward[fixingCond_nodes]
    fixingCond = hcat(fixingCond_nodes, ones(Int, length(fixingCond_nodes), 3))

    optLoad = 4 # 1=Line Loads; 2=Face Loads; 3=Face Loads-B; 4=Face Loads-C
    nodeVolume4ApplyingBC = zeros(Bool, mesh.resY + 1, mesh.resX + 1, mesh.resZ + 1)
    if optLoad == 1
        nodeVolume4ApplyingBC[1:nely+1, nelx+1, 1] .= true
    elseif optLoad == 2
        nodeVolume4ApplyingBC[round(Int, nely/3)*1:round(Int, nely/3)*2, nelx+1, round(Int, nelz/3)*1:round(Int, nelz/3)*2] .= true
    elseif optLoad == 3
        nodeVolume4ApplyingBC[1:nely+1, round(Int, nelx*11/12):nelx+1, 1] .= true
    elseif optLoad == 4
        nodeVolume4ApplyingBC[1:nely+1, nelx+1, 1:round(Int, nelz/6+1)] .= true
    end

    iLoad_nodes = findall(nodeVolume4ApplyingBC)
    iLoad_nodes = mesh.nodMapForward[iLoad_nodes]
    loadingCond = [hcat(iLoad_nodes, zeros(length(iLoad_nodes), 2), -ones(length(iLoad_nodes)) / length(iLoad_nodes))]

    objWeightingList = [1.0]
    passiveElements = Int[]

    return mesh, fixingCond, loadingCond, objWeightingList, passiveElements
end

function fea_apply_boundary_condition(mesh::CartesianMesh, loadingCond, fixingCond)
    # Pre-Check
    for ii in 1:length(loadingCond)
        iLoad = loadingCond[ii]
        nodesLoadedFixed = setdiff(fixingCond[:, 1], iLoad[:, 1])
        fixingCond = fixingCond[indexin(nodesLoadedFixed, fixingCond[:, 1]), :]
        # ... (unique and sort)
        if isempty(iLoad)
            @warn "No Loads!"
            return
        end
        if isempty(fixingCond)
            @warn "No Fixations!"
            return
        end
        loadingCond[ii] = iLoad
    end

    # Loading
    F = spzeros(mesh.numDOFs, length(loadingCond))
    for ii in 1:length(loadingCond)
        iLoad = loadingCond[ii]
        iFarr = spzeros(mesh.numNodes, 3)
        iFarr[iLoad[:, 1], :] = iLoad[:, 2:end]
        F[:, ii] = vec(iFarr')
    end

    # Fixing
    fixedDOFs = 3 * fixingCond[:, 1]
    fixedDOFs = fixedDOFs .- [2 1 0]
    fixedDOFs = vec(fixedDOFs')
    fixingState = fixingCond[:, 2:end]'
    fixedDOFs = fixedDOFs[vec(fixingState) .== 1]

    freeDOFs = trues(mesh.numDOFs)
    freeDOFs[fixedDOFs] .= false
    mesh.freeDOFs = freeDOFs

    mesh.fixedDOFs = falses(mesh.numDOFs)
    mesh.fixedDOFs[fixedDOFs] .= true

    return mesh, F, loadingCond, fixingCond
end

function fea_setup_voxel_based(mesh::CartesianMesh, poisson_ratio, cell_size)
    mesh.Ke = fea_voxel_based_element_stiffness_matrix(poisson_ratio, cell_size)

    # Placeholders for functions to be translated
    # solving_building_mesh_hierarchy()
    # solving_setup_ke_with_fixed_dofs()

    mesh.Ks = mesh.Ke

    return mesh
end

end # module
