module FEA

using LinearAlgebra
using SparseArrays
using ..Utils
using ..Solvers

export fea_voxel_based_element_stiffness_matrix, fea_voxel_based_discretization, create_voxel_fea_model, fea_apply_boundary_condition, fea_setup_voxel_based, adapt_bc_external_mdl, adapt_passive_elements_external_mdl

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

function create_voxel_fea_model(inputModel, coarsestResolutionControl)
    if typeof(inputModel) <: AbstractString
        # Read from file
        open(inputModel, "r") do f
            lines = readlines(f)
            line_idx = 1

            # Read header
            while !startswith(lines[line_idx], "Resolution:")
                line_idx += 1
            end
            res_line = split(lines[line_idx])
            nelx = parse(Int, res_line[2])
            nely = parse(Int, res_line[3])
            nelz = parse(Int, res_line[4])
            line_idx += 1

            # Read density values included flag
            density_values_included = parse(Bool, split(lines[line_idx])[2])
            line_idx += 2 # Skip "Solid_Voxels:" line

            # Read solid voxels
            num_solid_voxels = parse(Int, split(lines[line_idx])[2])
            line_idx += 1
            solid_voxels = zeros(Int, num_solid_voxels)
            density_layout = zeros(num_solid_voxels)
            for i in 1:num_solid_voxels
                line = split(lines[line_idx])
                solid_voxels[i] = parse(Int, line[1])
                if density_values_included
                    density_layout[i] = parse(Float64, line[2])
                end
                line_idx += 1
            end

            # Read passive elements
            line_idx += 1 # Skip "Passive elements:" line
            num_passive_elements = parse(Int, split(lines[line_idx])[2])
            line_idx += 1
            passive_elements = zeros(Int, num_passive_elements)
            for i in 1:num_passive_elements
                passive_elements[i] = parse(Int, lines[line_idx])
                line_idx += 1
            end

            # Read fixations
            line_idx += 1 # Skip "Fixations:" line
            num_fixed_nodes = parse(Int, split(lines[line_idx])[2])
            line_idx += 1
            fixing_cond = zeros(Int, num_fixed_nodes, 4)
            for i in 1:num_fixed_nodes
                line = split(lines[line_idx])
                fixing_cond[i, :] = [parse(Int, val) for val in line]
                line_idx += 1
            end

            # Read loads
            line_idx += 1 # Skip "Loads:" line
            num_loaded_nodes = parse(Int, split(lines[line_idx])[2])
            line_idx += 1
            loading_cond = [zeros(Float64, num_loaded_nodes, 4)]
            for i in 1:num_loaded_nodes
                line = split(lines[line_idx])
                loading_cond[1][i, :] = [parse(Float64, val) for val in line]
                line_idx += 1
            end

            # Read additional loads
            line_idx += 1 # Skip "Additional Loads:" line
            num_additional_loads = parse(Int, split(lines[line_idx])[2])
            line_idx += 1
            for _ in 1:num_additional_loads
                line_idx += 1 # Skip "Loads:" line
                num_loaded_nodes = parse(Int, split(lines[line_idx])[2])
                load_idx = parse(Int, split(lines[line_idx])[3])
                line_idx += 1
                new_load = zeros(Float64, num_loaded_nodes, 4)
                for i in 1:num_loaded_nodes
                    line = split(lines[line_idx])
                    new_load[i, :] = [parse(Float64, val) for val in line]
                    line_idx += 1
                end
                push!(loading_cond, new_load)
            end

            obj_weighting_list = ones(length(loading_cond)) / length(loading_cond)

            voxelized_volume = falses(nelx * nely * nelz)
            voxelized_volume[solid_voxels] .= true
            voxelized_volume = reshape(voxelized_volume, nely, nelx, nelz)

            mesh = fea_voxel_based_discretization(voxelized_volume, coarsestResolutionControl)

            # Adapt boundary conditions and passive elements
            if !isempty(fixing_cond)
                fixing_cond = adapt_bc_external_mdl(fixing_cond, [mesh.resX + 1, mesh.resY + 1, mesh.resZ + 1], (nelx, nely, nelz))
                fixing_cond[:, 1] = mesh.nodMapForward[fixing_cond[:, 1]]
            end
            if !isempty(loading_cond)
                for ii in 1:length(loading_cond)
                    iLoad = loading_cond[ii]
                    iLoad = adapt_bc_external_mdl(iLoad, [mesh.resX + 1, mesh.resY + 1, mesh.resZ + 1], (nelx, nely, nelz))
                    iLoad[:, 1] = mesh.nodMapForward[iLoad[:, 1]]
                    loading_cond[ii] = iLoad
                end
            end
            if !isempty(passive_elements)
                passive_elements = adapt_passive_elements_external_mdl(passive_elements, [mesh.resX, mesh.resY, mesh.resZ], (nelx, nely, nelz))
                passive_elements = mesh.eleMapForward[passive_elements]
            end

            meshHierarchy = [mesh]
            F = spzeros(mesh.numDOFs, length(loading_cond))
            for ii in 1:length(loading_cond)
                iLoad = loading_cond[ii]
                iFarr = spzeros(mesh.numNodes, 3)
                iFarr[Int.(iLoad[:, 1]), :] = iLoad[:, 2:end]
                F[:, ii] = vec(iFarr')
            end

            return meshHierarchy, F, passive_elements, density_layout, fixing_cond, loading_cond, obj_weighting_list
        end

    elseif typeof(inputModel) <: BitArray{3}
        # Built-in Cuboid Design Domain for Testing
        nely, nelx, nelz = size(inputModel)
        if nelx < 3 || nely < 3 || nelz < 3
            error("Inappropriate Input Model!")
        end

        mesh = fea_voxel_based_discretization(inputModel, coarsestResolutionControl)

        # Apply Boundary Conditions
        nodeVolume4ApplyingBC = zeros(Bool, mesh.resY + 1, mesh.resX + 1, mesh.resZ + 1)
        nodeVolume4ApplyingBC[1:nely+1, 1, 1:nelz+1] .= true
        fixingCond_nodes = findall(vec(nodeVolume4ApplyingBC))
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

        iLoad_nodes = findall(vec(nodeVolume4ApplyingBC))
        iLoad_nodes = mesh.nodMapForward[iLoad_nodes]
        loadingCond = [hcat(iLoad_nodes, zeros(length(iLoad_nodes), 2), -ones(length(iLoad_nodes)) / length(iLoad_nodes))]

        objWeightingList = [1.0]
        passiveElements = Int[]
        densityLayout = []

        meshHierarchy = [mesh]
        F = spzeros(mesh.numDOFs, length(loadingCond))
        for ii in 1:length(loadingCond)
            iLoad = loadingCond[ii]
            iFarr = spzeros(mesh.numNodes, 3)
            iFarr[Int.(iLoad[:, 1]), :] = iLoad[:, 2:end]
            F[:, ii] = vec(iFarr')
        end

        return meshHierarchy, F, passiveElements, densityLayout, fixingCond, loadingCond, objWeightingList
    end
end

function adapt_bc_external_mdl(srcBC, adjustedRes, original_res)
    nelx, nely, nelz = original_res
    nullNodeVolume = zeros((nelx + 1) * (nely + 1) * (nelz + 1))

    adjustedNnlx = adjustedRes[1]
    adjustedNnly = adjustedRes[2]
    adjustedNnlz = adjustedRes[3]
    nnlx = nelx + 1
    nnly = nely + 1
    nnlz = nelz + 1

    tmpBC = zeros(size(srcBC))
    srcBC = srcBC[sortperm(srcBC[:, 1]), :]
    nodeVolume = nullNodeVolume
    nodeVolume[srcBC[:, 1]] .= 1
    nodeVolume = reshape(nodeVolume, nely + 1, nelx + 1, nelz + 1)
    nodeVolume = cat(nodeVolume, zeros(nnly, adjustedNnlx - nnlx, nnlz), dims=2)
    nodeVolume = cat(nodeVolume, zeros(adjustedNnly - nnly, adjustedNnlx, nnlz), dims=1)
    nodeVolume = cat(nodeVolume, zeros(adjustedNnly, adjustedNnlx, adjustedNnlz - nnlz), dims=3)
    newLoadedNodes = findall(nodeVolume)
    tmpBC[:, 1] = newLoadedNodes

    for i in 2:4
        nodeVolume = nullNodeVolume
        nodeVolume[srcBC[:, 1]] = srcBC[:, i]
        nodeVolume = reshape(nodeVolume, nely + 1, nelx + 1, nelz + 1)
        nodeVolume = cat(nodeVolume, zeros(nnly, adjustedNnlx - nnlx, nnlz), dims=2)
        nodeVolume = cat(nodeVolume, zeros(adjustedNnly - nnly, adjustedNnlx, nnlz), dims=1)
        nodeVolume = cat(nodeVolume, zeros(adjustedNnly, adjustedNnlx, adjustedNnlz - nnlz), dims=3)
        nodeVolume = vec(nodeVolume)
        tmpBC[:, i] = nodeVolume[newLoadedNodes]
    end

    return tmpBC
end

function adapt_passive_elements_external_mdl(srcElesMapback, adjustedRes, original_res)
    nelx, nely, nelz = original_res
    nullVoxelVolume = zeros(nelx * nely * nelz)

    adjustedNelx = adjustedRes[1]
    adjustedNely = adjustedRes[2]
    adjustedNelz = adjustedRes[3]

    srcElesMapback = sort(srcElesMapback)
    voxelVolume = nullVoxelVolume
    voxelVolume[srcElesMapback] .= 1
    voxelVolume = reshape(voxelVolume, nely, nelx, nelz)
    voxelVolume = cat(voxelVolume, zeros(nely, adjustedNelx - nelx, nelz), dims=2)
    voxelVolume = cat(voxelVolume, zeros(adjustedNely - nely, adjustedNelx, nelz), dims=1)
    voxelVolume = cat(voxelVolume, zeros(adjustedNely, adjustedNelx, adjustedNelz - nnlz), dims=3)
    adjustedVoxelIndices = findall(voxelVolume)

    return adjustedVoxelIndices
end

function fea_apply_boundary_condition(meshHierarchy, loadingCond, fixingCond)
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
    F = spzeros(meshHierarchy[1].numDOFs, length(loadingCond))
    for ii in 1:length(loadingCond)
        iLoad = loadingCond[ii]
        iFarr = spzeros(meshHierarchy[1].numNodes, 3)
        iFarr[Int.(iLoad[:, 1]), :] = iLoad[:, 2:end]
        F[:, ii] = vec(iFarr')
    end

    # Fixing
    fixedDOFs = 3 * fixingCond[:, 1]
    fixedDOFs = fixedDOFs .- [2 1 0]
    fixedDOFs = vec(fixedDOFs')
    fixingState = fixingCond[:, 2:end]'
    fixedDOFs = fixedDOFs[vec(fixingState) .== 1]

    freeDOFs = trues(meshHierarchy[1].numDOFs)
    freeDOFs[fixedDOFs] .= false
    meshHierarchy[1].freeDOFs = freeDOFs

    meshHierarchy[1].fixedDOFs = falses(meshHierarchy[1].numDOFs)
    meshHierarchy[1].fixedDOFs[fixedDOFs] .= true

    return meshHierarchy, F, loadingCond, fixingCond
end

function fea_setup_voxel_based(meshHierarchy, poisson_ratio, cell_size, fixingCond, modulus)
    meshHierarchy[1].Ke = fea_voxel_based_element_stiffness_matrix(poisson_ratio, cell_size)

    meshHierarchy, _ = solving_building_mesh_hierarchy(meshHierarchy, numLevels_, nonDyadic_, eNodMatHalfTemp_)

    meshHierarchy[1].Ks = meshHierarchy[1].Ke

    # Preparation to apply for BCs directly on element stiffness matrix
    solving_setup_ke_with_fixed_dofs(meshHierarchy, fixingCond, modulus)

    return meshHierarchy
end

end # module