module TOP3D_XL

using LinearAlgebra
using SparseArrays
using Printf
using Statistics
using Makie
using ImageFiltering

include("Utils.jl")
using .Utils
export CartesianMesh, solving_mission_partition, common_recover_halfe_nod_mat, common_include_adjacent_elements, accumarray

include("Solvers.jl")
using .Solvers
export solving_pcg, solving_K_by_U_matrix_free, solving_v_cycle, solving_building_mesh_hierarchy, solving_setup_ke_with_fixed_dofs, solving_apply_bc_on_ele_stiff_mat_b, solving_mission_partition, solving_assemble_fea_stencil, solving_operator4multi_grid_restriction_and_interpolation, solving_sub_ele_nod_mat

include("FEA.jl")
using .FEA
export fea_voxel_based_element_stiffness_matrix, fea_voxel_based_discretization, create_voxel_fea_model, fea_apply_boundary_condition, fea_setup_voxel_based, adapt_bc_external_mdl, adapt_passive_elements_external_mdl

include("Optimization.jl")
using .Optimization
export top_opti_set_passive_elements, top_opti_compute_unit_compliance, top_opti_setup_pde_filter_matrix_free, top_opti_conduct_pde_filtering_matrix_free, mat_times_vec_matrix_free_b

include("IO.jl")
using .IO
export io_export_design_in_volume_nii, io_export_design_in_tri_surface_stl

include("Visualization.jl")
using .Visualization

# Export main function
export TOP3D_XL_TO, InitialSettings

# Physical Property
const modulus_ = 1.0
const poissonRatio_ = 0.3
const modulusMin_ = 1.0e-6 * modulus_
const SIMPpenalty_ = 3.0
const cellSize_ = 1.0

# Linear System Solver
const tol_ = 1.0e-3
const maxIT_ = 800
const weightFactorJacobi_ = 0.6
const coarsestResolutionControl_ = 50000
const typeVcycle_ = "Adapted"
const nonDyadic_ = 1

# Optimization
const specifyPassiveRegions_ = [0, 0, 0]

function InitialSettings()
    # In Julia, constants are defined at the top level of the module.
    # This function is kept for similarity with the MATLAB code.
end

function TOP3D_XL_TO(inputModel, V0, nLoop, rMin)
    tStartTotal = time()
    outPath = "./out/"
    if !isdir(outPath)
        mkdir(outPath)
    end
    InitialSettings()

    # Displaying Inputs
    println("==========================Displaying Inputs==========================")
    @printf("..............................................Volume Fraction: %6.4f\n", V0)
    @printf("..........................................Filter Radius: %6.4f Cells\n", rMin)
    @printf("................................................Cell Size: %6.4e\n", cellSize_)
    @printf("...............................................#MGCG Iterations: %4i\n", maxIT_)
    println(".....................................................V-cycle: ", typeVcycle_)
    @printf("...............................................Non-dyadic Strategy: %1i\n", nonDyadic_)
    @printf("...........................................Youngs Modulus: %6.4e\n", modulus_)
    @printf("....................................Youngs Modulus (Min.): %6.4e\n", modulusMin_)
    @printf("...........................................Poissons Ratio: %6.4e\n", poissonRatio_)

    # 0. Modeling
    tStart = time()
    meshHierarchy, F, passiveElements, densityLayout, fixingCond, loadingCond, objWeightingList = create_voxel_fea_model(inputModel, coarsestResolutionControl_)
    @printf("Preparing Voxel-based FEA Model Costs %10.1fs\n", time() - tStart)

    # 1. Pre. FEA
    meshHierarchy, F, loadingCond, fixingCond = fea_apply_boundary_condition(meshHierarchy, loadingCond, fixingCond)
    meshHierarchy = fea_setup_voxel_based(meshHierarchy, poissonRatio_, cellSize_, fixingCond, modulus_)

    # 2. Setup PDE filter
    PDEkernal4Filtering, diagPrecond4Filtering = top_opti_setup_pde_filter_matrix_free(rMin, meshHierarchy)

    # 3. prepare optimizer
    passiveElements = top_opti_set_passive_elements(meshHierarchy, loadingCond, fixingCond, passiveElements, specifyPassiveRegions_[1], specifyPassiveRegions_[2], specifyPassiveRegions_[3])
    numElements = meshHierarchy[1].numElements
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
    U = zeros(size(F))
    meshHierarchy[1].eleModulus = fill(modulus_, numElements)
    tSolvingFEAssemblingClock = time()
    meshHierarchy, cholFac, cholPermut = solving_assemble_fea_stencil(meshHierarchy, numLevels_, cholFac_, cholPermut_, isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_, uniqueKesFixed_, uniqueKesFree_, sonElesWithFixedDOFs_, mapUniqueKes_)
    itSolvingFEAssembling = time() - tSolvingFEAssemblingClock
    tSolvingFEAiterationClock = time()
    for ii in 1:size(F, 2)
        U[:, ii], _ = solving_pcg(solving_K_by_U_matrix_free, solving_v_cycle, F[:, ii], tol_, maxIT_, [false, true], U[:, ii])
    end
    itSolvingFEAiteration = time() - tSolvingFEAiterationClock
    ceList = top_opti_compute_unit_compliance(U, meshHierarchy, objWeightingList)
    cSolid = meshHierarchy[1].eleModulus' * ceList
    @printf("Compliance of Fully Solid Domain: %16.6e\n", cSolid)
    @printf(" It.: %4i Assembling Time: %4i s; Solver Time: %4i s.\n", 0, itSolvingFEAssembling, itSolvingFEAiteration)

    SIMP(xPhys) = modulusMin_ .+ xPhys.^SIMPpenalty_ .* (modulus_ - modulusMin_)
    DeSIMP(xPhys) = SIMPpenalty_ * (modulus_ - modulusMin_) .* xPhys.^(SIMPpenalty_ - 1)

    # 5. optimization
    while loop < nLoop && change > 0.0001 && sharpness > 0.01
        perIteCost = time()
        loop += 1

        # 5.1 & 5.2 FEA, objective and sensitivity analysis
        meshHierarchy[1].eleModulus = SIMP(xPhys)
        tSolvingFEAssemblingClock = time()
        meshHierarchy, cholFac, cholPermut = solving_assemble_fea_stencil(meshHierarchy, numLevels_, cholFac_, cholPermut_, isThisEle2ndLevelIncludingFixedDOFsOn1stLevel_, uniqueKesFixed_, uniqueKesFree_, sonElesWithFixedDOFs_, mapUniqueKes_)
        itSolvingFEAssembling = time() - tSolvingFEAssemblingClock
        tSolvingFEAiterationClock = time()
        for ii in 1:size(F, 2)
            U[:, ii], lssIts_ = solving_pcg(solving_K_by_U_matrix_free, solving_v_cycle, F[:, ii], tol_, maxIT_, [false, true], U[:, ii])
            push!(lssIts, lssIts_)
        end
        itSolvingFEAiteration = time() - tSolvingFEAiterationClock
        tOptimizationClock = time()
        ceList = top_opti_compute_unit_compliance(U, meshHierarchy, objWeightingList)
        ceNorm = ceList / cSolid
        cObj = meshHierarchy[1].eleModulus' * ceNorm
        cDesign = cObj * cSolid
        V = mean(xPhys)
        dc = -DeSIMP(xPhys) .* ceNorm
        dv = ones(numElements)
        itimeOptimization = time() - tOptimizationClock

        # 5.3 filtering/modification of sensitivity
        tPDEfilteringClock = time()
        ft = 1
        if ft == 1
            dc[:] = top_opti_conduct_pde_filtering_matrix_free(x .* dc, PDEkernal4Filtering, diagPrecond4Filtering, meshHierarchy) ./ max.(1e-3, x)
        elseif ft == 2
            dc = top_opti_conduct_pde_filtering_matrix_free(dc, PDEkernal4Filtering, diagPrecond4Filtering, meshHierarchy)
        end
        itimeDensityFiltering = time() - tPDEfilteringClock

        # 5.4 solve the optimization probelm
        tOptimizationClock = time()
        fval = mean(xPhys) - V0
        l1 = 0
        l2 = 1e9
        move = 0.2
        while (l2 - l1) / (l1 + l2) > 1e-6
            lmid = 0.5 * (l2 + l1)
            xnew = max.(0, max.(x .- move, min.(1, min.(x .+ move, x .* sqrt.(-dc ./ dv / lmid)))))
            xnew[passiveElements] .= 1.0
            gt = fval + sum((dv .* (xnew - x)))
            if gt > 0
                l1 = lmid
            else
                l2 = lmid
            end
        end
        change = maximum(abs.(xnew - x))
        x = xnew
        itimeOptimization += time() - tOptimizationClock

        # 5.5 filtering
        tPDEfilteringClock = time()
        if ft == 1
            xPhys = xnew
        elseif ft == 2
            xPhys[:] = top_opti_conduct_pde_filtering_matrix_free(xnew, PDEkernal4Filtering, diagPrecond4Filtering, meshHierarchy)
        end
        itimeDensityFiltering += time() - tPDEfilteringClock
        xPhys[passiveElements] .= 1.0
        sharpness = 4 * sum(xPhys .* (ones(numElements) - xPhys)) / numElements
        itimeTotal = time() - perIteCost

        # 5.6 write opti. history
        push!(cHist, cDesign)
        push!(volHist, V)
        push!(consHist, fval)
        push!(sharpHist, sharpness)
        iTimeStatistics = [itSolvingFEAssembling, itSolvingFEAiteration, itimeOptimization, itimeDensityFiltering, itimeTotal]
        push!(tHist, iTimeStatistics)

        # 5.7 print results
        @printf(" It.:%4i Obj.:%16.8e Vol.:%6.4e Sharp.:%6.4e Cons.:%4.2e Ch.:%4.2e\n",
            loop, cDesign, V, sharpness, fval, change)
        @printf(" It.: %i (Time)... Total per-It.: %8.2e s; Assemb.: %8.2e s; CG: %8.2e s; Opti.: %8.2e s; Filtering: %8.2e s.\n",
            loop, itimeTotal, itSolvingFEAssembling, itSolvingFEAiteration, itimeOptimization, itimeDensityFiltering)
    end

    # Output
    densityLayout_ = xPhys[:]
    fileName = joinpath(outPath, "DesignVolume.nii")
    io_export_design_in_volume_nii(fileName, meshHierarchy[1], densityLayout_)
    @printf("..........Solving FEA Costs: %10.4e s.\n", sum(tHist[:, 1:2]))
    @printf("..........Optimization (inc. sentivity analysis, update) Costs: %10.4e s.\n", sum(tHist[:, 3]))
    @printf("..........Performing PDE Filtering Costs: %10.4e s.\n", sum(tHist[:, 4]))
    @printf("..........Performing Topology Optimization Costs (in total): %10.4e s.\n", time() - tStartTotal)

    open(joinpath(outPath, "iters_Target.dat"), "w") do f
        for val in lssIts
            @printf(f, "%d\n", val)
        end
    end
    open(joinpath(outPath, "c_Target.dat"), "w") do f
        for val in cHist
            @printf(f, "%30.16e\n", val)
        end
    end
    open(joinpath(outPath, "sharp_Target.dat"), "w") do f
        for val in sharpHist
            @printf(f, "%30.16e\n", val)
        end
    end
    open(joinpath(outPath, "timing_Target.dat"), "w") do f
        for val in tHist
            @printf(f, "%16.6e\n", val[end])
        end
    end

    # Vis.
    allVoxels = zeros(size(meshHierarchy[1].eleMapForward))
    allVoxels[meshHierarchy[1].eleMapBack] = densityLayout_
    isovals = reshape(allVoxels, meshHierarchy[1].resY, meshHierarchy[1].resX, meshHierarchy[1].resZ)
    isovals = permutedims(isovals, (2, 1, 3))
    isovals = imfilter(isovals, Kernel.box((1,1,1)))
    fig = Figure()
    ax = Axis3(fig[1, 1])
    isosurface!(ax, isovals, 0.5)
    fileName = joinpath(outPath, "DesignVolume.stl")
    # io_export_design_in_tri_surface_stl(fileName, facesIsosurface, facesIsocap)

    display(fig)
end



end # module
