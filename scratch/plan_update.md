# Plan Update for Translating TOP3D_XL to Julia

This document summarizes the progress made so far in translating the TOP3D_XL MATLAB code to Julia and outlines a new, more granular plan for the remaining work.

## Progress Summary

- All functions from the MATLAB code have been translated to Julia.
- The project structure has been set up with the necessary modules.
- A test script `run_to.jl` has been created to run the topology optimization.

## Remaining Work & New Plan

The main remaining task is to debug the translated code and ensure it runs correctly.

**New Plan (November 4, 2025):**

1.  **Fix the `UndefVarError` for `coarsestResolutionControl_`:**
    *   [ ] Modify `FEA.jl`: `fea_voxel_based_discretization` to accept `coarsestResolutionControl` as an argument.
    *   [ ] Modify `FEA.jl`: `create_voxel_fea_model` to accept `coarsestResolutionControl` as an argument and pass it to `fea_voxel_based_discretization`.
    *   [ ] Modify `TOP3D_XL.jl`: Pass the `coarsestResolutionControl_` constant to `create_voxel_fea_model` in `TOP3D_XL_TO`.

2.  **Address other `UndefVarError`s:**
    *   [ ] Identify other variables that are defined as constants in `TOP3D_XL.jl` but used in other modules without being passed as arguments.
    *   [ ] Pass them as arguments to the functions that need them.

3.  **Run the `run_to.jl` test script and debug:**
    *   [ ] Run the script and identify the next error.
    *   [ ] Fix the error and repeat until the script runs without errors.

4.  **Implement missing functionalities:**
    *   [ ] Implement the file reading logic for `.TopVoxel` files in `create_voxel_fea_model`.
    *   [ ] Implement `adapt_bc_external_mdl` and `adapt_passive_elements_external_mdl` in `FEA.jl`.
    *   [ ] Implement `common_include_adjacent_elements` in `Utils.jl`.
    *   [ ] Implement `io_export_design_in_tri_surface_stl` in `IO.jl`.

5.  **Final verification:**
    *   [ ] Run the `run_to.jl` test script again to ensure everything is working as expected.
    *   [ ] Compare the output with the original MATLAB code if possible.
