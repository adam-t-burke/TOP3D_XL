module Visualization

using Makie
using ..Utils

export visualize_design

function visualize_design(densityLayout, mesh::CartesianMesh)
    allVoxels = zeros(mesh.resY, mesh.resX, mesh.resZ)
    allVoxels[mesh.eleMapBack] = densityLayout

    # The MATLAB code uses smooth3, which is not available in Julia's standard library.
    # We can consider adding a dependency for smoothing if needed, but for now, we'll skip it.

    scene = Scene(resolution = (800, 600))
    cam = cam3d!(scene)

    # Use Makie's isosurface function
    isosurface!(scene, allVoxels, 0.5, color = :green)

    # Adjust camera and lighting
    update_cam!(scene, cam, Vec3f0(55, 25, 25), Vec3f0(0, 0, 0))
    # Makie's lighting is different from MATLAB's, so we'll use a simple setup.

    return scene
end

end # module
