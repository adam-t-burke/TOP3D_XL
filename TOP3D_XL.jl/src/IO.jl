module IO

using NIfTI
using FileIO
using GeometryBasics
using ..Utils

export io_export_design_in_volume_nii, io_export_design_in_tri_surface_stl, io_export_design_in_tri_surface_stl

function io_export_design_in_volume_nii(fileName, meshHierarchy, densityLayout)
    V = zeros(length(meshHierarchy.eleMapForward))
    V[meshHierarchy.eleMapBack] = densityLayout
    V = reshape(V, meshHierarchy.resY, meshHierarchy.resX, meshHierarchy.resZ)
    niftiwrite(fileName, V)
end

function io_export_design_in_tri_surface_stl(fileName, facesIsosurface, facesIsocap)
    all_vertices = [facesIsosurface.vertices; facesIsocap.vertices]
    all_faces = [facesIsosurface.faces; size(facesIsosurface.vertices, 1) .+ facesIsocap.faces]
    mesh = Mesh(all_vertices, all_faces)
    save(fileName, mesh)
end

end # module