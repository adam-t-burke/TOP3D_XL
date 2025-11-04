push!(LOAD_PATH, "../src")

using TOP3D_XL

# Create a sample input
inputModel = trues(50, 100, 50)
V0 = 0.12
nLoop = 50
rMin = sqrt(3)

# Run the topology optimization
TOP3D_XL.TOP3D_XL_TO(inputModel, V0, nLoop, rMin)
