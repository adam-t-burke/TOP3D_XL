# GEMINI.md

This document provides additional information and examples for using TOP3D_XL.

## How to run TOP3D_XL

To run the `TOP3D_XL` function, you can use the following syntax:

```matlab
TOP3D_XL(inputModel, consType, V0, nLoop, rMin, varargin);
```

### Example

Run Topology Optimization on the Cuboid Design Domain with Built-in Boundary Conditions:

```matlab
TOP3D_XL(true(50,100,50), 'GLOBAL', 0.12, 50, sqrt(3));
```

## Running the example script

A convenience script `run_top3d_xl.m` has been created in the root directory. You can run this script directly in MATLAB to execute the example topology optimization.
