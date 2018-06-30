# Poisson-Solver

This repository contains a solver for the the modified Poisson-equation

-Δu(x) + αu(x) = f(x),  x∈Ω, ∂Ω	= 0, α	>0

where the domain Ω	is either square or cubic.	


The solver discretizes this equation using central finite differences to get a  linear system and then
runs the preconditioned conjugate gradient method to solve this system. 
For the preconditioner we use a tridiagonal matrix that we can efficiently invert using Cuda on a Nvidia graphic card.

I developed this code for my Bachelor thesis but did not look at it since then. It still seems to compile though. Maybe someone has a use for it.

## Directory structure
    ├── README.md
    ├── cublas_wrapper.h          # wrapper functions around some cublas functions
    ├── makefile
    ├── misc.h                    # some small helper functions
    ├── preconditioner.h          # Cuda kernels that solve solve the linear system for the preconditioner
    ├── solver.h                  # functions that apply the preconditioned conjugate gradient to solve the problem
    └── test.cu                   # demonstrates how to use the given code and runs tests for the different implementations.
