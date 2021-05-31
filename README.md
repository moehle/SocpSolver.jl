# SocpSolver.jl
`SocpSolver.jl` is a simple, concise interior point solver for solving optimization problems, including

- [second-order cone programs](https://en.wikipedia.org/wiki/Second-order_cone_programming)
- [linear programs](https://en.wikipedia.org/wiki/Linear_programming)
- [quadratic programs](https://en.wikipedia.org/wiki/Quadratic_programming)

It's written in pure Julia, and is best for small- and medium-sized problems.  It includes a wrapper to [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) for use with [JuMP](https://github.com/jump-dev/JuMP.jl) and [Convex.jl](https://github.com/jump-dev/Convex.jl).  At present, this package is still a prototype, and was written primarily as a learning experience for the author.  (But I do hope to keep polishing it over time.)

## License
`SocpSolver.jl` was written by [Nicholas Moehle](https://www.nicholasmoehle.com).  It is available under an MIT license, and relies only on packages with similarly liberal licenses.

## Algorithm
`SocpSolver.jl` is based on the primal--dual interior point algorithm `conelp` described in [this paper](http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf) by [Lieven Vandenberghe](http://www.seas.ucla.edu/~vandenbe/), with some algorithmic improvements described in [this paper](https://web.stanford.edu/~boyd/papers/pdf/ecos_ecc.pdf).

## Usage
Here is a minimal example using `SocpSolver.jl` with [Convex.jl](https://github.com/jump-dev/Convex.jl).

```
using Convex, SCS

m = 4;  n = 5
A = randn(m, n); b = randn(m, 1)
x = Variable(n)

problem = minimize(sumsquares(A * x - b), [x >= 0])
solve!(problem, SocpSolver.Optimizer())
```
