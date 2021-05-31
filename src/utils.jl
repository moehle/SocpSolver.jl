EPS = 1e-6

speye(n) = SparseMatrixCSC(Diagonal(ones(n)))
speye(T, n) = SparseMatrixCSC(Diagonal(ones(T, n)))

pos(x::Real) = max(0, x)
neg(x::Real) = max(0, -x)

struct QuadReflectionMatrix end
J = QuadReflectionMatrix()
*(J::QuadReflectionMatrix, x::Vector{T}) where T <: Real = [x[1]; -x[2:end]]
*(x::Vector{T}, J::QuadReflectionMatrix) where T <: Real = J*x
