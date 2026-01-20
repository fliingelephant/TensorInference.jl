using TensorInference, Test, LinearAlgebra, Graphs, BitBasis, Random

function gf2_basis(masks::Vector{T}, nbits::Int) where {T<:Integer}
    basis = T[]
    pivots = Int[]
    for mask in masks
        vec = mask
        for (b, p) in zip(basis, pivots)
            readbit(vec, p) == 0 && continue
            vec = vec ⊻ b
        end
        iszero(vec) && continue
        pivot = 0
        for i in nbits:-1:1
            if readbit(vec, i) == 1
                pivot = i
                break
            end
        end
        insert_at = findfirst(x -> x < pivot, pivots)
        if insert_at === nothing
            push!(basis, vec)
            push!(pivots, pivot)
        else
            insert!(basis, insert_at, vec)
            insert!(pivots, insert_at, pivot)
        end
    end
    return basis, pivots
end

function gf2_rank(masks::Vector{T}, nbits::Int) where {T<:Integer}
    basis, _ = gf2_basis(masks, nbits)
    return length(basis)
end

function reduce_with_basis(mask::T, basis::Vector{T}, pivots::Vector{Int}) where {T<:Integer}
    vec = mask
    for (b, p) in zip(basis, pivots)
        readbit(vec, p) == 0 && continue
        vec = vec ⊻ b
    end
    return vec
end

function cycle_mask(cycle::Vector{Int}, edge_index, ::Type{T}) where {T<:Integer}
    n = length(cycle)
    n == 0 && return zero(T)
    mask = zero(T)
    for i in 1:n
        u = cycle[i]
        v = cycle[i == n ? 1 : i + 1]
        idx = edge_index[(u, v)]
        mask = mask | bmask(T, idx)
    end
    return mask
end

@testset "cycle basis on Petersen graph" begin
    g = Graphs.SimpleGraphs.smallgraph(:petersen)
    basis = minimal_loops(g)
    rank = ne(g) - nv(g) + length(connected_components(g))
    @test length(basis.cycles) == rank
    lengths = count_ones.(basis.cycles)
    @test minimum(lengths) == 5
    @test all(len -> len >= 5, lengths)
    @test gf2_rank(basis.cycles, length(basis.edges)) == rank
    edge_index = Dict{Tuple{Int, Int}, Int}()
    for (i, (u, v)) in enumerate(basis.edges)
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    end
    masks = Set{eltype(basis.cycles)}()
    for cyc in simplecycles(DiGraph(g))
        length(cyc) < 3 && continue
        push!(masks, cycle_mask(cyc, edge_index, eltype(basis.cycles)))
    end
    basis_vecs, pivots = gf2_basis(basis.cycles, length(basis.edges))
    for mask in masks
        @test iszero(reduce_with_basis(mask, basis_vecs, pivots))
    end
end

function cycle_uai(tensors::Vector{Matrix{T}}) where {T}
    n = length(tensors)
    d1, d2 = size(tensors[1])
    d1 == d2 || throw(ArgumentError("tensors must be square"))
    cards = fill(d1, n)
    factors = Vector{TensorInference.Factor{T, 2}}(undef, n)
    for i in 1:n
        j = i == n ? 1 : i + 1
        size(tensors[i], 1) == d1 || throw(ArgumentError("dimension mismatch"))
        size(tensors[i], 2) == d1 || throw(ArgumentError("dimension mismatch"))
        factors[i] = TensorInference.Factor((i, j), tensors[i])
    end
    return TensorInference.UAIModel(n, cards, factors)
end

_scalar(x) = x isa AbstractArray && ndims(x) == 0 ? x[] : x

edge_key(u::Int, v::Int) = u < v ? (u, v) : (v, u)

function edge_list(g::SimpleGraph)
    eds = Tuple{Int, Int}[]
    for e in edges(g)
        u, v = edge_key(src(e), dst(e))
        push!(eds, (u, v))
    end
    sort!(eds)
    return eds
end

function graph_uai(g::SimpleGraph, bond_dim::Int; rng::AbstractRNG = Random.default_rng())
    eds = edge_list(g)
    edge_index = Dict{Tuple{Int, Int}, Int}()
    for (i, (u, v)) in enumerate(eds)
        edge_index[(u, v)] = i
        edge_index[(v, u)] = i
    end
    factors = TensorInference.Factor{Float64}[]
    for v in vertices(g)
        neis = sort!(collect(neighbors(g, v)))
        vars = [edge_index[(v, u)] for u in neis]
        tensor = rand(rng, ntuple(_ -> bond_dim, length(vars))...)
        push!(factors, TensorInference.Factor((vars...,), tensor))
    end
    return TensorInference.UAIModel(length(eds), fill(bond_dim, length(eds)), factors)
end

exact_weight(uai) = _scalar(probability(TensorNetworkModel(uai)))

function run_loop_expansion(uai; max_iter::Int = 500, tol::Real = 1e-8, K::Int = 1)
    bp = BeliefPropgation(uai)
    state, info = belief_propagate(bp; max_iter, tol)
    loops = loop_basis(bp)
    bp_weight = bp_vacuum_weight(bp, state)
    result = loop_expansion(bp, state; loops, K)
    return bp, state, info, loops, bp_weight, result
end

function random_cyclic_graph(n::Int, m::Int; rng::AbstractRNG = Random.default_rng(), max_tries::Int = 100)
    max_edges = n * (n - 1) ÷ 2
    m > max_edges && throw(ArgumentError("m must be <= $max_edges"))
    for _ in 1:max_tries
        g = SimpleGraph(n)
        pairs = [(u, v) for u in 1:n-1 for v in u+1:n]
        shuffle!(rng, pairs)
        for i in 1:m
            u, v = pairs[i]
            add_edge!(g, u, v)
        end
        if length(connected_components(g)) == 1 && ne(g) >= nv(g)
            return g
        end
    end
    error("failed to sample a connected cyclic graph after $max_tries attempts")
end

@testset "loop expansion on single cycle" begin
    A = [0.2 0.9; 0.9 0.2]
    tensors = [A for _ in 1:5]
    uai = cycle_uai(tensors)
    _, _, info, loops, bp_weight, result = run_loop_expansion(uai; max_iter=500, tol=1e-10, K=1)
    @test info.converged

    @test length(loops) == 1
    @test count_ones(loops[1].edges) == 5

    exact = tr(A^5)
    @test !isapprox(bp_weight, exact; atol=1e-6, rtol=1e-6)
    @test result.value ≈ exact atol=1e-6
end

@testset "loop expansion on Petersen random tensors" begin
    rng = MersenneTwister(17)
    g = Graphs.SimpleGraphs.smallgraph(:petersen)
    uai = graph_uai(g, 2; rng)
    _, _, info, loops, bp_weight, result = run_loop_expansion(uai; max_iter=500, tol=1e-8, K=1)
    @test info.converged

    @test !isempty(loops)

    exact = exact_weight(uai)
    @test !isapprox(bp_weight, exact; atol=1e-6, rtol=1e-6)
    @test isfinite(bp_weight) && isfinite(result.value) && isfinite(exact)
end

@testset "loop expansion on random simple graphs" begin
    for (n, m, seed) in ((6, 7, 23), (7, 9, 41))
        rng = MersenneTwister(seed)
        g = random_cyclic_graph(n, m; rng)
        uai = graph_uai(g, 2; rng)
        _, _, info, loops, bp_weight, result = run_loop_expansion(uai; max_iter=600, tol=1e-8, K=1)
        @test info.converged
        @test !isempty(loops)
        exact = exact_weight(uai)
        @test !isapprox(bp_weight, exact; atol=1e-6, rtol=1e-6)
        @test isfinite(bp_weight) && isfinite(result.value) && isfinite(exact)
    end
end
