"""
Loop series expansion utilities for belief propagation.
"""

using Graphs
using BitBasis

# --- Bitmask helpers ---

bitmask_type(nbits::Int) = LongLongUInt{max(1, cld(nbits, 64))}
edge_key(u::Int, v::Int) = u < v ? (u, v) : (v, u)

function mask_indices(mask::T, nbits::Int) where {T<:Integer}
    ids = Int[]
    for i in 1:nbits
        readbit(mask, i) == 1 && push!(ids, i)
    end
    return ids
end

function mask_highest(mask::T, nbits::Int) where {T<:Integer}
    for i in nbits:-1:1
        readbit(mask, i) == 1 && return i
    end
    return 0
end

# --- Graph loop basis (decoupled from tensors) ---

"""
    CycleBasis

A cycle basis represented by bitmasks over an explicit edge list.
"""
struct CycleBasis{INT<:Integer}
    edges::Vector{Tuple{Int, Int}}
    cycles::Vector{INT}
end

cycle_rank(g::SimpleGraph) = ne(g) - nv(g) + length(connected_components(g))

function edge_list(g::SimpleGraph)
    eds = Tuple{Int, Int}[]
    for e in edges(g)
        u, v = edge_key(src(e), dst(e))
        push!(eds, (u, v))
    end
    sort!(eds)
    return eds
end

function _edge_adjacency(eds::Vector{Tuple{Int, Int}}, nverts::Int)
    adj = [Vector{Tuple{Int, Int}}() for _ in 1:nverts]
    for (idx, (u, v)) in enumerate(eds)
        push!(adj[u], (v, idx))
        push!(adj[v], (u, idx))
    end
    return adj
end

function _shortest_path_masks(adj, start::Int, target::Int, skip_edge::Int, ::Type{INT}) where {INT<:Integer}
    n = length(adj)
    dist = fill(typemax(Int), n)
    parents = [Vector{Tuple{Int, Int}}() for _ in 1:n]
    queue = Vector{Int}(undef, n)
    head = 1
    tail = 1
    dist[start] = 0
    queue[1] = start

    while head <= tail
        u = queue[head]
        head += 1
        for (v, eidx) in adj[u]
            eidx == skip_edge && continue
            nd = dist[u] + 1
            if nd < dist[v]
                dist[v] = nd
                empty!(parents[v])
                push!(parents[v], (u, eidx))
                tail += 1
                queue[tail] = v
            elseif nd == dist[v]
                push!(parents[v], (u, eidx))
            end
        end
    end

    dist[target] == typemax(Int) && return INT[]
    masks = INT[]
    function backtrack(node::Int, mask::INT)
        if node == start
            push!(masks, mask)
            return
        end
        for (prev, eidx) in parents[node]
            backtrack(prev, mask | bmask(INT, eidx))
        end
    end
    backtrack(target, zero(INT))
    return masks
end

function _candidate_cycles(g::SimpleGraph, eds::Vector{Tuple{Int, Int}})
    nbits = length(eds)
    INT = bitmask_type(nbits)
    adj = _edge_adjacency(eds, nv(g))
    seen = Set{String}()
    candidates = INT[]
    weights = Int[]

    for (eidx, (u, v)) in enumerate(eds)
        paths = _shortest_path_masks(adj, u, v, eidx, INT)
        for path_mask in paths
            cycle = path_mask | bmask(INT, eidx)
            key = join(mask_indices(cycle, nbits), ",")
            key in seen && continue
            push!(seen, key)
            push!(candidates, cycle)
            push!(weights, count_ones(cycle))
        end
    end
    return candidates, weights
end

function _reduce(vec::INT, basis::Vector{INT}, pivots::Vector{Int}) where {INT<:Integer}
    for (b, p) in zip(basis, pivots)
        readbit(vec, p) == 0 && continue
        vec = vec ⊻ b
    end
    return vec
end

function _minimum_cycle_basis(candidates::Vector{INT}, weights::Vector{Int}, rank::Int, nbits::Int) where {INT<:Integer}
    rank == 0 && return INT[]
    order = sortperm(weights)
    basis = INT[]
    pivots = Int[]

    for idx in order
        vec = _reduce(candidates[idx], basis, pivots)
        iszero(vec) && continue
        pivot = mask_highest(vec, nbits)
        insert_at = findfirst(x -> x < pivot, pivots)
        if insert_at === nothing
            push!(basis, vec)
            push!(pivots, pivot)
        else
            insert!(basis, insert_at, vec)
            insert!(pivots, insert_at, pivot)
        end
        length(basis) == rank && break
    end

    length(basis) == rank || throw(ArgumentError("cycle basis incomplete: got $(length(basis)) of $rank"))
    return basis
end

"""
    minimal_loops(g::SimpleGraph) -> CycleBasis

Return a minimum cycle basis of `g`. Each cycle is a bitmask over the edge
list stored in `CycleBasis.edges`.
"""
function minimal_loops(g::SimpleGraph)
    eds = edge_list(g)
    rank = cycle_rank(g)
    INT = bitmask_type(length(eds))
    rank == 0 && return CycleBasis{INT}(eds, INT[])
    candidates, weights = _candidate_cycles(g, eds)
    cycles = _minimum_cycle_basis(candidates, weights, rank, length(eds))
    return CycleBasis{INT}(eds, cycles)
end

# --- Loop series contractions ---

"""
    LoopExcitation

A loop excitation represented by bitmasks for edges (variables) and tensors.
"""
struct LoopExcitation{E<:Integer, T<:Integer}
    edges::E
    tensors::T
end

loop_degree(loop::LoopExcitation) = count_ones(loop.edges)

struct LoopSeriesCache{T, VT <: AbstractVector{T}}
    message_in_norm::Vector{Vector{VT}}
    v2t_pos::Vector{Dict{Int, Int}}
    complement_proj::Vector{Union{Nothing, Matrix{T}}}
end

function LoopSeriesCache(bp::BeliefPropgation, state::BPState{T}) where {T}
    nvars = num_variables(bp)
    msg_norm = Vector{Vector{typeof(state.message_in[1][1])}}(undef, nvars)
    v2t_pos = Vector{Dict{Int, Int}}(undef, nvars)
    c_mats = Vector{Union{Nothing, Matrix{T}}}(undef, nvars)

    for v in 1:nvars
        msgs = state.message_in[v]
        msg_norm[v] = [copy(m) for m in msgs]
        pos = Dict{Int, Int}()
        for (idx, t) in enumerate(bp.v2t[v])
            pos[t] = idx
        end
        v2t_pos[v] = pos

        if length(msgs) == 2
            m1, m2 = msgs
            s = dot(m2, m1)
            iszero(s) && throw(ArgumentError("edge $v has zero message overlap"))
            scale1 = inv(sqrt(s))
            scale2 = inv(sqrt(conj(s)))
            msg_norm[v][1] = m1 .* scale1
            msg_norm[v][2] = m2 .* scale2
            d = length(m1)
            P = msg_norm[v][2] * msg_norm[v][1]'
            c_mats[v] = Matrix{T}(I, d, d) - P
        else
            c_mats[v] = nothing
        end
    end

    return LoopSeriesCache(msg_norm, v2t_pos, c_mats)
end

function tensor_connectivity_graph(bp::BeliefPropgation)
    g = SimpleGraph(num_tensors(bp))
    edge_to_var = Dict{Tuple{Int, Int}, Int}()

    for v in 1:num_variables(bp)
        tids = bp.v2t[v]
        length(tids) == 2 || continue
        t1, t2 = tids
        a, b = edge_key(t1, t2)
        if haskey(edge_to_var, (a, b))
            throw(ArgumentError("multiple variables between tensors $a and $b"))
        end
        edge_to_var[(a, b)] = v
        add_edge!(g, a, b)
    end

    return g, edge_to_var
end

function loop_basis(bp::BeliefPropgation)
    g, edge_to_var = tensor_connectivity_graph(bp)
    basis = minimal_loops(g)
    return loops_from_basis(bp, basis, edge_to_var)
end

function loops_from_basis(bp::BeliefPropgation, basis::CycleBasis{INT}, edge_to_var) where {INT<:Integer}
    nvars = num_variables(bp)
    nt = num_tensors(bp)
    EdgeMask = bitmask_type(nvars)
    TensorMask = bitmask_type(nt)
    loops = LoopExcitation{EdgeMask, TensorMask}[]

    for cycle in basis.cycles
        edge_mask = zero(EdgeMask)
        tensor_mask = zero(TensorMask)
        for edge_idx in mask_indices(cycle, length(basis.edges))
            u, v = basis.edges[edge_idx]
            a, b = edge_key(u, v)
            var = edge_to_var[(a, b)]
            edge_mask = edge_mask | bmask(EdgeMask, var)
            tensor_mask = tensor_mask | bmask(TensorMask, u, v)
        end
        push!(loops, LoopExcitation(edge_mask, tensor_mask))
    end

    return loops
end

_scalar(x) = x isa AbstractArray && ndims(x) == 0 ? x[] : x

function _message_to_tensor(bp::BeliefPropgation, state::BPState, cache::LoopSeriesCache, v::Int, t::Int)
    tids = bp.v2t[v]
    if length(tids) == 2
        idx = cache.v2t_pos[v][t]
        idx_other = idx == 1 ? 2 : 1
        return cache.message_in_norm[v][idx_other]
    elseif length(tids) == 1
        idx = cache.v2t_pos[v][t]
        return state.message_out[v][idx]
    else
        throw(ArgumentError("loop corrections require variables of degree 1 or 2; variable $v has degree $(length(tids))"))
    end
end

function _reduced_tensor(bp::BeliefPropgation, state::BPState, cache::LoopSeriesCache, loop_mask::T, t::Int) where {T<:Integer}
    vars = bp.t2v[t]
    ixs = Vector{Vector{Int}}()
    tensors = Any[]
    push!(ixs, vars)
    push!(tensors, bp.tensors[t])

    keep_vars = Int[]
    for v in vars
        if readbit(loop_mask, v) == 1
            push!(keep_vars, v)
        else
            msg = _message_to_tensor(bp, state, cache, v, t)
            push!(ixs, [v])
            push!(tensors, msg)
        end
    end

    code = EinCode(ixs, keep_vars)
    return code(tensors...), keep_vars
end

"""
    loop_weight(bp::BeliefPropgation, state::BPState, loop::LoopExcitation; optimizer=nothing)

Compute the weight of a single loop excitation by contracting the tensors
in the loop with BP messages absorbed on external edges and `I - P` projectors
inserted on loop edges.
"""
function loop_weight(bp::BeliefPropgation, state::BPState, loop::LoopExcitation; optimizer=nothing)
    cache = LoopSeriesCache(bp, state)
    return loop_weight(bp, state, loop, cache; optimizer)
end

function loop_weight(bp::BeliefPropgation, state::BPState, loop::LoopExcitation, cache::LoopSeriesCache; optimizer=nothing)
    nvars = num_variables(bp)
    tensors = Any[]
    labels = Vector{Vector{Int}}()

    for t in 1:num_tensors(bp)
        readbit(loop.tensors, t) == 1 || continue
        reduced, keep_vars = _reduced_tensor(bp, state, cache, loop.edges, t)
        tensor_labels = Int[]
        for v in keep_vars
            t1, t2 = bp.v2t[v]
            push!(tensor_labels, t == t1 ? v : v + nvars)
        end
        push!(tensors, reduced)
        push!(labels, tensor_labels)
    end

    for v in mask_indices(loop.edges, nvars)
        C = cache.complement_proj[v]
        C === nothing && throw(ArgumentError("loop edge $v is not degree-2"))
        push!(tensors, C)
        push!(labels, [v, v + nvars])
    end

    code = EinCode(labels, Int[])
    if optimizer !== nothing
        size_dict = OMEinsum.get_size_dict(labels, tensors)
        code = optimize_code(code, size_dict, optimizer)
    end
    return _scalar(code(tensors...))
end

"""
    bp_vacuum_weight(bp::BeliefPropgation, state::BPState)

Compute the BP vacuum contribution by contracting each tensor with incoming
messages on all edges and multiplying the resulting scalars.
"""
function bp_vacuum_weight(bp::BeliefPropgation, state::BPState)
    cache = LoopSeriesCache(bp, state)
    return bp_vacuum_weight(bp, state, cache)
end

function bp_vacuum_weight(bp::BeliefPropgation, state::BPState, cache::LoopSeriesCache)
    empty_loop = zero(bitmask_type(num_variables(bp)))
    vals = map(1:num_tensors(bp)) do t
        reduced, _ = _reduced_tensor(bp, state, cache, empty_loop, t)
        _scalar(reduced)
    end
    return prod(vals)
end

function _disjoint_loop_sum(loops::Vector{LoopExcitation{E, T}}, weights::Vector, K::Int, edge_zero::E, tensor_zero::T) where {E<:Integer, T<:Integer}
    isempty(loops) && return zero(weights[1])
    K <= 0 && return zero(weights[1])
    total = zero(weights[1])

    function backtrack(start::Int, depth::Int, used_edges::E, used_tensors::T, prod_weight)
        depth > 0 && (total += prod_weight)
        depth == K && return
        for i in start:length(loops)
            loop = loops[i]
            iszero(loop.edges & used_edges) || continue
            iszero(loop.tensors & used_tensors) || continue
            backtrack(i + 1, depth + 1, used_edges | loop.edges, used_tensors | loop.tensors, prod_weight * weights[i])
        end
    end

    backtrack(1, 0, edge_zero, tensor_zero, one(weights[1]))
    return total
end

"""
    loop_expansion(bp::BeliefPropgation, state::BPState; loops=loop_basis(bp), K::Int=1, optimizer=nothing)

Compute a loop series correction to the BP vacuum contribution. The correction
includes all disjoint combinations of up to `K` loops. Returns a named tuple
with the BP vacuum weight, correction, total value, and per-loop weights.
"""
function loop_expansion(bp::BeliefPropgation, state::BPState; loops = loop_basis(bp), K::Int = 1, optimizer = nothing)
    cache = LoopSeriesCache(bp, state)
    bp_weight = bp_vacuum_weight(bp, state, cache)

    if isempty(loops) || K <= 0
        return (bp_weight = bp_weight, correction = zero(bp_weight), value = bp_weight, loop_weights = typeof(bp_weight)[])
    end

    loop_weights = [loop_weight(bp, state, loop, cache; optimizer) for loop in loops]
    edge_zero = zero(bitmask_type(num_variables(bp)))
    tensor_zero = zero(bitmask_type(num_tensors(bp)))
    correction = _disjoint_loop_sum(loops, loop_weights, K, edge_zero, tensor_zero)
    return (bp_weight = bp_weight, correction = correction, value = bp_weight + correction, loop_weights = loop_weights)
end
