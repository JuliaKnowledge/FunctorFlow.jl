# ============================================================================
# test_learn.jl — categorical deep learning: backprop as a functor (FinVect_n)
# ============================================================================

using Test
using FunctorFlow
const Cat = FunctorFlow.Cat

@testset "Forward pass is functorial" begin
    W1 = Cat.LinMap(7, 3, 2, [1 0 2; 0 1 1])
    W2 = Cat.LinMap(7, 2, 1, reshape([1, 1], 1, 2))
    net = Cat.lin_compose(W1, W2)
    x = [1, 2, 3]
    # forward(g∘f) = forward(g) ∘ forward(f)
    @test Cat.forward(net, x) == Cat.forward(W2, Cat.forward(W1, x))
    # identity layer is neutral
    @test Cat.forward(Cat.lin_id(7, 3), x) == mod.(x, 7)
end

@testset "Backprop is the transpose functor (chain rule)" begin
    W1 = Cat.LinMap(7, 3, 2, [1 0 2; 0 1 1])
    W2 = Cat.LinMap(7, 2, 1, reshape([1, 1], 1, 2))
    # (g∘f)ᵀ = fᵀ∘gᵀ  — backprop reverses the network
    @test Cat.transpose_is_functorial(W1, W2)
    # the backward pass agrees whole-network vs layer-by-layer (chain rule on vjps)
    net = Cat.lin_compose(W1, W2)
    cot = [3]
    @test Cat.backward(net, cot) == Cat.backward(W1, Cat.backward(W2, cot))
    # transpose is an involution
    @test Cat.lin_transpose(Cat.lin_transpose(W1)) == W1
    # reverse_derivative is the transpose
    @test Cat.reverse_derivative(W1) == Cat.lin_transpose(W1)
end

@testset "FinVect_n category laws; composition order matters" begin
    A = Cat.LinMap(7, 2, 2, [1 1; 0 1])
    B = Cat.LinMap(7, 2, 2, [1 0; 1 1])
    @test Cat.finvect_category_laws([A, B, Cat.lin_compose(A, B), Cat.lin_id(7, 2)])
    # composition is NOT commutative — order matters, which is why backprop must reverse it
    @test Cat.lin_compose(A, B) != Cat.lin_compose(B, A)
end

@testset "2-layer backprop demo" begin
    d = Cat.backprop_demo()
    @test d.forward == [5]                 # over ℤ₇
    @test d.chain_rule_ok
    @test d.transpose_functorial
    @test d.category_laws
end

@testset "Backprop chain-rule Lean certificate renders" begin
    W1 = Cat.LinMap(7, 3, 2, [1 0 2; 0 1 1])
    W2 = Cat.LinMap(7, 2, 1, reshape([1, 1], 1, 2))
    cert = render_backprop_certificate(W1, W2)
    @test occursin("chainRuleHolds", cert)
    @test occursin("backprop_chain_rule", cert)
    @test occursin("native_decide", cert)
end
