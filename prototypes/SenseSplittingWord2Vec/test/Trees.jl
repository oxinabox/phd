using Trees
using Base.Test

@testset "Trees" begin
	x = BranchNode(
	[BranchNode([],"11"),BranchNode([
			BranchNode([],"121"),BranchNode([],"122"),
			],"12")],
		"1"
	)
	sol = BranchNode([BranchNode([],"L11"),BranchNode([
			BranchNode([],"L121"),BranchNode([],"L122"),
			],"x12")],
		"x1")

	y=transform_tree(x, leaf_transform = word->"L"*word, internal_transform = dummy -> "x"*dummy)

	@test y == sol
	@test leaves_of(y) |> collect == leaves_of(sol)|> collect
	@test get_paths(y) |> collect == get_paths(sol) |> collect
	@test y[2].data == sol[2].data
end
