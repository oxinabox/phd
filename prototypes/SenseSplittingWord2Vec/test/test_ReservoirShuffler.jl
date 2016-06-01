using Base.Test
using Utils
using ReservoirShuffle


@testset "Reservoir  Shuffle" begin
	srand(10)
	@test ReservoirShuffler(collect(1:1000), 10) ≅ 1:1000
	@test ReservoirShuffler(collect(1:1000), 1024) ≅ 1:1000
	@test ReservoirShuffler(collect(1:1000), 1) |> collect == collect(1:1000) # with reservoir of 1 it is exactly the same
	@test ReservoirShuffler(collect(1:1000), 10) |>collect != collect(1:1000) #Not same order
	@test ReservoirShuffler(collect(1:1000), 1024) |> collect != collect(1:1000) #not same order
	

end
