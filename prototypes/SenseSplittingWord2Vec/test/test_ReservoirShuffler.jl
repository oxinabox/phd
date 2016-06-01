using Base.Test
using Utils
using ReservoirShuffle


@testset "Reservoir  Shuffle" begin
	srand(15)
	@test ReservoirShuffler(10:10:100, 64) |> Set == 10:10:100 |> Set

	@test ReservoirShuffler(1:10, 2) |> Set == 1:10 |> Set
	@test ReservoirShuffler(1:100, 2) |> Set == 1:100 |> Set
	
	@test ReservoirShuffler(collect(1:1000), 10) |> Set == Set(1:1000)
	@test ReservoirShuffler(collect(1:1000), 1024) |> Set == Set(1:1000)
	@test ReservoirShuffler(collect(1:1000), 1) |> collect == collect(1:1000) # with reservoir of 1 it is exactly the same
	@test ReservoirShuffler(collect(1:1000), 10) |>collect != collect(1:1000) #Not same order
	@test ReservoirShuffler(collect(1:1000), 1024) |> collect != collect(1:1000) #not same order
	
	@test ReservoirShuffler(1:1_000_000, 1024) |> Set == Set(1:1_000_000)

	@test ReservoirShuffler(1:1_000_000,64) |> Set == Set(1:1_000_000)
	
end
