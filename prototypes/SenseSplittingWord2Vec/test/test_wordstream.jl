push!(LOAD_PATH,"../src/")
using FactCheck

using WordStreams

const data_raw = "the king and his men are sailing on the seven seas to find the land of gold"
const data_words = split(data_raw)
const data = IOBuffer(data_raw)

const filedata_path = "./data/text8_tiny"
const filedata_data = split(readstring(open(filedata_path,"r"))) 

facts("Should Read") do
	@fact collect(words_of(data)) --> data_words "IOBuffer"
	@fact collect(words_of(filedata_path)) --> filedata_data "From File"
	@fact collect(words_of(open(filedata_path,"r"))) --> filedata_data "From Open File"
end


facts("Should get a sliding window") do 
	ww=words_of(data)
	windows = collect(sliding_window(ww,lsize=1,rsize=1))
	@fact windows[1] --> AbstractString["the","king","and"]
	@fact windows[2] --> AbstractString["king","and", "his"]

	bigger_windows = collect(sliding_window(ww,lsize=2,rsize=2))
	@fact bigger_windows[end] --> AbstractString["find","the","land","of","gold"]
end

facts("Should report progress") do
	function check_progress_facts(ww)		
		oldprog=0.0
		for (prog,word) in collect(enumerate_progress(ww))
			@fact prog --> greater_than_or_equal(oldprog) "progress must monotonically increase"
			oldprog=prog
		end
		
		@fact collect(enumerate_progress(ww))[end][1] --> 1.0 "Must end at 1.0"

		for ((prog,p_word),s_word) in zip(enumerate_progress(ww), ww)
			@fact p_word --> s_word
		end
	end

	context("On words") do
		ww=words_of(data)
		check_progress_facts(ww)
	end

	context("On Window") do
		ww=words_of(data)
		check_progress_facts(sliding_window(ww))
	end

end





#TODO Add tests of subsampling
