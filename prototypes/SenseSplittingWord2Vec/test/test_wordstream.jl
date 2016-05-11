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

facts("Should Filter") do
	
	@fact collect(words_of(data,Dict("the"=>0.01,"men"=>0.01,"sailing"=>0.01, "gold"=>0.01 ) )) --> ["the", "men","sailing","the", "the","gold"] "Filter only, not subsapling"
end




facts("Should get a sliding window") do 
	ww=words_of(data)
	windows = collect(sliding_window(ww,lsize=1,rsize=1))
	@fact windows[1] --> AbstractString["the","king","and"]
	@fact windows[2] --> AbstractString["king","and", "his"]

	bigger_windows = collect(sliding_window(ww,lsize=2,rsize=2))
	@fact bigger_windows[end] --> AbstractString["find","the","land","of","gold"]
end




#TODO Add tests of subsampling
