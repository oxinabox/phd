module WordStreams
using PooledElements
export words_of, WordStream, SlidingWindow, sliding_window, subsampling_prob

"""
Probability of removing an word that has `word_distr` distribution.
If `subsampling_rate` is zero, then this always returns 0.0 (rather than the expected 1.0)
"""
function subsampling_prob(subsampling_rate, word_distr)
	if subsampling_rate>0.0
		prob = clamp(1.0 - sqrt(subsampling_rate/word_distr), 0.0,1.0)
	else
		#Setting Rate to Zero is shorthand for "We are not doing Subsampling"
		0.0
	end
end

const WHITESPACE = (' ', '\n', '\r')
type WordStream{S<:String, F<:AbstractFloat}
    source::Union{IO, String}
    # filter configuration
    rate::AbstractFloat    #if rate > 0, words will be subsampled according to distr
    filter::Bool    # if filter is true, only words present in the keys(distr) will be considered
    distr::Dict{S, F}
end

function words_of(source::Union{IO,String}; subsampling=(0.0,false,nothing))
    rate, filter, subsampling_distr = subsampling
    distr = (rate==0.0 && !filter) ? Dict{String,Float32}() : subsampling_distr
 	WordStream(source, rate, filter, distr)
end

"""Filters out all word not in the filter_dist -- only the keys are used, values are ignored"""
function words_of{S<:String, F<:AbstractFloat}(source::Union{IO,String}, filter_distr::Dict{S,F})
	filter = true
	rate = -1.0
 	WordStream(source, rate, filter,filter_distr)
end



"Returns the next word without considering rate. Mutated the sourcepointer state"
function unrated_next_word!(ws::WordStream, fp)
    filter_out(word) = (ws.filter && !haskey(ws.distr, word))
    
    next_word = IOBuffer()
    while(!eof(fp))
        c = read(fp, Char)       
        if c∈WHITESPACE
           s = takebuf_string(next_word)
            if s == "" || filter_out(s)
                continue
           else
                return pstring(s)#
           end
        else #Non Whitespace character
            write(next_word,c)
        end
    end
    #Hit EOF
    s = takebuf_string(next_word)
    filter_out(s) ? pstring("") : pstring(s)
end


function Base.start(ws::WordStream)
    if isa(ws.source, String)
        fp=open(ws.source)
    else
        @assert(typeof(ws.source)<:IO)
        fp=deepcopy(ws.source)  #WARN: This may not actually work. Deepcopy on IOStream seems to be broken -- [*OX] 22/4/16
	end
    (unrated_next_word!(ws,fp), fp)
end

function Base.done(ws::WordStream, state)
    (next_word, fp) =state
    next_word=="" #Done when there is no next word.
end

Base.iteratorsize(::WordStream) = Base.SizeUnknown()
Base.eltype{S,F}(::Type{WordStream{S,F}})=S
function Base.next(ws::WordStream, state)
    (next_word, fp) = state
    while(!eof(fp))
        if ws.rate > 0
            prob = subsampling_prob(ws.rate, ws.distr[next_word])
			if(rand()<prob)
				#Skip this word
                next_word=unrated_next_word!(ws,fp) #Advance to next word, skipping this one
                continue
            end
        end
        next_next_word = unrated_next_word!(ws,fp)
        return (next_word,(next_next_word,fp))
    end
    #Hit EOF
    #This is the last word we are getting, do not skip it.
    #This will throw off subsampling some tiny factor. Negligible
    return ((next_word),("",fp))
end
######################################################

type SlidingWindow
    ws::WordStream
    lsize::Int64
    rsize::Int64
end


Base.iteratorsize(::SlidingWindow) = Base.SizeUnknown()
window_length(window) = window.lsize + 1 + window.rsize

function Base.start(window::SlidingWindow)
    ws_state = start(window.ws)
    words = String["" for ii in 1:window_length(window)]
    
    for ii in 0:window_length(window)-2
	    _,(words,ws_state) = next(window, (words, ws_state))
    end
    (words, ws_state)
end

function Base.done(window::SlidingWindow, state)
    (words, ws_state) = state
    done(window.ws, ws_state) #CHECKME: Is this getting all the words I want at the end?
end

function Base.next(window::SlidingWindow, state)
    (words, ws_state) = state        
    next_word, ws_state = next(window.ws, ws_state)
    push!(words, next_word)    
    words = words[2:end] #PREMOPT: could use a circular bufffer
    
    (filter(w->length(w)>0, words), (words,ws_state))
end

function sliding_window(words; lsize=5, rsize=5)
    SlidingWindow(words, lsize, rsize)
end


end #End Module
