module WordStreams
using PooledElements
export words_of, WordStream, SlidingWindow, sliding_window, enumerate_progress


const WHITESPACE = (' ', '\n', '\r')
type WordStream
    source::Union{IO, AbstractString}
    # filter configuration
    rate::Float64    #if rate > 0, words will be subsampled according to distr
    filter::Bool    # if filter is true, only words present in the keys(distr) will be considered
    distr::Dict{AbstractString, Float64}
end

function words_of(file::Union{IO,AbstractString}; subsampling=(0,false,nothing))
    rate, filter, subsampling_distr = subsampling
    distr = (rate==0 && !filter) ? Dict{AbstractString,Float64}() : subsampling_distr
 	WordStream(file, rate, filter, distr)
end


"Returns the next word without considering rate. Mutated the filepointer state"
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
                return s
           end
        else #Non Whitespace character
            write(next_word,c)
        end
    end
    #Hit EOF
    s = takebuf_string(next_word)
    filter_out(s) ? "" : s
end


function Base.start(ws::WordStream)
    if isa(ws.source, AbstractString)
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
Base.eltype(::Type{WordStream})=AbstractString
function Base.next(ws::WordStream, state)
    (next_word, fp) = state
    while(!eof(fp))
        if ws.rate > 0
            prob = (sqrt(ws.distr[next_word] / ws.rate) + 1) * ws.rate / ws.distr[next_word]
            if(prob < rand())
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
    return (pstring(next_word),("",fp))
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
    words = AbstractString["" for ii in 1:window_length(window)]
    
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

###########################################



type ProgressIter{T}
	state2complete_at :: Function
	state2completion_level :: Function
	source_iter::T
end


function ProgressIter(ws::WordStream)
	function s2c_at(state)
		fp = state[end]
		mark(fp)
		complete_at = position(seekend(fp))
		reset(fp)
		complete_at
	end,	
	function s2c_lvl(state)
		position(state[end])
	end
	ProgressIter(s2c_at, s2c_lvl, ws) 
end

function ProgressIter(win::SlidingWindow)
	ws_prog_iter = ProgressIter(win.ws)
	s2c_at(state)  = ws_prog_iter.state2complete_at(state[end])
	s2c_lvl(state) = ws_prog_iter.state2completion_level(state[end])

	ProgressIter(s2c_at, s2c_lvl, win)
end


"""Similar to enumerate, but showing portion of completion. 
Returns an iterator, that yields `(p,x[i])` 
where x[i] is the it element of `x`, and `p≈i/length(x)`.
**Note:**This is only a rough estimate of progress."""
enumerate_progress(x) = ProgressIter(x)

function Base.start(prog::ProgressIter)
	source_initial_state =  start(prog.source_iter)
	prog.state2complete_at(source_initial_state), source_initial_state
end

Base.done(prog::ProgressIter, state) = done(prog.source_iter,state[end])
Base.iteratorsize(::ProgressIter) = Base.SizeUnknown()
#Base.eltype(::Type{ProgressIter{T}})=Tuple{Float64,eltype(::T)}

function Base.next(prog::ProgressIter, state)
	complete_at, source_state = state	
	value, source_state = next(prog.source_iter, source_state)
	cur_level = prog.state2completion_level(source_state)
	(cur_level/complete_at, value), (complete_at, source_state)

end


end #End Module
