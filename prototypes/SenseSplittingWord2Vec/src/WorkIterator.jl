
"A kind of async parallel iterator map, that does not quiet  presever order."
immutable WorkFarmerIterator{T}
    func :: Function #Must take an tuple of const_data, and an element of source_iter and do work on them
    source_iter
	const_data::T
	channel_sizes::Int64
end

type WorkFarmerState
    pending::RemoteChannel
    complete::RemoteChannel
    job_submitter::Future
end

function Base.start(iter::WorkFarmerIterator)

    o_pending = RemoteChannel(()->Channel(iter.channel_sizes),myid())
    o_complete = RemoteChannel(()->Channel(iter.channel_sizes),myid())

    o_job_submitter = remotecall(myid(), o_pending, iter.source_iter) do pending, source_iter
		#Can prob do this without a remotecall
      for wk in source_iter
          put!(pending, wk)
      end
      true
    end

    # Fire off workers that Read Pending (until closed exception)
    # And write to complete
	@sync for pid in workers()
        @async remotecall(pid, iter.func, iter.const_data, o_pending,o_complete) do func, const_data, pending, complete			
			try
				while(true)
					wk = take!(pending) #Block til work arrives
					res = func(const_data, wk)
					put!(complete, res) #Block til I can hand work over
				end			
			catch ee
				#Break out of loop when a stream is closed and we try to read pending`
				#Or in the case of a unfortunte timing error we try to write to complete
				if !(ee|>typeof ==InvalidStateException && ee.state == :closed)
					rethrow()
				end
				#otherwise eat the InvalidStateException
			end
		end
	end

    #Register Finalizer
	state=WorkFarmerState(o_pending,o_complete, o_job_submitter)
	function finalize_state(st::WorkFarmerState)
		close(st.pending)
		close(st.complete)
	end
	finalizer(state, finalize_state)
	state
end

function Base.done(iter::WorkFarmerIterator, state::WorkFarmerState)
	#Check that job_submitter is done
	isdone= (isready(state.job_submitter) #Finish submitting
			&&  !isready(state.complete) #and complete work closed or empty
			&&  !isready(state.pending)  #and pending empty 
			)
	#We can't actually check if the RemoteChannel is closed.
	if isdone
		finalize(state) #Done not hurt to finalize it twice
	end
	isdone
end

function Base.next(iter::WorkFarmerIterator, state::WorkFarmerState)
	#Read complete -- this is a blocking operation so if nothing is ready we'll sit here til it is.
	res = take!(state.complete)
	(res,state)
end

Base.iteratorsize(iter::WorkFarmerIterator) = Base.iteratorsize(iter.source_iter)

Base.length(iter::WorkFarmerIterator) = Base.length(iter.source_iter)

