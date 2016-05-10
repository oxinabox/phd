# Print Hello World

immutable WorkFarmerIterator
    func :: Function
    source_iter
    channel_sizes::Int64
end

type WorkFarmerState
    pending::RemoteChannel
    complete::RemoteChannel
    job_submitter::Future
end

function Base.start(iter::WorkFarmerIterator)

    o_pending = RemoteChannel(iter.channel_sizes)
    o_complete = RemoteChannel(iter.channel_sizes)

    o_job_submitter = remotecall(myid(), o_pending) do pending #Can prob do this without a remotecall
      for wk in source_iter
          put!(pending, wk)
      end
      true
    end

    # Fire off workers that Read Pending (until closed exception)
    # And write to complete
        remotecall(default_worker_pool()) do
    end

    #Register Finalizer

    WorkFarmerState(o_pending,o_complete, o_job_submitter)
end

Base.done()
#Check that job_submitter is done
#and complete empty (or closed)
#and pending empty (or closed)
#This does leave it with a Potential Glitch that the last few elements might be
#missed as they are still being worked on, but I don't actually care

Base.next()
#Read complete -- this is a blocking operation so if nothing is ready we'll sit here til it is.

HasLength = HasLength(source)
