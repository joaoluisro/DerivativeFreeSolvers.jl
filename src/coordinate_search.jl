export coordinate_search

"""
	The coordinate search is a derivative free optimization method.
	It minimizes a certain function f by walking on a grid and taking steps towards smaller function values.
	When it can no longer find any smaller values, it reduces the grid by a factor of β, repeating the process.
	coordinate_search(nlp; options...)
	where "nlp" is an "AbstractNLPModel".
	standard options :
	tol = 1e-12
	α = 1.0
	β = 2.0
	max_time = 1e-1
	max_eval = 500
"""

function coordinate_search(nlp :: AbstractNLPModel;
			   tol :: Real = 1e-12,
			   α :: Real = 1.0,
			   β :: Real = 2.0,
			   x :: AbstractVector = nlp.meta.x0,
			   max_time :: Real  = 1e-1,
			   max_eval :: Int = 500)
	
  start_time = time()
  el_time = 0.0
  tired = neval_obj(nlp) > max_eval || el_time > max_time
  optimal = α <= tol
  T = eltype(x)
  status =:unknown
  k = 0
  @info log_header([:iter, :f, :H], [Int, T, T], hdr_override=Dict(:f=>"f(x)", :H=>"α"))

  while !(optimal || tired)
    f = obj(nlp,x)
    @info log_row(Any[k, f, α])

    for i in 1:nlp.meta.nvar
      success = false
      for s in [-1,1]
        xt = copy(x)
        xt[i] += α * s
        if (obj(nlp,x) < f)
          x = xt
          success = true
	  break
	end
    end
			
      if (success == false)
        α /= β
      end
			
      k += 1
    end
		
    optimal = α <= tol
    el_time = time() - start_time
    tired = neval_obj(nlp) > max_eval || el_time > max_time
		
  end

  if optimal
    status =:unknown
  elseif tired
    if (neval_obj(nlp) > max_eval && max_eval >= 0)
      status =:max_eval
    elseif el_time >= max_time
      status =:max_time
    end
  end

  return GenericExecutionStats(status, nlp, solution=x, objective=f,
				iter = k, elapsed_time = el_time)
end
