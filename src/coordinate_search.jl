export coordinate_search

"""
	The coordinate search is a derivative free optimization method.
	It minimizes a certain function f by walking on a grid and taking a step where the function values are smaller.
	When it can no longer find any smaller values, it reduces the grid by a factor of 2, repeating the process.

"""

function coordinate_search(nlp :: AbstractNLPModel;
						tol = 1e-12,
						k = 0,
						α = 1.0,
						xk = nlp.meta.x0,
						el_time = 0.0,
						max_time :: Real  = 1e-1,
						max_eval = 500)


	start_time = time()
	tired = neval_obj(nlp) > max_eval || el_time > max_time
	optimal = α <= tol
	T = eltype(xk)
	status =:unknown
	@info log_header([:iter, :f, :H],[Int, T, T], hdr_override=Dict(:f=>"f(x)", :H=>"α"))


    while !(optimal || tired)
		@info log_row([k, obj(nlp,xk), α])

        for i in 1:nlp.meta.nvar
            success = false
            for s in [-1,1]
                xt = copy(xk)
                xt[i] += α * s
                if (obj(nlp,xt) < obj(nlp,xk))
                    xk = xt
                    success = true
                    break
                end
            end

			if (success == false)
				α /= 2
			end
			k += 1
        end
		tired = neval_obj(nlp) > max_eval || el_time > max_time
		optimal = α <= tol
		el_time = time() - start_time
    end

	if (optimal)
		status =:first_order
	elseif (tired)
		if (neval_obj(nlp) > max_eval && max_eval >= 0)
			status =:max_eval
		elseif (el_time >= max_time)
			status =:max_time
		end
	end

	return GenericExecutionStats(status,nlp,solution=xk,objective=obj(nlp,xk),
								iter = k,elapsed_time = el_time)
end
