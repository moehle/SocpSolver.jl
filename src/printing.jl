function print_first_line(stats::Stats, settings::Settings)
    if settings.verbose
        printline = "\n"
        printline *= """
                 -------------------------------------------------
                   SocpSolver v0.0.1 - (c) Nicholas Moehle, 2021  
                 -------------------------------------------------
        """ 
        printline *= @sprintf "iter  "
        printline *= @sprintf "resid  "
        printline *= @sprintf "best  "
        printline *= @sprintf "  μ    "
        printline *= @sprintf "  σ    "
        printline *= @sprintf "  α    " 
        printline *= @sprintf " reg   " 
        printline *= @sprintf "|  IR iter  | "
        printline *= @sprintf " prec"
        printline *= "\n"
        print(printline)
    end
end


function print_stats_iter(stats::Stats, settings::Settings)
    if settings.verbose
        num_digits = ceil(log10(settings.max_iters))
        i = stats.iters
        printline = ""
        printline *= @sprintf "%4d  " stats.iters
        printline *= @sprintf "%5.0e  " stats.residual[i]
        if stats.best[i]
            printline *= @sprintf " xx   "
        else
            printline *= @sprintf "      "
        end
        printline *= @sprintf "%5.0e  " stats.μ[i]
        printline *= @sprintf "%5.0e  " stats.σ[i]
        printline *= @sprintf "%5.0e  " stats.α_comb[i]
        printline *= @sprintf "%5.0e  " stats.regularization[i]
        if i % 2 == 1
            printline *= @sprintf "/  " 
        else
            printline *= @sprintf "\\  " 
        end
        printline *= @sprintf "%1d  " stats.ir_iters_cache[i]
        printline *= @sprintf "%1d  " stats.ir_iters_aff[i]
        printline *= @sprintf "%1d  " stats.ir_iters_comb[i]
        if i % 2 == 1
            printline *= @sprintf "\\ " 
        else
            printline *= @sprintf "/ " 
        end
        printline *= @sprintf " f64 "
        printline *= "\n"
        print(printline)
    end
end
