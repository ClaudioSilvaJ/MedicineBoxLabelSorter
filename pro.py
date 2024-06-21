import pstats


stats = pstats.Stats('resultados.prof')
stats.sort_stats('cumulative')
stats.print_stats()
stats.sort_stats('time')
stats.print_stats()
