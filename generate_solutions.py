import algorithms

path = 'data/no_rep_chromo_bench/chr'

for i in range(1, 24):
    if i == 23:
        i = 'X'
    data = path + str(i)
    algorithms.get_solutions_for_all(data)

