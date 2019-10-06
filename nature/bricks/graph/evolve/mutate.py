from random import choice

from nature import insert_motifs

MUTATION_FNS = [insert_motifs]
N_MUTATIONS = 2


def mutate(G):
    for mutation_number in range(N_MUTATIONS):
        G = choice(MUTATION_FNS)(G)
    return G
