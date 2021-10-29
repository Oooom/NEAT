var max_pop = 200
var comp_thresh = 3.0

var c_m = 0.4
var c_um = 1

var weight_mutation_chance = 80
var weight_mutation_chance_new_random = 5
var weight_mutation_chance_uniformly_perturb = 95

var add_node_mutation_chance = 0.1
var add_link_mutation_chance = 5

var interspecies_mating_chance = 0.1

var disable_offspring_gene_chance = 75

var crossover_occurrence_chance = 75

var stagnate_pop_after_gens = 20
var stagnate_specie_after_gens = 15

var top_n_percent_for_mating = 10

export { 
    c_m, c_um, 
    weight_mutation_chance, weight_mutation_chance_new_random, weight_mutation_chance_uniformly_perturb, 
    add_node_mutation_chance, add_link_mutation_chance, 
    disable_offspring_gene_chance, 
    crossover_occurrence_chance, interspecies_mating_chance, 
    max_pop, 
    comp_thresh,
    stagnate_pop_after_gens, stagnate_specie_after_gens,
    top_n_percent_for_mating
}