// populate initial generation of entities
/*
generate
    - calculate fitness
    - speciate
    - share fitness
    - reproduce

*/

import {Entity} from "./entity.mjs"
import {initialGenome, calculateFitness} from "./xor_support.mjs"
import {createContext, percent, chooseRandomly} from "./lib.mjs" 
import * as params from "./parameters.mjs"

var max_pop     = 150
var comp_thresh = 3.0 

var generation = []
var history    = []
var species    = []
var ctxt       = createContext()



function init(){
    generation = []

    for(var i = 0; i < max_pop; i++){
        generation.push(new Entity(initialGenome(ctxt), calculateFitness))
    }
}

function generate(){

    // measure performance of the neural network

    for(var entity of generation){
        entity.nn.calculateFitness()
    }

    // init specie data

    for(var specie of species){
        if(specie.champion){
            specie.count = 0
            specie.sum_fitness = 0
            specie.members = []
        }
    }


    // placing entities into species

    for(var entity of generation) {
        var allotted = false

        for(var i = 0; i < species.length; i++){
            var specie = species[i]

            if(entity.genome.distanceFrom(specie.champion.genome) < comp_thresh){
                allotted = true
                
                // for calculating shared fitness and proportionate offsprings production
                specie.count++
                specie.sum_fitness += entity.nn.fitness

                // for members to mate intra-species
                specie.members.push(entity)

                if(entity.nn.fitness > specie.champion.nn.fitness){

                    if( (!specie.next_champion) || (specie.next_champion && entity.nn.fitness > specie.next_champion.nn.fitness) ){
                        specie.next_champion = entity
                    }

                    specie.max_fitness = entity.nn.fitness
                }

                break
            }

        }

        if(!allotted){
            species.push({
                id         : ctxt.getSpecieID(),
                champion   : entity,
                count      : 1,
                sum_fitness: entity.nn.fitness,
                members    : [entity],
                max_fitness: entity.nn.fitness
            })
        }

    }

    species = species.filter((s)=>s.members.length > 0)


    var to_remove = []

    // checking if species are stagnant and eliminating
    for(var specie of species){
        ctxt.tickSpecie(specie.id, specie.max_fitness)
        
        
        if(ctxt.eliminateSpecie(specie.id)){
            to_remove.push(specie)
        }
    }

    for(var specie of to_remove){
        species.splice(species.indexOf(specie), 1)
    }

    // checking if population is stagnant and retaining top two species    
    species.sort((s1, s2)=>s1.max_fitness - s2.max_fitness).reverse()

    var max_fitness = species[0].max_fitness

    ctxt.tickPop(max_fitness)
    if(ctxt.refocusPop()){
        species.splice(2)
    }

    var next_gen = []
    var sum_species_fitness = 0

    // sharing fitness

    for(var specie of species){

        specie.shared_fitness = specie.sum_fitness / specie.count

        sum_species_fitness += specie.shared_fitness

        specie.members.sort( (a, b) => a.nn.fitness - b.nn.fitness ).reverse()

        if(specie.count >= 5){

            if(specie.next_champion){
                specie.champion = specie.next_champion
            }

            next_gen.push(specie.champion)
            specie.next_champion = null
        }
    }

    var retained_count = next_gen.length
    var offsprings_to_produce = max_pop - retained_count

    var total_allotted = 0

    // tasking species to create offsprings proportionate to their shared fitness

    for(var i = 0; i < species.length - 1; i++){
        var specie = species[i]

        specie.offsprings_to_produce = Math.floor(offsprings_to_produce * ( specie.shared_fitness / sum_species_fitness ))

        total_allotted += specie.offsprings_to_produce

        if(specie.members.length - specie.offsprings_to_produce > 2){
            specie.members.splice(specie.members.length - specie.offsprings_to_produce)
        }
    }

    species[species.length - 1].offsprings_to_produce = offsprings_to_produce - total_allotted

    // reproduce

    for(var specie of species){

        for(var i = 0; i < specie.offsprings_to_produce; i++){
            var is_sexual = percent(params.crossover_occurrence_chance)

            var child_entity = null

            if(is_sexual){
                var parentA = chooseRandomly(specie.members)
                var parentB = parentA

                while (parentB == parentA){
                    if( species.length > 1 && percent( params.interspecies_mating_chance ) ){
                        var random_specie = specie

                        while(random_specie == specie){
                            random_specie = chooseRandomly(species)
                        }

                        parentB = chooseRandomly(random_specie.members)
                    }else{
                        parentB = chooseRandomly(specie.members)
                    }
                }

                child_entity = new Entity(parentA.genome.crossover(parentB.genome), calculateFitness)

            }else{
                child_entity = new Entity( chooseRandomly(specie.members).genome.clone(), calculateFitness )
            }

            next_gen.push(child_entity)
        }

    }

    for(var i = retained_count; i < next_gen.length; i++){
        var child = next_gen[i]

        child.genome.mutate()
    }

    var stats = {
        species: [],
        pop    : []
    }

    for(var i = 0; i < species.length; i++){
        var specie = species[i]

        stats.species.push({id: specie.id, count: specie.count, shared_fitness: specie.shared_fitness, champion: specie.champion.clone()})
    }

    for(var entity of generation){
        stats.pop.push({ fitness: entity.nn.fitness, nodes: entity.genome.nodes.length, connections: entity.genome.connections.length })
    }

    history.push(stats)

    console.log(max_fitness + " vs " + ctxt.globalStats.max_fitness)

    generation = next_gen
    ctxt.resetCombn()
}

init()

var flag = true

while(flag){
    generate()
}
