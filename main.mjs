// populate initial generation of entities
/*
generate
    - calculate fitness
    - speciate
    - share fitness
    - reproduce

*/

import {Entity} from "./entity.mjs"
import {initialGenome} from "./xor_support.mjs"

var max_pop     = 150
var comp_thresh = 3.0 

var generation = []
var species    = []


function init(){
    generation = []

    for(var i = 0; i < max_pop; i++){
        generation.push(new Entity(initialGenome()))
    }
}

function generate(){
    for(var entity of generation){
        entity.nn.calculateFitness()
    }

    for(var specie of species){
        if(specie.champion){
            specie.count = 1
            specie.sum_fitness = specie.champion.nn.fitness
            specie.members = []
        }
    }

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

                if(entity.nn.fitness > specie.champion){
                    specie.next_champion = entity
                }

                break
            }

        }

        if(!allotted){
            specie.push({champion: entity, count: 1, sum_fitness: entity.nn.fitness, members: [entity]})
        }

    }

    var next_gen = []
    var sum_species_fitness = 0

    for(var specie of species){

        specie.shared_fitness = specie.sum_fitness / specie.count

        sum_species_fitness += specie.shared_fitness

        if(specie.count >= 5){

            if(specie.next_champion){
                specie.champion = specie.next_champion
            }

            next_gen.push(specie.champion)
        }
    }

    var offsprings_to_produce = max_pop - next_gen.length

    var total_allotted = 0

    for(var i = 0; i < species.length - 1; i++){
        var specie = species[i]

        specie.offsprings_to_produce = Math.floor(offsprings_to_produce * ( specie.shared_fitness / sum_species_fitness ))

        total_allotted += specie.offsprings_to_produce
    }

    species[species.length - 1].offsprings_to_produce = offsprings_to_produce - total_allotted

    // reproduce
}
