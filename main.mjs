import {Entity} from "./entity.mjs"
import * as pattern from "./pattern_support.mjs"
import * as xor     from "./xor_support.mjs"
import {createContext, percent, chooseRandomly} from "./lib.mjs" 
import * as params from "./parameters.mjs"

var generation = []
var history    = []
var species    = []
var ctxt       = createContext()

var gen_count = 0

function initXOR(){
    generation = []
    history    = []
    species    = []
    ctxt       = createContext()
    ctxt.calculateFitness = xor.calculateFitness

    for(var i = 0; i < params.max_pop; i++){
        generation.push(new Entity(xor.initialGenome(ctxt), ctxt.calculateFitness))
    }

    gen_count = 0
}

function initPATTERN(){
    generation = []
    history    = []
    species    = []
    ctxt       = createContext()
    ctxt.calculateFitness = pattern.calculateFitness

    for(var i = 0; i < params.max_pop; i++){
        generation.push(new Entity(pattern.initialGenome(ctxt), ctxt.calculateFitness))
    }

    gen_count = 0
}

function chooseRandomlyFromTopNPercentProportionateToFitness(members, n_percent){
    var limit = Math.max(2, Math.floor(members.length * n_percent / 100) )
    var total_fitness = 0

    for(var i = 0; i < limit; i++){
        total_fitness += members[i].nn.fitness
    }

    var ball = Math.random() * total_fitness

    var selected = null

    var accum = 0
    for(var wheel = 0; wheel < limit; wheel++){
        if(ball >= accum && ball <= accum + members[wheel].nn.fitness){
            selected = members[wheel]

            break
        }else{
            accum += members[wheel].nn.fitness
        }
    }

    return selected
}

function generateXOR(callback){

    // measure performance of the neural network

    for(var entity of generation){
        entity.nn.calculateFitness()
    }

    // init specie data

    for(var specie of species){
        specie.members     = []
        specie.sum_fitness = 0
    }


    // placing entities into species

    for(var entity of generation) {
        var allotted = false

        for(var i = 0; i < species.length; i++){
            var specie = species[i]

            if(entity.genome.distanceFrom(specie.champion.genome) < params.comp_thresh){
                allotted = true

                // for members to mate intra-species
                specie.members.push(entity)

                if(entity.nn.fitness > specie.champion.nn.fitness){
                    if ( (!specie.next_champion) || (entity.nn.fitness > specie.next_champion.nn.fitness)) {
                        specie.next_champion = entity
                    }
                }
                
                if(entity.nn.fitness > specie.max_fitness){
                    specie.max_fitness = entity.nn.fitness
                }

                specie.sum_fitness += entity.nn.fitness

                break
            }

        }

        if(!allotted){
            species.push({
                id         : ctxt.getSpecieID(),
                champion   : entity,
                members    : [entity],
                max_fitness: entity.nn.fitness,
                sum_fitness: entity.nn.fitness
            })
        }

    }

    species = species.filter((s)=>s.members.length > 0)

    // adjusted fitness
    for(var specie of species){

        if(specie.next_champion){
            specie.champion = specie.next_champion
        }

        var sum_adjusted_fitness = 0

        for(var entity of specie.members){
            entity.nn.adjusted_fitness = entity.nn.fitness / specie.members.length

            sum_adjusted_fitness += entity.nn.adjusted_fitness
        }

        specie.avg_adjusted_fitness = sum_adjusted_fitness / specie.members.length

        specie.members.sort( (a, b) => a.nn.fitness - b.nn.fitness ).reverse()
    }


    var to_remove = []

    // checking if species are stagnant and eliminating
    for(var specie of species){
        ctxt.tickSpecie(specie.id, specie.max_fitness)
        
        
        if(species.length > 1){
            if(ctxt.eliminateSpecie(specie.id)){
                to_remove.push(specie)
            }
        }
    }

    for(var specie of to_remove){
        species.splice(species.indexOf(specie), 1)
    }

    // checking if population is stagnant and retaining top two species    
    species.sort((s1, s2)=>s1.max_fitness - s2.max_fitness).reverse()

    var max_fitness = species[0].max_fitness
    var global_champion = species[0].champion

    ctxt.tickPop(max_fitness)
    if(ctxt.refocusPop()){
        species.splice(2)
    }

    var next_gen = []
    var global_sum_adjusted_fitness = 0

    for(var specie of species){
        global_sum_adjusted_fitness += specie.avg_adjusted_fitness

        if(specie.members.length > 5){
            next_gen.push(specie.champion)
        }
    }

    var retained_count = next_gen.length

    // tasking species to create offsprings proportionate to their avg adjusted fitness

    for(var i = 0; i < species.length; i++){
        var specie = species[i]

        specie.offsprings_to_produce = Math.floor( (params.max_pop - retained_count) * ( specie.avg_adjusted_fitness / global_sum_adjusted_fitness ))
    }

    // reproduce

    for(var specie of species){

        for(var i = 0; i < specie.offsprings_to_produce; i++){
            var is_sexual = specie.members.length > 1 && percent(params.crossover_occurrence_chance)

            var child_entity = null

            if(is_sexual){

                try{
                    var parentA = chooseRandomlyFromTopNPercentProportionateToFitness(specie.members, 10)
                    var parentB = parentA
    
                    while (parentB == parentA){
                        if( species.length > 1 && percent( params.interspecies_mating_chance ) ){
                            var random_specie = specie
    
                            while(random_specie == specie){
                                random_specie = chooseRandomly(species)
                            }
    
                            parentB = chooseRandomlyFromTopNPercentProportionateToFitness(random_specie.members, 10)
                        }else{
                            parentB = chooseRandomlyFromTopNPercentProportionateToFitness(specie.members, 10)
                        }
                    }
    
                    child_entity = new Entity(parentA.genome.crossover(parentB.genome), ctxt.calculateFitness)
                }catch(e){
                    child_entity = new Entity( chooseRandomly(specie.members).genome.clone(), ctxt.calculateFitness )    
                }

            }else{
                child_entity = new Entity( chooseRandomly(specie.members).genome.clone(), ctxt.calculateFitness )
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

        stats.species.push({id: specie.id, champion: specie.champion.clone()})
    }

    for(var entity of generation){
        stats.pop.push({ fitness: entity.nn.fitness, nodes: entity.genome.nodes.length, connections: entity.genome.connections.length })
    }

    history.push(stats)

    console.log("\nGen " + history.length)

    console.log(max_fitness + " vs " + ctxt.globalStats.max_fitness)

    var ans1 = Math.round(global_champion.nn.predict([0, 0])[0])
    var ans2 = Math.round(global_champion.nn.predict([0, 1])[0])
    var ans3 = Math.round(global_champion.nn.predict([1, 0])[0])
    var ans4 = Math.round(global_champion.nn.predict([1, 1])[0])


    console.log("0 0 -> " + ans1)
    console.log("0 1 -> " + ans2)
    console.log("1 0 -> " + ans3)
    console.log("1 1 -> " + ans4)

    console.log("hidden nodes " + (global_champion.genome.nodes.length - 4) )

    generation = next_gen
    ctxt.resetCombn()
}

function generatePATTERN(callback){

    // measure performance of the neural network
    var solution_found_this_gen = false

    for(var entity of generation){
        entity.is_solution = entity.nn.calculateFitness()
        entity.baggage_nodes = 0
        entity.baggage_conns = 0
        entity.baggage = 0

        var enabled_nodes = new Set()

        for(var connection of entity.genome.connections){
            if(!connection.is_disabled){
                enabled_nodes.add(connection.from)
                enabled_nodes.add(connection.to)

                entity.baggage++
                entity.baggage_conns++
            }
        }

        entity.baggage += enabled_nodes.size
        entity.baggage_nodes = enabled_nodes.size

        if(entity.is_solution){
            solution_found_this_gen = true
        }
    }

    // init specie data

    for(var specie of species){
        specie.members      = []
        specie.sum_fitness  = 0
        specie.has_solution = false
    }


    // placing entities into species

    for(var entity of generation) {
        var allotted = false

        for(var i = 0; i < species.length; i++){
            var specie = species[i]

            if(entity.genome.distanceFrom(specie.champion.genome) < params.comp_thresh){
                allotted = true

                // for members to mate intra-species
                specie.members.push(entity)

                if(entity.nn.fitness > specie.champion.nn.fitness){
                    if ( (!specie.next_champion) || (entity.nn.fitness > specie.next_champion.nn.fitness)) {
                        specie.next_champion = entity
                    }
                }
                
                if(entity.nn.fitness > specie.max_fitness){
                    specie.max_fitness = entity.nn.fitness
                }

                specie.sum_fitness += entity.nn.fitness

                if(entity.is_solution){
                    specie.has_solution = true
                }

                break
            }

        }

        if(!allotted){
            species.push({
                id         : ctxt.getSpecieID(),
                champion   : entity,
                members    : [entity],
                max_fitness: entity.nn.fitness,
                sum_fitness: entity.nn.fitness
            })
        }

    }

    species = species.filter((s)=>s.members.length > 0)

    // adjusted fitness
    for(var specie of species){

        if(specie.next_champion){
            specie.champion = specie.next_champion
        }

        var sum_adjusted_fitness = 0

        for(var entity of specie.members){
            entity.nn.adjusted_fitness = entity.nn.fitness / specie.members.length

            sum_adjusted_fitness += entity.nn.adjusted_fitness
        }

        specie.avg_adjusted_fitness = sum_adjusted_fitness / specie.members.length

        specie.members.sort( (a, b) => a.nn.fitness - b.nn.fitness ).reverse()
    }


    var to_remove = []

    // checking if species are stagnant and eliminating
    for(var specie of species){
        ctxt.tickSpecie(specie.id, specie.max_fitness)
        
        
        if(species.length > 1){
            if(ctxt.eliminateSpecie(specie.id)){
                to_remove.push(specie)
            }
        }
    }

    for(var specie of to_remove){
        species.splice(species.indexOf(specie), 1)
    }

    // checking if population is stagnant and retaining top two species    
    species.sort((s1, s2)=>s1.max_fitness - s2.max_fitness).reverse()

    var max_fitness = species[0].max_fitness
    var global_champion = species[0].champion

    ctxt.tickPop(max_fitness)
    if(ctxt.refocusPop()){
        species.splice(2)
    }

    var next_gen = []
    var global_sum_adjusted_fitness = 0

    for(var specie of species){
        global_sum_adjusted_fitness += specie.avg_adjusted_fitness

        if(specie.members.length > 5){
            next_gen.push(specie.champion)
        }
    }

    var retained_count = next_gen.length

    // tasking species to create offsprings proportionate to their avg adjusted fitness

    for(var i = 0; i < species.length; i++){
        var specie = species[i]

        specie.offsprings_to_produce = Math.floor( (params.max_pop - retained_count) * ( specie.avg_adjusted_fitness / global_sum_adjusted_fitness ))
    }

    // reproduce

    for(var specie of species){

        for(var i = 0; i < specie.offsprings_to_produce; i++){
            var is_sexual = specie.members.length > 1 && percent(params.crossover_occurrence_chance)

            var child_entity = null

            if(is_sexual){

                try{
                    var parentA = chooseRandomlyFromTopNPercentProportionateToFitness(specie.members, 10)
                    var parentB = parentA
    
                    while (parentB == parentA){
                        if( species.length > 1 && percent( params.interspecies_mating_chance ) ){
                            var random_specie = specie
    
                            while(random_specie == specie){
                                random_specie = chooseRandomly(species)
                            }
    
                            parentB = chooseRandomlyFromTopNPercentProportionateToFitness(random_specie.members, 10)
                        }else{
                            parentB = chooseRandomlyFromTopNPercentProportionateToFitness(specie.members, 10)
                        }
                    }
    
                    child_entity = new Entity(parentA.genome.crossover(parentB.genome), ctxt.calculateFitness)
                }catch(e){
                    child_entity = new Entity(chooseRandomly(specie.members).genome.clone(), ctxt.calculateFitness )
                }

            }else{
                child_entity = new Entity(chooseRandomly(specie.members).genome.clone(), ctxt.calculateFitness )
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

        var best = null
        var worst = null

        if(specie.has_solution){
            var members = specie.members.filter((entity)=>entity.is_solution)
            
            members.sort((a, b)=>a.baggage - b.baggage)
    
            best = members[0].serializableClone()
            worst = members[members.length - 1].serializableClone()
        }

        stats.species.push({id: specie.id, members: specie.members.length, best: best, worst: worst, avg_adjusted_fitness: specie.avg_adjusted_fitness})
    }

    for(var entity of generation){
        stats.pop.push({ fitness: entity.nn.fitness, nodes: entity.baggage_nodes, conns: entity.baggage_conns, is_solution = entity.is_solution })
    }

    callback(stats, solution_found_this_gen)

    var ans1 = Math.round(global_champion.nn.predict([0])[0])
    var ans2 = Math.round(global_champion.nn.predict([0])[0])
    var ans3 = Math.round(global_champion.nn.predict([0])[0])
    var ans4 = Math.round(global_champion.nn.predict([0])[0])
    var ans5 = Math.round(global_champion.nn.predict([0])[0])
    var ans6 = Math.round(global_champion.nn.predict([0])[0])

    console.log(`GEN ${++gen_count}    Pattern: ${ans1} ${ans2} ${ans3} ${ans4} ${ans5} ${ans6}`)

    generation = next_gen
    ctxt.resetCombn()
}

export {initPATTERN, generatePATTERN}
