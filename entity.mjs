import {NeuralNetwork} from "./nn.mjs"

class Entity{
    constructor(genome, calculateFitness){
        this.genome = genome
        this.nn     = new NeuralNetwork(genome)
        this.nn.calculateFitness = calculateFitness
    }

    clone(){
        return new Entity(this.genome.clone(), this.calculateFitness)
    }

    serializableClone(){
        return {
            genome: this.genome.serializableClone(),
            fitness: this.nn.fitness
        }
    }
}

export {Entity}