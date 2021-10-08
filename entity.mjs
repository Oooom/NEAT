import {NeuralNetwork} from "./nn.mjs"
import {calculateFitness} from "./xor_support.mjs"

class Entity{
    constructor(genome){
        this.genome = genome
        this.nn     = new NeuralNetwork(genome)
        this.nn.calculateFitness = calculateFitness
    }
}

export {Entity}