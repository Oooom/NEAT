import {Genome, NodeGene, ConnectGene} from "./genome.mjs"
import {sigmoid}               from "./nn.mjs"

function clamp(num){
    if(num > 1){
        return 1
    }

    if(num < 0){
        return 0
    }

    return num
}

function calculateFitness(){
    var ans1 = clamp(this.predict([0])[0])
    var ans2 = clamp(this.predict([0])[0])
    var ans3 = clamp(this.predict([0])[0])
    var ans4 = clamp(this.predict([0])[0])
    var ans5 = clamp(this.predict([0])[0])
    var ans6 = clamp(this.predict([0])[0])

    this.fitness = (6 - ( (ans1) + (1 - ans2) + (ans3) + (1 - ans4) + (ans5) + (1 - ans6))) ** 2

    if(Math.round(ans1) == 0 && Math.round(ans2) == 1 && Math.round(ans3) == 0 && Math.round(ans4) == 1 && Math.round(ans5) == 0 && Math.round(ans6) == 1){
        console.log("solution found.... ")
    }
}

function initialGenome(ctxt){
    var i1 = new NodeGene("i1", "input", sigmoid)

    var b = new NodeGene("b", "bias")
    b.op = 1

    var o1 = new NodeGene("o1", "output", sigmoid)

    var i1_o1 = new ConnectGene("i1", "o1", NaN, 1, false)
    i1_o1.mutateRandomly()
    var b_o1  = new ConnectGene("b" , "o1", NaN, 3, false)
    b_o1.mutateRandomly()

    var genome = new Genome([i1, b, o1], [i1_o1, b_o1], ctxt)

    ctxt.innov = 3

    return genome
}

export {calculateFitness, initialGenome}
