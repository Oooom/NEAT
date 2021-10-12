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
    var ans1 = clamp(this.predict([0, 0])[0])
    var ans2 = clamp(this.predict([0, 1])[0])
    var ans3 = clamp(this.predict([1, 0])[0])
    var ans4 = clamp(this.predict([1, 1])[0])

    this.fitness = (4 - ( (ans1 - 0) + (1 - ans2) + (1 - ans3) + (ans4 - 0))) ** 2

    if(Math.round(ans1) == 0 && Math.round(ans2) == 1 && Math.round(ans3) == 1 && Math.round(ans4) == 0){
        return true
    }

    return false
}

function initialGenome(ctxt){
    var i1 = new NodeGene("i1", "input", sigmoid)
    var i2 = new NodeGene("i2", "input", sigmoid)

    var b = new NodeGene("b", "bias")
    b.op = 1

    var o1 = new NodeGene("o1", "output", sigmoid)

    var i1_o1 = new ConnectGene("i1", "o1", NaN, 1, false)
    i1_o1.mutateRandomly()
    var i2_o1 = new ConnectGene("i2", "o1", NaN, 2, false)
    i2_o1.mutateRandomly()
    var b_o1  = new ConnectGene("b" , "o1", NaN, 3, false)
    b_o1.mutateRandomly()

    var genome = new Genome([i1, i2, b, o1], [i1_o1, i2_o1, b_o1], ctxt)

    ctxt.innov = 4

    return genome
}

function solutionGenome(ctxt){
    var i1 = new NodeGene("i1", "input")
    var i2 = new NodeGene("i2", "input")

    var h1 = new NodeGene(ctxt.getHiddenNodeID("i1", "o1"), "hidden", sigmoid)
    var h2 = new NodeGene(ctxt.getHiddenNodeID("i2", "o1"), "hidden", sigmoid)

    var o1 = new NodeGene("o1", "output", sigmoid)

    var bh1 = new NodeGene("bh1", "bias")
    bh1.op = -4.8419018689338325
    
    var bh2 = new NodeGene("bh2", "bias")
    bh2.op = -2.1328499034609063
    
    var bo1 = new NodeGene("bo1", "bias")
    bo1.op = -2.8804196258629196


    var i1_h1 = new ConnectGene("i1", "h1", 3.191352509627136, 1, false)
    var i1_h2 = new ConnectGene("i1", "h2", 5.491641199017287, 2, false)
    
    var i2_h1 = new ConnectGene("i2", "h1", 3.179256100003881, 3, false)
    var i2_h2 = new ConnectGene("i2", "h2", 5.444467626110183, 4, false)
    
    var h1_o1 = new ConnectGene("h1", "o1", -6.998221924119023, 5, false)
    var h2_o1 = new ConnectGene("h2", "o1",  6.488756231728473, 6, false)

    var bh1_h1 = new ConnectGene("bh1", "h1", 1, 7, false)
    var bh2_h2 = new ConnectGene("bh2", "h2", 1, 8, false)
    var bo1_o1 = new ConnectGene("bo1", "o1", 1, 9, false)

    var genome = new Genome([i1, i2, h1, h2, o1, bh1, bh2, bo1], [i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1], ctxt)

    ctxt.innov = 10

    return genome
}

export {calculateFitness, initialGenome, solutionGenome}
