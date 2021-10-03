import assert from "assert"
import { ConnectGene } from "../genome.mjs"
import { Genome } from "../genome.mjs"
import { NodeGene } from "../genome.mjs"
import { sigmoid, steepSigmoid } from "../nn.mjs"
import {NeuralNetwork} from "../nn.mjs"


describe("loadOPofIP tests", function(){
    
    var n1 = new NodeGene("i-1", "input")
    var n2 = new NodeGene("i-2", "input")
    
    var g1 = new Genome([n1], [])
    var g2 = new Genome([n2, new NodeGene("bias", "bias"), n1], [])
    
    it("single node", function () {
        var nn1 = new NeuralNetwork(g1)

        nn1.loadOPofIP([0.5])

        assert.equal(n1.op, 0.5)
    })
    
    it("multiple nodes, jumbled order", function () {
        var nn2 = new NeuralNetwork(g2)

        nn2.loadOPofIP([0.5, 1.5])

        assert.equal(n1.op, 1.5)
        assert.equal(n2.op, 0.5)
    })
})

describe("detectCycle tests", function(){
    
    it("single node self loop", function () {
        
        var h1 = new NodeGene("h-1", "hidden")
        var self_connection = new ConnectGene("h-1", "h-1", 0, 0, false)

        var genome = new Genome([h1], [self_connection])
        
        assert.equal(genome.detectCycle(self_connection), true, "self loop not detected as cycle")
    })
    
    it("multi node feed forward", function () {

        var h1 = new NodeGene("h-1", "hidden")
        var h2 = new NodeGene("h-2", "hidden")
        var h3 = new NodeGene("h-3", "hidden")
        var h1_h2 = new ConnectGene("h-1", "h-2", 0, 0, false)
        var h2_h3 = new ConnectGene("h-2", "h-3", 0, 0, false)

        var genome = new Genome([h1, h2, h3], [h1_h2, h2_h3])

        assert.equal(genome.detectCycle(h2_h3), false, "false cycle detected")
    })

    it("multi node multi feed forward", function () {
        
        var h1 = new NodeGene("h-1", "hidden")
        var h2 = new NodeGene("h-2", "hidden")
        var h3 = new NodeGene("h-3", "hidden")
        var h1_h2 = new ConnectGene("h-1", "h-2", 0, 0, false)
        var h2_h3 = new ConnectGene("h-2", "h-3", 0, 0, false)
        var h1_h3 = new ConnectGene("h-1", "h-3", 0, 0, false)

        var genome = new Genome([h1, h2, h3], [h1_h2, h2_h3, h1_h3])

        assert.equal(genome.detectCycle(h1_h3), false, "false cycle detected")
    })

    it("multi node multi feed forward with loop", function () {
        
        var h1 = new NodeGene("h-1", "hidden")
        var h2 = new NodeGene("h-2", "hidden")
        var h3 = new NodeGene("h-3", "hidden")
        var h1_h2 = new ConnectGene("h-1", "h-2", 0, 0, false)
        var h2_h3 = new ConnectGene("h-2", "h-3", 0, 0, false)
        var h1_h3 = new ConnectGene("h-1", "h-3", 0, 0, false)
        var h3_h1 = new ConnectGene("h-3", "h-1", 0, 0, false)

        var genome = new Genome([h1, h2, h3], [h1_h2, h2_h3, h1_h3, h3_h1])

        assert.equal(genome.detectCycle(h3_h1), true, "cycle not detected")
    })

    it("multi node multi feed forward with unrelated loop", function () {

        var h1 = new NodeGene("h-1", "hidden")
        var h2 = new NodeGene("h-2", "hidden")
        var h3 = new NodeGene("h-3", "hidden")
        var h1_h2 = new ConnectGene("h-1", "h-2", 0, 0, false)
        var h2_h3 = new ConnectGene("h-2", "h-3", 0, 0, false)
        var h1_h3 = new ConnectGene("h-1", "h-3", 0, 0, false)
        var h3_h1 = new ConnectGene("h-3", "h-1", 0, 0, false)

        h3_h1.is_recurrent = true

        var genome = new Genome([h1, h2, h3], [h1_h2, h2_h3, h1_h3, h3_h1])

        assert.equal(genome.detectCycle(h1_h3), false, "false cycle detected")
    })
    
    it("two nodes with cycle", function () {

        var h1 = new NodeGene("h-1", "hidden")
        var h2 = new NodeGene("h-2", "hidden")
        var h1_h2 = new ConnectGene("h-1", "h-2", 0, 0, false)
        var h2_h1 = new ConnectGene("h-2", "h-1", 0, 0, false)

        var genome = new Genome([h1, h2], [h1_h2, h2_h1])

        assert.equal(genome.detectCycle(h2_h1), true, "cycle not detected")
    })

})

describe("evaluation tests", function(){
    it("minimal xor", function () {
        var i1 = new NodeGene("i-1", "input")
        var i2 = new NodeGene("i-2", "input")

        var h1 = new NodeGene("h-1", "hidden", sigmoid)
        var h2 = new NodeGene("h-2", "hidden", sigmoid)

        var o1 = new NodeGene("o-1", "output", sigmoid)

        var bh1 = new NodeGene("b-h1", "bias")
        bh1.op = -4.8419018689338325
        
        var bh2 = new NodeGene("b-h2", "bias")
        bh2.op = -2.1328499034609063
        
        var bo1 = new NodeGene("b-o1", "bias")
        bo1.op = -2.8804196258629196


        var i1_h1 = new ConnectGene("i-1", "h-1", 3.191352509627136, 0, false)
        var i1_h2 = new ConnectGene("i-1", "h-2", 5.491641199017287, 0, false)
        
        var i2_h1 = new ConnectGene("i-2", "h-1", 3.179256100003881, 0, false)
        var i2_h2 = new ConnectGene("i-2", "h-2", 5.444467626110183, 0, false)
        
        var h1_o1 = new ConnectGene("h-1", "o-1", -6.998221924119023, 0, false)
        var h2_o1 = new ConnectGene("h-2", "o-1",  6.488756231728473, 0, false)

        var bh1_h1 = new ConnectGene("b-h1", "h-1", 1, 0, false)
        var bh2_h2 = new ConnectGene("b-h2", "h-2", 1, 0, false)
        var bo1_o1 = new ConnectGene("b-o1", "o-1", 1, 0, false)

        var genome = new Genome([i1, i2, h1, h2, o1, bh1, bh2, bo1], [i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1])
        var nn     = new NeuralNetwork(genome)

        var ans1 = Math.round(nn.predict([0, 0])[0])
        var ans2 = Math.round(nn.predict([0, 1])[0])
        var ans3 = Math.round(nn.predict([1, 0])[0])
        var ans4 = Math.round(nn.predict([1, 1])[0])

        assert.equal(ans1, 0, "0-0 -> 0 case wrong")
        assert.equal(ans2, 1, "0-1 -> 1 case wrong")
        assert.equal(ans3, 1, "1-0 -> 1 case wrong")
        assert.equal(ans4, 0, "1-1 -> 0 case wrong")
    })

    it("self loop recurrent neuron", function () {
        var n = new NodeGene("o-1", "output", sigmoid)
        var bias = new NodeGene("b-o1", "bias")
        bias.op = 1

        var o1_o1 = new ConnectGene("o-1", "o-1", 1, 0, false)
        o1_o1.is_recurrent = true

        var b_o1  = new ConnectGene("b-o1", "o-1", 1, 0, false)

        var genome = new Genome([n, bias], [o1_o1, b_o1])

        var nn = new NeuralNetwork(genome)

        var op1 = nn.predict([])[0]
        var op2 = nn.predict([])[0]
        var op3 = nn.predict([])[0]

        assert.equal(op1, 0.7310585786300049, "recurrence sanity check output not matching")
        assert.equal(op2, 0.8495477739862124, "recurrence sanity check output not matching")
        assert.equal(op3, 0.8640739977337843, "recurrence sanity check output not matching")

    })
})
