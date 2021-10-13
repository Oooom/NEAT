import assert from "assert"
import { ConnectGene } from "../genome.mjs"
import { Genome } from "../genome.mjs"
import { NodeGene } from "../genome.mjs"
import { sigmoid, steepSigmoid } from "../nn.mjs"
import {NeuralNetwork} from "../nn.mjs"
import {initialGenome, solutionGenome} from "../xor_support.mjs"

import {createContext} from "../lib.mjs"


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

describe("genome compatibility tests", function () {

    it("distance with self should be 0", function(){
        
        // COPIED FROM MINIMAL XOR TEST
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


        var i1_h1 = new ConnectGene("i-1", "h-1", 3.191352509627136, 1, false)
        var i1_h2 = new ConnectGene("i-1", "h-2", 5.491641199017287, 2, false)
        
        var i2_h1 = new ConnectGene("i-2", "h-1", 3.179256100003881, 3, false)
        var i2_h2 = new ConnectGene("i-2", "h-2", 5.444467626110183, 4, false)
        
        var h1_o1 = new ConnectGene("h-1", "o-1", -6.998221924119023, 5, false)
        var h2_o1 = new ConnectGene("h-2", "o-1",  6.488756231728473, 6, false)

        var bh1_h1 = new ConnectGene("b-h1", "h-1", 1, 7, false)
        var bh2_h2 = new ConnectGene("b-h2", "h-2", 1, 8, false)
        var bo1_o1 = new ConnectGene("b-o1", "o-1", 1, 9, false)

        var genome = new Genome([i1, i2, h1, h2, o1, bh1, bh2, bo1], [i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1])
        // COPY END

        assert.equal(genome.distanceFrom(genome), 0, "distance with self not 0")
    })

    it("distance with identical not 0", function(){
        
        // COPIED FROM MINIMAL XOR TEST
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


        var i1_h1 = new ConnectGene("i-1", "h-1", 3.191352509627136, 1, false)
        var i1_h2 = new ConnectGene("i-1", "h-2", 5.491641199017287, 2, false)
        
        var i2_h1 = new ConnectGene("i-2", "h-1", 3.179256100003881, 3, false)
        var i2_h2 = new ConnectGene("i-2", "h-2", 5.444467626110183, 4, false)
        
        var h1_o1 = new ConnectGene("h-1", "o-1", -6.998221924119023, 5, false)
        var h2_o1 = new ConnectGene("h-2", "o-1",  6.488756231728473, 6, false)

        var bh1_h1 = new ConnectGene("b-h1", "h-1", 1, 7, false)
        var bh2_h2 = new ConnectGene("b-h2", "h-2", 1, 8, false)
        var bo1_o1 = new ConnectGene("b-o1", "o-1", 1, 9, false)

        var genome1 = new Genome([i1, i2, h1, h2, o1, bh1, bh2, bo1], [i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1])
        var genome2 = new Genome(JSON.parse(JSON.stringify([i1, i2, h1, h2, o1, bh1, bh2, bo1])), JSON.parse(JSON.stringify([i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1])))
        // COPY END

        assert.equal(genome1.distanceFrom(genome2), 0, "distance with identical not 0")
    })

    it("distance with nothing in common", function () {

        // COPIED FROM MINIMAL XOR TEST
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


        var i1_h1 = new ConnectGene("i-1", "h-1", 3.191352509627136, 1, false)
        var i1_h2 = new ConnectGene("i-1", "h-2", 5.491641199017287, 2, false)

        var i2_h1 = new ConnectGene("i-2", "h-1", 3.179256100003881, 3, false)
        var i2_h2 = new ConnectGene("i-2", "h-2", 5.444467626110183, 4, false)

        var h1_o1 = new ConnectGene("h-1", "o-1", -6.998221924119023, 5, false)
        var h2_o1 = new ConnectGene("h-2", "o-1", 6.488756231728473, 6, false)

        var bh1_h1 = new ConnectGene("b-h1", "h-1", 1, 7, false)
        var bh2_h2 = new ConnectGene("b-h2", "h-2", 1, 8, false)
        var bo1_o1 = new ConnectGene("b-o1", "o-1", 1, 9, false)

        var genome1 = new Genome([i1, i2, h1, h2, o1, bh1, bh2, bo1], [i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1])
        var genome2 = new Genome(JSON.parse(JSON.stringify([i1, i2, h1, h2, o1, bh1, bh2, bo1])), JSON.parse(JSON.stringify([i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1])))
        
        for(var i = 0; i < genome2.connections.length; i++){
            genome2.connections[i].innov *= 10
        }
        
        // COPY END

        assert.equal(genome1.distanceFrom(genome2), 18, "distance with totally different should be same as number of genes in them")
    })

    it("distance with some in common, with those in common having same weight", function () {

        // COPIED FROM MINIMAL XOR TEST
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


        var i1_h1 = new ConnectGene("i-1", "h-1", 3.191352509627136, 1, false)
        var i1_h2 = new ConnectGene("i-1", "h-2", 5.491641199017287, 2, false)

        var i2_h1 = new ConnectGene("i-2", "h-1", 3.179256100003881, 3, false)
        var i2_h2 = new ConnectGene("i-2", "h-2", 5.444467626110183, 4, false)

        var h1_o1 = new ConnectGene("h-1", "o-1", -6.998221924119023, 5, false)
        var h2_o1 = new ConnectGene("h-2", "o-1", 6.488756231728473, 6, false)

        var bh1_h1 = new ConnectGene("b-h1", "h-1", 1, 7, false)
        var bh2_h2 = new ConnectGene("b-h2", "h-2", 1, 8, false)
        var bo1_o1 = new ConnectGene("b-o1", "o-1", 1, 9, false)

        var genome1 = new Genome([i1, i2, h1, h2, o1, bh1, bh2, bo1], [i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1])
        
        var new_conns = JSON.parse(JSON.stringify([i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1]))
        
        for(var i = 0; i < 4; i++){
            new_conns[i].innov *= 10
        }

        var genome2 = new Genome(JSON.parse(JSON.stringify([i1, i2, h1, h2, o1, bh1, bh2, bo1])), new_conns)

        // COPY END

        assert.equal(genome1.distanceFrom(genome2), 8 + 0, "distance with matching genomes having same weight should be equal to unmatching genome count")
    })
    
    it("distance with all in common, but weight differences", function () {

        // COPIED FROM MINIMAL XOR TEST
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


        var i1_h1 = new ConnectGene("i-1", "h-1", 10, 1, false)
        var i1_h2 = new ConnectGene("i-1", "h-2", 10, 2, false)

        var i2_h1 = new ConnectGene("i-2", "h-1", 10, 3, false)
        var i2_h2 = new ConnectGene("i-2", "h-2", 10, 4, false)

        var h1_o1 = new ConnectGene("h-1", "o-1", 10, 5, false)
        var h2_o1 = new ConnectGene("h-2", "o-1", 10, 6, false)

        var bh1_h1 = new ConnectGene("b-h1", "h-1", 10, 7, false)
        var bh2_h2 = new ConnectGene("b-h2", "h-2", 10, 8, false)
        var bo1_o1 = new ConnectGene("b-o1", "o-1", 10, 9, false)

        var genome1 = new Genome([i1, i2, h1, h2, o1, bh1, bh2, bo1], [i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1])
        
        var new_conns = JSON.parse(JSON.stringify([i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1]))
        
        for(var i = 0; i < new_conns.length; i++){
            new_conns[i].weight = i
        }

        var genome2 = new Genome(JSON.parse(JSON.stringify([i1, i2, h1, h2, o1, bh1, bh2, bo1])), new_conns)

        // COPY END

        assert.equal(genome1.distanceFrom(genome2), 0.4 * 6)
    })

    it("distance with some common and some weight differences", function () {

        // COPIED FROM MINIMAL XOR TEST
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


        var i1_h1 = new ConnectGene("i-1", "h-1", 1, 1, false)
        var i1_h2 = new ConnectGene("i-1", "h-2", 2, 2, false)

        var i2_h1 = new ConnectGene("i-2", "h-1", 3, 3, false)
        var i2_h2 = new ConnectGene("i-2", "h-2", 4, 4, false)

        var h1_o1 = new ConnectGene("h-1", "o-1", 5, 5, false)
        var h2_o1 = new ConnectGene("h-2", "o-1", 6, 6, false)

        var bh1_h1 = new ConnectGene("b-h1", "h-1", 7, 7, false)
        var bh2_h2 = new ConnectGene("b-h2", "h-2", 8, 8, false)
        var bo1_o1 = new ConnectGene("b-o1", "o-1", 9, 9, false)

        var genome1 = new Genome([i1, i2, h1, h2, o1, bh1, bh2, bo1], [i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1])

        var new_conns = JSON.parse(JSON.stringify([i1_h1, i1_h2, i2_h1, i2_h2, h1_o1, h2_o1, bh1_h1, bh2_h2, bo1_o1]))

        for (var i = 0; i < new_conns.length; i++) {
            if(i % 2 == 0){
                new_conns[i].innov *= 10
            }else{
                new_conns[i].weight += 1
            }
        }

        var genome2 = new Genome(JSON.parse(JSON.stringify([i1, i2, h1, h2, o1, bh1, bh2, bo1])), new_conns)

        // COPY END

        assert.equal(genome1.distanceFrom(genome2), 10 + 0.4 * 1)
    })

})

describe("mutation add connection tests", function(){
    it("calling mutation on solution xor genome repeatedly for 100 times and comparing all possible outcomes", function(){

        var comb = {}

        for (var iter = 0; iter < 100; iter++){
            var ctxt = createContext()

            var genome = solutionGenome(ctxt)

            genome.mutateAddConnection()

            assert.equal(genome.connections.length, 10, "connection not added")

            var new_conn = genome.connections[genome.connections.length - 1]
            for (var i = 0; i < genome.connections.length - 1; i++){
                assert.equal(new_conn.from == genome.connections[i].from && new_conn.to == genome.connections[i].to, false, "duplicate connection added")
            }

            assert.equal(genome.getNode(new_conn.to).type == "input" || genome.getNode(new_conn.to).type == "bias", false, "connection to input or bias made")

            if(new_conn.from == new_conn.to){
                assert.equal(new_conn.is_recurrent, true, "self loop not detected as recurrent")
            }

            var combination = new_conn.from + " " + new_conn.to

            switch(combination){
                case "o1 h1":
                case "o1 h2":
                case "o1 o1":
                case "h1 h1":
                case "h2 h2":
                        assert.equal(new_conn.is_recurrent, true, "recurrent not detected")
                        assert.equal(genome.has_any_enabled_recurrent_neurons, true, "genome not aware of newly enabled recurrent neuron")
                    break
                default:
                        assert.equal(new_conn.is_recurrent, false, `false recurrent between ${new_conn.from} & ${new_conn.to} detected`)
                        assert.equal(genome.has_any_enabled_recurrent_neurons, false, "genome does not have any recurrent neuron but still it thinks it has one enabled")
                    break
            }

            comb[combination] = true
        }

        assert.equal(Object.keys(comb).length, 15, "all combinations did not occur, only these occurred: " + Object.keys(comb))
    })
})

describe("mutate add node tests", function(){
    it("calling mutation on solution xor genome repeatedly for 100 times and comparing all possible outcomes", function(){
        var comb = {}

        for (var iter = 0; iter < 100; iter++){
            var ctxt = createContext()

            var genome = solutionGenome(ctxt)

            genome.mutateAddNode()

            assert.equal(genome.nodes.length, 9, "node not added")
            assert.equal(genome.connections.length, 11, "connections not added")

            var new_node = genome.nodes[genome.nodes.length - 1]
            var prev_conn = genome.connections[genome.connections.length - 2]
            var next_conn = genome.connections[genome.connections.length - 1]

            var broken_conn = genome.connections[genome.connections.findIndex((conn)=>conn.from == prev_conn.from && conn.to == next_conn.to)]

            assert(broken_conn, "broken connection does not exist")
            assert.equal(broken_conn.is_disabled, true, "broken connection not disabled")
            assert.equal(prev_conn.weight, 1, "previous connection should have 1 weight")
            assert.equal(next_conn.weight, broken_conn.weight, "next connection should have broken conn weight")

            assert.equal(prev_conn.is_recurrent, false, "prev conn can never be recurrent")
            assert.equal(next_conn.is_recurrent, broken_conn.is_recurrent, "next conn and broken conn should have same recurrency")

            assert.equal(new_node.id, "h3", "new node not added with proper id")

            var combination = broken_conn.from + " " + broken_conn.to
            comb[combination] = true
        }

        assert.equal(Object.keys(comb).length, 9, "all combinations did not occur, only these occurred: " + Object.keys(comb))
    })
})

describe("crossover tests", function(){

    it("combined info of nodes and connections should be present", function(){

        var ctxt = createContext()

        var g1 = solutionGenome(ctxt)
        g1.fitness = 1
        var r_conn1 = new ConnectGene("o1", "h1", 0, ++ctxt.innov, false)
        r_conn1.is_recurrent = true
        g1.addConnection(r_conn1)

        var g2 = solutionGenome(ctxt)
        g2.fitness = 0
        var r_conn2 = new ConnectGene("o1", "h1", 0, ++ctxt.innov+1, false)
        r_conn2.is_recurrent = true
        g2.addConnection(r_conn2)

        var child1 = g1.crossover(g2)
        var child2 = g2.crossover(g1)

        assert.equal(g1.distanceFrom(g2), 2)
        assert.equal(child1.distanceFrom(child2), 0)

        for(var i = 0; i < child1.connections.length; i++){
            child1.getNode(child1.connections[i].from)
            child1.getNode(child1.connections[i].to)
        }

        var nodes = new Set(child1.nodes.map((n)=>n.id))

        assert.equal(child1.nodes.length, nodes.size, "nodes in genome are not unique")
    })  

    it("both parents have same fitness", function () {
        
        var comb = {}

        for(var i = 0; i < 100; i++){
            var ctxt = createContext()
    
            var g1 = solutionGenome(ctxt)
            g1.fitness = 0
            var r_conn1 = new ConnectGene("o1", "h1", 0, 10, false)
            r_conn1.is_recurrent = true
            g1.addConnection(r_conn1)

            for(var j = 0; j < g1.connections.length; j++){
                g1.connections[j].innov += 10
            }

    
            var g2 = solutionGenome(ctxt)
            g2.fitness = 0
            var r_conn2 = new ConnectGene("o1", "h1", 0, 10, false)
            r_conn2.is_recurrent = true
            g2.addConnection(r_conn2)
    
            var child1 = g1.crossover(g2)

            for(var j = 0; j < child1.connections.length; j++){
                comb[child1.connections[j].innov] = true
            }
        }

        assert.equal(Object.keys(comb).length, 10, "not all combinations were found. Only these were found + " + JSON.stringify(comb))
    })

    it("parent A has higher fitness", function () {

    })

    it("parent B has higher fitness", function () {

    })

    it("recurrent nodes in child", function(){

    })

})

/*
    TO-DO:
    + add enabling of genes in addConnection
    + something related to mutation Power
    + random entity as representative of a specie
    + if a specie is eliminated due to stagnation, replace its population with initial genomes
    + add test to check matched and unmatched count in distance function
    + dynamic thresholding for compatibility


    + generation graphs - histogram which plots fitness max, min, mean, solution
    + node graph - drop down by iterating over keys of history and render best worst as per specie... create and keep entities so that network can be tested
*/
