import {c_m, c_um} from "./parameters.mjs"
import { getUniformRandomFromRange, getUniformRandomFromRangeInt, getNormalRandom} from "./lib.mjs"
import { sigmoid } from "./nn.mjs"
class ConnectGene {
    constructor(from, to, weight, innov, disabled) {
        this.from = from
        this.to = to
        this.weight = weight
        this.innov = innov
        this.is_recurrent = false
        this.is_disabled = disabled
    }

    mutateRandomly(){
        this.weight = getUniformRandomFromRange(-1, 1)
    }

    mutateNormally(){
        this.weight += getNormalRandom()
    }
}

class NodeGene {
    constructor(id, type, activation_fn) {
        this.id = id
        this.type = type
        this.activation_fn = activation_fn

        // transient data
        this.op      = null
        this.prev_op = null
    }
}

class Genome {
    constructor(nodes, connections, ctxt) {
        this.nodes = nodes
        this.connections = []

        this.ctxt = ctxt

        this.id_to_ref = {}
        this.ip_node_ids = []
        this.op_node_ids = []

        var max = 0

        for (var node of this.nodes) {
            this.id_to_ref[node.id] = node
            if (node.type == "input") {
                this.ip_node_ids.push(node.id)
            }
            if (node.type == "output") {
                this.op_node_ids.push(node.id)
            }

            if(node.type == "hidden"){
                var num = parseInt(node.id.substring(1))

                if (num > max){
                    max = num
                }
            }
        }

        this.hidden_id_last = max


        this.from_connections_of = {}
        this.to_connections_of = {}

        this.has_any_enabled_recurrent_neurons = false

        for (var connection of connections) {
            this.addConnection(connection)

            if(!connection.is_disabled && connection.is_recurrent){
                this.has_any_enabled_recurrent_neurons = true
            }
        }

        this.connections.sort((a, b) => a.innov - b.innov)
    }

    addConnection(connection){
        this.connections.push(connection)

        if (this.from_connections_of[connection.from]) {
            this.from_connections_of[connection.from].push(connection)
        } else {
            this.from_connections_of[connection.from] = [connection]
        }

        if (this.to_connections_of[connection.to]) {
            this.to_connections_of[connection.to].push(connection)
        } else {
            this.to_connections_of[connection.to] = [connection]
        }
    }

    addNode(node){
        this.nodes.push(node)
    }

    getNode(id){
        if(!this.id_to_ref[id]){
            throw new Error("Why is reference of " + id + " not present ?")
        }

        return this.id_to_ref[id]
    }

    detectCycle(connection) {
        var traversed = []

        var to_traverse = [connection]

        while (to_traverse.length > 0) {
            var next = to_traverse.shift()

            if (traversed.indexOf(next) != -1) {
                return true
            } else {
                traversed.push(next)
            }

            if (this.from_connections_of[next.to]){
                for(var connection of this.from_connections_of[next.to]){
                    if (!connection.is_recurrent && to_traverse.indexOf(connection) == -1 ){
                        to_traverse.push(connection)
                    }
                }
            }
        }

        return false
    }

    mutateAddConnection() {
        var found = false

        while(!found){
            var from = this.nodes[getUniformRandomFromRangeInt(0, this.nodes.length)]
            var to   = this.nodes[getUniformRandomFromRangeInt(0, this.nodes.length)]

            var is_bias_or_input = (to.type == "bias" || to.type == "input")
            var is_connection_absent = (this.from_connections_of[from.id] == undefined || this.from_connections_of[from.id].findIndex((conn) => conn.to == to.id) == -1)

            if (!is_bias_or_input && is_connection_absent ){
                var new_conn = new ConnectGene(from.id, to.id, NaN, ++this.ctxt.innov, false)
                new_conn.mutateRandomly()

                this.addConnection(new_conn)

                new_conn.is_recurrent = this.detectCycle(new_conn)

                if(new_conn.is_recurrent){
                    this.has_any_enabled_recurrent_neurons = true
                }

                found = true
            }
        }
    }

    mutateAddNode() {
        var found = false

        while(!found){
            var conn = this.connections[getUniformRandomFromRangeInt(0, this.connections.length)]

            if(conn.is_disabled == false){
                var new_node = new NodeGene("h"+(++this.hidden_id_last), "hidden", sigmoid)
                
                var prev_conn = new ConnectGene(conn.from, new_node.id, 1, ++this.ctxt.innov, false)
                var next_conn = new ConnectGene(new_node.id, conn.to, conn.weight, ++this.ctxt.innov, false)
                next_conn.is_recurrent = conn.is_recurrent

                conn.is_disabled = true

                this.addConnection(prev_conn)
                this.addConnection(next_conn)
                this.addNode(new_node)

                found = true
            }
        }
    }

    distanceFrom(genome){
        var unmatched_count = 0
        var matched_count   = 0
        var sum_weight_difference_of_matched = 0

        var this_i = 0
        var other_i = 0

        while(this_i < this.connections.length || other_i < genome.connections.length){

            if(this_i == this.connections.length || other_i == this.connections.length){
                if (this_i == this.connections.length) {
                    other_i++
                    unmatched_count++
                }
                else if (other_i == genome.connections.length) {
                    this_i++
                    unmatched_count++
                }
            }else{
                if (this.connections[this_i].innov == genome.connections[other_i].innov) {
                    sum_weight_difference_of_matched += Math.abs(this.connections[this_i].weight - genome.connections[other_i].weight)

                    matched_count++

                    this_i++
                    other_i++
                }else{
                    if (this.connections[this_i].innov < genome.connections[other_i].innov) {
                        this_i++
                        unmatched_count++
                    } else {
                        other_i++
                        unmatched_count++
                    }
                }
            }

        }                

        if(unmatched_count + matched_count * 2 != this.connections.length + genome.connections.length){
            throw new Error("Matched and Unmatched Count were different from total gene length in distance calculation")
        }

        if( matched_count > 0 ){
            return c_um * unmatched_count + c_m * (sum_weight_difference_of_matched / matched_count)
        }else{
            return c_um * unmatched_count
        }
    }

}

export { Genome, NodeGene, ConnectGene }
