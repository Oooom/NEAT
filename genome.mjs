import {c_m, c_um} from "./parameters.mjs"
import { getUniformRandomFromRange, getUniformRandomFromRangeInt, getNormalRandom, percent} from "./lib.mjs"
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

    clone(){
        var cl = new ConnectGene(this.from, this.to, this.weight, this.innov, this.disabled)

        cl.is_recurrent = this.is_recurrent

        return cl
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

    clone(){
        var cl = new NodeGene(this.id, this.type, this.activation_fn)

        return cl
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
                var new_conn = new ConnectGene(from.id, to.id, NaN, this.ctxt.getInnov(from.id, to.id), false)
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
                
                var prev_conn = new ConnectGene(conn.from, new_node.id, 1, this.ctxt.getInnov(conn.from, new_node.id), false)
                var next_conn = new ConnectGene(new_node.id, conn.to, conn.weight, this.ctxt.getInnov(new_node.id, conn.to), false)
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

    crossover(other){
        var percent_of_baggage_from_this = this.fitness == other.fitness ? 50 : this.fitness > other.fitness ? 100 : 0
        
        var nodes = new Set()
        var connections = []

        var this_i  = 0
        var other_i = 0

        var matching_genes = []
        var unmatching_genes = {from_this: [], from_other: []}

        while(this_i < this.connections.length || other_i < other.connections.length){

            if(this_i == this.connections.length || other_i == this.connections.length){
                if (this_i == this.connections.length) {
                    unmatching_genes.from_other.push(other_i)
                    other_i++
                }
                else if (other_i == other.connections.length) {
                    unmatching_genes.from_this.push(this_i)
                    this_i++
                }
            }else{
                if (this.connections[this_i].innov == other.connections[other_i].innov) {
                    matching_genes.push( {this_idx: this_i, other_idx: other_i} )

                    this_i++
                    other_i++
                }else{
                    if (this.connections[this_i].innov < other.connections[other_i].innov) {
                        unmatching_genes.from_this.push(this_i)
                        this_i++
                    } else {
                        unmatching_genes.from_other.push(other_i)
                        other_i++
                    }
                }
            }

        }

        for(var i = 0; i < matching_genes.length; i++){
            var this_idx = matching_genes[i].this_idx
            var other_idx = matching_genes[i].other_idx

            if( percent(50) ){    
                connections.push(this.connections[this_idx] )
                nodes.add( this.getNode( this.connections[this_idx].from ) )
                nodes.add( this.getNode( this.connections[this_idx].to   ) )
            }else{
                connections.push( other.connections[other_idx] )
                nodes.add( other.getNode( other.connections[other_idx].from ) )
                nodes.add( other.getNode( other.connections[other_idx].to   ) )
            }

            if (other.connections[other_idx].is_disabled || this.connections[this_idx].is_disabled ){
                if( percent(75) ){
                    connections[connections.length - 1].is_disabled = true
                }
            }
        }

        for(var this_idx of unmatching_genes.from_this){
            if( percent(percent_of_baggage_from_this) ){
                connections.push( this.connections[this_idx] )
                nodes.add( this.getNode( this.connections[this_idx].from ) )
                nodes.add( this.getNode( this.connections[this_idx].to   ) )
            }
        }

        for(var other_idx of unmatching_genes.from_other){
            if( !percent(percent_of_baggage_from_this) ){
                connections.push( other.connections[other_idx] )
                nodes.add( other.getNode( other.connections[other_idx].from ) )
                nodes.add( other.getNode( other.connections[other_idx].to   ) )
            }
        }
        
        var node_clones = Array.from(nodes).map((node)=>node.clone())
        var conn_clones = connections.map((conn)=>conn.clone())

        var child = new Genome(node_clones, conn_clones)

        return child
    }

}

export { Genome, NodeGene, ConnectGene }
