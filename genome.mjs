import * as params from "./parameters.mjs"
import { getUniformRandomFromRange, chooseRandomly, getNormalRandom, percent} from "./lib.mjs"
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
    
    // mutateNormally(){
    //     if(percent(50)){
    //         this.weight += this.weight * 0.2
    //     }else{
    //         this.weight -= this.weight * 0.2
    //     }
    // }

    clone(){
        var cl = new ConnectGene(this.from, this.to, this.weight, this.innov, this.is_disabled)

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

        if(this.type == "bias"){
            cl.op = this.op
        }

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

        for (var node of this.nodes) {
            this.id_to_ref[node.id] = node
            if (node.type == "input") {
                this.ip_node_ids.push(node.id)
            }
            if (node.type == "output") {
                this.op_node_ids.push(node.id)
            }
        }


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
        this.id_to_ref[node.id] = node
    }

    getNode(id){
        if(!this.id_to_ref[id]){
            // console.error("Why is reference of " + id + " not present ?")

            return null
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

    getUnconnectedNodeList(){
        var nodes = []

        var node_ids = this.nodes.map((n)=>n.id)

        for(var node of node_ids){
            var unconnected = node_ids.filter( (n_id) => {
                var is_not_input_or_bias = !(this.getNode(n_id).type == "input" || this.getNode(n_id).type == "bias")
                
                var is_connection_absent

                if(this.from_connections_of[node]){
                    is_connection_absent = this.from_connections_of[node].findIndex((conn) => conn.to == n_id) == -1
                }else{
                    is_connection_absent = true
                } 

                return is_not_input_or_bias && is_connection_absent
            })

            if(unconnected.length > 0){
                nodes.push({from: node, to: unconnected})
            }
        }

        return nodes
    }

    mutateAddConnection() {

        var unconnected_node_combns = this.getUnconnectedNodeList()

        if(unconnected_node_combns.length > 1){
            var from = chooseRandomly( unconnected_node_combns )
            var to   = chooseRandomly( from.to )
    
            from = from.from
    
            var new_conn = new ConnectGene(from, to, NaN, this.ctxt.getInnov(from, to), false)
            new_conn.mutateRandomly()
    
            this.addConnection(new_conn)
    
            new_conn.is_recurrent = this.detectCycle(new_conn)
    
            if(new_conn.is_recurrent){
                this.has_any_enabled_recurrent_neurons = true
            }
        }

    }

    mutateAddNode() {
        var enabled_connections = this.connections.filter((conn)=>conn.is_disabled == false)

        if(enabled_connections.length > 0){
            var conn = chooseRandomly(enabled_connections)
    
            var new_node = new NodeGene( this.ctxt.getHiddenNodeID(conn.from, conn.to) , "hidden", sigmoid)
            
            var prev_conn = new ConnectGene(conn.from, new_node.id, 1, this.ctxt.getInnov(conn.from, new_node.id), false)
            var next_conn = new ConnectGene(new_node.id, conn.to, conn.weight, this.ctxt.getInnov(new_node.id, conn.to), false)
            next_conn.is_recurrent = conn.is_recurrent
    
            conn.is_disabled = true
    
            this.addConnection(prev_conn)
            this.addConnection(next_conn)
            this.addNode(new_node)
        }
    }

    distanceFrom(genome){
        var unmatched_count = 0
        var matched_count   = 0
        var sum_weight_difference_of_matched = 0

        var this_i = 0
        var other_i = 0

        while(this_i < this.connections.length || other_i < genome.connections.length){

            if(this_i == this.connections.length || other_i == genome.connections.length){
                if (this_i == this.connections.length) {
                    if (!genome.connections[other_i].is_disabled)
                        unmatched_count++
                    other_i++
                }
                else if (other_i == genome.connections.length) {
                    if(!this.connections[this_i].is_disabled)
                        unmatched_count++
                    this_i++
                }
            }else{
                if (this.connections[this_i].innov == genome.connections[other_i].innov) {
                    
                    if(!this.connections[this_i].is_disabled && !genome.connections[other_i].is_disabled){
                        sum_weight_difference_of_matched += Math.abs(this.connections[this_i].weight - genome.connections[other_i].weight)
    
                        matched_count++
                    }else{
                        if( !(this.connections[this_i].is_disabled && genome.connections[other_i].is_disabled) ){
                            unmatched_count++
                        }
                    }

                    this_i++
                    other_i++
                }else{
                    if (this.connections[this_i].innov < genome.connections[other_i].innov) {
                        if(!this.connections[this_i].is_disabled)
                            unmatched_count++
                        this_i++
                    } else {
                        if (!genome.connections[other_i].is_disabled)
                            unmatched_count++
                        other_i++
                    }
                }
            }

        }                

        if( matched_count > 0 ){
            return params.c_um * unmatched_count + params.c_m * (sum_weight_difference_of_matched / matched_count)
        }else{
            return params.c_um * unmatched_count
        }
    }

    crossover(other){
        var percent_of_baggage_from_this = this.fitness == other.fitness ? 
                                                    this.connections.length < other.connections.length ? 
                                                        100 : 0
                                                    : 
                                                    this.fitness > other.fitness ? 
                                                        100 : 0
        
        var nodes = new Set()
        var connections = []

        var this_i  = 0
        var other_i = 0

        var matching_genes = []
        var unmatching_genes = {from_this: [], from_other: []}

        while(this_i < this.connections.length || other_i < other.connections.length){

            if(this_i == this.connections.length || other_i == other.connections.length){
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
                nodes.add( this.connections[this_idx].from )
                nodes.add( this.connections[this_idx].to   )
            }else{
                connections.push( other.connections[other_idx] )
                nodes.add( other.connections[other_idx].from )
                nodes.add( other.connections[other_idx].to   )
            }

            if (other.connections[other_idx].is_disabled || this.connections[this_idx].is_disabled ){
                if( percent(params.disable_offspring_gene_chance) ){
                    connections[connections.length - 1].is_disabled = true
                }
            }
        }

        for(var this_idx of unmatching_genes.from_this){
            if( percent(percent_of_baggage_from_this) ){
                connections.push( this.connections[this_idx] )
                nodes.add( this.connections[this_idx].from )
                nodes.add( this.connections[this_idx].to   )
            }
        }

        for(var other_idx of unmatching_genes.from_other){
            if( !percent(percent_of_baggage_from_this) ){
                connections.push( other.connections[other_idx] )
                nodes.add( other.connections[other_idx].from )
                nodes.add( other.connections[other_idx].to   )
            }
        }

        var node_clones = Array.from(nodes).map((nodeid)=> (this.getNode(nodeid) || other.getNode(nodeid)).clone())
        var conn_clones = connections.map((conn)=>conn.clone())

        var child = new Genome(node_clones, conn_clones, this.ctxt)

        return child
    }

    mutate(){

        for(var connection of this.connections){
            if(!connection.is_disabled){
                if ( percent(params.weight_mutation_chance) ){
                    if (percent(params.weight_mutation_chance_uniformly_perturb)){
                        connection.mutateNormally()
                    }

                    if ( percent(params.weight_mutation_chance_new_random) ){
                        connection.mutateRandomly()
                    }
                }
            }
        }

        if ( percent(params.add_node_mutation_chance) ){
            this.mutateAddNode()
        }

        if ( percent(params.add_link_mutation_chance) ){
            this.mutateAddConnection()
        }

    }

    clone(){
        var node_clones = this.nodes      .map((node)=>node.clone())
        var conn_clones = this.connections.map((conn)=>conn.clone())

        return new Genome(node_clones, conn_clones, this.ctxt)
    }

}

export { Genome, NodeGene, ConnectGene }
