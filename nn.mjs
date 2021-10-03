function sigmoid(x){
    return 1 / (1 + Math.E ** -x)
}

function steepSigmoid(x){
    return 1 / (1 + Math.E ** -4.9*x)
}

class NeuralNetwork{

    constructor(genome){
        this.nodes = genome.nodes
        this.connections = genome.connections

        this.id_to_ref    = genome.id_to_ref
        this.ip_node_ids  = genome.ip_node_ids
        this.op_node_ids  = genome.op_node_ids

        this.from_connections_of = genome.from_connections_of
        this.to_connections_of   = genome.to_connections_of

        this.has_any_enabled_recurrent_neurons = genome.has_any_enabled_recurrent_neurons
    }

    loadOPofIP(ips){
        var i = 0
        var iter = 0

        while (i < ips.length && iter < this.ip_node_ids.length) {
            this.getNode(this.ip_node_ids[iter]).op = ips[i]

            i++
            iter++
        }

        if(i != ips.length || iter != this.ip_node_ids.length){
            throw new Error("OP of IP not correctly loaded")
        }
    }

    getNode(id){
        if(!this.id_to_ref[id]){
            throw new Error("Why is reference of " + id + " not present ?")
        }

        return this.id_to_ref[id]
    }

    calculateOutputOfNode(node_id){
        
        var weighted_ip = 0

        var curr_node = this.getNode(node_id)

        for(var connection of this.to_connections_of[node_id]){
            var prev_node = this.getNode(connection.from)

            var ip = null

            if(prev_node.op == null){

                if(connection.is_recurrent){
                    if(prev_node.prev_op != null){
                        ip = prev_node.prev_op
                    }else{
                        ip = 0
                    }
                }else{
                    this.calculateOutputOfNode(prev_node.id) 
                    ip = prev_node.op
                }

            }else{
                ip = prev_node.op
            }

            weighted_ip += ip * connection.weight
        }

        curr_node.op = curr_node.activation_fn(weighted_ip)

    }

    calculateOutputOfOutputNodes(){
        var outputs = []
        for(var node_id of this.op_node_ids){
            this.calculateOutputOfNode(node_id)
            outputs.push(this.getNode(node_id).op)
        }

        return outputs
    }

    stepForward(){
        for(var node of this.nodes){
            if( !(node.type == "input" || node.type == "bias") ){
                node.prev_op = node.op
                node.op      = null
            }
        }
    }

    predict(ips){
        this.loadOPofIP(ips)
        var outputs = this.calculateOutputOfOutputNodes()
        this.stepForward()

        return outputs
    }
}

export { NeuralNetwork, sigmoid, steepSigmoid }
