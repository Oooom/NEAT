function sigmoid(x){
    return 1 / (1 + Math.E ** -x)
}

function steepSigmoid(x){
    return 1 / (1 + Math.E ** -4.9*x)
}

/*
    node (id, type, activation_fn)
    connection (id, disabled, weight, innov)
*/

class Node{
    constructor(type, id, activation_fn){
        this.type = type
        this.id   = id
        this.activation_fn = activation_fn
        
        //transient variables
        this.op   = null
        this.prev_op = null
        this.ip   = null
    }
}

class NeuralNetwork{

    constructor(genome){
        this.nodes = nodes
        this.connections = connections

        this.id_to_ref    = {}
        this.ip_node_ids  = []

        this.from_connections_of = {}
        this.to_connections_of = {}

        for(var node of this.nodes){
            this.id_to_ref[node.id] = node
            if(node.type == "input"){
                this.ip_node_ids.push(node.id)
            }
        }
        
        for(var connection of this.connections){
            this.from_connections_of[connection.from] = connection
            this.to_connections_of[connection.to]     = connection
        }
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

    getNextNodesOf(node_id){

        if(this.getNode(node_id).type == "output"){
            throw new Error("Why Asking For Next Nodes of OUTPUT node ?")
        }

        var next_nodes = []

        for(var connection of this.connections){
            if(connection.from == node_id){
                next_nodes.push(connection.to)
            }
        }

    }

    evaluate(node_id){
        var to_evaluate_first = []
        var connections       = []

        for(var connection of this.connections){
            if(connection.to == node_id){

            }
        }
    }

    predict(ips){
        
        this.loadOPofIP(ips)

    }
}

export { NeuralNetwork, Node, sigmoid, steepSigmoid }
