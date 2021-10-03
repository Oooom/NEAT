class ConnectGene {
    constructor(from, to, weight, innov, disabled) {
        this.from = from
        this.to = to
        this.weight = weight
        this.innov = innov
        this.is_recurrent = false
        this.is_disabled = disabled
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
    constructor(nodes, connections) {
        this.nodes = nodes
        this.connections = connections

        this.nodes = nodes
        this.connections = connections

        this.id_to_ref = {}
        this.ip_node_ids = []
        this.op_node_ids = []

        this.from_connections_of = {}
        this.to_connections_of = {}

        this.has_any_enabled_recurrent_neurons = false

        for (var node of this.nodes) {
            this.id_to_ref[node.id] = node
            if (node.type == "input") {
                this.ip_node_ids.push(node.id)
            }
            if (node.type == "output") {
                this.op_node_ids.push(node.id)
            }
        }

        for (var connection of this.connections) {
            if (this.from_connections_of[connection.from]){
                this.from_connections_of[connection.from].push(connection)
            }else{
                this.from_connections_of[connection.from] = [connection]
            }

            if (this.to_connections_of[connection.to]) {
                this.to_connections_of[connection.to].push(connection)
            } else {
                this.to_connections_of[connection.to] = [connection]
            }

            if(!connection.is_disabled && connection.is_recurrent){
                this.has_any_enabled_recurrent_neurons = true
            }
        }
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

    mutationAddConnection() {

    }

    mutationAddNode() {

    }
}

export { Genome, NodeGene, ConnectGene }
