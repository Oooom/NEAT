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

        this.from_connections_of = {}
        this.to_connections_of = {}

        for (var node of this.nodes) {
            this.id_to_ref[node.id] = node
            if (node.type == "input") {
                this.ip_node_ids.push(node.id)
            }
        }

        for (var connection of this.connections) {
            this.from_connections_of[connection.from] = connection
            this.to_connections_of[connection.to] = connection
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

            for(var connection of this.from_connections_of[next.to]){
                if(!connection.is_recurrent && to_traverse.indexOf(connection) == -1){
                    to_traverse.push(connection)
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
