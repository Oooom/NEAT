import { init, generate } from "./main.mjs";
import * as pattern from "./pattern_support.mjs"
import * as xor from "./xor_support.mjs"

var ctxt = {
    what   : null,
    solved : false,
    task   : null
}

onmessage = function(e) {
    switch(e.data.type){
        case "init":
            handleINIT(e.data.what)
        break

        case "generate":
            handleGENERATE(e.data)    
        break
    }
}

function handleINIT(what){
    console.log(what + " init")
    
    ctxt.what = what
    
    switch(what){
        case "PATTERN":
            ctxt.task = pattern
            break
        
        case "XOR":
            ctxt.task = xor
            break
        
        default:
            throw new Error("Unidentified Task")
    }

    init(ctxt.task)
}

function handleGENERATE(data){
    if(ctxt.what){

        ctxt.solved = false

        if(data.how_many == "till_solution"){
            while(!ctxt.solved){
                generate(ctxt.task, generationTick)
            }
        }else{
            for(var i = 0; i < data.how_many; i++)
                generate(ctxt.task, generationTick)
        }


    }else{
        console.error("init not done")
    }
}

function generationTick(stats, solved){
    postMessage({type: "gen_stats", stats: stats})
    ctxt.solved = solved
}