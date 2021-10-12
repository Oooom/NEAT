import { initPATTERN, generatePATTERN } from "./main.mjs";

var ctxt = {
    what: null,
    solved: false
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
    
    if(what == "PATTERN"){
        initPATTERN()
    }
}

function handleGENERATE(data){
    if(ctxt.what){

        ctxt.solved = false

        if(data.how_many == "till_solution"){
            while(!ctxt.solved){
                generatePATTERN(generationTick)
            }
        }else{
            if(ctxt.what == "PATTERN"){
                for(var i = 0; i < data.how_many; i++)
                    generatePATTERN(generationTick)
            }
        }


    }else{
        console.error("init not done")
    }
}

function generationTick(stats, solved){
    postMessage({type: "gen_stats", stats: stats})
    ctxt.solved = solved
}