function getUniformRandomFromRange(min, max){
    return Math.random() * (max - min) + min
}

function getUniformRandomFromRangeInt(min, max){
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min) + min);
}

// function getNormalRandom() {
//     var u = 0, v = 0;
//     while(u === 0) u = Math.random();
//     while(v === 0) v = Math.random();
//     return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
// }
var g = {}
let y2 = 0
function getNormalRandom(mean, sd = 1) {
    let y1, x1, x2, w;
    if (g._gaussian_previous) {
        y1 = y2;
        g._gaussian_previous = false;
    } else {
        do {
            x1 = Math.random() * 2 - 1;
            x2 = Math.random() * 2 - 1;
            w = x1 * x1 + x2 * x2;
        } while (w >= 1);
        w = Math.sqrt(-2 * Math.log(w) / w);
        y1 = x1 * w;
        y2 = x2 * w;
        g._gaussian_previous = true;
    }

    const m = mean || 0;
    return y1 * sd + m;
}

function chooseRandomly(list){
    return list[ getUniformRandomFromRangeInt(0, list.length) ]
}

function percent(num){
    return Math.random() * 100 < num
}

function createContext(){
    return {
        innov: 0,
        combn: {},
        
        lastHiddenNodeID: 0,
        hiddenNodeCombn: {},

        lastSpecieID: 0,
        specieStats: {},

        globalStats: {
            timer: 0,
            max_fitness: -Infinity
        },

        getInnov: function(from, to){
            
            if(this.combn[from]){

                if(!this.combn[from][to]){
                    this.combn[from][to] = ++this.innov
                }

            }else{
                this.combn[from] = {[to]: ++this.innov}
            }


            return this.combn[from][to]
        },
        getHiddenNodeID: function(from, to){
            if(!this.hiddenNodeCombn[from]){
                this.hiddenNodeCombn[from] = {}
            }
            if(!this.hiddenNodeCombn[from][to]){
                this.hiddenNodeCombn[from][to] = ++this.lastHiddenNodeID
            }

            return "h" + this.hiddenNodeCombn[from][to]
        },
        getSpecieID: function(){
            return "S" + (++this.lastSpecieID)
        },
        tickSpecie: function(specie_id, max_fitness){
            if(!this.specieStats[specie_id]){
                this.specieStats[specie_id] = {
                    timer: 0,
                    max_fitness: max_fitness 
                }
            }else{
                if(max_fitness > this.specieStats[specie_id].max_fitness){
                    this.specieStats[specie_id].timer = 0
                    this.specieStats[specie_id].max_fitness = max_fitness
                }else{
                    this.specieStats[specie_id].timer++
                }
            }
        },
        eliminateSpecie: function(specie_id){
            if(this.specieStats[specie_id].timer > 10){
                delete this.specieStats[specie_id]

                return true
            }else{
                return false
            }
        },
        tickPop: function(max_fitness){
            if(max_fitness > this.globalStats.max_fitness){
                this.globalStats.timer = 0
                this.globalStats.max_fitness = max_fitness
            }else{
                this.globalStats.timer++
            }
        },
        refocusPop: function(){
            if (this.globalStats.timer > 15) {
                this.globalStats.timer = 0

                return true
            } else {
                return false
            }
        },
        resetCombn: function(){
            this.combn = {}
        },
    }
}

export {getUniformRandomFromRange, getUniformRandomFromRangeInt, getNormalRandom, percent, createContext, chooseRandomly}