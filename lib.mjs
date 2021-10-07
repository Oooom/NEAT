function getUniformRandomFromRange(min, max){
    return Math.random() * (max - min) + min
}

function getUniformRandomFromRangeInt(min, max){
    min = Math.ceil(min);
    max = Math.floor(max);
    return Math.floor(Math.random() * (max - min) + min);
}

function getNormalRandom() {
    var u = 0, v = 0;
    while(u === 0) u = Math.random();
    while(v === 0) v = Math.random();
    return Math.sqrt( -2.0 * Math.log( u ) ) * Math.cos( 2.0 * Math.PI * v );
}

function percent(num){
    return Math.random() * 100 < num
}

function createContext(){
    return {
        innov: 0,
        combn: {},
        getInnov: function(from, to){
            
            if(this.combn[from]){

                if(!this.combn[from][to]){
                    this.combn[from][to] = ++this.innov
                }

            }else{
                this.combn[from] = {[to]: ++this.innov}
            }


            return this.combn[from][to]
        }
    }
}

export {getUniformRandomFromRange, getUniformRandomFromRangeInt, getNormalRandom, percent, createContext}