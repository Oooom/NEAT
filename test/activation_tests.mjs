import assert from "assert"
import {sigmoid, steepSigmoid} from "../nn.mjs"


describe("activation function tests", function(){
    it("sigmoid test", function () {
        assert.equal( sigmoid(1) - 0.731 < 0.001,  true)
    })
    it("steepSigmoid test", function () {
        assert.equal( steepSigmoid(1) - 0.992 < 0.001,  true)
    })
})