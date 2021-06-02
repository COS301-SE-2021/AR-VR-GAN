'use strict'

const supertest = require('supertest')
const server = require('../../Server/Server')

describe('/image',() =>{
    let request
    let data

    before(() =>{
        const app = server()
        request = supertest.agent(app)
        data = {
            "x": "1",
            "y": "5"
        }
    })

    it("returns the x co-ordinate", () =>{
        return request.post("/image").send(data).expect(data.x)
    })
})