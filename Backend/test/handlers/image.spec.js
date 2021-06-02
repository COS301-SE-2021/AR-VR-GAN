'use strict'

const supertest = require('supertest')
const server = require('../../Server/Server')

/**
 * testing the co-ordinates have been passed through the body of the POST request
 */
describe('/image',() =>{
    let request
    let data

    //before the testing commences 
    before(() =>{
        const app = server()
        request = supertest.agent(app)
        data = {
            "x": "1",
            "y": "5"
        }
    })

    //test that the /image POST request is correct
    it("respond with JSON containing the body", () =>{
        return request.post("/image").send(data).expect(data).expect('Content-Type', /json/).expect(200)
    })
})