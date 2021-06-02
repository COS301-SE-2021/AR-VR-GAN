'use strict'

const express = require('express')
const image = require('./handlers/image')

/**
 * 
 * @returns returns the express app
 */
function Server () {
    const app = express()

    //allows for us to use req.body
    app.use(express.json());
    app.use(express.urlencoded());

    //calls the image function if the post /image is called
    app.post('/image', (req,res) =>{
        image(req.body,req,res)
    })

    
    return app
}

module.exports = Server;