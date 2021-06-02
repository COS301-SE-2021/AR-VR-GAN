'use strict'

const express = require('express')
const model = require('./handlers/model')

function Server () {
    const app = express()
    app.get('/model', model)
    return app
}

module.exports = Server;