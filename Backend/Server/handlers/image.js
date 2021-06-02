'use strict'

function image(data,req,res){
    res.status(200).send(data.x)
}

module.exports = image