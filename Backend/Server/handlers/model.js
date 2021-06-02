'use strict'
/**
 * 
 * @param {*} data 
 * @param {*} req 
 * @param {*} res 
 */
function model(data,req,res){
    res.status(200).send(data)
}

module.exports = model