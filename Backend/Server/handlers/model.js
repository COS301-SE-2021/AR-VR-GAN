'use strict'
/**
 * handles the training of the neueral network from a passed through file
 * @param {the image dataset} data 
 * @param {*} req 
 * @param {*} res 
 */
function model(data,req,res){
    //TODO: Train the model with the given in file
    res.status(200).send(data)
}

module.exports = model