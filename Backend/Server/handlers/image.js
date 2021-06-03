'use strict'
/**
 * Handles the Post request of a user updating their co-ordinates and will return a generated image
 * @param {has the data passed through from the POST request} data 
 * @param {request object} req 
 * @param {response object} res 
 */
function image(data,req,res){
    // TODO: communicate with Neural network and Unreal scripts
    res.status(200).send(data)
}

module.exports = image