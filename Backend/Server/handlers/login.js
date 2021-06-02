'use strict'
/**
 * Handles the Post request of a user loging In
 * @param {has the data passed through from the POST request} data 
 * @param {request object} req 
 * @param {response object} res 
 */
function login(data,req,res){
    // TODO: communicate with Neural network and Unreal scripts
    res.status(200).send(data)
}

module.exports = login