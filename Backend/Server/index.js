'use strict'

const server = require('./Server')
const PORT = process.env.PORT || 3000

const app = server()
app.listen(PORT, () => {
    console.log(`Listening on Port ${PORT}`)
})

