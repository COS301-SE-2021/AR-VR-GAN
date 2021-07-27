import * as mongoose from 'mongoose';

export const UserSchema = new mongoose.Schema({
    username: String,
    email: String,
    password: String,
    isAdmin: {
        type: Boolean,
        default: false
    }
});