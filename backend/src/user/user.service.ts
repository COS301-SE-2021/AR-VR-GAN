import { Injectable } from '@nestjs/common';
import { User } from './interfaces/user.interface';
import { UserResponse } from './dto/user-response.to';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import * as bcrypt from 'bcrypt';

@Injectable()
export class UserService {
    constructor(@InjectModel('User') private readonly userModel: Model<User>) {}

    async getAllUsers(): Promise<User[]> {
        return await this.userModel.find();
    }

    async getUserById(id: string): Promise<User> {
        return await this.userModel.findOne({ _id: id });
    }

    /**
     * The method allows a new user to be registered.
     * 
     * @param user 
     * @returns A message containing a success variable and a descriptive message.
     */
    async registerUser(user: User): Promise<UserResponse> {
        if (user == null) {
            return new UserResponse(false, 'Please send a username, password and email address.');
        }

        if (!(('username' in user) && ('email' in user) && ('password' in user))) {
            return new UserResponse(false, 'Please send a username, password and email address.');
        }

        const userWithUsername = await this.userModel.findOne(
            { username: user.username }
        );

        if (userWithUsername != null) {
            // The user with the specified username already exists.
            return new UserResponse(false, 'This username is already taken.');
        }

        const userWithEmail = await this.userModel.findOne(
            { email: user.email }
        );

        if (userWithEmail != null) {
            // The user with the specified email already exists.
            return new UserResponse(false, 'This email is already taken.');
        }

        const saltOrRounds = 10;
        const hash = await bcrypt.hash(user.password, saltOrRounds);
        user.password = hash;
        const newUser = new this.userModel(user);
        const newUserSaved =  await newUser.save();

        if (newUserSaved === newUser) {
            return new UserResponse(true, 'The user was registered successfully.');
        } else {
            return new UserResponse(false, 'The user was not registered successfully.');
        }
    }

    async deleteUserById(id: string): Promise<User> {
        return await this.userModel.findByIdAndRemove(id);
    }

    async updateUserWithId(id: string, user: User): Promise<User> {
        return await this.userModel.findByIdAndUpdate(id, user, { new: true });
    }
}
