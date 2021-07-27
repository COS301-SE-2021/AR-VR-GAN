import { Injectable } from '@nestjs/common';
import { User } from './interfaces/user.interface';
import { UserResponse } from './dto/user-response.dto';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import * as bcrypt from 'bcrypt';
import { LoginUserDto } from './dto/login-user.dto';
import { JwtService } from '@nestjs/jwt';

@Injectable()
export class UserService {
    constructor(@InjectModel('User') private readonly userModel: Model<User>, private readonly jwtService: JwtService) {}

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

    /**
     * Login a user with a given username and password.
     * 
     * @param user 
     * @returns A variable that indicates a successful login and a JWTToken for the user's session.
     */
    async loginUser(user: LoginUserDto): Promise<UserResponse> {
        if (user == null) {
            return new UserResponse(false, 'Please send a username and password.');
        }

        if (!(('username' in user) && ('password' in user))) {
            return new UserResponse(false, 'Please send a username and password.');
        }

        const userWithUsername = await this.userModel.findOne(
            { username: user.username }
        );

        if (userWithUsername == null) {
            return new UserResponse(false, 'The user with the specified username does not exist.');
        }

        const isMatch = await bcrypt.compare(user.password, userWithUsername.password);

        if (!isMatch) {
            return new UserResponse(false, 'The password is incorrect.');
        }
        
        const payload = { username: userWithUsername.username, sub: userWithUsername.id };
        var JWTToken = this.jwtService.sign(payload);

        var userWithJWTToken = await this.userModel.findOne(
            { jwtToken: JWTToken }
        );

        while (userWithJWTToken != null) {
            JWTToken = this.jwtService.sign(payload);

            userWithJWTToken = await this.userModel.findOne(
                { jwtToken: JWTToken }
            );
        }

        var updateToken = await this.userModel.updateOne({ username: userWithUsername.username }, { jwtToken: JWTToken });

        while (updateToken.nModified != 1) {
            updateToken = await this.userModel.updateOne({ username: userWithUsername.username }, { jwtToken: JWTToken });
        }

        var updateLastLoggedIn = await this.userModel.updateOne({ username: userWithUsername.username }, { lastLoggedIn: Date.now() });

        while (updateLastLoggedIn.nModified != 1) {
            updateLastLoggedIn = await this.userModel.updateOne({ username: userWithUsername.username }, { lastLoggedIn: Date.now() });
        }

        return new UserResponse(true, JWTToken);
    }

    async getAllUsers(): Promise<User[]> {
        return await this.userModel.find();
    }

    async getUserById(id: string): Promise<User> {
        return await this.userModel.findOne({ _id: id });
    }

    async deleteUserById(id: string): Promise<User> {
        return await this.userModel.findByIdAndRemove(id);
    }

    async updateUserWithId(id: string, user: User): Promise<User> {
        return await this.userModel.findByIdAndUpdate(id, user, { new: true });
    }
}
