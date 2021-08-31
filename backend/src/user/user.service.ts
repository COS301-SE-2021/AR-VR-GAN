import { Injectable, UnauthorizedException } from '@nestjs/common';
import { JwtService } from '@nestjs/jwt';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';
import * as bcrypt from 'bcrypt';
import { User } from './interfaces/user.interface';
import { LoginUserDto } from './dto/login-user.dto';
import { GetAllUsersResponse } from './dto/get-all-users.dto';
import { RegisterUserDto } from './dto/register-user.dto';
import { GetUserByUsernameResponse } from './dto/get-user-by-usernameResp.dto';
import { UpdateUserByUsernameDto } from './dto/update-user-by-username.dto';
import { UserResponse } from './dto/user-response.dto';
import config from '../config/keys';
import { AuthenticateUserDto , AuthenticateUserResponseDto } from './dto/authenticate-user.dto';

@Injectable()
export class UserService {
    constructor(@InjectModel('User') private readonly userModel: Model<User>, private readonly jwtService: JwtService) {}

    /**
     * The method allows a new user to be registered.
     * 
     * @param user 
     * @returns A message containing a success variable and a descriptive message.
     */
    async registerUser(user: RegisterUserDto): Promise<UserResponse> {
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

        const hash = await bcrypt.hash(user.password, config.saltOrRounds);
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

    /**
     * Returns a list of all users in the database if an admin user makes the request.
     * 
     * @param jwtToken 
     * @returns 
     */
    async getAllUsers(jwtToken: string): Promise<GetAllUsersResponse> {
        if (jwtToken == null) {
            return new GetAllUsersResponse(false, 'Please provide a valid JWTToken.', null);
        }

        if (jwtToken.length < 1) {
            return new GetAllUsersResponse(false, 'Please provide a valid JWTToken.', null);
        }

        const userWithJWTToken = await this.userModel.findOne(
            { jwtToken : jwtToken }
        );

        if (userWithJWTToken == null) {
            return new GetAllUsersResponse(false, 'This JWTToken does not exist.', null);
        }

        if (!userWithJWTToken.isAdmin) {
            return new GetAllUsersResponse(false, 'The JWTToken does not belong to an admin user.', null);
        }

        const users = await this.userModel.find();
        
        return new GetAllUsersResponse(true, 'The list of all users is attatched.', users);
    }

    /**
     * Return the user with the given username provided that the user requesting it is an admin.
     * 
     * @param jwtToken 
     * @param username 
     * @returns 
     */
    async getUserByUsername(jwtToken: string, username: string): Promise<GetUserByUsernameResponse> {
        if (jwtToken == null) {
            return new GetUserByUsernameResponse(false, 'Please provide a valid JWTToken.', null);
        }

        if (jwtToken.length < 1) {
            return new GetUserByUsernameResponse(false, 'Please provide a valid JWTToken.', null);
        }

        const userWithJWTToken = await this.userModel.findOne(
            { jwtToken : jwtToken }
        );

        if (userWithJWTToken == null) {
            return new GetUserByUsernameResponse(false, 'This JWTToken does not exist.', null);
        }

        if (!userWithJWTToken.isAdmin) {
            return new GetUserByUsernameResponse(false, 'The JWTToken does not belong to an admin user.', null);
        }

        const userWithUsername = await this.userModel.findOne(
            { username : username }
        );

        if (userWithUsername == null) {
            return new GetUserByUsernameResponse(false, 'There is no user with the given username.', null);
        }

        return new GetUserByUsernameResponse(true, 'The required user is attatched.', userWithUsername);
    }

    /**
     * This function returns the user with the given JWTToken.
     * 
     * @param jwtToken 
     * @returns 
     */
    async getUserByJWTToken(jwtToken: string): Promise<GetUserByUsernameResponse> {
        if (jwtToken == null) {
            return new GetUserByUsernameResponse(false, 'Please provide a valid JWTToken.', null);
        }

        if (jwtToken.length < 1) {
            return new GetUserByUsernameResponse(false, 'Please provide a valid JWTToken.', null);
        }

        const userWithJWTToken = await this.userModel.findOne(
            { jwtToken : jwtToken }
        );

        if (userWithJWTToken == null) {
            return new GetUserByUsernameResponse(false, 'This JWTToken does not exist.', null);
        }

        return new GetUserByUsernameResponse(true, 'The required user is attatched.', userWithJWTToken);
    }

    /**
     * Remove a user with a given username provided that the user requesting the action is an admin user.
     * 
     * @param jwtToken 
     * @param username 
     * @returns 
     */
    async deleteUserByUsername(jwtToken: string, username: string): Promise<UserResponse> {
        if (jwtToken == null) {
            return new UserResponse(false, 'Please provide a valid JWTToken.');
        }

        if (jwtToken.length < 1) {
            return new UserResponse(false, 'Please provide a valid JWTToken.');
        }

        const userWithJWTToken = await this.userModel.findOne(
            { jwtToken : jwtToken }
        );

        if (userWithJWTToken == null) {
            return new UserResponse(false, 'This JWTToken does not exist.');
        }

        if (!userWithJWTToken.isAdmin) {
            return new UserResponse(false, 'The JWTToken does not belong to an admin user.');
        }

        const userWithUsername = await this.userModel.findOne(
            { username : username }
        );

        if (userWithUsername == null) {
            return new UserResponse(false, 'There is no user with the given username.');
        }

        await this.userModel.findByIdAndDelete(userWithUsername.id);

        return new UserResponse(true, 'The user has been removed.');
    }

    /**
     * Allow an admin user to update any user's details and also allow a user to update their own details.
     * 
     * @param updateUserWithUsernameDto 
     * @returns 
     */
    async updateUserWithUsername(updateUserWithUsernameDto: UpdateUserByUsernameDto): Promise<UserResponse> {
        if (updateUserWithUsernameDto.jwtToken == null) {
            return new UserResponse(false, 'Please provide a valid JWTToken.');
        }

        if (updateUserWithUsernameDto.currentUsername == null) {
            return new UserResponse(false, 'Please provide a valid current username.');
        }

        if (updateUserWithUsernameDto.jwtToken.length < 1) {
            return new UserResponse(false, 'Please provide a valid JWTToken.');
        }

        if (updateUserWithUsernameDto.currentUsername.length < 1) {
            return new UserResponse(false, 'Please provide a valid current username.');
        }

        const userWithJWTToken = await this.userModel.findOne(
            { jwtToken : updateUserWithUsernameDto.jwtToken }
        );

        if (userWithJWTToken == null) {
            return new UserResponse(false, 'This JWTToken does not exist.');
        }

        if (!userWithJWTToken.isAdmin) {
            if (updateUserWithUsernameDto.currentUsername != userWithJWTToken.username) {
                return new UserResponse(false, 'You are not an admin user, you may only update your own account.');
            }
        }

        const userWithCurrentUsername = await this.userModel.findOne(
            { username: updateUserWithUsernameDto.currentUsername }
        );

        if (userWithCurrentUsername == null) {
            return new UserResponse(false, 'The user with the username you are trying to update does not exist.');
        }

        const currentUserId = userWithCurrentUsername.id;

        var message = '';

        if (updateUserWithUsernameDto.newUsername != null) {
            if (updateUserWithUsernameDto.newUsername.length > 0) {
                const userWithUsername = await this.userModel.findOne(
                    { username : updateUserWithUsernameDto.newUsername }
                );

                if (userWithUsername != null) {
                    message += 'The username you are trying to update to is already taken. ';
                } else {
                    var updateUsername = await this.userModel.updateOne({ _id: currentUserId }, { username: updateUserWithUsernameDto.newUsername });

                    while (updateUsername.nModified != 1) {
                        updateUsername = await this.userModel.updateOne({ _id: currentUserId }, { username: updateUserWithUsernameDto.newUsername });
                    }

                    message += `The username was updated to ${updateUserWithUsernameDto.newUsername}. `;
                }
            }
        }

        if (updateUserWithUsernameDto.newEmail != null) {
            if (updateUserWithUsernameDto.newEmail.length > 0) {
                const userWithEmail = await this.userModel.findOne(
                    { email : updateUserWithUsernameDto.newEmail }
                );

                if (userWithEmail != null) {
                    message += 'The email you are trying to update to is already taken. ';
                } else {
                    var updateEmail = await this.userModel.updateOne({ _id: currentUserId }, { email: updateUserWithUsernameDto.newEmail });

                    while (updateEmail.nModified != 1) {
                        updateEmail = await this.userModel.updateOne({ _id: currentUserId }, { email: updateUserWithUsernameDto.newEmail });
                    }

                    message += `The email was updated to ${updateUserWithUsernameDto.newEmail}. `;
                }
            }
        }

        if (updateUserWithUsernameDto.newPassword != null) {
            if (updateUserWithUsernameDto.newPassword.length > 0) {
                const hash = await bcrypt.hash(updateUserWithUsernameDto.newPassword, config.saltOrRounds);

                var updatePassword = await this.userModel.updateOne({ _id: currentUserId }, { password: hash });

                while (updatePassword.nModified != 1) {
                    updatePassword = await this.userModel.updateOne({ _id: currentUserId }, { password: hash });
                }

                message += `The password was updated to ${updateUserWithUsernameDto.newPassword}. `;
            }
        }

        if (message.length <= 1) {
            return new UserResponse(false, 'No fields were requested to be updated.');
        }

        return new UserResponse(true, message);
    }

    /**
     * Aunthenticates a user by verifying a jwt token
     * @param user 
     */
    async authenticateUser(user: AuthenticateUserDto): Promise<AuthenticateUserResponseDto> {
        try{
            const data = await this.jwtService.verifyAsync(user.jwtToken);

            if(!data){
                const resp = new AuthenticateUserResponseDto(false);
                return resp;
            }

            const resp = new AuthenticateUserResponseDto(true);
            return resp;
        }
        catch(e){
            const resp = new AuthenticateUserResponseDto(false);
            return resp;
        }
        
    }
}