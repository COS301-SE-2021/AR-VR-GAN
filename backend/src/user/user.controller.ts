import { Body, Controller, Post } from '@nestjs/common';
import { RegisterUserDto } from './dto/register-user.dto';
import { UserService } from './user.service';
import { UserResponse } from './dto/user-response.dto';
import { LoginUserDto } from './dto/login-user.dto';
import { GetAllUsersDto, GetAllUsersResponse } from './dto/get-all-users.dto';
import { GetUserByUsernameDto } from './dto/get-user-by-username.dto';
import { GetUserByUsernameResponse } from './dto/get-user-by-usernameResp.dto';
import { UpdateUserByUsernameDto } from './dto/update-user-by-username.dto';
import { GetUserByJWTTokenDto } from './dto/get-user-by-jwtToken.dto';

@Controller('user')
export class UserController {
    constructor(private readonly userService: UserService) {}

    /**
     * handles the post request to register a user
     * @param registerUserDto 
     * @returns 
     */
    @Post('/register')
    registerUser(@Body() registerUserDto: RegisterUserDto): Promise<UserResponse> {
        return this.userService.registerUser(registerUserDto);
    }

    /**
     * handles the post request for logging in a user
     * @param loginUserDto 
     * @returns 
     */
    @Post('/login')
    loginUser(@Body() loginUserDto: LoginUserDto): Promise<UserResponse> {
        return this.userService.loginUser(loginUserDto);
    }

    /**
     * handles the post request for retrieving all the users
     * @param getAllUsersDto 
     * @returns 
     */
    @Post('/getAllUsers')
    getAllUsers(@Body() getAllUsersDto: GetAllUsersDto): Promise<GetAllUsersResponse> {
        return this.userService.getAllUsers(getAllUsersDto.jwtToken);
    }

    /**
     * handles the post request to retrieve a user by a given username
     * @param getUserByUsernameDto 
     * @returns 
     */
    @Post('/getUserByUsername')
    getUserByUsername(@Body() getUserByUsernameDto : GetUserByUsernameDto): Promise<GetUserByUsernameResponse> {
        return this.userService.getUserByUsername(getUserByUsernameDto.jwtToken, getUserByUsernameDto.username);
    }

    /**
     * handles the post request to delete a user by username
     * @param deleteUserByUsername 
     * @returns 
     */
    @Post('/deleteUserByUsername')
    deleteUserByUsername(@Body() deleteUserByUsername: GetUserByUsernameDto): Promise<UserResponse> {
        return this.userService.deleteUserByUsername(deleteUserByUsername.jwtToken, deleteUserByUsername.username);
    }

    /**
     * handles the post request for updating a user with a username
     * @param updateUserByUsername 
     * @returns 
     */
    @Post('/updateUserWithUsername')
    updateUserWithUsername(@Body() updateUserByUsername: UpdateUserByUsernameDto): Promise<UserResponse> {
        return this.userService.updateUserWithUsername(updateUserByUsername);
    }

    @Post('/getUserByJWTToken')
    getUserByJWTToken(@Body() getUserByJWTTokenDto: GetUserByJWTTokenDto): Promise<GetUserByUsernameResponse> {
        return this.userService.getUserByJWTToken(getUserByJWTTokenDto.jwtToken);
    }
}
