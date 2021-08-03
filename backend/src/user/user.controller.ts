import { Body, Controller, Param, Post } from '@nestjs/common';
import { RegisterUserDto } from './dto/register-user.dto';
import { UserService } from './user.service';
import { User } from './interfaces/user.interface';
import { UserResponse } from './dto/user-response.dto';
import { LoginUserDto } from './dto/login-user.dto';
import { GetAllUsersDto, GetAllUsersResponse } from './dto/get-all-users.dto';
import { GetUserByUsernameDto } from './dto/get-user-by-username.dto';
import { GetUserByUsernameResponse } from './dto/get-user-by-username.dto';
import { UpdateUserByUsernameDto } from './dto/update-user-by-username.dto';

@Controller('user')
export class UserController {
    constructor(private readonly userService: UserService) {}

    @Post('/register')
    registerUser(@Body() registerUserDto: RegisterUserDto): Promise<UserResponse> {
        return this.userService.registerUser(registerUserDto);
    }

    @Post('/login')
    loginUser(@Body() loginUserDto: LoginUserDto): Promise<UserResponse> {
        return this.userService.loginUser(loginUserDto);
    }

    @Post('/getAllUsers')
    getAllUsers(@Body() getAllUsersDto: GetAllUsersDto): Promise<GetAllUsersResponse> {
        return this.userService.getAllUsers(getAllUsersDto.jwtToken);
    }

    @Post('/getUserByUsername')
    getUserByUsername(@Body() getUserByUsernameDto : GetUserByUsernameDto): Promise<GetUserByUsernameResponse> {
        return this.userService.getUserByUsername(getUserByUsernameDto.jwtToken, getUserByUsernameDto.username);
    }

    @Post('/deleteUserByUsername')
    deleteUserByUsername(@Body() deleteUserByUsername: GetUserByUsernameDto): Promise<UserResponse> {
        return this.userService.deleteUserByUsername(deleteUserByUsername.jwtToken, deleteUserByUsername.username);
    }

    @Post('/updateUserWithUsername')
    updateUserWithUsername(@Body() updateUserByUsername: UpdateUserByUsernameDto): Promise<UserResponse> {
        return this.userService.updateUserWithUsername(updateUserByUsername);
    }
}
