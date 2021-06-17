import { Body, Controller, Delete, Get, Param, Post, Put } from '@nestjs/common';
import { RegisterUserDto } from './dto/register-user.dto';
import { UserService } from './user.service';
import { User } from './interfaces/user.interface';

@Controller('user')
export class UserController {
    constructor(private readonly userService: UserService) {}

    @Get()
    getAllUsers(): Promise<User[]> {
        return this.userService.getAllUsers();
    }

    @Get(':id')
    getUserById(@Param('id') id): Promise<User> {
        return this.userService.getUserById(id);
    }

    @Post()
    registerUser(@Body() registerUserDto: RegisterUserDto): Promise<User> {
        return this.userService.registerUser(registerUserDto);
    }

    @Delete(':id')
    deleteUserById(@Param('id') id): Promise<User> {
        return this.userService.deleteUserById(id);
    }

    @Put(':id')
    updateUserWithId(@Param('id') id, @Body() updateUserDto: RegisterUserDto): Promise<User> {
        return this.userService.updateUserWithId(id, updateUserDto);
    }
}
