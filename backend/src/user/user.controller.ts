import { Body, Controller, Delete, Get, Param, Post, Put } from '@nestjs/common';
import { RegisterUserDto } from './dto/register-user.dto';

@Controller('user')
export class UserController {
    @Get()
    getAllUsers(): string {
        return 'All users';
    }

    @Get(':id')
    getUserById(@Param('id') id): string {
        return `User ID: ${id}`;
    }

    @Post()
    registerUser(@Body() registerUserDto: RegisterUserDto): string {
        return `Username: ${registerUserDto.username}, Email: ${registerUserDto.email}, Password: ${registerUserDto.password}`;
    }

    @Delete(':id')
    deleteUserById(@Param('id') id): string {
        return `Delete user with ID: ${id}`;
    }

    @Put(':id')
    updateUserWithId(@Param('id') id, @Body() updateItemDto: RegisterUserDto): string {
        return `Update user with ID: ${id}, Username: ${updateItemDto.username}, Email: ${updateItemDto.email}, Password: ${updateItemDto.password}`;
    }
}
