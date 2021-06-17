import { Injectable } from '@nestjs/common';
import { User } from './interfaces/user.interface';
import { InjectModel } from '@nestjs/mongoose';
import { Model } from 'mongoose';

@Injectable()
export class UserService {
    constructor(@InjectModel('User') private readonly userModel: Model<User>) {}

    async getAllUsers(): Promise<User[]> {
        return await this.userModel.find();
    }

    async getUserById(id: string): Promise<User> {
        return await this.userModel.findOne({ _id: id });
    }

    async registerUser(user: User): Promise<User> {
        const newUser = new this.userModel(user);
        return await newUser.save();
    }

    async deleteUserById(id: string): Promise<User> {
        return await this.userModel.findByIdAndRemove(id);
    }

    async updateUserWithId(id: string, user: User): Promise<User> {
        return await this.userModel.findByIdAndUpdate(id, user, { new: true });
    }
}
