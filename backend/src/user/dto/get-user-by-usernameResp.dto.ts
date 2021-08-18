import { User } from "../interfaces/user.interface";
export class GetUserByUsernameResponse {
    readonly success: boolean;
    readonly message: string;
    readonly user: User;

    constructor(success: boolean, message: string, user: User) {
        this.success = success;
        this.message = message;
        this.user = user;
    }
}