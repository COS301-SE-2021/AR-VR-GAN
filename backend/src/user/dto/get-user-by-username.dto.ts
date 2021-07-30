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

export class GetUserByUsernameDto {
    readonly jwtToken: string;
    readonly username: string;

    constructor(jwtToken: string, username: string) {
        this.jwtToken = jwtToken;
        this.username = username;

    }
}