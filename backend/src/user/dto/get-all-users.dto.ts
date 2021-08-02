import { User } from "../interfaces/user.interface";

export class GetAllUsersDto {
    readonly jwtToken: string;
    
    constructor(jwtToken: string) {
        this.jwtToken = jwtToken;
    }
}

export class GetAllUsersResponse {
    readonly success: boolean;
    readonly message: string;
    readonly users: User[];

    constructor(success: boolean, message: string, users: User[]) {
        this.success = success;
        this.message = message;
        this.users = users;
    }
}
  