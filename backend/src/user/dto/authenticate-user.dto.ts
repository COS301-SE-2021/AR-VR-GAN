export class AuthenticateUserDto {
    readonly jwtToken: string;
    
    constructor(jwtToken: string) {
        this.jwtToken = jwtToken;
    }
}

export class AuthenticateUserResponseDto {
    readonly success: boolean;
    readonly message: string;
    
    constructor(success: boolean, message: string) {
        this.success = success;
        this.message = message;
    }
}