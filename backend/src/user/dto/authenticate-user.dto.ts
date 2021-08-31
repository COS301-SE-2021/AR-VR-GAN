export class AuthenticateUserDto {
    readonly jwtToken: string;
    
    constructor(jwtToken: string) {
        this.jwtToken = jwtToken;
    }
}

export class AuthenticateUserResponseDto {
    readonly success: boolean;
    
    constructor(success: boolean) {
        this.success = success;
    }
}