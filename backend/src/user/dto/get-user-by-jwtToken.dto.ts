export class GetUserByJWTTokenDto {
    readonly jwtToken: string;

    constructor(jwtToken: string) {
        this.jwtToken = jwtToken;
    }
}