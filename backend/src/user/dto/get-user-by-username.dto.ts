
export class GetUserByUsernameDto {
    readonly jwtToken: string;
    readonly username: string;
    

    constructor(jwtToken:string,username: string) {
        this.jwtToken = jwtToken;
        this.username = username;
        
    }
}

