export class LoginUserDto {
    readonly username: string;
    readonly password: string;

    constructor(username: string, password:string) {
        this.username = username;
        this.password = password;
    }
}
  