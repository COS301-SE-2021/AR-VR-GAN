export class RegisterUserDto {
    readonly username: string;
    readonly email: string;
    password: string;


    constructor(username: string, email: string, password: string) {
        this.username = username;
        this.email = email;
        this.password = password;
    }
}
  