export class sendEmailDto {
    readonly username: string;
    readonly email: string;
    readonly modelName: string;

    constructor(name: string , email: string, modelName: string) {
        this.username = name;
        this.email = email;
        this.modelName = modelName;
    }
}