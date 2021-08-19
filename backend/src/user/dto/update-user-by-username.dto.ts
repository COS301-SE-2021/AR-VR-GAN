export class UpdateUserByUsernameDto {
    readonly jwtToken: string;              // User who is doing the update.
    readonly currentUsername: string;       // The user whose details you want to update.
    readonly newUsername: string;
    readonly newPassword: string;
    readonly newEmail: string;

    constructor(jwtToken: string, currentUsername: string, newUsername: string, newPassword:string, newEmail:string) {
        this.jwtToken = jwtToken;
        this.currentUsername = currentUsername;
        this.newUsername = newUsername
        this.newPassword = newPassword
        this.newEmail= newEmail
    }
}