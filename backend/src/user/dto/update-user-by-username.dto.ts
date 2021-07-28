export class UpdateUserByUsernameDto {
    readonly jwtToken: string;              // User who is doing the update.
    readonly currentUsername: string;       // The user whose details you want to update.
    
    readonly newUsername: string;
    readonly newPassword: string;
    readonly newEmail: string;
}