export interface User {
    id?: string;
    username: string;
    email: string;
    password: string;
    isAdmin: boolean;
    jwtToken: string;
    lastLoggedIn: number;
}