export class loadModelResponseDto {
    readonly succesful: boolean;
    readonly message: string;

    constructor(succesful: boolean, message: string) {
        this.succesful = succesful;
        this.message = message;
    }
}