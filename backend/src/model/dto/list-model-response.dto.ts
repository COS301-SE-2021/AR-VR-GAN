export class listModelsResponseDto {
    readonly models: string[];

    constructor(models: string[]) {
        this.models = models;
    }
}