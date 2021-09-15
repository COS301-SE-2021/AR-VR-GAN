export class listModelsResponseDto {
    readonly modelName: string;
    readonly modelDetails: any;

    constructor(modelName: string,modelDetails: any) {
        this.modelName = modelName;
        this.modelDetails = modelDetails;
    }
}