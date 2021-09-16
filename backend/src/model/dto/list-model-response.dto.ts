export class listModelsResponseDto {
    readonly models: string[];
    readonly modelDetails: any;

    constructor(modelName: string[],modelDetails: any) {
        this.models = modelName;
        this.modelDetails = modelDetails;
    }
}