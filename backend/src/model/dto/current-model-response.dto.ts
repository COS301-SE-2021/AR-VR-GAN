export class currentModelResponseDto {
    readonly modelName: string;
    readonly modelDetails: any;

    constructor(modelName: string,modelDetails: any) {
        this.modelName = modelName;
        this.modelDetails = modelDetails;
    }
}