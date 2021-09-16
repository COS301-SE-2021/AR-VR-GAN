import internal from "stream";

export class trainModelDto {
    readonly modelName: string;
    readonly trainingEpochs: number;
    readonly latentSize: number;
    readonly datasetName: string;
    readonly beta: number;
    readonly modelType: string;
    readonly jwtToken: string;

    constructor(modelName: string,trainingEpochs: number,latentSize: number,datasetName: string,beta: number,modelType: string,jwtToken: string) {
        this.modelName= modelName;
        this.trainingEpochs= trainingEpochs;
        this.latentSize= latentSize;
        this.datasetName= datasetName;
        this.beta= beta;
        this.modelType= modelType;
        this.jwtToken= jwtToken;
    }
}
