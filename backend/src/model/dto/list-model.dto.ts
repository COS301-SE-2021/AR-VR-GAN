export class listModelsDto {
    readonly default: boolean;
    readonly saved: boolean;

    constructor(standard : boolean , saved: boolean) {
        this.default = standard;
        this.saved = saved;
    }
}