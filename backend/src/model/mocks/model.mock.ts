import { join } from "path";
import { listModelsResponseDto } from "../dto/list-model-response.dto";
import { loadModelResponseDto } from "../dto/load-model-response.dto";

export const MockModelService= {

    handleCoords: jest.fn((dto) => {
        let sum = 0;

        for (let i = 0; i < dto.data.length; i++) {
            sum += dto.data[i]
        }
        return sum;
    }),

    runPython: jest.fn((request) => {
        var myPythonScriptPath = join(__dirname, './py-script.py');

        const spawnSync = require("child_process").spawnSync;

        var commaSplitList = request.data.toString().split(',');

        var process = spawnSync('python',[myPythonScriptPath,commaSplitList]);

        return process.stdout.toString()
    }),

    proxy: jest.fn(async (request) => {
        let sum = 0;

        for (let i = 0; i < request.data.length; i++) {
            sum += request.data[i]
        }
        return sum;
    }),

    loadModel: jest.fn((request) => {
        if (request == null)
        {
            const resp = new loadModelResponseDto(false,"The request body was left empty!");
            return resp;
        }
        return request.modelName;
    }),

    listModels: jest.fn((request) => {
        if (request == null)
        {
            const resp = new listModelsResponseDto(null,null);
            return resp;
        }
        if(request.saved == true && request.default == false)
        {
            const resp = new listModelsResponseDto(["saved"],"saved model");
            return resp;
        }
        if(request.default == true && request.saved == false)
        {
            const resp = new listModelsResponseDto(["saved"],"default model");
            return resp;
        }
        const resp = new listModelsResponseDto(["model"],"all models");
        return resp;
    })
  }
