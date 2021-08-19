import { join } from "path";

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
    })
  }
