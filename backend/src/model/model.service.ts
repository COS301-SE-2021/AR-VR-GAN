import { Injectable } from '@nestjs/common';
import { Response } from './interfaces/response.interface';
import { Request } from './interfaces/request.interface';
import { PythonShell } from 'python-shell';
import { join } from 'path';

@Injectable()
export class ModelService {
    public handleCoords(request: Request): number {
        let sum = 0;

        for (let i = 0; i < request.data.length; i++) {
            sum += request.data[i]
        }

        return  sum;
    }

    public runPython(request: Request): number {

        var myPythonScriptPath = './mocks/py-script.py';
        var options = {
            mode: "text",
            pythonOptions: ['-u'],
            scriptPath: myPythonScriptPath
        }

        console.log("here");
        PythonShell.run(join(__dirname, '../../src/model/mocks/py-script.py'), null, function (err, results) {
            if (err) throw err;

            console.log('results: ', results);
        });

        return 2;
    }
}
