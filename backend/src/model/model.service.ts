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

    public runPython(request: Request): string[] {

        var myPythonScriptPath = join(__dirname, '../../src/model/mocks');
        let options = {
            pythonOptions: ['-u'], // get print results in real-time
            scriptPath: join(__dirname, '../../src/model/mocks'),
            args: [request.data.toString()]
          };

        let num: any
        PythonShell.run('py-script.py', options, function (err, results) {
            if (err) throw err;
            
            console.log(results[0]);
            num = results[0];
        });

        return num;
    }
}
