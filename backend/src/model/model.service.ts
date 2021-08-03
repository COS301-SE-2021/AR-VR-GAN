import { Injectable } from '@nestjs/common';
import { Request } from './interfaces/request.interface';
import { join } from 'path';

@Injectable()
export class ModelService {
    private num: string
    
    public handleCoords(request: Request): number {
        let sum = 0;

        for (let i = 0; i < request.data.length; i++) {
            sum += request.data[i]
        }

        return sum;
    }

    public runPython(request: Request): string {    
        var myPythonScriptPath = join(__dirname, '../../src/model/mocks/py-script.py');

        const spawn = require("child_process").spawn;

        var process = spawn('python',[myPythonScriptPath,request.data.toString()]);
          
        process.stdout.on('data',async data =>{
              this.num = data.toString().trim()
              return this.num;
        })

        return this.num;
    }
}
