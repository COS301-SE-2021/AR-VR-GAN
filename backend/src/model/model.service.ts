import { Injectable } from '@nestjs/common';
import { Response } from './interfaces/response.interface';
import { Request } from './interfaces/request.interface';
import { PythonShell } from 'python-shell';
import { join } from 'path';
//import { spawn } from 'child_process';

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
        let options = {
            pythonOptions: ['-u'], // get print results in real-time
            scriptPath: join(__dirname, '../../src/model/mocks'),
            args: [request.data.toString()]
          };

        
        // PythonShell.run('py-script.py', options, function (err, results) {
        //     if (err) throw err;
            
        //     console.log(results[0]);
        //     num = results[0];
        // });

        const spawn = require("child_process").spawn;
          var process = spawn('python',[myPythonScriptPath,request.data.toString()]);
          process.stdout.on('data',async data =>{
              this.num = data.toString().trim()
              console.log(data.toString().trim());
              return this.num;
          })

        

        return this.num;
    }
}
