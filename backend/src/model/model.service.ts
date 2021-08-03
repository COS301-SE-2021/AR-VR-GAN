import { Injectable } from '@nestjs/common';
import { Request } from './interfaces/request.interface';
import { join } from 'path';

@Injectable()
export class ModelService {
    private num: string
    /**
     * handles the coordinates that will be sent from the front end
     * @param request 
     * @returns 
     */
    public handleCoords(request: Request): number {
        let sum = 0;

        for (let i = 0; i < request.data.length; i++) {
            sum += request.data[i]
        }

        return sum;
    }

    /**
     * executes the python script and returns the data returned from the script
     * @param request 
     * @returns 
     */
    public runPython(request: Request): string {

        var myPythonScriptPath = join(__dirname, '../../src/model/mocks/py-script.py');

        const spawn = require("child_process").spawn;

        var process = spawn('python',[myPythonScriptPath,request.data.toString()]);
        
        //on data recieved from the python script
        process.stdout.on('data',async data =>{
              this.num = data.toString().trim()
              return this.num;
        })

        //If an error occurs in the python script
        process.stderr.on('data',async data =>{
            console.log(data.toString())
      })

        return this.num;
    }
}
