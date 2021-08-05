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

        var myPythonScriptPath = join(__dirname, '../../../generativeModelFiles/modelGenerator.py');
        var myPythonModelPath = join(__dirname, '../../../generativeModelFiles/defaultModels/Epochs-50.pt');

        const spawn = require("child_process").spawn;

        var commaSplitList = request.data.toString().split(',');
        var coord1 = parseFloat(commaSplitList[0]);
        var coord2 = parseFloat(commaSplitList[1]);
        var coord3 = parseFloat(commaSplitList[2]);

        var process = spawn('python',["-W ignore",myPythonScriptPath,"--coordinates",coord1,coord2,coord3,"--model",myPythonModelPath]);
        
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
