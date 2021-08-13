import { Injectable,Inject } from '@nestjs/common';
import { Request } from './interfaces/request.interface';
import { join } from 'path';
import { ChildProcess } from 'child_process';

import { Client, ClientGrpc } from '@nestjs/microservices';
import { microserviceOptions } from './grpc.options';
import { ModelGeneration,RequestProxy } from './grpc.interface';
import { Observable, ReplaySubject, Subject } from 'rxjs';
import { toArray } from 'rxjs/operators';

@Injectable()
export class ModelService {

    constructor(@Inject('MODEL_PACKAGE') private readonly client: ClientGrpc) {}
    private grpcService: ModelGeneration;

    onModuleInit() {
        this.grpcService = this.client.getService<ModelGeneration>('ModelGeneration');
    }

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

    // /**
    //  * executes the python script and returns the data returned from the script -- spawn
    //  * @param request 
    //  * @returns 
    //  */
    // public runPython(request: Request): string {

    //         var myPythonScriptPath = join(__dirname, '../../../generativeModelFiles/modelGenerator.py');
    //         var myPythonModelPath = join(__dirname, '../../../generativeModelFiles/defaultModels/Epochs-50.pt');
    
    //         const spawn = require("child_process").spawn;
    
    //         var commaSplitList = request.data.toString().split(',');
    //         var coord1 = parseFloat(commaSplitList[0]);
    //         var coord2 = parseFloat(commaSplitList[1]);
    //         var coord3 = parseFloat(commaSplitList[2]);
    //         var process = null;
    //         var process = spawn('python',["-W ignore",myPythonScriptPath,"--coordinates",coord1,coord2,coord3,"--model",myPythonModelPath],{ encoding : 'utf8' });
            
            
    //         //on data recieved from the python script
    //         process.stdout.on('data', async data =>{
    //               this.num = data.toString().trim()
    //               return this.num;
    //         })
    
    //         //If an error occurs in the python script
    //         process.stderr.on('data',async data =>{
    //             console.log(data.toString())
    //         })
    
    //         process.on('close',async data =>{
    //             return this.num;
    //         })
    
    //         return this.num;
    
    //     }

    /**
     * executes the python script and returns the data returned from the script -- spawnSync
     * @param request 
     * @returns 
     */
    public runPython(request: Request): any {

        var myPythonScriptPath = join(__dirname, '../../../generativeModelFiles/modelGenerator.py');
        var myPythonModelPath = join(__dirname, '../../../generativeModelFiles/defaultModels/Epochs-50.pt');

        const spawnSync = require("child_process").spawnSync;

        var commaSplitList = request.data.toString().split(',');
        var coord1 = parseFloat(commaSplitList[0]);
        var coord2 = parseFloat(commaSplitList[1]);
        var coord3 = parseFloat(commaSplitList[2]);

        var process = spawnSync('python',["-W ignore",myPythonScriptPath,"--coordinates",coord1,coord2,coord3,"--model",myPythonModelPath]);
        console.log(process.stderr.toString())
        return process.stdout.toString()
    }

    // /**
    //  * executes the python script and returns the data returned from the script -- execFile
    //  * @param request 
    //  * @returns 
    //  */
    // public async runPython(request: Request): Promise<string> {

    //     var myPythonScriptPath = join(__dirname, '../../../generativeModelFiles/modelGenerator.py');
    //     var myPythonModelPath = join(__dirname, '../../../generativeModelFiles/defaultModels/Epochs-50.pt');

    //     var commaSplitList = request.data.toString().split(',');
    //     var coord1 = parseFloat(commaSplitList[0]);
    //     var coord2 = parseFloat(commaSplitList[1]);
    //     var coord3 = parseFloat(commaSplitList[2]);

    //     const util = require('util');
    //     const execFile = util.promisify(require('child_process').execFile);
    //     async function getImage(){
    //         const { stdout } = await execFile('python',["-W ignore",myPythonScriptPath,"--coordinates",coord1,coord2,coord3,"--model",myPythonModelPath]);
    //         return stdout;
    //     }
    //     return await getImage();

    // }
    
    // /**
    //  * executes the python script and returns the data returned from the script -- spawnSync
    //  * @param request 
    //  * @returns 
    //  */
    //  public async proxy(request: Request): Promise<any> {
    //     const req = new ReplaySubject<Request>();
    //     req.next({data : request.data})
    //     req.complete()
    //     return  this.grpcService.generateImage(req).toPromise();
    // }

    /**
     * executes the python script and returns the data returned from the script -- spawnSync
     * @param request 
     * @returns 
     */
         public async proxy(request: Request): Promise<any> {
            const ids$ = new ReplaySubject<RequestProxy>();
            ids$.next({ vector: request.data });
            ids$.complete();
            const stream =await this.grpcService.generateImage(ids$.asObservable());
            return stream.toPromise();
        }
    

}
