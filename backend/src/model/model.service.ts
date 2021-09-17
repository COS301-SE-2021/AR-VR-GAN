import { Injectable,Inject } from '@nestjs/common';
import { Request } from './interfaces/request.interface';
import { join } from 'path';
import { ClientGrpc } from '@nestjs/microservices';
import { ModelGeneration,RequestProxy } from './grpc.interface';
import { ReplaySubject, Subject} from 'rxjs';
import { loadModelDto } from './dto/load-model.dto';
import { loadModelResponseDto } from './dto/load-model-response.dto';
import { listModelsDto } from './dto/list-model.dto';
import { listModelsResponseDto } from './dto/list-model-response.dto';
import { currentModelResponseDto } from './dto/current-model-response.dto';
import { currentModelDto } from './dto/current-model.dto';
import { MailService } from '../mail/mail.service';
import { sendEmailDto } from 'src/mail/dto/send-email.dto';
import { trainModelResponseDto } from './dto/train-model-response.dto';
import { trainModelDto } from './dto/train-model.dto';
import { UserService } from '../user/user.service';

@Injectable()
export class ModelService {

    constructor(@Inject('MODEL_PACKAGE') private readonly client: ClientGrpc,private mailService: MailService,private userService: UserService) {}
    private grpcService: ModelGeneration;

    onModuleInit() {
        this.grpcService = this.client.getService<ModelGeneration>('ModelGeneration');
    }

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
        //console.log(process.stderr.toString())
        return process.stdout.toString()
    }

    /**
     * acts as a client to the python grpc server to retrieve the image byte array
     * @param request the coordinates from the user to be send to the model
     * @returns image byte array
     */
    public proxy(request: Request): Promise<any> {
        const subject = new ReplaySubject<RequestProxy>();
        subject.next({ vector: request.data });
        subject.complete();
        const stream =this.grpcService.generateImage(subject.asObservable());
        return stream.toPromise();
    }

    public loadModel(request: loadModelDto): loadModelResponseDto {
        return this.grpcService.loadModel(request);     
    }

    public async listModels(request: listModelsDto): Promise<listModelsResponseDto> {
        const data = await this.grpcService.listModels(request); 
        return data.toPromise();                  
    }

    public currentModel(request: currentModelDto): currentModelResponseDto {
        return this.grpcService.currentModel(request);    
    }

    public sendEmail(request: sendEmailDto){
        this.mailService.sendConfirmationEmail(request);
    }
    
    public async trainModel(request: trainModelDto): Promise<trainModelResponseDto> {
        const response = await this.grpcService.trainModel(request);
        response.subscribe( async data => {
            let userResponse = await this.userService.getUserByJWTToken(request.jwtToken);

            let emailDto = new sendEmailDto(userResponse.user.username, userResponse.user.email, request.modelName);
            this.sendEmail(emailDto);
        });

        return response;
    }

}
