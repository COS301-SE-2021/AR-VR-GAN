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
import { sendEmailDto } from '../mail/dto/send-email.dto';
import { trainModelResponseDto } from './dto/train-model-response.dto';
import { trainModelDto } from './dto/train-model.dto';
import { UserService } from '../user/user.service';

@Injectable()
export class ModelService {
    alreadyTraining: boolean = false;

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

    /**
     * this function handles the changing between models
     * @param request the request object will hold the name of the model to load 
     * @returns a boolean value based on whether or not the change was succesful
     */
    public loadModel(request: loadModelDto): loadModelResponseDto {
        if (request == null)
        {
            const resp = new loadModelResponseDto(false,"The request body was left empty!");
            return resp;
        }
        return this.grpcService.loadModel(request);     
    }

    /**
     * this function handles a request to list all the models it will retrieve a list and and a byte array
     * @param request the request object will hold the boolean values for which models to list "default" or "saved"
     * @returns a promise of the list of models that contains all the details of each model
     */
    public async listModels(request: listModelsDto): Promise<listModelsResponseDto> {
        if (request == null)
        {
            const resp = new listModelsResponseDto(null,null);
            return resp;
        }
        const data = await this.grpcService.listModels(request); 
        return data.toPromise();                  
    }

    /**
     * this function handles a request to retrive the current model name and details
     * @param request 
     * @returns the name and details of the current loaded model
     */
    public currentModel(request: currentModelDto): currentModelResponseDto {
        if (request == null)
        {
            const resp = new currentModelResponseDto(null,null);
            return resp;
        }
        return this.grpcService.currentModel(request);    
    }

    /**
     * This function handles the requests that will send emails 
     * @param request holds the details of the user and the model that was being trained
     */
    public sendEmail(request: sendEmailDto){
        if(request != null)
        {
            this.mailService.sendConfirmationEmail(request);
        }
    }
    
    /**
     * This function handles a request to train a customized model to a user specification,
     * once the model is trained the function will request an email to be sent
     * @param request holds all the parameterss that will be needed.
     * @returns a boolean for whether or not the operation was succesful or not
     */
    public async trainModel(request: trainModelDto): Promise<trainModelResponseDto> {
        if (request == null) {
            let resp = new trainModelResponseDto(false, "Please send a valid request object.");
            return resp;
        }

        let userResponse = await this.userService.getUserByJWTToken(request.jwtToken);

        if (userResponse.user == null) {
            let resp = new trainModelResponseDto(false, "Please login again, your JWT Token has expired.");
            return resp;
        }

        if (this.alreadyTraining) {
            let resp = new trainModelResponseDto(false, "There is currently a model being trained. Please try again later.");
            return resp;
        }

        this.alreadyTraining = true;
        let trainResponse = await this.grpcService.trainModel(request).toPromise();
        this.alreadyTraining = false;

        if (trainResponse.succesful) {
            let emailDto = new sendEmailDto(userResponse.user.username, userResponse.user.email, request.modelName);
            this.sendEmail(emailDto);

            return new trainModelResponseDto(true, `The model, ${request.modelName}, was trained successfully.`);
        } else {
            return new trainModelResponseDto(false, `The model, ${request.modelName}, was not trained successfully.`);
        } 
    }
}
