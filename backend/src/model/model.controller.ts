import { Body, Controller, Post} from '@nestjs/common';
import { Client, ClientGrpc, GrpcStreamMethod, Transport } from '@nestjs/microservices';
import { Response } from './interfaces/response.interface'
import { ResponsePython } from './interfaces/responsePython.interface';
import { Request } from './interfaces/request.interface'
import { ModelService } from './model.service';
import { Observable,Subject } from 'rxjs';
import { join } from 'path';
import { loadModelDto } from './dto/load-model.dto';
import { loadModelResponseDto } from './dto/load-model-response.dto';
import { listModelsDto } from './dto/list-model.dto';
import { listModelsResponseDto } from './dto/list-model-response.dto';
import { currentModelResponseDto } from './dto/current-model-response.dto';
import { currentModelDto } from './dto/current-model.dto';
import { sendEmailDto } from '../mail/dto/send-email.dto';
import { trainModelDto } from './dto/train-model.dto';
import { trainModelResponseDto } from './dto/train-model-response.dto';
import { UserService } from '../user/user.service';


@Controller('model')
export class ModelController {
    @Client({
        transport: Transport.GRPC,
        options: {
          package: 'model',
          protoPath: join(__dirname, '../../src/model/model.proto')
        },
      })
      client: ClientGrpc;

    constructor(private modelService: ModelService) {}

    @GrpcStreamMethod()
    handleCoords(messages: Observable<Request>): Observable<Response> {
        const subject = new Subject<Response>();
        
        const onNext = (message: Request) => {
            subject.next({
                data: this.modelService.handleCoords(message)
            });
        };

        const onComplete = () => subject.complete();

        messages.subscribe({
            next: onNext,
            complete: onComplete,
        });

        return subject.asObservable();
    }

    /**
     * handles the streaming of coordinates to a python script
     * @param messages 
     * @returns 
     */
    @GrpcStreamMethod('ModelController', 'RunPython')
    runPython(messages: Observable<Request>): Observable<ResponsePython> {
        const subject = new Subject<Response>();

        const onNext =(message: Request) => {
            //when recievimg data from the python we need to remove the [] using replace
            const bufferArray = this.modelService.runPython(message).replace(/[\[\]']+/g,'').split(",");
            
            subject.next({
                data: bufferArray
            });
        };

        const onComplete = () => subject.complete();

        messages.subscribe({
            next: onNext,
            complete: onComplete,
        });

        return subject.asObservable();
    }

    /**
     * handles the streaming of coordinates to a python script using grpc
     * @param messages 
     * @returns 
     */
    @GrpcStreamMethod('ModelController', 'Proxy')
    proxy(messages: Observable<Request>): Observable<ResponsePython> {
        const subject = new Subject<Response>();
    
        const onNext =async (message: Request) => {
            var data =await this.modelService.proxy(message)
            subject.next({
                data: data.image
            });
        };
    
        const onComplete = () => subject.complete();

        messages.subscribe({
            next: onNext,
            complete: onComplete,
        });

        return subject.asObservable();
    }

    /**
     * handles the post request to change the model 
     * @param model holds the name of the model to change to
     * @returns true if the operation was a success
     */
    @Post('/loadModel')
    loadModel(@Body() model: loadModelDto): loadModelResponseDto {
        return this.modelService.loadModel(model);
    }

    /**
     * handles the post requests to list all the models
     * @param request holds the parameters of which models to list
     * @returns the names and all the details of each model
     */
    @Post('/listModels')
    async listModels(@Body() request: listModelsDto): Promise<listModelsResponseDto> {
        const data = await this.modelService.listModels(request);
        const details = data.modelDetails
        for (var i in details) {
            details[i] = JSON.parse(details[i].toString());
        }
        return data;
    }

    /**
     * handles the post request to get the current model and details
     * @returns the current model and details
     */
    @Post('/currentModel')
    currentModel(): currentModelResponseDto {
        const request = new currentModelDto();
        return this.modelService.currentModel(request);
    }

    /**
     * handles a post request to send an email to a user
     * @param request contains the user details
     */
    @Post('/sendEmail')
    sendEmail(@Body() request: sendEmailDto) {
        this.modelService.sendEmail(request);
    }
    
    /**
     * handles a post request to train a model
     * @param request all the details the user specifies to train a model
     * @returns true if the operation was succesful
     */
    @Post('/trainModel')
    async trainModel(@Body() request: trainModelDto): Promise<trainModelResponseDto> {
        let response = await this.modelService.trainModel(request);  
        return response;
    }
}
