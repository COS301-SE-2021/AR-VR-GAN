import { Controller, UnauthorizedException} from '@nestjs/common';
import { Client, ClientGrpc, GrpcStreamMethod, Transport } from '@nestjs/microservices';
import { Response } from './interfaces/response.interface'
import { ResponsePython } from './interfaces/responsePython.interface';
import { Request } from './interfaces/request.interface'
import { ModelService } from './model.service';
import { Observable,Subject } from 'rxjs';
import { join } from 'path';
import { UserService } from '../user/user.service';
import { AuthenticateUserDto} from '../user/dto/authenticate-user.dto';

@Controller('model')
export class ModelController {
    @Client({
        transport: Transport.GRPC,
        options: {
          package: 'model',
          protoPath: join(__dirname, '../../src/model/model.proto'),
          //protoPath: join(__dirname, '../model/model.proto'),
        },
      })
      client: ClientGrpc;

    constructor(private modelService: ModelService,private readonly userService: UserService) {}

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
            const authenticateDto = new AuthenticateUserDto(message.jwt);
            const success =await  this.userService.authenticateUser(authenticateDto);
            
    
            if( success.success == false)
            {
                //invalid jwt token code goes here
            }
            else
            {
                var data =await this.modelService.proxy(message)
                subject.next({
                    data: data.image
                });
            }
        };
    
        const onComplete = () => subject.complete();

        messages.subscribe({
            next: onNext,
            complete: onComplete,
        });

        return subject.asObservable();
    }

}
