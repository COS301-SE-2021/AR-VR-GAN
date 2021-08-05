import { Controller } from '@nestjs/common';
import { GrpcMethod, GrpcStreamMethod } from '@nestjs/microservices';
import { Response } from './interfaces/response.interface'
import { ResponsePython } from './interfaces/responsePython.interface';
import { Request } from './interfaces/request.interface'
import { ModelService } from './model.service';
import { Observable, Subject } from 'rxjs';
import { join } from 'path';
import * as fs from 'fs';


@Controller('model')
export class ModelController {
    constructor(private modelService: ModelService) {}

    @GrpcStreamMethod('ModelController', 'HandleCoords')
    handleCoords(messages: Observable<Request>): Observable<Response> {
        const subject = new Subject<Response>();

        const onNext = (message: Request) => {
            var imageNumber = Math.floor(this.modelService.handleCoords(message) % 10);
            var img = fs.readFileSync(join(__dirname, `../../uploads/${imageNumber}.jpg`));

            subject.next({
                data: img
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
     * handles the grpc request to run the python script be calling the service
     * @param request 
     * @returns 
     */
    // @GrpcMethod('ModelController', 'RunPython')
    // runPython(request: Request): ResponsePython {
    //     return { data : this.modelService.runPython(request) };
    // }

    /**
     * handles the streaming of coordinates to a python script
     * @param messages 
     * @returns 
     */
    @GrpcStreamMethod('ModelController', 'RunPython')
    runPython(messages: Observable<Request>): Observable<ResponsePython> {
        const subject = new Subject<Response>();

        const onNext = (message: Request) => {
            subject.next({
                data: this.modelService.runPython(message)
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
     * handles the post requests to call the pytho scripts
     * @param request 
     * @returns 
     */
    // @Post('testPython')
    // testPython(@Body() request: Request): ResponsePython {
    //     return  this.runPython(request);
    // }
}
