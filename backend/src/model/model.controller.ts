import { Body, Controller, Post, Logger } from '@nestjs/common';
import { GrpcMethod,GrpcStreamMethod } from '@nestjs/microservices';
import { Response } from './interfaces/response.interface'
import { ResponsePython } from './interfaces/responsePython.interface';
import { Request } from './interfaces/request.interface'
import { ModelService } from './model.service';
import { Observable, ReplaySubject, Subject } from 'rxjs';
import { createReadStream } from 'fs';
import { join } from 'path';
import * as fs from 'fs';

@Controller('model')
export class ModelController {
    constructor(private modelService: ModelService) {}

    // @GrpcStreamMethod()
    // handleCoords(messages: Observable<Request>): Observable<Response> {
    //     const subject = new Subject<Response>();

    //     const onNext = (message: Request) => {
    //         subject.next({
    //             data: this.modelService.handleCoords(message)
    //         });
    //     };

    //     const onComplete = () => subject.complete();

    //     messages.subscribe({
    //         next: onNext,
    //         complete: onComplete,
    //     });

    //     return subject.asObservable();
    // }

    /**
     * streaming the image back to the client
     * @param messages 
     * @returns 
     */
    @GrpcStreamMethod()
    handleCoords(messages: Observable<Request>): Observable<Response> {
        const subject = new Subject<Response>();
        const img = fs.readFileSync(join(__dirname, '../../uploads/capstone.jpg'));
        const onNext = (message: Request) => {
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

    @GrpcMethod('ModelController', 'RunPython')
    runPython(request: Request): ResponsePython {
        return { data : this.modelService.runPython(request) };
    }
}
