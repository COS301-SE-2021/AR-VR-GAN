import { Controller } from '@nestjs/common';
import { GrpcMethod, GrpcStreamMethod } from '@nestjs/microservices';
import { Response } from './interfaces/response.interface'
import { ResponsePython } from './interfaces/responsePython.interface';
import { Request } from './interfaces/request.interface'
import { ModelService } from './model.service';
import { Observable, Subject } from 'rxjs';
import * as fs from 'fs';
import { join } from 'path';

@Controller('model')
export class ModelController {
    constructor(private modelService: ModelService) {}

    @GrpcStreamMethod()
    handleCoords(messages: Observable<Request>): Observable<Response> {
        const subject = new Subject<Response>();

        const img = fs.readFileSync(join(__dirname, '../../uploads/0.jpg'));

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
