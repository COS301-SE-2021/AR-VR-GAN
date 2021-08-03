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

    @GrpcMethod('ModelController', 'RunPython')
    runPython(request: Request): ResponsePython {
        return { data : this.modelService.runPython(request) };
    }
}
