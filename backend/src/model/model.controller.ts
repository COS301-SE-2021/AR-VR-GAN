import { Body, Controller, Post, Logger } from '@nestjs/common';
import { GrpcMethod,GrpcStreamMethod } from '@nestjs/microservices';
import { Response } from './interfaces/response.interface'
import { ResponsePython } from './interfaces/responsePython.interface';
import { Request } from './interfaces/request.interface'
import { ModelService } from './model.service';
import { Observable, ReplaySubject, Subject } from 'rxjs';

@Controller('model')
export class ModelController {
    constructor(private modelService: ModelService) {}

    // @GrpcMethod('ModelController', 'HandleCoords')
    // handleCoords(request: Request): Response {
    //     return { sum : this.modelService.handleCoords(request) };
    // }

    // @GrpcStreamMethod('ModelController', 'HandleCoords')
    // handleCoords(request: Observable<Request>): Observable<Response> {
    //     return ({ sum : this.modelService.handleCoords(Observable<request>) }).asObservable();
    // }
    @GrpcStreamMethod('ModelController', 'HandleCoords')
    handleCoords(data$: Observable<Request>): Observable<Response> {
      const hero$ = new Subject<Response>();
  
      const onNext = (request: Request) => {
        const item = { sum : this.modelService.handleCoords(request) };
        hero$.next(item);
      };
      const onComplete = () => hero$.complete();
      data$.subscribe({
        next: onNext,
        complete: onComplete,
      });

      return hero$.asObservable();
    }

    @GrpcMethod('ModelController', 'RunPython')
    runPython(request: Request): ResponsePython {
        return { data : this.modelService.runPython(request) };
    }

    // @Post('testGRPC')
    // testGRPC(@Body() request: Request): Response {
    //     return this.handleCoords(request);
    // }

    @Post('testPython')
    testPython(@Body() request: Request): ResponsePython {
        return  this.runPython(request);
    }
}
