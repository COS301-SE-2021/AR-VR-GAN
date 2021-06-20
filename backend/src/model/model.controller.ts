import { Body, Controller, Post, Logger } from '@nestjs/common';
import { GrpcMethod } from '@nestjs/microservices';
import { Response } from './interfaces/response.interface'
import { ResponsePython } from './interfaces/responsePython.interface';
import { Request } from './interfaces/request.interface'
import { ModelService } from './model.service';

@Controller('model')
export class ModelController {
    constructor(private modelService: ModelService) {}

    @GrpcMethod('ModelController', 'HandleCoords')
    handleCoords(request: Request): Response {
        return { sum : this.modelService.handleCoords(request) };
    }

    @GrpcMethod('ModelController', 'RunPython')
    runPython(request: Request): ResponsePython {
        return { data : this.modelService.runPython(request) };
    }

    @Post('testGRPC')
    testGRPC(@Body() request: Request): Response {
        return this.handleCoords(request);
    }

    @Post('testPython')
    testPython(@Body() request: Request): ResponsePython {
        return  this.runPython(request);
    }
}
