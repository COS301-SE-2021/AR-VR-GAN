import { Body, Controller, Post } from '@nestjs/common';
import { GrpcMethod } from '@nestjs/microservices';
import { Response } from './interfaces/response.interface'
import { Request } from './interfaces/request.interface'
import { ModelService } from './model.service';

@Controller('model')
export class ModelController {

    constructor(private modelService: ModelService) {}


    @GrpcMethod('ModelController', 'HandleCoords')
    handleCoords(request: Request): Response {
        return { sum : this.modelService.handleCoords(request) };
    }

    @Post('testGRPC')
    testGRPC(@Body() request: Request): Response {
        return this.handleCoords(request);
    }
}
