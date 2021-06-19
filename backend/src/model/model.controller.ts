import { Body, Controller, Post } from '@nestjs/common';
import { GrpcMethod } from '@nestjs/microservices';
import { Response } from './interfaces/response.interface'
import { Request } from './interfaces/request.interface'

@Controller('model')
export class ModelController {
    @GrpcMethod('ModelController', 'HandleCoords')
    handleCoords(request: Request): Response {
        let sum = 0;

        for (let i = 0; i < request.data.length; i++) {
            sum += request.data[i]
        }

        return { sum };
    }

    @Post('testGRPC')
    testGRPC(@Body() request: Request): Response {
        return this.handleCoords(request);
    }
}
