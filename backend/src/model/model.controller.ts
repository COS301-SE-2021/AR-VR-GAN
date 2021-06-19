import { Body, Controller, Post } from '@nestjs/common';
import { GrpcMethod } from '@nestjs/microservices';

interface Request {
    data: number[];
}

interface Response {
    sum: number;
}

@Controller('model')
export class ModelController {
    @GrpcMethod('ModelController', 'HandleCoords')
    handleCoords(request: Request, metadata: any): Response {
        let sum = 0;

        for (let i = 0; i < request.data.length; i++) {
            sum += request.data[i]
        }

        return { sum };
    }

    @Post('testGRPC')
    testGRPC(@Body() request: Request): Response {
        return this.handleCoords(request, "");
    }
}
