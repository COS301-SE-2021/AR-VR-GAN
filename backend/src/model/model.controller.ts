import { Controller } from '@nestjs/common';
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
        return {sum: 3};
    }
}
