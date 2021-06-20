import { Injectable } from '@nestjs/common';
import { Response } from './interfaces/response.interface'
import { Request } from './interfaces/request.interface'

@Injectable()
export class ModelService {
    public handleCoords(request: Request): number {
        let sum = 0;

        for (let i = 0; i < request.data.length; i++) {
            sum += request.data[i]
        }

        return sum;
    }
}
