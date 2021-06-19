import { Injectable } from '@nestjs/common';
import { ICoOrdinates } from './interfaces/ICoOrdinates.interface';

@Injectable()
export class ModelService {

    public getCoOrdinates(coOrds: ICoOrdinates): number[]{
        // var ret: number[];                                      //just for testing
        // ret = [1,2,3];                                          //just for testing
        console.log(coOrds.coOrdinates)
        return coOrds.coOrdinates;
    }
}
