import { Injectable } from '@nestjs/common';

@Injectable()
export class ModelService {
    
    public getCoOrdinates(coOrds: Number[]){
        var ret: Number[];                                      //just for testing
        ret = [1,2,3];                                          //just for testing
        return ret;
    }
}
