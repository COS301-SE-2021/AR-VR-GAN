import { Controller,Logger } from '@nestjs/common';
import { GrpcMethod } from '@nestjs/microservices';
import { ICoOrdinates } from './interfaces/ICoOrdinates.interface';
import { ModelService } from './model.service';

@Controller('model')
export class ModelController {
    //to output to terminal
    private logger = new Logger('ModelController');
    constructor(private modelService: ModelService){}

    @GrpcMethod('ModelController','getCoOrdinates')

    getMappedCoOrds( coOrds : ICoOrdinates ) : ICoOrdinates{                            //not sure if this function decleration needs to be the same as the proto
        this.logger.log('Getting mapped Coordinates for ' + coOrds.toString);
        return { coOrdinates : this.modelService.getMappedCoOrds(coOrds.coOrdinates)};  //should return x,y,z interface or number array interface?
    }



}
