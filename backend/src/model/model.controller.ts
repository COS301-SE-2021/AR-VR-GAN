import { Controller,Get,Logger,Param,OnModuleInit , Inject } from '@nestjs/common';
import { GrpcMethod,ClientGrpc,GrpcStreamMethod ,Client} from '@nestjs/microservices';
import { ICoOrdinates } from './interfaces/ICoOrdinates.interface';
import { ModelService } from './model.service';
import { Observable, ReplaySubject, Subject } from 'rxjs';
import { toArray } from 'rxjs/operators';
import { grpcClientOptions } from '../grpc-client.options';
//import { Client } from 'grpc';

// @Controller('model')
// export class ModelController {
//     //to output to terminal
//     private logger = new Logger('ModelController');
//     constructor(private modelService: ModelService){}

//     @GrpcMethod('ModelController','getCoOrdinates')
//     getCoOrdinates(coOrds : ICoOrdinates ) : ICoOrdinates{                            //not sure if this function decleration needs to be the same as the proto
//         this.logger.log('Getting mapped Coordinates for ' + coOrds.toString);
//         return { coOrdinates : this.modelService.getCoOrdinates(coOrds.coOrdinates)};  //should return x,y,z interface or number array interface?
//     }
// }

//////////////////////////////////////////////////////////////////////////////////////

interface ModelSer
{
    getCoOrdinates(coOrds : ICoOrdinates) : ICoOrdinates;
}



@Controller('model')
export class ModelController implements OnModuleInit{ 
    private x: number[] = [1,2,3];
    private list: ICoOrdinates = {coOrdinates : this.x};
    //to output to terminal
    private modelService: ModelSer;

    //constructor(private service: ModelService){}
    

    constructor(@Inject('MODEL_PACKAGE') private readonly client: ClientGrpc,private service: ModelService) {}

    //private grpcService: IGrpcService;

    onModuleInit() {
        this.modelService = this.client.getService<ModelSer>('ModelController');
      }

    @Get()
    getCoOrd() : ICoOrdinates{
        console.log(this.list)
        return this.modelService.getCoOrdinates(this.list)
    }

    @GrpcMethod('ModelController','getCoOrdinates')
    getCoOrdinates(coOrds : ICoOrdinates ) : ICoOrdinates{                            //not sure if this function decleration needs to be the same as the proto
        console.log(coOrds.coOrdinates)
        return { coOrdinates : this.service.getCoOrdinates(coOrds)};  //should return x,y,z interface or number array interface?
    }
}

