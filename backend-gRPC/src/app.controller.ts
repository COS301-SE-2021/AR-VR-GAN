import { Controller, Post, Body, OnModuleInit, Logger } from '@nestjs/common';
import { Client, ClientGrpc } from '@nestjs/microservices';
import { microserviceOptions } from './grpc.options';
import { IGrpcService } from './grpc.interface';

@Controller()
export class AppController implements OnModuleInit {
  @Client(microserviceOptions)
  private client: ClientGrpc;
  private logger = new Logger('AppController');
  private grpcService: IGrpcService;

  onModuleInit() {
    this.grpcService = this.client.getService<IGrpcService>('ModelController');
  }

  @Post('testGRPC')
  async accumulate(@Body() data: RequestDto)  {
    this.logger.log(`The numbers to be added are: ${data.data}`);
    return this.grpcService.handleCoords(data); 
  }
}

interface RequestDto {
  data: number[];
}