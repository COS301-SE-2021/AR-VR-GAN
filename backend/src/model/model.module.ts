import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { ModelController } from './model.controller';
import { ModelService } from './model.service';
import { grpcClientOptions } from '../grpc-client.options';
import { ClientsModule } from '@nestjs/microservices';

@Module({
  imports: [],
  controllers: [ModelController],
  providers: [ModelService]
})
export class ModelModule {}
