import { Module } from '@nestjs/common';
import { ModelController } from './model.controller';
import { ModelService } from './model.service';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { join } from 'path';

@Module({
  imports: [
    ClientsModule.register([
      {
        name: 'MODEL_PACKAGE',
        transport: Transport.GRPC,
        options: {
          package: 'ModelGenerator',
          protoPath: join(__dirname, '../../../generativeModelFiles/modelGenerator.proto'),
          url: "127.0.0.1:50051"
          
        },
      },
    ]),
  ],
  controllers: [ModelController],
  providers: [ModelService],
  exports: [ModelService]
})
export class ModelModule {}
