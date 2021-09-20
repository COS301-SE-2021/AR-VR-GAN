import { Module } from '@nestjs/common';
import { ModelController } from './model.controller';
import { ModelService } from './model.service';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { join } from 'path';
import { MailModule } from '../mail/mail.module';
import { UsersModule } from '../user/user.module';

const URL = process.env.URL || "0.0.0.0:50051"
@Module({
  imports: [MailModule, UsersModule,
      ClientsModule.register([
      {
        name: 'MODEL_PACKAGE',
        transport: Transport.GRPC,
        options: {
          package: 'ModelGenerator',
          protoPath: join(__dirname, '../../../backend/src/model/modelGenerator.proto'),
          url: URL,
          credentials: null
  
        },
      },
    ]),
  ],
  controllers: [ModelController],
  providers: [ModelService],
  exports: [ModelService]
})
export class ModelModule {}
