import { Module } from '@nestjs/common';
import { ModelController } from './model.controller';
import { ModelService } from './model.service';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { join } from 'path';
import { UsersModule } from '../user/user.module';
import { UserService } from '../user/user.service';
import { MongooseModule } from '@nestjs/mongoose';
import { UserSchema } from '../user/schemas/user.schema';
import { JwtModule } from '@nestjs/jwt';
import config from '../config/keys';

@Module({
  imports: [MongooseModule.forFeature([{ name: 'User', schema: UserSchema }]),
  JwtModule.register({
    secret: config.jwtSecret,
    signOptions: { expiresIn: '7200s'}
  }),
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
  providers: [ModelService,UserService],
  exports: [ModelService]
})
export class ModelModule {}
