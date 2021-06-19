import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MongooseModule } from '@nestjs/mongoose';
import { UsersModule } from './user/user.module';
import { ModelController } from './model/model.controller';
import { ModelService } from './model/model.service';
import config from './config/keys'
import { ModelModule } from './model/model.module';
import { grpcClientOptions } from './grpc-client.options';
import { ClientsModule } from '@nestjs/microservices';

@Module({
  imports: [UsersModule, MongooseModule.forRoot(config.mongoURI),    ClientsModule.register([
    {
      name: 'MODEL_PACKAGE',
      ...grpcClientOptions,
    },
  ]),],
  controllers: [AppController, ModelController],
  providers: [AppService, ModelService],
})
export class AppModule {}
