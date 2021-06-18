import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MongooseModule } from '@nestjs/mongoose';
import { UsersModule } from './user/user.module';
import { ModelController } from './model/model.controller';
import { ModelService } from './model/model.service';
import config from './config/keys'

@Module({
  imports: [UsersModule, MongooseModule.forRoot(config.mongoURI)],
  controllers: [AppController, ModelController],
  providers: [AppService, ModelService],
})
export class AppModule {}
