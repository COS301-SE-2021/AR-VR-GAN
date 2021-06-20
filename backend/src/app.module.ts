import { Module } from '@nestjs/common';
import { AppController } from './app.controller';
import { AppService } from './app.service';
import { MongooseModule } from '@nestjs/mongoose';
import { UsersModule } from './user/user.module';
import { ModelModule } from './model/model.module';
import { UploadModule } from './upload/upload.module';
import config from './config/keys'


@Module({
  imports: [UploadModule, ModelModule, UsersModule, MongooseModule.forRoot(config.mongoURI)],
  controllers: [AppController],
  providers: [AppService],
})
export class AppModule {}
