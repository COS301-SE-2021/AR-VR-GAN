import { Module } from '@nestjs/common';
import { MongooseModule } from '@nestjs/mongoose';
import { UsersModule } from './user/user.module';
import { ModelModule } from './model/model.module';
import { UploadModule } from './upload/upload.module';
import { MailModule } from './mail/mail.module';
import config from './config/keys'


@Module({
  imports: [UploadModule, ModelModule, UsersModule, MongooseModule.forRoot(config.mongoURI), MailModule]
})
export class AppModule {}
