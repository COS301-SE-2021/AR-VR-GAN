import { Module } from '@nestjs/common';
import { ModelModule } from '../model/model.module';
import { UploadController } from './upload.controller';
import { UploadService } from './upload.service';

@Module({
  imports: [ModelModule],
  controllers: [UploadController],
  providers: [UploadService],
})
export class UploadModule {}
