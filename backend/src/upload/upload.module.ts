import { Module } from '@nestjs/common';
import { UploadController } from './upload.controller';
import { UploadService } from './upload.service';
import { ModelModule } from 'src/model/model.module';

@Module({
  imports: [ModelModule],
  controllers: [UploadController],
  providers: [UploadService],
})
export class UploadModule {}
