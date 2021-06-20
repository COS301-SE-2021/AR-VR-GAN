import { Test, TestingModule } from '@nestjs/testing';
import { UploadController } from './upload.controller';
import { MockUploadService } from './mocks/upload.mock';
import { UploadService } from './upload.service';


describe('UploadController', () => {
  let controller: UploadController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [UploadController],
      providers: [UploadService]
    }).overrideProvider(UploadService).useValue(MockUploadService).compile();

    controller = module.get<UploadController>(UploadController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
