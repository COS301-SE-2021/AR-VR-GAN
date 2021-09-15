import { Test, TestingModule } from '@nestjs/testing';
import { UploadController } from './upload.controller';
import { MockUploadService } from './mocks/upload.mock';
import { UploadService } from './upload.service';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { join } from 'path';
import { ModelService } from '../../src/model/model.service';


describe('UploadController', () => {
  let controller: UploadController;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      controllers: [UploadController],
      providers: [UploadService,ModelService],
      imports: [
        ClientsModule.register([
          {
            name: 'MODEL_PACKAGE',
            transport: Transport.GRPC,
            options: {
              package: 'ModelGenerator',
              protoPath: join(__dirname, '../../../backend/src/model/modelGenerator.proto'),
              url: "127.0.0.1:50051"
              
            },
          },
        ]),
      ]
    }).overrideProvider(UploadService).useValue(MockUploadService).compile();

    controller = module.get<UploadController>(UploadController);
  });

  it('should be defined', () => {
    expect(controller).toBeDefined();
  });
});
