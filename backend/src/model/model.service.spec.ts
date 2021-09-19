import { ClientsModule, Transport } from '@nestjs/microservices';
import { Test, TestingModule } from '@nestjs/testing';
import { join } from 'path';
import { ModelService } from './model.service';
import { Request } from './interfaces/request.interface';
import { MailModule } from '../mail/mail.module';
import { UsersModule } from '../user/user.module';
import { MongooseModule } from '@nestjs/mongoose';
import config from '../config/keys';

describe('ModelService', () => {
  let service: ModelService;

  beforeEach(async () => {
    const module: TestingModule = await Test.createTestingModule({
      imports: [MailModule,UsersModule,
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
      ],
      providers: [ModelService],
    }).compile();

    service = module.get<ModelService>(ModelService);
  });

  it('should be defined', () => {
    expect(service).toBeDefined();
  });

  // it('should sum up the data', async() => {
  //   const dto = { data : [1,2,3] }
  //   let sum = 0;

  //   for (let i = 0; i < dto.data.length; i++) {
  //       sum += dto.data[i]
  //   }

  //   expect(service.handleCoords(dto)).toEqual(sum);
  // });

  // it('should run python the python script', () => {
  //   const dto = {data: [1.1,1.1,1.1]}
  //   expect(service.runPython(dto)).toBeDefined();
  // });
});
