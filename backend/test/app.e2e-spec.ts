import { Test} from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import {ModelController} from '../src/model/model.controller';
import { ClientProxy, ClientsModule, Transport } from '@nestjs/microservices';
import { join } from 'path';
import * as ProtoLoader from '@grpc/proto-loader';
import * as GRPC from '@grpc/grpc-js';
import { ModelService } from '../src/model/model.service';
import { Response } from '../src/model/interfaces/response.interface'
import { fail } from 'assert';
import { UploadController } from '../src/upload/upload.controller';
import { UploadService } from '../src/upload/upload.service';
import { UploadModule } from '../src/upload/upload.module';
import { MailModule } from '../src/mail/mail.module';
import { UsersModule } from '../src/user/user.module';

describe('GRPC transport', () => {
  let server;
  let app: INestApplication;
  let client: any;

  beforeEach(async () => {
    const module = await Test.createTestingModule({
      controllers: [ModelController],
      providers: [ModelService],
      imports: [MailModule,UsersModule,
        ClientsModule.register([
          {
            name: 'MODEL_PACKAGE',
            transport: Transport.GRPC,
            options: {
              package: 'ModelGenerator',
              protoPath: join(__dirname, '../../generativeModelFiles/modelGenerator.proto'),
              url: "127.0.0.1:50051"
              
            },
          },
        ]),
      ],
    }).compile();

    app = module.createNestApplication();
    server = app.getHttpAdapter().getInstance();

    app.connectMicroservice({
      transport: Transport.GRPC,
      options: {
        package: ['model'],
        protoPath: [
          join(__dirname, '../src/model/model.proto'),
        ],
      },
    });
    // Start gRPC microservice
    await app.startAllMicroservices();
    await app.init();

    // Load proto-buffers for test gRPC dispatch
    const proto = ProtoLoader.loadSync(
      join(__dirname, '../src/model/model.proto'),
    ) as any;
    // Create Raw gRPC client object
    const protoGRPC = GRPC.loadPackageDefinition(proto) as any;
    // Create client connected to started services at standard 5000 port
    client = new protoGRPC.model.ModelController(
      'localhost:5000',
      GRPC.credentials.createInsecure(),
    );
  });

  it('should be defined', () => {    
    expect(app).toBeDefined();
  })

  it('GRPC streaming the coordinates', async () => {
    const dto = {data:[1,2,3]}
    const callHandler = client.handleCoords(dto);

    var sum = 0;
    for (let i = 0; i < dto.data.length; i++) {
      sum += dto.data[i]
    }

    callHandler.on('data', (msg: Response) => {
      expect(msg.data).toEqual(sum);
      callHandler.cancel();
    });

    callHandler.on('error', (err: any) => {
      // We want to fail only on real errors while Cancellation error
      // is expected
      if (String(err).toLowerCase().indexOf('cancelled') === -1) {
        fail('gRPC Stream error happened, error: ' + err);
      }
    });

    return new Promise((resolve,reject) => {
      callHandler.write(dto);
      setTimeout(() => resolve(callHandler), 1000);
    });
  });

  it('GRPC streaming the coordinates to run python script', async () => {
    const dto = {data:[1,2,3]}
    const callHandler = client.runPython(dto);

    callHandler.on('data', (msg: Response) => {
      expect(msg.data).toBeDefined();
      callHandler.cancel();
    });

    callHandler.on('error', (err: any) => {
      // We want to fail only on real errors while Cancellation error
      // is expected
      if (String(err).toLowerCase().indexOf('cancelled') === -1) {
        fail('gRPC Stream error happened, error: ' + err);
      }
    });

    return new Promise((resolve,reject) => {
      callHandler.write(dto);
      setTimeout(() => resolve(callHandler), 1000);
    });
  });

  afterEach(async () => {
    await app.close();
  });

});

describe('E2E FileTest', () => {
  let app: INestApplication;
  let client: ClientProxy;

  beforeAll(async () => {
    const moduleRef = await Test.createTestingModule({
      controllers: [UploadController],
      providers: [UploadService,ModelService],
      imports: [
        UploadModule,MailModule,UsersModule,
        ClientsModule.register([
          { name: 'UploadService', 
            transport: Transport.TCP,
            options:{
              port: 3000 }
          }
        ]),
        ClientsModule.register([
          {
            name: 'MODEL_PACKAGE',
            transport: Transport.GRPC,
            options: {
              package: 'ModelGenerator',
              protoPath: join(__dirname, '../../generativeModelFiles/modelGenerator.proto'),
              url: "127.0.0.1:50051"
              
            },
          },
        ]),
      ],
    }).compile();

    app = moduleRef.createNestApplication();

    app.connectMicroservice({
      transport: Transport.TCP,
    });

    app.enableCors({
      allowedHeaders: '*',
      origin: '*'
    })


    await app.startAllMicroservicesAsync();
    await app.init();


    client = app.get('UploadService');
    await client.connect();
  });

  it('server runs for TCP connection', () => {    
    expect(app).toBeDefined();
  })

  it('TCP connection success', () => {    
    expect(client).toBeDefined();
  })

  afterEach(async () => {
    await app.close();
  });

});
