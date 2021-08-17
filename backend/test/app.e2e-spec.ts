import { Test, TestingModule } from '@nestjs/testing';
import { INestApplication } from '@nestjs/common';
import * as request from 'supertest';
import { AppModule } from '../src/app.module';

import {ModelController} from '../src/model/model.controller';
import { ClientProxy, ClientsModule, Transport } from '@nestjs/microservices';
import { join } from 'path';
import * as ProtoLoader from '@grpc/proto-loader';
import * as GRPC from '@grpc/grpc-js';
import { ModelService } from '../src/model/model.service';
import { Response } from '../src/model/interfaces/response.interface'
import { fail } from 'assert';
//import { expect } from 'chai';
import { readFileSync } from 'fs';
import { UploadController } from '../src/upload/upload.controller';
import { UploadService } from '../src/upload/upload.service';
import { UploadModule } from '../src/upload/upload.module';
import { Observable } from 'rxjs';

// describe('AppController (e2e)', () => {
//   let app: INestApplication;

//   beforeEach(async () => {
//     const moduleFixture: TestingModule = await Test.createTestingModule({
//       imports: [AppModule],
//     }).compile();

//     app = moduleFixture.createNestApplication();
//     await app.init();
//   });

//   it('/ (GET)', () => {
//     return request(app.getHttpServer())
//       .get('/')
//       .expect(200)
//       .expect('Hello World!');
//   });
// });

describe('GRPC transport', () => {
  let server;
  let app: INestApplication;
  let client: any;

  beforeEach(async () => {
    const module = await Test.createTestingModule({
      controllers: [ModelController,UploadController],
      providers: [ModelService,UploadService],
      imports: [
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

  it('GRPC Sending and receiving Stream from RX handler', async () => {
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
      providers: [UploadService],
      imports: [
        UploadModule,
        ClientsModule.register([
          { name: 'UploadService', 
            transport: Transport.TCP,
            options:{
              port: 3000 }
          }
        ]),
      ],
    }).compile();

    app = moduleRef.createNestApplication();

    app.connectMicroservice({
      transport: Transport.TCP,
    });

    await app.startAllMicroservicesAsync();
    await app.init();

    app.enableCors();

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
