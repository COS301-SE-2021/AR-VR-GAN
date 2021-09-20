import { Test } from '@nestjs/testing';
import { ModelController } from './model.controller';
import { MockModelService } from './mocks/model.mock';
import { ModelService } from './model.service';
import { INestApplication } from '@nestjs/common';
import { ClientsModule, Transport } from '@nestjs/microservices';
import { join } from 'path';
import * as ProtoLoader from '@grpc/proto-loader';
import * as GRPC from '@grpc/grpc-js';
import { Response } from './interfaces/response.interface'
import { MailModule } from '../mail/mail.module';
import { UsersModule } from '../user/user.module';
import { loadModelDto } from './dto/load-model.dto';
import { trainModelDto } from './dto/train-model.dto';

describe('test grpc on model controller', () => {
  let server;
  let app: INestApplication;
  let client: any;
  let controller: ModelController;

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
              protoPath: join(__dirname, '../../../backend/src/model/modelGenerator.proto'),
              url: "127.0.0.1:50051"
              
            },
          },
        ]),
      ],
    }).overrideProvider(ModelService).useValue(MockModelService).compile();

    app = module.createNestApplication();
    server = app.getHttpAdapter().getInstance();

    app.connectMicroservice({
      transport: Transport.GRPC,
      options: {
        package: ['model'],
        protoPath: [
          join(__dirname, './model.proto'),
        ],
      },
    });
    // Start gRPC microservice
    await app.startAllMicroservices();
    await app.init();
    // Load proto-buffers for test gRPC dispatch
    const proto = ProtoLoader.loadSync(
      join(__dirname, '../model/model.proto'),
    ) as any;
    // Create Raw gRPC client object
    const protoGRPC = GRPC.loadPackageDefinition(proto) as any;
    // Create client connected to started services at standard 5000 port
    client = new protoGRPC.model.ModelController(
      'localhost:5000',
      GRPC.credentials.createInsecure(),
    );

    controller = module.get<ModelController>(ModelController);
  });
  

  it('should be defined', () => {    
    expect(app).toBeDefined();
  })

  it('should be defined', () => {    
    expect(controller).toBeDefined();
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
      if (String(err).toLowerCase().indexOf('cancelled') === -1) {
        fail('error: ' + err);
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
      //callHandler.cancel();
    });

    return new Promise((resolve,reject) => {
      callHandler.write(dto);
      setTimeout(() => resolve(callHandler), 1000);
    });
  });

  it('should reach the proxy through grpc streaming', async () => {
    const dto = {data:[1,2,3]}
    const callHandler = client.proxy(dto);

    var sum = 0;
    for (let i = 0; i < dto.data.length; i++) {
      sum += dto.data[i]
    }

    callHandler.on('data', (msg: Response) => {
      expect(msg).toBeDefined();
      callHandler.cancel();
    });

    callHandler.on('error', (err: any) => {
      //callHandler.cancel();
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


describe('testing post request points', () => {
  let controller: ModelController;

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
              protoPath: join(__dirname, '../../../backend/src/model/modelGenerator.proto'),
              url: "127.0.0.1:50051"
              
            },
          },
        ]),
      ],
    }).overrideProvider(ModelService).useValue(MockModelService).compile();

    controller = module.get<ModelController>(ModelController);
  });

  it('should be defined', () => {    
    expect(controller).toBeDefined();
  })

  it('should return the name of the loaded model', () => {
    const name = "test model";
    const dto = new loadModelDto(name)    
    expect(controller.loadModel(dto)).toEqual(name);
  })

  it('should return the name was not specified', () => {
    const message = "The model name was not specified!";
    const dto = new loadModelDto(null)    
    expect(controller.loadModel(dto).message).toEqual(message);
  })

  it('should return the request body was not specified', () => {
    const message = "The request body was left empty!";
    const dto = null   
    expect(controller.loadModel(dto).message).toEqual(message);
  })

  it('should return the current model', () => {  
    const name = "current model";  
    expect(controller.currentModel().modelName).toEqual(name);
  })

  it('should train a model',async () => {  
    const name = "modelToTrain";  
    const dto =new trainModelDto(name,1,1,"mnist",1,"CAE","jwt");
    expect(await (await controller.trainModel(dto)).message).toEqual(name+" trained");
  })

  it('should return an invalid name error',async () => {   
    const expected = "Please send a valid model name.";
    const dto =new trainModelDto(null,1,1,"mnist",1,"CAE","jwt");
    expect(await (await controller.trainModel(dto)).message).toEqual(expected);
  })

  it('should return an invalid epochs error',async () => {   
    const expected = "Please send a valid training epochs value.";
    const name = "modelToTrain";
    const dto =new trainModelDto(name,null,1,"mnist",1,"CAE","jwt");
    expect(await (await controller.trainModel(dto)).message).toEqual(expected);
  })

  it('should return an invalid latent size error',async () => {   
    const expected = "Please send a valid latent size.";
    const name = "modelToTrain";
    const dto =new trainModelDto(name,1,null,"mnist",1,"CAE","jwt");
    expect(await (await controller.trainModel(dto)).message).toEqual(expected);
  })

  it('should return an invalid dataset name error',async () => {   
    const expected = "Please send a valid dataset name.";
    const name = "modelToTrain";
    const dto =new trainModelDto(name,1,1,null,1,"CAE","jwt");
    expect(await (await controller.trainModel(dto)).message).toEqual(expected);
  })

  it('should return an invalid beta value error',async () => {   
    const expected = "Please send a valid beta value.";
    const name = "modelToTrain";
    const dto =new trainModelDto(name,1,1,"mnist",null,"CAE","jwt");
    expect(await (await controller.trainModel(dto)).message).toEqual(expected);
  })

  it('should return an invalid dataset model type error',async () => {   
    const expected = "Please send a valid model type.";
    const name = "modelToTrain";
    const dto =new trainModelDto(name,1,1,"mnist",1,null,"jwt");
    expect(await (await controller.trainModel(dto)).message).toEqual(expected);
  })

  it('should return an invalid jwt Token error',async () => {   
    const expected = "Please send a valid jwt Token.";
    const name = "modelToTrain";
    const dto =new trainModelDto(name,1,1,"mnist",1,"CAE",null);
    expect(await (await controller.trainModel(dto)).message).toEqual(expected);
  })

});
