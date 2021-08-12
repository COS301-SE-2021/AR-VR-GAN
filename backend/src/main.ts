import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { MicroserviceOptions, Transport } from '@nestjs/microservices';
import { join } from 'path';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  /**
   * hadles the TCP communications on server, POST/GET requests that are done to the server 
   * runs on port 3000
   */
  const microserviceTCP = app.connectMicroservice<MicroserviceOptions>({
    transport: Transport.TCP,
    options: {
      port: 3000,
    }
  });

  /**
   * handles the GRPC communication on the server
   * runs on port 3001
   */
  const microserviceGRPC = app.connectMicroservice<MicroserviceOptions>({
    transport: Transport.GRPC,  
    options: {
      package: 'model', 
      protoPath: join(__dirname, '../src/model/model.proto'),
      url: '127.0.0.1:3001', 
    },
  });  
  

  await app.startAllMicroservicesAsync();                                
  await app.listen(3000);
}

bootstrap();