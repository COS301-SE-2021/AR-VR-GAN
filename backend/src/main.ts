import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { MicroserviceOptions, Transport } from '@nestjs/microservices';
import { join } from 'path';

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  const microserviceTCP = app.connectMicroservice<MicroserviceOptions>({
    transport: Transport.TCP,
    options: {
      port: 3000,
    }
  });

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