import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { Transport } from '@nestjs/microservices';
import { join } from 'path';

const microserviceOptions = {
  transport: Transport.GRPC,  
  options: {
    package: 'model', 
    protoPath: join(__dirname, '../src/model/model.proto'), 
  },
};

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  
  app.connectMicroservice(microserviceOptions);              
  
  await app.startAllMicroservices();                                 
  await app.listen(3000);
}
bootstrap();