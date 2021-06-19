import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ModelModule } from './model/model.module';
import { Transport } from '@nestjs/microservices';
import { join } from 'path';
import { Logger } from '@nestjs/common';

// async function bootstrap() {
//   const app = await NestFactory.create(AppModule);
//   // await app.listen(3000);

//   const configService = app.get(ConfigService);
  
//   await app.connectMicroservice<MicroserviceOptions>({
//     transport: Transport.GRPC,
//     options: {
//       package: 'model',
//       protoPath: join(process.cwd(), 'src/model/model.proto'),
//       url: configService.get('localhost:5000')
//     },
//   });
 
//   app.startAllMicroservices();
// }
// bootstrap();

const logger = new Logger('Main');
const microserviceOptions = {
  transport: Transport.GRPC,  
  options: {
    package: 'model', 
    protoPath: join(__dirname, '../src/model/model.proto'), 
  },
};

async function bootstrap() {
  const app = await NestFactory.createMicroservice(ModelModule, microserviceOptions);
  app.listen(() => {
    logger.log('Microservice is listening...');
  });
}
bootstrap();
