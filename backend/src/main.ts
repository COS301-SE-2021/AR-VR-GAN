import { NestFactory } from '@nestjs/core';
import { AppModule } from './app.module';
import { ModelModule } from './model/model.module';
import { Transport } from '@nestjs/microservices';
import { join } from 'path';

//ethan added
const microserviceOptions = {
  Transport : Transport.GRPC,
  options: {
      package: 'coOrds',                                              //the package name from the proto file
      protoPath: join(__dirname,'./dto/coOrdinates.proto')            //links the proto file

  },
};

async function bootstrap() {
  const app = await NestFactory.create(AppModule);
  app.connectMicroservice(microserviceOptions)                        //connects the grpc microService with the proto
  await app.startAllMicroservices();                                  //start tyhe microservice
  await app.listen(3000);
}
bootstrap();
