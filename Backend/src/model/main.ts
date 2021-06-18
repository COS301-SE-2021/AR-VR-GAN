import { NestFactory } from '@nestjs/core';
import { ModelModule } from './model.module';
import { Transport } from '@nestjs/microservices';
import { join } from 'path';
import { Logger } from '@nestjs/common';

const logger = new Logger('Main');
const microserviceOptions = {
    Transport : Transport.GRPC,
    options: {
        package: 'coOrds',                                              //the package name from the proto file
        protoPath: join(__dirname,'./dto/coOrdinates.proto')            //links the proto file

    },
};

async function bootstrap() {
    const app = await NestFactory.createMicroservice(ModelModule , microserviceOptions);
    app.listen( () => {
        logger.log('Microservice is listening');
    });
  }
  bootstrap();