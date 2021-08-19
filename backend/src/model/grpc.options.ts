import { ClientOptions, Transport } from '@nestjs/microservices';
import { join } from 'path';

export const microserviceOptions: ClientOptions = {
  transport: Transport.GRPC,
  options: {
    package: 'ModelGenerator',
    protoPath: join(__dirname, '../../../generativeModelFiles/modelGenerator.proto'),
    url: "0.0.0.0:50051"
  },
};