import { ClientOptions, Transport } from '@nestjs/microservices';
import { join } from 'path';

export const grpcClientOptions: ClientOptions = {
  transport: Transport.GRPC,
  options: {
    package: 'coOrds', 
    protoPath: join(__dirname, './model/dto/coOrdinates.proto'),
  },
};
