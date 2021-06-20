// package: model
// file: model/model.proto

import * as model_model_pb from "../model/model_pb";
import {grpc} from "@improbable-eng/grpc-web";

type ModelControllerHandleCoords = {
  readonly methodName: string;
  readonly service: typeof ModelController;
  readonly requestStream: false;
  readonly responseStream: false;
  readonly requestType: typeof model_model_pb.RequestDto;
  readonly responseType: typeof model_model_pb.ResponseDto;
};

export class ModelController {
  static readonly serviceName: string;
  static readonly HandleCoords: ModelControllerHandleCoords;
}

export type ServiceError = { message: string, code: number; metadata: grpc.Metadata }
export type Status = { details: string, code: number; metadata: grpc.Metadata }

interface UnaryResponse {
  cancel(): void;
}
interface ResponseStream<T> {
  cancel(): void;
  on(type: 'data', handler: (message: T) => void): ResponseStream<T>;
  on(type: 'end', handler: (status?: Status) => void): ResponseStream<T>;
  on(type: 'status', handler: (status: Status) => void): ResponseStream<T>;
}
interface RequestStream<T> {
  write(message: T): RequestStream<T>;
  end(): void;
  cancel(): void;
  on(type: 'end', handler: (status?: Status) => void): RequestStream<T>;
  on(type: 'status', handler: (status: Status) => void): RequestStream<T>;
}
interface BidirectionalStream<ReqT, ResT> {
  write(message: ReqT): BidirectionalStream<ReqT, ResT>;
  end(): void;
  cancel(): void;
  on(type: 'data', handler: (message: ResT) => void): BidirectionalStream<ReqT, ResT>;
  on(type: 'end', handler: (status?: Status) => void): BidirectionalStream<ReqT, ResT>;
  on(type: 'status', handler: (status: Status) => void): BidirectionalStream<ReqT, ResT>;
}

export class ModelControllerClient {
  readonly serviceHost: string;

  constructor(serviceHost: string, options?: grpc.RpcOptions);
  handleCoords(
    requestMessage: model_model_pb.RequestDto,
    metadata: grpc.Metadata,
    callback: (error: ServiceError|null, responseMessage: model_model_pb.ResponseDto|null) => void
  ): UnaryResponse;
  handleCoords(
    requestMessage: model_model_pb.RequestDto,
    callback: (error: ServiceError|null, responseMessage: model_model_pb.ResponseDto|null) => void
  ): UnaryResponse;
}

