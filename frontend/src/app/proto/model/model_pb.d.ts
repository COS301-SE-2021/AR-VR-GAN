// package: model
// file: model/model.proto

import * as jspb from "google-protobuf";

export class RequestDto extends jspb.Message {
  clearDataList(): void;
  getDataList(): Array<number>;
  setDataList(value: Array<number>): void;
  addData(value: number, index?: number): number;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): RequestDto.AsObject;
  static toObject(includeInstance: boolean, msg: RequestDto): RequestDto.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: RequestDto, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): RequestDto;
  static deserializeBinaryFromReader(message: RequestDto, reader: jspb.BinaryReader): RequestDto;
}

export namespace RequestDto {
  export type AsObject = {
    dataList: Array<number>,
  }
}

export class ResponseDto extends jspb.Message {
  getSum(): number;
  setSum(value: number): void;

  serializeBinary(): Uint8Array;
  toObject(includeInstance?: boolean): ResponseDto.AsObject;
  static toObject(includeInstance: boolean, msg: ResponseDto): ResponseDto.AsObject;
  static extensions: {[key: number]: jspb.ExtensionFieldInfo<jspb.Message>};
  static extensionsBinary: {[key: number]: jspb.ExtensionFieldBinaryInfo<jspb.Message>};
  static serializeBinaryToWriter(message: ResponseDto, writer: jspb.BinaryWriter): void;
  static deserializeBinary(bytes: Uint8Array): ResponseDto;
  static deserializeBinaryFromReader(message: ResponseDto, reader: jspb.BinaryReader): ResponseDto;
}

export namespace ResponseDto {
  export type AsObject = {
    sum: number,
  }
}

