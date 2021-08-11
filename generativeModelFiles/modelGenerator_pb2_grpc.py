# Generated by the gRPC Python protocol compiler plugin. DO NOT EDIT!
"""Client and server classes corresponding to protobuf-defined services."""
import grpc

import modelGenerator_pb2 as modelGenerator__pb2


class ModelGenerationStub(object):
    """Missing associated documentation comment in .proto file."""

    def __init__(self, channel):
        """Constructor.

        Args:
            channel: A grpc.Channel.
        """
        self.GenerateImage = channel.stream_stream(
                '/ModelGenerator.ModelGeneration/GenerateImage',
                request_serializer=modelGenerator__pb2.ImageRequest.SerializeToString,
                response_deserializer=modelGenerator__pb2.ImageResponse.FromString,
                )
        self.LoadDataset = channel.unary_unary(
                '/ModelGenerator.ModelGeneration/LoadDataset',
                request_serializer=modelGenerator__pb2.LoadDatasetRequest.SerializeToString,
                response_deserializer=modelGenerator__pb2.LoadDatasetResponse.FromString,
                )
        self.TrainModel = channel.unary_unary(
                '/ModelGenerator.ModelGeneration/TrainModel',
                request_serializer=modelGenerator__pb2.TrainModelRequest.SerializeToString,
                response_deserializer=modelGenerator__pb2.TrainModelResponse.FromString,
                )
        self.LoadModel = channel.unary_unary(
                '/ModelGenerator.ModelGeneration/LoadModel',
                request_serializer=modelGenerator__pb2.LoadModelRequest.SerializeToString,
                response_deserializer=modelGenerator__pb2.LoadModelResponse.FromString,
                )


class ModelGenerationServicer(object):
    """Missing associated documentation comment in .proto file."""

    def GenerateImage(self, request_iterator, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def LoadDataset(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def TrainModel(self, request, context):
        """Missing associated documentation comment in .proto file."""
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')

    def LoadModel(self, request, context):
        """- Delete Model // Could be used later
        """
        context.set_code(grpc.StatusCode.UNIMPLEMENTED)
        context.set_details('Method not implemented!')
        raise NotImplementedError('Method not implemented!')


def add_ModelGenerationServicer_to_server(servicer, server):
    rpc_method_handlers = {
            'GenerateImage': grpc.stream_stream_rpc_method_handler(
                    servicer.GenerateImage,
                    request_deserializer=modelGenerator__pb2.ImageRequest.FromString,
                    response_serializer=modelGenerator__pb2.ImageResponse.SerializeToString,
            ),
            'LoadDataset': grpc.unary_unary_rpc_method_handler(
                    servicer.LoadDataset,
                    request_deserializer=modelGenerator__pb2.LoadDatasetRequest.FromString,
                    response_serializer=modelGenerator__pb2.LoadDatasetResponse.SerializeToString,
            ),
            'TrainModel': grpc.unary_unary_rpc_method_handler(
                    servicer.TrainModel,
                    request_deserializer=modelGenerator__pb2.TrainModelRequest.FromString,
                    response_serializer=modelGenerator__pb2.TrainModelResponse.SerializeToString,
            ),
            'LoadModel': grpc.unary_unary_rpc_method_handler(
                    servicer.LoadModel,
                    request_deserializer=modelGenerator__pb2.LoadModelRequest.FromString,
                    response_serializer=modelGenerator__pb2.LoadModelResponse.SerializeToString,
            ),
    }
    generic_handler = grpc.method_handlers_generic_handler(
            'ModelGenerator.ModelGeneration', rpc_method_handlers)
    server.add_generic_rpc_handlers((generic_handler,))


 # This class is part of an EXPERIMENTAL API.
class ModelGeneration(object):
    """Missing associated documentation comment in .proto file."""

    @staticmethod
    def GenerateImage(request_iterator,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.stream_stream(request_iterator, target, '/ModelGenerator.ModelGeneration/GenerateImage',
            modelGenerator__pb2.ImageRequest.SerializeToString,
            modelGenerator__pb2.ImageResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def LoadDataset(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ModelGenerator.ModelGeneration/LoadDataset',
            modelGenerator__pb2.LoadDatasetRequest.SerializeToString,
            modelGenerator__pb2.LoadDatasetResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def TrainModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ModelGenerator.ModelGeneration/TrainModel',
            modelGenerator__pb2.TrainModelRequest.SerializeToString,
            modelGenerator__pb2.TrainModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)

    @staticmethod
    def LoadModel(request,
            target,
            options=(),
            channel_credentials=None,
            call_credentials=None,
            insecure=False,
            compression=None,
            wait_for_ready=None,
            timeout=None,
            metadata=None):
        return grpc.experimental.unary_unary(request, target, '/ModelGenerator.ModelGeneration/LoadModel',
            modelGenerator__pb2.LoadModelRequest.SerializeToString,
            modelGenerator__pb2.LoadModelResponse.FromString,
            options, channel_credentials,
            insecure, call_credentials, compression, wait_for_ready, timeout, metadata)
