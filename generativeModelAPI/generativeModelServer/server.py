import grpc
import concurrent
from concurrent import futures

import modelGenerator_pb2
import modelGenerator_pb2_grpc
from generativeModelFiles.modelGenerator import ModelGenerator

class ModelGenerationServicer(modelGenerator_pb2_grpc.ModelGenerationServicer):
    m_generator = ModelGenerator()
    def GenerateImage(self, request, context):
        pass

    def LoadDataset(self, request, context):
        pass

    def TrainModel(self, request, context):
        pass

    def LoadModel(self, request, context):
        print(f"The Model: {request.modelName}")

        response = modelGenerator_pb2.LoadModelResponse()
        response.successful = True
        response.message = "Successful"
        return response

if __name__ == "__main__":
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    modelGenerator_pb2_grpc.add_ModelGenerationServicer_to_server(ModelGenerationServicer(), server)
    port_number = server.add_insecure_port('[::]:50051') # Change this to secure_port when we are not in development
    print(f"Model Generator Server Started. Listening on port {port_number}")
    server.start()
    server.wait_for_termination()