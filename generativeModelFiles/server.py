import grpc
import asyncio
import concurrent
from concurrent import futures
import sys
import pathlib
# sys.path.append(str(pathlib.Path(__file__).parent.parent)+"\generativeModelFiles")


import modelGenerator_pb2
import modelGenerator_pb2_grpc
from modelGenerator import ModelGenerator
m_generator = ModelGenerator()
global_vector = []

class ModelGenerationServicer(modelGenerator_pb2_grpc.ModelGenerationServicer):

    async def get_image_data(self, request, context):
        async for item in request:
            # print(type(item))
            # print(item.vector)
            # global_vector = item.vector
            write_task = asyncio.create_task(self.write_image_data(context, item.vector))
            await write_task

    async def write_image_data(self, context, latent_vector):
        # print(f"write : {latent_vector}")
        if latent_vector == []:
            latent_vector = [0.00 for x in range(m_generator.model.retrieve_latent_size())]
        generated_image = bytes(m_generator.generateImage(latent_vector))
        await context.write(modelGenerator_pb2.ImageResponse(width=0, height=0, image=generated_image))
        # await asyncio.sleep(0.0515)

    async def GenerateImage(self, request: modelGenerator_pb2.ImageRequest, context):
        read_task = asyncio.create_task(self.get_image_data(request, context))
        await read_task

    def LoadDataset(self, request, context):
        pass

    def TrainModel(self, request, context):
        pass

    def LoadModel(self, request, context):
        print(f"The Model: {request.modelName}")

        response = modelGenerator_pb2.LoadModelResponse()
        response.succesful = True
        try:
            m_generator.loadModel(request.modelName)
        except:
            response.succesful = False
            response.message = f"Unable to load {request.modelName}, try another model."
            return response
        response.message = "Successful: "+ str(m_generator.model.retrieve_latent_size())
        # response.message = "Successful: "
        return response


async def serve():
    server = grpc.aio.server()
    modelGenerator_pb2_grpc.add_ModelGenerationServicer_to_server(ModelGenerationServicer(), server)
    server.add_insecure_port("[::]:50051")
    print(f"Model Generator Server Started. Listening on port 50051")
    await server.start()
    await server.wait_for_termination()

    # ThreadPoolExecutor(max_workers=20))
    # modelGenerator_pb2_grpc.add_ModelGenerationServicer_to_server(ModelGenerationServicer(), server)
    # port_number = server.add_insecure_port('[::]:50051') # Change this to secure_port when we are not in development
    # print(f"Model Generator Server Started. Listening on port {port_number}")
    # server.start()
    # server.wait_for_termination()

if __name__ == "__main__":
    # m_generator.loadModel("./generativeModelFiles/defaultModels/Epochs-50.pt")
    # server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
    # modelGenerator_pb2_grpc.add_ModelGenerationServicer_to_server(ModelGenerationServicer(), server)
    # port_number = server.add_insecure_port('[::]:50051') # Change this to secure_port when we are not in development
    # print(f"Model Generator Server Started. Listening on port {port_number}")
    # server.start()
    # server.wait_for_termination()
    asyncio.run(serve())