from concurrent import futures
import torch

from torchvision import transforms
from DataLoaders import DataLoaders
import grpc
import asyncio

import modelGenerator_pb2
import modelGenerator_pb2_grpc
from modelGenerator import ModelGenerator
import json
# m_generator = ModelGenerator()
global_vector = []
SAVED_MODELS_DIR ="./savedModels/"
class ModelGenerationServicer(modelGenerator_pb2_grpc.ModelGenerationServicer):
    def __init__(self) -> None:
        super().__init__()
        kwargs = {'num_workers': 1, 'pin_memory': True} if torch.cuda.is_available() else {}

        # To create a custom dataset go to https://pytorch.org/tutorials/beginner/data_loading_tutorial.html
        im_transform = transforms.Compose([
            # transforms.Resize((32,32)),
            transforms.ToTensor(),
        ])
        self.data_loaders = DataLoaders(im_transform, kwargs)
        self.m_generator = ModelGenerator(self.data_loaders)
        self.m_generator.loadModel("Beta-1-CIFAR-20.pt")

    async def get_image_data(self, request, context):
        async for item in request:
            write_task = asyncio.create_task(self.write_image_data(context, item.vector))
            await write_task

    async def write_image_data(self, context, latent_vector):
        if latent_vector == []:
            latent_vector = [0.00 for x in range(self.m_generator.model.retrieve_latent_size())]
        generated_image = bytes(self.m_generator.generateImage(latent_vector))
        await context.write(modelGenerator_pb2.ImageResponse(width=0, height=0, image=generated_image))

    async def GenerateImage(self, request: modelGenerator_pb2.ImageRequest, context):
        read_task = asyncio.create_task(self.get_image_data(request, context))
        await read_task

    def LoadDataset(self, request, context):
        pass

    def ListModels(self, request, context):
        defaults: bool = request.default
        saved: bool = request.saved

        default_list: list = []
        saved_list: list = []

        if defaults:
            default_list = self.m_generator.get_available_models()
        if saved:
            saved_list = self.m_generator.get_available_models(False)
        
        total_list: list = default_list + saved_list
        total_list = list(set(total_list)) # Removes duplicates from the list

        temp_mg = ModelGenerator(self.data_loaders)
        
        response = modelGenerator_pb2.ListModelsResponse()

        for model in total_list:
            temp_mg.loadModel(model)
            details = temp_mg.model.details()
            new_dict = {x:str(details[x]) for x in details}
            if "beta_value" not in details:
                new_dict["beta_value"] = "-1"
            response.modelDetails[model] = json.dumps(new_dict).encode('utf-8')
            
        response.models.extend(total_list)
        return response

    def TrainModel(self, request, context):
        modelName: str = request.modelName
        epochs: int = request.trainingEpochs
        latentSize: int = request.latentSize
        datasetName: str = request.datasetName 
        beta: int = request.beta
        model_type: str = request.modelType
        if request.modelType == "":
            model_type = "cvae"
        
        temp = ModelGenerator(self.data_loaders)
        response = modelGenerator_pb2.TrainModelResponse()
        response.succesful = True
        try:
            temp.set_latent_size(latent_size=latentSize)
            temp.train_model(epochs=epochs, latent_vector=latentSize, dataset=datasetName.lower(), model_type=model_type, beta=beta, name=modelName)
            response.message = temp.saveModel(SAVED_MODELS_DIR+modelName)
        except Exception as e:
            response.succesful = False
            response.message = str(e)

        return response


    def LoadModel(self, request, context):
        response = modelGenerator_pb2.LoadModelResponse()
        response.succesful = True
        try:
            self.m_generator.loadModel(request.modelName)
        except Exception as e:
            print(e)
            response.succesful = False
            response.message = f"Unable to load {request.modelName}, try another model."
            return response
        response.message = "Successful: Latent size "+ str(self.m_generator.model.retrieve_latent_size())
        return response

    def CurrentModel(self, request, context):
        response = modelGenerator_pb2.CurrentModelResponse()
        details = self.m_generator.model.details()
        response.modelName = details['name']
        for x in details:
            response.modelDetails[x] = str(details[x])

        return response

async def serve():
    server = grpc.aio.server(futures.ThreadPoolExecutor(max_workers=100))
    modelGenerator_pb2_grpc.add_ModelGenerationServicer_to_server(ModelGenerationServicer(), server)
    server.add_insecure_port("[::]:50051")
    print(f"Model Generator Server Started. Listening on port 50051")
    await server.start()
    await server.wait_for_termination()

if __name__ == "__main__":
    # m_generator.loadModel("./defaultModels/Epochs-50.pt")
    asyncio.run(serve())