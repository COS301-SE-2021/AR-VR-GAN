# import grpc
# from concurrent import futures

# import generativeModelServer.modelGenerator_pb2_grpc
# from generativeModelServer.server import ModelGenerationServicer

# server = grpc.server(futures.ThreadPoolExecutor(max_workers=20))
# generativeModelServer.modelGenerator_pb2_grpc.add_ModelGenerationServicer_to_server(ModelGenerationServicer(), server)
# port_number = server.add_insecure_port('[::]:50051') # Change this to secure_port when we are not in development
# print(f"Model Generator Server Started. Listening on port {port_number}")
# server.start()
# server.wait_for_termination()