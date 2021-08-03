using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Text;
using System.Threading.Tasks;
using Google.Protobuf.Collections;
using UnityEngine;
using Grpc.Core;
using Model;
using UnityEngine.UI;

public class GRPCClient : MonoBehaviour
{
    /**
     * GRPC Client Variables
     */
    private readonly ModelController.ModelControllerClient client;
    private readonly Channel channel;
    private readonly string server = "127.0.0.1:3001";

    internal GRPCClient()
    {
        channel = new Channel(server, ChannelCredentials.Insecure);
        client = new ModelController.ModelControllerClient(channel);
    }

    public async Task HandleCoords(Material mat, GameObject plane)
    {
        double[] arr1 = {1.0, 0.0, 0.0};

        try
        {
            var requests = new List<RequestDto>
            {
                Request(arr1)
            };

            using (var call = client.HandleCoords())
            {
                var responseReaderTask = Task.Run(async () =>
                {
                    while (await call.ResponseStream.MoveNext())
                    {
                        var note = call.ResponseStream.Current;
                        print(note.Data.Length);

                        byte [] bytes = note.Data.ToByteArray();
                        LoadNewTexture(bytes, mat, plane);

                    }
                    
                    
                });

                foreach (RequestDto request in requests)
                {
                    print("Request: " + request.Data);

                    await call.RequestStream.WriteAsync(request);
                }

                await call.RequestStream.CompleteAsync();
                await responseReaderTask;

                print("Finished");
            }

        }
        catch (RpcException e)
        {
            print("GRPC Failed");
        }
    }

    public void LoadNewTexture(byte [] arr, Material mat, GameObject plane)
    {
        var tex = new Texture2D(1, 1);
                       
        tex.LoadImage(arr);
        print("image loaded");
        mat.mainTexture = tex;

        MeshRenderer mr = plane.GetComponent<MeshRenderer>();
        mr.material = mat;

        print("material changed");
    }
    

    public RequestDto Request(double[] coords)
    {
        return new RequestDto
        {
            Data = {coords}
        };
    }
    
    
}
