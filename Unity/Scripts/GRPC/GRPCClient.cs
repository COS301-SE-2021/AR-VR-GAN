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

public class GRPCClient
{
    /**
     * GRPC Client Variables
     */
    private readonly ModelController.ModelControllerClient client;
    private readonly Channel channel;
    private readonly string server = "127.0.0.1:3001";
    //private Texture2D tex;
    private byte [] bytes;
    public GRPCClient()
    {
        channel = new Channel(server, ChannelCredentials.Insecure);
        client = new ModelController.ModelControllerClient(channel);
    }

    public async Task<byte[]> HandleCoords(Vector3 coords)
    {
        //Vector3 coords = camera.transform.position;
        double[] arr1 = {coords.x, coords.y, coords.z};
        //double[] arr1 = { 0.5, 0.6, 0.7};

        try
        {
            var requests = new List<RequestDto>
            {
                Request(arr1)
            };

            using (var call = client.Proxy())
            {
                var responseReaderTask = Task.Run(async () =>
                {
                    while (await call.ResponseStream.MoveNext())
                    {
                        var note = call.ResponseStream.Current;
                        //print(note.Data.Length);
                        
                        bytes = note.Data.ToByteArray();
                        //System.IO.File.WriteAllBytes("./Assets/Scripts/GRPC/image.jpg", bytes);

                    }
                    
                    
                });

                foreach (RequestDto request in requests)
                {
                    //print("Request: " + request.Data);

                    await call.RequestStream.WriteAsync(request);
                }

                await call.RequestStream.CompleteAsync();
                await responseReaderTask;

                

                //tex = new Texture2D(2, 2);
                //tex.LoadImage(bytes);
                //plane.GetComponent<Renderer>().material.mainTexture = tex;
                //print("Finished");
            }

        }
        catch (RpcException e)
        {
            Debug.Log("GRPC Failed");
        }

        return bytes;
    }
/*
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
*/

    public byte[] returnByteArray()
    {
        return bytes;
    }
    

    public RequestDto Request(double[] coords)
    {
        return new RequestDto
        {
            Data = {coords}
        };
    }
    
    
}
