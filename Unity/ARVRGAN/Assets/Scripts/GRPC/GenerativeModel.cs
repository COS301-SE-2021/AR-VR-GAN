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

public class GenerativeModel : MonoBehaviour
{
    /**
     * GRPC Client Variables
     */
    private readonly ModelController.ModelControllerClient client;
    private readonly Channel channel;
    private readonly string server = "127.0.0.1:3001";

    internal GenerativeModel()
    {
        channel = new Channel(server, ChannelCredentials.Insecure);
        client = new ModelController.ModelControllerClient(channel);
    }

    public async Task FetchImagePython()
    {
        double [] dataPoints = {2, 3, 4};

        RequestDto request = new RequestDto(Request(dataPoints));

        using (var call = client.RunPython())
        {
            var response = client.RunPython();

            var responseReaderTask = Task.Run(async () =>
            {
                while (await call.ResponseStream.MoveNext())
                {
                    var note = call.ResponseStream.Current;
                    print(note.Data);
                   
                    byte[] bytes = Encoding.ASCII.GetBytes(note.Data);

                    //MemoryStream ms = new MemoryStream(bytes);
                   
                    //System.IO.File.WriteAllBytes("./Assets/Scripts/GRPC/image.jpg", bytes);
                    //byte [] bytes = note.Data.ToByteArray();
                    //LoadNewTexture(bytes, mat, plane);

                }
            });

            await call.RequestStream.WriteAsync(request);
            await call.RequestStream.CompleteAsync();
        }

    }
    
    public RequestDto Request(double[] coords)
    {
        return new RequestDto
        {
            Data = {coords}
        };
    }
}
