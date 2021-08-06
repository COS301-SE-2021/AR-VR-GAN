using System;
using System.Collections;
using System.Collections.Generic;
using System.IO;
using System.Linq;
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
    private byte [] bytes;

    internal GenerativeModel()
    {
        channel = new Channel(server, ChannelCredentials.Insecure);
        client = new ModelController.ModelControllerClient(channel);
    }

    public async void FetchImagePython()
    {
        print("Starting python script");
        double [] dataPoints = {1.0, 1.3, 1.2};

        RequestDto request = new RequestDto(Request(dataPoints));

        using (var call = client.RunPython())
        {
            var response = client.RunPython();

            var responseReaderTask = Task.Run(async () =>
            {
                while (await call.ResponseStream.MoveNext())
                {
                    var note = call.ResponseStream.Current;
                    //print("stream received");
                    print(note.Data);

                    String [] byt = note.Data.Split(',');
                    //print(note.Data[0]);
                    bytes = byt.Select(byte.Parse).ToArray();

                // //MemoryStream ms = new MemoryStream(bytes);

                }
            });

            await call.RequestStream.WriteAsync(request);
            await call.RequestStream.CompleteAsync();
            await responseReaderTask;
            print(bytes);
            //System.IO.File.WriteAllBytes("./Assets/Scripts/GRPC/image.jpg", bytes);
            //         //byte [] bytes = note.Data.ToByteArray();
            //         //LoadNewTexture(bytes, mat, plane);

            print("done");
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
