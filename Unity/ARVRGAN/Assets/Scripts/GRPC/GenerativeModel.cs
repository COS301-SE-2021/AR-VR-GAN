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
    private Texture2D tex;

    internal GenerativeModel()
    {
        channel = new Channel(server, ChannelCredentials.Insecure);
        client = new ModelController.ModelControllerClient(channel);
    }

    public async void FetchImagePython(GameObject plane, GameObject camera)
    {
        print("Starting python script");
        Vector3 coords = camera.transform.position;
        double[] arr1 = {coords.x*2.0, coords.y*3.0, coords.z*2};
        print(coords);
        //double [] arr1 = {1.0, 1.3, 1.2};

        //RequestDto arr1 = new RequestDto(Request(dataPoints));

        using (var call = client.RunPython())
        {
            //var response = client.RunPython();

            var requests = new List<RequestDto>
            {
                Request(arr1)
            };

            var responseReaderTask = Task.Run(async () =>
            {
                while (await call.ResponseStream.MoveNext())
                {
                    var note = call.ResponseStream.Current;
                    //print("stream received");
                    print(note.Data[0]);
                    //System.IO.File.WriteAllBytes("./Assets/Scripts/GRPC/image.jpg", note.Data.ToByteArray());

                    //String [] byt = note.Data.Split(',');
                    //print(note.Data[0]);
                    bytes = note.Data.ToByteArray();


                // //MemoryStream ms = new MemoryStream(bytes);

                }
            });

            foreach (RequestDto request in requests)
            {
                await call.RequestStream.WriteAsync(request);
            }
            await call.RequestStream.CompleteAsync();
            await responseReaderTask;
            tex = new Texture2D(2, 2);
            tex.LoadImage(bytes);
            //tex.Apply();
            //tex.EncodeToJPG();
            plane.GetComponent<Renderer>().material.mainTexture = tex;

            //print(bytes.ToString());
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
