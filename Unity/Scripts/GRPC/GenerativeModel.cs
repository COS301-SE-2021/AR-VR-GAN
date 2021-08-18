using System.Collections.Generic;
using System.Threading.Tasks;
using UnityEngine;
using Grpc.Core;
using Model;
using ModelGenerator;

public class GenerativeModel
{
    /**
     * GRPC Client Variables
     */
    //private readonly ModelController.ModelControllerClient client;
    private readonly ModelGeneration.ModelGenerationClient clientPy;
    private readonly Channel channel;
    //private readonly string server = "127.0.0.1:3001";
    private readonly string pythonServer = "127.0.0.1:50051";
    private byte [] bytes;
    private Texture2D tex;

    internal GenerativeModel()
    {
        channel = new Channel(pythonServer, ChannelCredentials.Insecure);
        //client = new ModelController.ModelControllerClient(channel);
        clientPy = new ModelGeneration.ModelGenerationClient(channel);
    }

    public async void FetchImagePython(GameObject plane, GameObject camera)
    {
        //print("Starting python script");
        Vector3 coords = camera.transform.position;
        float[] arr1 = {(coords.x+5)/10, (coords.y)/2, (coords.z+5)/10};
        //print(coords);
        //double [] arr1 = {1.0, 1.3, 1.2};

        //RequestDto arr1 = new RequestDto(Request(dataPoints));

        using (var call = clientPy.GenerateImage())
        {
            //var response = client.RunPython();

            var requests = new List<ImageRequest>
            {
                RequestPy(arr1)
            };

            var responseReaderTask = Task.Run(async () =>
            {
                while (await call.ResponseStream.MoveNext())
                {
                    var note = call.ResponseStream.Current;
                    //print("stream received");
                    //print(note.Image[0]);
                    //System.IO.File.WriteAllBytes("./Assets/Scripts/GRPC/image.jpg", note.Data.ToByteArray());

                    //String [] byt = note.Data.Split(',');
                    //print(note.Data[0]);
                    bytes = note.Image.ToByteArray();


                // //MemoryStream ms = new MemoryStream(bytes);

                }
            });

            foreach (ImageRequest request in requests)
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

            //print("done");
        }

    }
    
    public RequestDto Request(double[] coords)
    {
        return new RequestDto
        {
            Data = {coords}
        };
    }
    
    public ImageRequest RequestPy(float[] coords)
    {
        return new ImageRequest()
        {
            Vector = { coords }
        };
    }
}
