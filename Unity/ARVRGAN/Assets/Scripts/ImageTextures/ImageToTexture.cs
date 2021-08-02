using System;
using System.Collections;
using System.Collections.Generic;
using OVRSimpleJSON;
using UnityEngine;
using UnityEngine.UI;
using UnityEngine.Networking;

public class ImageToTexture : MonoBehaviour
{
    //public InputField plane;
    // Update is called once per frame
    void Update()
    {
        PostData();
    }

    private void PostData() => StartCoroutine(PostImage());
    
    IEnumerator PostImage()
    {
        string url = "http://localhost:3000/model/testGRPC";
        WWWForm form = new WWWForm();
        float[] arr = { 1f,2f,3f};
        form.AddField("data", "[1,2,3]");

        using (UnityWebRequest request = UnityWebRequest.Post(url, form))
        {
            //request.SetRequestHeader("Content-Type", "application/json");
            yield return request.SendWebRequest();
            if (request.isNetworkError || request.isHttpError)
                print("Server Connection Error");
            else
                print(request.downloadHandler.text); 
        }
        
    }
    
    
    
}
