using System;
using System.Collections;
using System.Text;
using UnityEngine;
using UnityEngine.Networking;
using UnityEngine.UI;

public class Login : MonoBehaviour
{
    public Text username;
    public Text password;
    
    public void login()
    {
        StartCoroutine(MainLogin());
    }
    
    /**
     * Login Function
     * Post request to backend server
     */
    public IEnumerator MainLogin()
    {
        /*
        print("here1");
        JSON obj = new JSON();
        obj.username = username.text;
        obj.password = password.text;

        string json = JsonUtility.ToJson(obj);

        UnityWebRequest webRequest = new UnityWebRequest("http://localhost:3000/user/login/", "POST");
        byte[] encodedPayload = new System.Text.UTF8Encoding().GetBytes(json);
        webRequest.uploadHandler = (UploadHandler) new UploadHandlerRaw(encodedPayload);
        webRequest.downloadHandler = (DownloadHandler) new DownloadHandlerBuffer();
        webRequest.SetRequestHeader("Content-Type", "application/json");
        webRequest.SetRequestHeader("cache-control", "no-cache");
         
        UnityWebRequestAsyncOperation requestHandel = webRequest.SendWebRequest();
        requestHandel.completed += delegate(AsyncOperation pOperation) {
            Debug.Log(webRequest.responseCode);
            Debug.Log(webRequest.downloadHandler.text);
        };
        */
        print("starting request");
        JSON js = new JSON();
        js.username = "mattwoodx";
        js.password = "password";
       // string json = "{ \"username\": \"mattwoodx\", \"password\": \"password\"}";

        //byte[] sendForm = System.Text.Encoding.UTF8.GetBytes(JsonUtility.ToJson(js));

        var request = new UnityWebRequest ("http://localhost:3000/user/login/", "POST");
        byte[] bodyRaw = Encoding.UTF8.GetBytes(JsonUtility.ToJson(js));
        request.uploadHandler = (UploadHandler) new UploadHandlerRaw(bodyRaw);
        request.downloadHandler = (DownloadHandler) new DownloadHandlerBuffer();
        request.SetRequestHeader("Content-Type", "application/json");
        print("here");
        yield return request.SendWebRequest();
        print("done");
        if (request.error != null)
        {
            print("error");
        }
        else
        {
            print("complete");
        }
    }
        
    

    [Serializable]
    public class JSON
    {
        public string username;
        public string password;
    }
    


}
