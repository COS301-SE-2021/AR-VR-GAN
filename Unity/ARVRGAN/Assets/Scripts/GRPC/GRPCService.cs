using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using UnityEngine.UI;

public class GRPCService : MonoBehaviour
{
    //private Image;
    private GRPCClient client;
    //public Material imgPlane;
    public GameObject plane;
    
    // Start is called before the first frame update
    void Start()
    {
        plane.GetComponent<Renderer>().material.color = Color.cyan;
        client = new GRPCClient();
        fetchImage(plane);
        //GenerativeModel gm = new GenerativeModel();
        //gm.FetchImagePython();
    }

    public void fetchImage(GameObject plane)
    {
        client.HandleCoords(plane);
    }
}
