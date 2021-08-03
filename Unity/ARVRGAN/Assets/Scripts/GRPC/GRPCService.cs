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
    public Material mat;
    
    // Start is called before the first frame update
    void Start()
    {
        client = new GRPCClient();
        getImage(mat, plane);
    }

    public void getImage(Material mat, GameObject plane)
    {
        fetchImage(mat, plane);
    }

    public void fetchImage(Material mat, GameObject plane)
    {
        var image = client.HandleCoords(mat, plane);
    }
}
